from typing import List, Tuple

import time
import cv2
import numpy as np
import depthai as dai

try:
    from pupil_apriltags import Detector as AprilTagDetector
except Exception:
    AprilTagDetector = None


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    # Robust ordering using centroid angles
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = points[order]
    # Ensure first is top-left by shifting so smallest (x+y) is first, then rotate
    sums = ordered.sum(axis=1)
    start = int(np.argmin(sums))
    ordered = np.roll(ordered, -start, axis=0)
    # Make second be top-right by checking the two options
    if ordered[1][1] > ordered[3][1]:
        # Flip order to keep clockwise TL,TR,BR,BL
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered.astype(np.float32)


def _order_tag_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """Return a single tag's corners ordered as TL, TR, BR, BL.

    The incoming corner order from the detector is not guaranteed. We re-order
    using the common approach of splitting by y (top/bottom) and then by x
    (left/right). This makes the mapping stable for downstream logic.
    """
    pts = corners.astype(np.float32).reshape(4, 2)
    # Sort by y to separate top vs bottom
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    # Sort each pair by x for left/right
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return ordered


class AprilTagWarpNode(dai.node.ThreadedHostNode):
    """Detect 4 AprilTags and publish a perspective-rectified crop as an image.

    Input: dai.ImgFrame (BGR)
    Output: dai.ImgFrame (BGR, interleaved)

    Robust like AprilTagAnnotationNode:
    - non-blocking IO queues to avoid backpressure
    - decision-margin filtering
    - high-res fallback detector when needed
    - short-term tag persistence to bridge brief dropouts
    - last-good warp hold to keep downstream nodes fed
    """

    class _SimpleDetection:
        __slots__ = ("tag_id", "center", "corners", "decision_margin")
        def __init__(self, tag_id: int, center: Tuple[float, float], corners: np.ndarray, decision_margin: float):
            self.tag_id = int(tag_id)
            self.center = (float(center[0]), float(center[1]))
            self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            self.decision_margin = float(decision_margin)

    def __init__(
        self,
        out_width: int,
        out_height: int,
        families: str = "tag36h11",
        quad_decimate: float = 1.0,
        tag_size: float = 0.1,
        z_offset: float = 0.01,
        quad_sigma: float = 0.0,
        decode_sharpening: float = 0.25,
        refine_edges: bool = True,
        decision_margin: float = 5.0,
        persistence_seconds: float = 2.0,
        hold_last_warp_seconds: float = 1.0,
        # --- new robustness knobs ---
        emergency_low_dm_factor: float = 0.6,
        panic_after_frames: int = 8,
        max_upscale: float = 2.0,
        enable_bilateral_preproc: bool = True,
        enable_tophat_preproc: bool = True,
        enable_adaptive_thresh: bool = True,
    ) -> None:
        super().__init__()

        # IO setup: non-blocking queues with small buffers to keep camera flowing
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        try:
            self.input.setBlocking(False)
            self.input.setQueueSize(1)
        except AttributeError:
            pass

        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        try:
            self.out.setBlocking(False)
            self.out.setQueueSize(2)
        except AttributeError:
            pass

        self.out_w = int(out_width)
        self.out_h = int(out_height)
        self.families = families
        self.quad_decimate = quad_decimate if quad_decimate is not None and quad_decimate >= 0.5 else 1.0
        self.quad_sigma = quad_sigma
        self.decode_sharpening = decode_sharpening
        self.refine_edges = refine_edges
        self.decision_margin = float(decision_margin)
        self.persistence_seconds = float(persistence_seconds)
        self.hold_last_warp_seconds = float(hold_last_warp_seconds)

        # Robustness toggles/params
        self.enable_clahe = True
        self.enable_unsharp = True
        self.enable_multiscale = True
        # broaden variant scales for extreme cases; values > 1.0 imply upscaling
        self.variant_scales = (1.25, 1.5, 1.75, 2.0)
        self.max_upscale = float(max_upscale)
        self.enable_bilateral_preproc = bool(enable_bilateral_preproc)
        self.enable_tophat_preproc = bool(enable_tophat_preproc)
        self.enable_adaptive_thresh = bool(enable_adaptive_thresh)
        self._detector_sharp = None  # persistent sharper-decoder detector

        self.tag_size = tag_size  # Tag size in meters
        self.z_offset = z_offset  # Z-axis offset in meters
        # Margins/paddings tweak the final crop to match panel framing
        self.margin = 0.01
        self.padding_left = -0.008
        self.padding_right = -0.005
        self.bottom_right_y_offset = 0.016
        self.bottom_y_offset = 0.01

        self._detector = None
        self._detector_highres = None  # persistent fallback detector (decimate=1.0)
        self._detector_ultra = None    # persistent panic-mode detector (decimate=0.5)

        # Persistence structures
        # tag_id -> {"corners": np.ndarray(4,2), "center": (x,y), "timestamp": float}
        self._remembered_tags: dict[int, dict] = {}

        # Last successful warped frame for brief hold when tags blink out
        self._last_warp_frame: np.ndarray | None = None
        self._last_warp_time: float = 0.0

        # Track consecutive no-detection frames to trigger panic mode
        self._no_detect_streak: int = 0
        self.emergency_low_dm_factor = float(emergency_low_dm_factor)
        self.panic_after_frames = int(panic_after_frames)

        # Camera intrinsics (approximate) for pose nudging
        self.camera_matrix = np.array([
            [1400.0, 0.0, 960.0],
            [0.0, 1400.0, 540.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    def _preprocess_variants(self, gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Generate a set of robust grayscale variants.
        We keep everything 8-bit and conservative to avoid artifacts.
        Returns list of (name, image) pairs.
        """
        variants: list[tuple[str, np.ndarray]] = [("raw", gray)]

        # CLAHE + gamma + unsharp (existing pipeline)
        try:
            g2 = self._apply_clahe_and_gamma(gray)
            variants.append(("clahe", g2))
        except Exception:
            g2 = gray

        # Bilateral smoothing preserves edges while denoising speckle
        if self.enable_bilateral_preproc:
            try:
                gb = cv2.bilateralFilter(g2, 7, 40, 40)
                variants.append(("bilateral", gb))
            except Exception:
                pass

        # Top-hat / black-hat to fight non-uniform illumination and glare
        if self.enable_tophat_preproc:
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                top = cv2.morphologyEx(g2, cv2.MORPH_TOPHAT, kernel)
                black = cv2.morphologyEx(g2, cv2.MORPH_BLACKHAT, kernel)
                # Boost bright small-scale structures (like tag edges) and suppress halos
                boosted = cv2.addWeighted(g2, 1.0, top, 0.8, 0)
                black_scaled = cv2.convertScaleAbs(black, alpha=0.5, beta=0)
                boosted = cv2.subtract(boosted, black_scaled)
                variants.append(("tophat", boosted))
            except Exception:
                pass

        # Binary adaptive threshold variant can help on washed-out scenes
        if self.enable_adaptive_thresh:
            try:
                at = cv2.adaptiveThreshold(
                    g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
                )
                variants.append(("athresh", at))
            except Exception:
                pass

        return variants

    def _ensure_detectors(self) -> None:
        """Lazily (and idempotently) construct detector variants."""
        self._lazy_init()

        # Sharper decoder variant (more aggressive decode sharpening)
        if self._detector_sharp is None:
            try:
                self._detector_sharp = AprilTagDetector(
                    families=self.families,
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=int(self.refine_edges),
                    decode_sharpening=max(0.5, float(self.decode_sharpening) * 2.0),
                )
            except Exception:
                self._detector_sharp = None

        # High-res fallback (quad_decimate = 1.0)
        if self._detector_highres is None:
            try:
                self._detector_highres = AprilTagDetector(
                    families=self.families,
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=self.quad_sigma,
                    refine_edges=int(self.refine_edges),
                    decode_sharpening=self.decode_sharpening,
                )
            except Exception:
                self._detector_highres = None

        # Ultra mode (panic): even higher res (decimate=0.5)
        if self._detector_ultra is None:
            try:
                self._detector_ultra = AprilTagDetector(
                    families=self.families,
                    nthreads=2,
                    quad_decimate=0.5,
                    quad_sigma=self.quad_sigma,
                    refine_edges=int(self.refine_edges),
                    decode_sharpening=max(0.75, float(self.decode_sharpening) * 2.0),
                )
            except Exception:
                self._detector_ultra = None

    def _detect_with_engine(self, img: np.ndarray, engine) -> list["AprilTagWarpNode._SimpleDetection"]:
        try:
            dets = engine.detect(img)
        except Exception:
            dets = None
        return self._wrap_detections(dets, 1.0, 1.0)

    def _detect_scaled(self, img: np.ndarray, scale: float, engine) -> list["AprilTagWarpNode._SimpleDetection"]:
        try:
            g_up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return []
        try:
            dets = engine.detect(g_up)
        except Exception:
            dets = None
        return self._wrap_detections(dets, scale, scale)

    def build(self, frames: dai.Node.Output) -> "AprilTagWarpNode":
        frames.link(self.input)
        return self

    def _lazy_init(self) -> None:
        if self._detector is None:
            if AprilTagDetector is None:
                raise RuntimeError("pupil-apriltags is not installed. Add it to requirements.txt")
            safe_decimate = self.quad_decimate if self.quad_decimate and self.quad_decimate >= 0.5 else 1.0
            self._detector = AprilTagDetector(
                families=self.families,
                nthreads=2,
                quad_decimate=safe_decimate,
                quad_sigma=self.quad_sigma,
                refine_edges=int(self.refine_edges),
                decode_sharpening=self.decode_sharpening,
            )
            self.quad_decimate = float(safe_decimate)

    def _estimate_pose_and_apply_offset(self, corners_2d: np.ndarray) -> np.ndarray:
        """Estimate pose of an AprilTag and apply z-offset to its corners.

        Accepts the raw 2D corners (BL, BR, TR, TL from detector), computes a pose,
        applies z-offset, and returns adjusted corners in 2D.
        """
        half_size = float(self.tag_size) / 2.0
        object_points = np.array([
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, corners_2d.astype(np.float32), self.camera_matrix, self.dist_coeffs
        )
        if not success:
            return corners_2d.astype(np.float32)

        tvec_offset = tvec.copy()
        tvec_offset[2] += float(self.z_offset)
        offset_corners_2d, _ = cv2.projectPoints(
            object_points, rvec, tvec_offset, self.camera_matrix, self.dist_coeffs
        )
        return offset_corners_2d.reshape(-1, 2).astype(np.float32)

    def _dst_quad(self) -> np.ndarray:
        # Calculate margin/padding in pixels
        margin_pixels = self.margin * self.out_h
        padding_left_pixels = self.padding_left * self.out_w
        padding_right_pixels = self.padding_right * self.out_w
        bottom_y_offset_pixels = self.bottom_y_offset * self.out_h
        br_y_offset_pixels = self.bottom_right_y_offset * self.out_h
        return np.array(
            [
                [padding_left_pixels, margin_pixels],
                [self.out_w - 1.0 - padding_right_pixels, margin_pixels],
                [self.out_w - 1.0 - padding_right_pixels, self.out_h - 1.0 - margin_pixels + bottom_y_offset_pixels + br_y_offset_pixels],
                [padding_left_pixels, self.out_h - 1.0 - margin_pixels + bottom_y_offset_pixels],
            ],
            dtype=np.float32,
        )

    def _apply_clahe_and_gamma(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive contrast + gentle sharpening for low light / glare.
        Keeps output 8-bit and avoids over-amplifying noise.
        """
        g = gray
        try:
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            g = clahe.apply(g)
        except Exception:
            pass

        # Adaptive gamma: brighten dark frames, tame overexposed ones
        mean = float(np.mean(g)) if g.size else 128.0
        gamma = 1.0
        if mean < 70.0:
            gamma = 0.7
        elif mean > 180.0:
            gamma = 1.3
        if abs(gamma - 1.0) > 0.05:
            inv = 1.0 / max(gamma, 1e-6)
            lut = (np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255)).astype(np.uint8)
            g = cv2.LUT(g, lut)

        # Light denoise + unsharp mask (optional)
        g = cv2.medianBlur(g, 3)
        if self.enable_unsharp:
            blur = cv2.GaussianBlur(g, (0, 0), 1.0)
            g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
        return g

    def _wrap_detections(self, dets, sx: float = 1.0, sy: float = 1.0) -> List["AprilTagWarpNode._SimpleDetection"]:
        out = []
        if dets is None:
            return out
        for det in dets:
            try:
                c = np.asarray(det.corners, dtype=np.float32).reshape(4, 2).copy()
                c[:, 0] /= max(sx, 1e-9)
                c[:, 1] /= max(sy, 1e-9)
                cx, cy = det.center
                cx = float(cx) / max(sx, 1e-9)
                cy = float(cy) / max(sy, 1e-9)
                out.append(self._SimpleDetection(det.tag_id, (cx, cy), c, getattr(det, "decision_margin", 0.0)))
            except Exception:
                continue
        return out

    def _dedupe_best(self, lists: List[List["AprilTagWarpNode._SimpleDetection"]], min_dm: float | None = None) -> List["AprilTagWarpNode._SimpleDetection"]:
        if min_dm is None:
            min_dm = self.decision_margin
        best: dict[int, "AprilTagWarpNode._SimpleDetection"] = {}
        for L in lists:
            for d in L:
                if d.decision_margin < float(min_dm):
                    continue
                cur = best.get(d.tag_id)
                if cur is None or d.decision_margin > cur.decision_margin:
                    best[d.tag_id] = d
        return list(best.values())

    def _detect_apriltags(self, gray: np.ndarray) -> List["AprilTagWarpNode._SimpleDetection"]:
        """Bulletproof multi-variant detection.

        Strategy:
        - Try several carefully chosen pre-processing variants (CLAHE, bilateral, tophat, adaptive threshold).
        - For each variant, run base/sharp/high-res engines.
        - Multi-scale upscaling passes for tiny/far tags.
        - If we fail for several consecutive frames, enter a panic mode that relaxes
          the decision margin and uses an ultra-high-resolution detector.
        """
        self._ensure_detectors()

        min_dm = float(self.decision_margin)
        results: List[List["AprilTagWarpNode._SimpleDetection"]] = []
        variants = self._preprocess_variants(gray)

        # Engines in descending order of speed; we will early-exit on success
        engines = [self._detector, self._detector_sharp, self._detector_highres]
        engines = [e for e in engines if e is not None]

        # 1) Straight detects on each preproc variant
        for _, img in variants:
            for eng in engines:
                dets = self._detect_with_engine(img, eng)
                if dets:
                    results.append(dets)
                    dedup = self._dedupe_best(results, min_dm)
                    if len(dedup) >= 4:
                        self._no_detect_streak = 0
                        return dedup

        # 2) Multi-scale upscaling passes (helps tiny/far tags)
        if self.enable_multiscale:
            pref_engine = self._detector_sharp or self._detector_highres or self._detector
            if pref_engine is not None:
                for _, img in variants:
                    for s in self.variant_scales:
                        if s <= 1.01:
                            continue
                        if s > self.max_upscale:
                            continue
                        dets = self._detect_scaled(img, float(s), pref_engine)
                        if dets:
                            results.append(dets)
                            dedup = self._dedupe_best(results, min_dm)
                            if len(dedup) >= 4:
                                self._no_detect_streak = 0
                                return dedup

        # If we got here we didn't get 4 solid tags this frame
        self._no_detect_streak += 1

        # 3) Panic mode: relax margin + ultra detector on strongest variants
        if self._no_detect_streak >= self.panic_after_frames:
            emergency_dm = max(2.0, float(self.decision_margin) * float(self.emergency_low_dm_factor))
            strong_variants = []
            # Prefer CLAHE / bilateral / tophat if present
            for name, img in variants:
                if name in ("clahe", "bilateral", "tophat"):
                    strong_variants.append(img)
            if not strong_variants:
                strong_variants = [v for _, v in variants]

            # Run ultra detector at native and max upscale
            if self._detector_ultra is not None:
                for img in strong_variants[:2]:  # cap work
                    dets = self._detect_with_engine(img, self._detector_ultra)
                    if dets:
                        results.append(dets)
                    if self.enable_multiscale and self.max_upscale > 1.01:
                        s = float(min(self.max_upscale, 2.0))
                        dets2 = self._detect_scaled(img, s, self._detector_ultra)
                        if dets2:
                            results.append(dets2)

            dedup = self._dedupe_best(results, emergency_dm)
            if len(dedup) >= 4:
                # keep streak but reset after success so we don't stay in panic forever
                self._no_detect_streak = 0
                return dedup

        # Return best we managed to find (possibly < 4); caller handles persistence/hold
        return self._dedupe_best(results, min_dm)

    def _send_img(self, bgr: np.ndarray, src: dai.ImgFrame) -> None:
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888i)
        img.setWidth(self.out_w)
        img.setHeight(self.out_h)
        img.setData(bgr.tobytes())
        img.setSequenceNum(src.getSequenceNum())
        img.setTimestamp(src.getTimestamp())
        self.out.send(img)

    def run(self) -> None:
        self._lazy_init()
        dst_quad = self._dst_quad()

        while self.isRunning():
            # Drain to the latest frame (non-blocking)
            frame_msg: dai.ImgFrame | None = None
            try:
                while True:
                    try:
                        m = self.input.tryGet()
                    except AttributeError:
                        if not self.input.has():
                            break
                        m = self.input.get()
                    if m is None:
                        break
                    frame_msg = m
            except Exception:
                frame_msg = None

            if frame_msg is None:
                time.sleep(0.001)
                continue

            bgr = frame_msg.getCvFrame()
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            now = time.time()

            # 1) Detect tags (robust multi-variant pipeline)
            detections = self._detect_apriltags(gray)

            # 2) Update persistence from current detections
            for det in detections:
                # Pose-adjusted corners for accuracy, then order
                offset_corners = self._estimate_pose_and_apply_offset(np.array(det.corners, dtype=np.float32))
                ordered_tag = _order_tag_corners_tl_tr_br_bl(offset_corners)
                self._remembered_tags[int(det.tag_id)] = {
                    "corners": ordered_tag,
                    "center": tuple(map(float, det.center)),
                    "timestamp": now,
                }

            # Remove expired remembered tags
            expired = [tid for tid, info in self._remembered_tags.items() if (now - float(info.get("timestamp", 0.0))) > self.persistence_seconds]
            for tid in expired:
                del self._remembered_tags[tid]

            # 3) Build candidate set using current + remembered tags
            # Prefer the freshest center coordinates for selection
            candidates_centers = []
            candidates_entries = []

            # From current frame first
            for det in detections:
                tid = int(det.tag_id)
                info = self._remembered_tags.get(tid)
                if info is None:
                    continue
                candidates_centers.append(np.array(info["center"], dtype=np.float32))
                candidates_entries.append((tid, info))

            # Then add any remaining remembered tags (not already included)
            for tid, info in self._remembered_tags.items():
                if all(t != tid for t, _ in candidates_entries):
                    candidates_centers.append(np.array(info["center"], dtype=np.float32))
                    candidates_entries.append((tid, info))

            if len(candidates_entries) >= 4:
                centers = np.stack(candidates_centers, axis=0)
                c_mean = centers.mean(axis=0)
                dists = np.linalg.norm(centers - c_mean, axis=1)
                idx = np.argsort(dists)[-4:]
                chosen_infos = [candidates_entries[int(i)][1] for i in idx]

                # For each chosen tag, take the inner corner relative to global centroid
                corner_candidates = []
                for info in chosen_infos:
                    ordered_tag = np.asarray(info["corners"], dtype=np.float32)
                    tag_center = np.asarray(info["center"], dtype=np.float32)
                    is_left = tag_center[0] < c_mean[0]
                    is_top = tag_center[1] < c_mean[1]
                    if is_top and is_left:
                        chosen_corner = ordered_tag[1]  # TR
                    elif is_top and not is_left:
                        chosen_corner = ordered_tag[0]  # TL
                    elif not is_top and not is_left:
                        chosen_corner = ordered_tag[3]  # BL
                    else:
                        chosen_corner = ordered_tag[2]  # BR
                    corner_candidates.append(chosen_corner)

                src_pts = _order_points_clockwise(np.array(corner_candidates, dtype=np.float32))
                area = cv2.contourArea(src_pts.reshape(-1, 1, 2))
                if area >= 10.0:
                    H = cv2.getPerspectiveTransform(src_pts, dst_quad)
                    warped = cv2.warpPerspective(bgr, H, (self.out_w, self.out_h))
                    self._last_warp_frame = warped
                    self._last_warp_time = now
                    self._send_img(warped, frame_msg)
                    continue

            # 4) If not enough tags: briefly hold last good warp to keep stream alive
            if (
                self._last_warp_frame is not None and
                (now - self._last_warp_time) <= self.hold_last_warp_seconds
            ):
                self._send_img(self._last_warp_frame, frame_msg)
                continue

            # Otherwise, yield and wait for more frames/tags (no output this cycle)
            time.sleep(0.001)
            continue