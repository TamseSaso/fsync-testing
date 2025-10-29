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

        self.tag_size = tag_size  # Tag size in meters
        self.z_offset = z_offset  # Z-axis offset in meters
        # Margins/paddings tweak the final crop to match panel framing
        self.margin = 0.01
        self.padding_left = -0.008
        self.padding_right = -0.01
        self.bottom_right_y_offset = 0.016
        self.bottom_y_offset = 0.01

        self._detector = None
        self._detector_highres = None  # persistent fallback detector

        # Persistence structures
        # tag_id -> {"corners": np.ndarray(4,2), "center": (x,y), "timestamp": float}
        self._remembered_tags: dict[int, dict] = {}

        # Last successful warped frame for brief hold when tags blink out
        self._last_warp_frame: np.ndarray | None = None
        self._last_warp_time: float = 0.0

        # Camera intrinsics (approximate) for pose nudging
        self.camera_matrix = np.array([
            [1400.0, 0.0, 960.0],
            [0.0, 1400.0, 540.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

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

            # 1) Detect tags
            detections = []
            all_dets = self._detector.detect(gray)
            # Filter by decision margin
            for det in all_dets:
                if getattr(det, "decision_margin", 0.0) >= self.decision_margin:
                    detections.append(det)

            # Optional high-res fallback if decimated detector returns nothing
            if not detections and (self.quad_decimate is None or self.quad_decimate > 1.2):
                if self._detector_highres is None:
                    self._detector_highres = AprilTagDetector(
                        families=self.families,
                        nthreads=2,
                        quad_decimate=1.0,
                        quad_sigma=self.quad_sigma,
                        refine_edges=int(self.refine_edges),
                        decode_sharpening=self.decode_sharpening,
                    )
                for det in self._detector_highres.detect(gray):
                    if getattr(det, "decision_margin", 0.0) >= self.decision_margin:
                        detections.append(det)

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