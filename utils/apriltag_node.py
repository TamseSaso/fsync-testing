from typing import List, Tuple

import cv2
import depthai as dai

from depthai_nodes.utils import AnnotationHelper

try:
    from pupil_apriltags import Detector as AprilTagDetector
except Exception as exc:  # pragma: no cover
    AprilTagDetector = None


class AprilTagAnnotationNode(dai.node.ThreadedHostNode):
    """Host node that detects AprilTags in incoming frames and outputs viewer annotations.

    Input: dai.ImgFrame
    Output: dai.Buffer (annotations built via AnnotationHelper)

    Optionally waits to emit annotations until a specified number of unique tags are detected in the current frame.
    """

    def __init__(
        self, 
        families: str = "tag36h11", 
        max_tags: int = 4, 
        quad_decimate: float = 0.8,
        quad_sigma: float = 0.5,
        decode_sharpening: float = 0.6,
        refine_edges: bool = True,
        decision_margin: float = 5.0,
        persistence_seconds: float = 240.0,
        wait_for_n_tags: int | None = 4
    ) -> None:
        super().__init__()

        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        # Avoid back‑pressure on the camera stream if tag processing is slow
        self.input.setBlocking(False)
        self.input.setQueueSize(1)

        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        # Do not block on output (drop if the consumer is late)
        self.out.setBlocking(False)
        self.out.setQueueSize(2)

        self.families = families
        self.max_tags = max_tags
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.decode_sharpening = decode_sharpening
        self.refine_edges = refine_edges
        self.decision_margin = decision_margin
        self.persistence_seconds = persistence_seconds
        self.wait_for_n_tags = wait_for_n_tags
        self._detector = None
        self._detector_highres = None  # persistent high-res fallback to avoid per-frame ctor/dtor
        
        # Tag persistence: remember last known positions
        # Format: {tag_id: {"corners": [...], "timestamp": time, "center": (x, y)}}
        self._remembered_tags = {}
        self._tag_first_seen = {}  # Track when each tag was first detected

    def build(self, frames: dai.Node.Output) -> "AprilTagAnnotationNode":
        frames.link(self.input)
        return self

    def _lazy_init(self) -> None:
        if self._detector is None:
            if AprilTagDetector is None:
                raise RuntimeError(
                    "pupil-apriltags is not installed. Add 'pupil-apriltags' to requirements.txt."
                )
            # Ensure reasonable decimation
            safe_decimate = self.quad_decimate if self.quad_decimate and self.quad_decimate >= 0.5 else 1.0
            self._detector = AprilTagDetector(
                families=self.families,
                nthreads=2,
                quad_decimate=safe_decimate,
                quad_sigma=self.quad_sigma,
                refine_edges=int(self.refine_edges),
                decode_sharpening=self.decode_sharpening,
            )
            self.quad_decimate = safe_decimate

    @staticmethod
    def _norm_rect(corners: List[Tuple[float, float]], width: int, height: int) -> Tuple[float, float, float, float]:
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        xmin = max(0.0, min(xs) / max(1, width))
        ymin = max(0.0, min(ys) / max(1, height))
        xmax = min(1.0, max(xs) / max(1, width))
        ymax = min(1.0, max(ys) / max(1, height))
        return xmin, ymin, xmax, ymax

    def run(self) -> None:
        self._lazy_init()
        import time
        
        while self.isRunning():
            # Drain to latest frame (non-blocking)
            frame_msg: dai.ImgFrame | None = None
            try:
                # Prefer tryGet() if available
                while True:
                    try:
                        m = self.input.tryGet()
                    except AttributeError:
                        # Fallback if tryGet() isn't available on this binding
                        if not self.input.has():
                            break
                        m = self.input.get()
                    if m is None:
                        break
                    frame_msg = m
            except Exception:
                frame_msg = None

            if frame_msg is None:
                # No new frame right now; yield a bit to avoid busy spinning
                time.sleep(0.001)
                continue

            bgr = frame_msg.getCvFrame()
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            current_time = time.time()

            all_detections = self._detector.detect(gray)

            # Filter detections by decision_margin threshold
            detections = [det for det in all_detections if det.decision_margin >= self.decision_margin]

            # Fallback: if none detected and decimate > 1.2, run a persistent high-res detector
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
                detections = self._detector_highres.detect(gray)
                detections = [det for det in detections if det.decision_margin >= self.decision_margin]

            # If requested, wait until we see the desired number of unique tags in the *current* frame
            if self.wait_for_n_tags is not None and self.wait_for_n_tags > 0:
                unique_current_ids = {int(det.tag_id) for det in detections}
                if len(unique_current_ids) < int(self.wait_for_n_tags):
                    # Not enough tags yet; skip output this frame
                    continue

            # Update remembered tags with current detections
            detected_ids = set()
            img_h, img_w = gray.shape[:2]

            for det in detections:
                tag_id = det.tag_id
                detected_ids.add(tag_id)
                corners = [(float(pt[0]), float(pt[1])) for pt in det.corners]
                center = det.center

                self._remembered_tags[tag_id] = {
                    "corners": corners,
                    "center": center,
                    "timestamp": current_time,
                    "img_w": img_w,
                    "img_h": img_h,
                }

                if tag_id not in self._tag_first_seen:
                    self._tag_first_seen[tag_id] = current_time

            # Add persistent tags that weren't detected this frame but are still fresh
            persistent_detections = []
            expired_tags = []

            for tag_id, tag_info in self._remembered_tags.items():
                age = current_time - tag_info["timestamp"]

                if tag_id not in detected_ids:
                    if age <= self.persistence_seconds:
                        persistent_detections.append({
                            "tag_id": tag_id,
                            "corners": tag_info["corners"],
                            "center": tag_info["center"],
                            "img_w": tag_info["img_w"],
                            "img_h": tag_info["img_h"],
                            "is_persistent": True,
                        })
                    else:
                        expired_tags.append(tag_id)

            for tag_id in expired_tags:
                del self._remembered_tags[tag_id]
                if tag_id in self._tag_first_seen:
                    del self._tag_first_seen[tag_id]

            # Build annotations from both current and persistent detections
            annotations = AnnotationHelper()

            for det in detections:
                corners = [(float(pt[0]), float(pt[1])) for pt in det.corners]
                xmin, ymin, xmax, ymax = self._norm_rect(corners, img_w, img_h)
                annotations.draw_rectangle((xmin, ymin), (xmax, ymax))
                annotations.draw_text(
                    text=f"ID {det.tag_id}",
                    position=(min(max(0.0, xmin + 0.005), 0.98), max(0.0, ymin + 0.02)),
                    size=18,
                )

            for persist_det in persistent_detections:
                xmin, ymin, xmax, ymax = self._norm_rect(
                    persist_det["corners"],
                    persist_det["img_w"],
                    persist_det["img_h"],
                )
                persistent_color = (0.0, 1.0, 0.0, 0.5)
                annotations.draw_rectangle((xmin, ymin), (xmax, ymax), outline_color=persistent_color)
                annotations.draw_text(
                    text=f"ID {persist_det['tag_id']} 📍",
                    position=(min(max(0.0, xmin + 0.005), 0.98), max(0.0, ymin + 0.02)),
                    size=18,
                )

            annotations_msg = annotations.build(
                timestamp=frame_msg.getTimestamp(dai.CameraExposureOffset.END),
                sequence_num=frame_msg.getSequenceNum(),
            )

            self.out.send(annotations_msg)
