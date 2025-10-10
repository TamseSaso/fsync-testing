from typing import List, Tuple

import cv2
import depthai as dai

from depthai_nodes.utils import AnnotationHelper
from .apriltag_shared import get_detector


class AprilTagAnnotationNode(dai.node.ThreadedHostNode):
    """Host node that detects AprilTags in incoming frames and outputs viewer annotations.

    Input: dai.ImgFrame
    Output: dai.Buffer (annotations built via AnnotationHelper)
    """

    def __init__(self, families: str = "tag36h11", max_tags: int = 4, quad_decimate: float = 1.0) -> None:
        super().__init__()

        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, False)])

        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, False)])

        self.families = families
        self.max_tags = max_tags
        self.quad_decimate = quad_decimate
        self._detector = None
        self._lock = None

    def build(self, frames: dai.Node.Output) -> "AprilTagAnnotationNode":
        frames.link(self.input)
        return self

    def _lazy_init(self) -> None:
        if self._detector is None:
            # Ensure reasonable decimation
            safe_decimate = self.quad_decimate if self.quad_decimate and self.quad_decimate >= 0.5 else 1.0
            self._detector, self._lock = get_detector(self.families, safe_decimate)
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
        while self.isRunning():
            frame_msg: dai.ImgFrame = self.input.get()
            bgr = frame_msg.getCvFrame()
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            with self._lock:
                detections = self._detector.detect(gray)[: self.max_tags]

            annotations = AnnotationHelper()
            img_h, img_w = gray.shape[:2]

            for det in detections:
                corners = [(float(pt[0]), float(pt[1])) for pt in det.corners]
                xmin, ymin, xmax, ymax = self._norm_rect(corners, img_w, img_h)
                annotations.draw_rectangle((xmin, ymin), (xmax, ymax))

                annotations.draw_text(
                    text=f"ID {det.tag_id}",
                    position=(min(max(0.0, xmin + 0.005), 0.98), max(0.0, ymin + 0.02)),
                    size=18,
                )

            annotations_msg = annotations.build(
                timestamp=frame_msg.getTimestamp(),
                sequence_num=frame_msg.getSequenceNum(),
            )

            self.out.send(annotations_msg)


