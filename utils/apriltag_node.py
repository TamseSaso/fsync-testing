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
    """

    def __init__(self, families: str = "tag36h11", max_tags: int = 4, quad_decimate: float = 1.0) -> None:
        super().__init__()

        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])

        self.families = families
        self.max_tags = max_tags
        self.quad_decimate = quad_decimate
        self._detector = None
        self._detector_highres = None  # persistent high-res fallback to avoid per-frame ctor/dtor

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
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
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
        while self.isRunning():
            frame_msg: dai.ImgFrame = self.input.get()
            bgr = frame_msg.getCvFrame()
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            detections = self._detector.detect(gray)[: self.max_tags]
            # Fallback: if none detected and decimate > 1.2, run a persistent high-res detector
            if not detections and (self.quad_decimate is None or self.quad_decimate > 1.2):
                if self._detector_highres is None:
                    self._detector_highres = AprilTagDetector(
                        families=self.families,
                        nthreads=2,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=True,
                        decode_sharpening=0.25,
                    )
                detections = self._detector_highres.detect(gray)[: self.max_tags]

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


