from typing import List, Tuple

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
    """

    def __init__(self, out_width: int, out_height: int, families: str = "tag36h11", quad_decimate: float = 1.0, tag_size: float = 0.1, z_offset: float = 0.01) -> None:
        super().__init__()

        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.out_w = int(out_width)
        self.out_h = int(out_height)
        self.families = families
        self.quad_decimate = quad_decimate if quad_decimate is not None and quad_decimate >= 0.5 else 1.0
        self.tag_size = tag_size  # Tag size in meters
        self.z_offset = z_offset  # Z-axis offset in meters
        self.margin = 0.008  # Hardcoded margin as fraction of height for top/bottom (1%)
        self.padding_left = 0.0  # Hardcoded left padding as fraction of width 
        self.padding_right = -0.012  # Hardcoded right padding as fraction of width
        self._detector = None
        
        # Default camera parameters for 1920x1080 resolution (approximate values)
        # These should ideally be calibrated for the specific camera
        self.camera_matrix = np.array([
            [1400.0, 0.0, 960.0],    # fx, 0, cx
            [0.0, 1400.0, 540.0],    # 0, fy, cy
            [0.0, 0.0, 1.0]          # 0, 0, 1
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no distortion

    def build(self, frames: dai.Node.Output) -> "AprilTagWarpNode":
        frames.link(self.input)
        return self

    def _lazy_init(self) -> None:
        if self._detector is None:
            if AprilTagDetector is None:
                raise RuntimeError("pupil-apriltags is not installed. Add it to requirements.txt")
            self._detector = AprilTagDetector(
                families=self.families,
                nthreads=2,
                quad_decimate=self.quad_decimate,
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
            )

    def _estimate_pose_and_apply_offset(self, detection, image_shape) -> np.ndarray:
        """Estimate pose of AprilTag and apply z-offset to its corners."""
        # Get the 2D corner points
        corners_2d = np.array(detection.corners, dtype=np.float32)
        
        # Define 3D object points for the AprilTag (centered at origin)
        half_size = self.tag_size / 2.0
        object_points = np.array([
            [-half_size, -half_size, 0.0],  # Bottom-left
            [half_size, -half_size, 0.0],   # Bottom-right
            [half_size, half_size, 0.0],    # Top-right
            [-half_size, half_size, 0.0]    # Top-left
        ], dtype=np.float32)
        
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            object_points, corners_2d, self.camera_matrix, self.dist_coeffs
        )
        
        if not success:
            return corners_2d  # Fallback to original corners if pose estimation fails
        
        # Apply z-offset by modifying the translation vector
        tvec_offset = tvec.copy()
        tvec_offset[2] += self.z_offset  # Move away from camera by z_offset
        
        # Project the offset 3D points back to 2D
        offset_corners_2d, _ = cv2.projectPoints(
            object_points, rvec, tvec_offset, self.camera_matrix, self.dist_coeffs
        )
        
        return offset_corners_2d.reshape(-1, 2).astype(np.float32)

    def _create_imgframe(self, bgr: np.ndarray, src: dai.ImgFrame) -> dai.ImgFrame:
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888i)
        img.setWidth(self.out_w)
        img.setHeight(self.out_h)
        img.setData(bgr.tobytes())
        img.setSequenceNum(src.getSequenceNum())
        img.setTimestamp(src.getTimestamp())
        return img

    def run(self) -> None:
        self._lazy_init()
        
        # Calculate margin in pixels (for top/bottom) and padding in pixels (for left/right)
        margin_pixels = self.margin * self.out_h
        padding_left_pixels = self.padding_left * self.out_w
        padding_right_pixels = self.padding_right * self.out_w
        
        # Create destination quad with margin on top/bottom and separate padding on left/right
        dst_quad = np.array(
            [
                [padding_left_pixels, margin_pixels],  # Top-left with left padding and top margin
                [self.out_w - 1.0 - padding_right_pixels, margin_pixels],  # Top-right with right padding and top margin  
                [self.out_w - 1.0 - padding_right_pixels, self.out_h - 1.0 - margin_pixels],  # Bottom-right with right padding and bottom margin
                [padding_left_pixels, self.out_h - 1.0 - margin_pixels],  # Bottom-left with left padding and bottom margin
            ],
            dtype=np.float32,
        )

        while self.isRunning():
            frame_msg: dai.ImgFrame = self.input.get()
            bgr = frame_msg.getCvFrame()
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            detections = self._detector.detect(gray)

            if len(detections) >= 4:
                # Choose 4 detections with largest spread (convex hull approach)
                centers = np.array([det.center for det in detections], dtype=np.float32)
                c_mean = centers.mean(axis=0)

                # Pick inner corner for each of the 4 farthest detections from centroid
                dists = np.linalg.norm(centers - c_mean, axis=1)
                idx = np.argsort(dists)[-4:]
                chosen = [detections[int(i)] for i in idx]

                corner_candidates = []
                for det in chosen:
                    # Pose-adjusted corners for accuracy
                    offset_corners = self._estimate_pose_and_apply_offset(det, gray.shape)

                    # Order this tag's corners into TL, TR, BR, BL for stable mapping
                    ordered_tag = _order_tag_corners_tl_tr_br_bl(offset_corners)

                    # Determine which quadrant this tag is in relative to the global centroid
                    tag_center = np.array(det.center, dtype=np.float32)
                    is_left = tag_center[0] < c_mean[0]
                    is_top = tag_center[1] < c_mean[1]

                    # Map by quadrant to the inner corner of the LED panel
                    # TL tag -> use TR, TR tag -> use TL, BR tag -> use BL, BL tag -> use BR
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

                # Validate geometry to avoid singular matrices
                area = cv2.contourArea(src_pts.reshape(-1, 1, 2))
                if area < 10.0:
                    continue

                H = cv2.getPerspectiveTransform(src_pts, dst_quad)
                warped = cv2.warpPerspective(bgr, H, (self.out_w, self.out_h))
                out_msg = self._create_imgframe(warped, frame_msg)
                self.out.send(out_msg)
            else:
                # If not enough tags, skip sending to keep stream stable
                continue


