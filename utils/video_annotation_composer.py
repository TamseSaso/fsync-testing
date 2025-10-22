import cv2
import numpy as np
import depthai as dai


class VideoAnnotationComposer(dai.node.ThreadedHostNode):
    """
    Composites video frames with annotation overlays into a single output stream.
    
    Input 1: dai.ImgFrame (video frames)
    Input 2: dai.Buffer (annotations from AnnotationHelper)
    Output: dai.ImgFrame (video with annotations overlaid)
    """

    def __init__(self) -> None:
        super().__init__()
        
        # Create inputs for video and annotations
        self.video_input = self.createInput()
        self.video_input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.annotations_input = self.createInput()
        self.annotations_input.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        
        # Create output for composited video
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        # Store latest annotations
        self.latest_annotations = None

    def build(self, video_output: dai.Node.Output, annotations_output: dai.Node.Output) -> "VideoAnnotationComposer":
        video_output.link(self.video_input)
        annotations_output.link(self.annotations_input)
        return self

    def run(self) -> None:
        while self.isRunning():
            try:
                import time

                # Drain annotations to the latest (non-blocking)
                try:
                    while True:
                        try:
                            ann = self.annotations_input.tryGet()
                        except AttributeError:
                            if not self.annotations_input.has():
                                break
                            ann = self.annotations_input.get()
                        if ann is None:
                            break
                        self.latest_annotations = ann
                except Exception:
                    pass

                # Drain video to the latest (non-blocking)
                video_msg: dai.ImgFrame | None = None
                try:
                    while True:
                        try:
                            m = self.video_input.tryGet()
                        except AttributeError:
                            if not self.video_input.has():
                                break
                            m = self.video_input.get()
                        if m is None:
                            break
                        video_msg = m
                except Exception:
                    video_msg = None

                if video_msg is None:
                    # No new frame yet; avoid busy spin
                    time.sleep(0.001)
                    continue

                # Get the BGR frame
                bgr_frame = video_msg.getCvFrame()
                if bgr_frame is None:
                    continue

                # Apply annotations if available
                if self.latest_annotations is not None:
                    try:
                        bgr_frame = self._draw_annotations_on_frame(bgr_frame, self.latest_annotations)
                    except Exception as e:
                        print(f"Warning: Failed to process annotations: {e}")

                # Create output frame message as BGR888 interleaved (matches OpenCV buffer)
                h, w = bgr_frame.shape[:2]
                output_msg = dai.ImgFrame()
                output_msg.setType(dai.ImgFrame.Type.BGR888i)
                output_msg.setWidth(w)
                output_msg.setHeight(h)
                output_msg.setTimestamp(video_msg.getTimestamp())
                output_msg.setSequenceNum(video_msg.getSequenceNum())
                output_msg.setData(bgr_frame.tobytes())

                self.out.send(output_msg)

            except Exception as e:
                print(f"Error in VideoAnnotationComposer: {e}")
                continue

    def _draw_annotations_on_frame(self, frame: np.ndarray, annotations_msg: dai.Buffer) -> np.ndarray:
        """Draw dai.Buffer (annotations from AnnotationHelper) onto a frame using OpenCV."""
        h, w = frame.shape[:2]
        
        try:
            # Iterate through all annotations in the message
            for annotation in annotations_msg.annotations:
                # Draw polylines/rectangles
                for points_annot in annotation.points:
                    points = [(int(pt.x * w), int(pt.y * h)) for pt in points_annot.points]
                    
                    if len(points) >= 2:
                        # Convert color from dai.Color to OpenCV BGR
                        outline_color = self._dai_color_to_bgr(points_annot.outlineColor)
                        fill_color = self._dai_color_to_bgr(points_annot.fillColor) if points_annot.fillColor.a > 0 else None
                        thickness = max(1, int(points_annot.thickness))
                        
                        # Draw filled polygon if fill color is specified
                        if fill_color and points_annot.fillColor.a > 0:
                            overlay = frame.copy()
                            pts_array = np.array(points, dtype=np.int32)
                            cv2.fillPoly(overlay, [pts_array], fill_color)
                            # Blend with alpha
                            alpha = points_annot.fillColor.a
                            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                        
                        # Draw outline
                        if points_annot.type == dai.PointsAnnotationType.LINE_LOOP or len(points) > 2:
                            cv2.polylines(frame, [np.array(points, dtype=np.int32)], 
                                        isClosed=(points_annot.type == dai.PointsAnnotationType.LINE_LOOP),
                                        color=outline_color, thickness=thickness)
                        elif points_annot.type == dai.PointsAnnotationType.LINE_STRIP:
                            for i in range(len(points) - 1):
                                cv2.line(frame, points[i], points[i+1], outline_color, thickness)
                
                # Draw text annotations
                for text_annot in annotation.texts:
                    x = int(text_annot.position.x * w)
                    y = int(text_annot.position.y * h)
                    text = text_annot.text
                    font_size = text_annot.fontSize / 32.0  # Scale down from visualizer size
                    text_color = self._dai_color_to_bgr(text_annot.textColor)
                    
                    # Draw background if specified
                    if text_annot.backgroundColor.a > 0:
                        bg_color = self._dai_color_to_bgr(text_annot.backgroundColor)
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1
                        )
                        cv2.rectangle(frame, 
                                    (x - 2, y - text_height - baseline),
                                    (x + text_width + 2, y + baseline),
                                    bg_color, -1)
                    
                    # Draw text
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                              font_size, text_color, 1, cv2.LINE_AA)
                
                # Draw circles
                for circle_annot in annotation.circles:
                    center = (int(circle_annot.position.x * w), int(circle_annot.position.y * h))
                    radius = int(circle_annot.diameter / 2 * min(w, h))
                    outline_color = self._dai_color_to_bgr(circle_annot.outlineColor)
                    thickness = max(1, int(circle_annot.thickness))
                    
                    # Draw filled circle if fill color specified
                    if circle_annot.fillColor.a > 0:
                        fill_color = self._dai_color_to_bgr(circle_annot.fillColor)
                        cv2.circle(frame, center, radius, fill_color, -1)
                    
                    # Draw outline
                    cv2.circle(frame, center, radius, outline_color, thickness)
        
        except Exception as e:
            print(f"Error drawing annotations: {e}")
        
        return frame

    def _dai_color_to_bgr(self, color: dai.Color) -> tuple:
        """Convert dai.Color to OpenCV BGR tuple."""
        # dai.Color has r, g, b, a in range [0, 1]
        return (int(color.b * 255), int(color.g * 255), int(color.r * 255))
