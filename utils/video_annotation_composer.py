import cv2
import numpy as np
import depthai as dai


class VideoAnnotationComposer(dai.node.ThreadedHostNode):
    """
    Composites video frames with annotation overlays into a single output stream.
    
    Input 1: dai.ImgFrame (video frames)
    Input 2: dai.ImgAnnotations (annotations from AnnotationHelper)
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
                # Try to get the latest annotations (non-blocking)
                try:
                    annotations_msg = self.annotations_input.tryGet()
                    if annotations_msg is not None:
                        self.latest_annotations = annotations_msg
                except:
                    pass
                
                # Get video frame (blocking)
                video_msg: dai.ImgFrame = self.video_input.get()
                if video_msg is None:
                    continue
                
                # Get the BGR frame
                bgr_frame = video_msg.getCvFrame()
                if bgr_frame is None:
                    continue
                
                # Apply annotations if available
                if self.latest_annotations is not None:
                    try:
                        # Parse and draw the annotations manually using OpenCV
                        bgr_frame = self._draw_annotations_on_frame(bgr_frame, self.latest_annotations)
                    except Exception as e:
                        print(f"Warning: Failed to process annotations: {e}")
                
                # Create output frame message
                output_msg = dai.ImgFrame()
                output_msg.setData(bgr_frame.flatten())
                output_msg.setType(video_msg.getType())
                output_msg.setWidth(bgr_frame.shape[1])
                output_msg.setHeight(bgr_frame.shape[0])
                output_msg.setTimestamp(video_msg.getTimestamp())
                output_msg.setSequenceNum(video_msg.getSequenceNum())
                
                self.out.send(output_msg)
                
            except Exception as e:
                print(f"Error in VideoAnnotationComposer: {e}")
                continue

    def _draw_annotations_on_frame(self, frame: np.ndarray, annotations_msg: dai.ImgAnnotations) -> np.ndarray:
        """Draw dai.ImgAnnotations onto a frame using OpenCV."""
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
