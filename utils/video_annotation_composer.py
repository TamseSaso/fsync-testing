import cv2
import numpy as np
import depthai as dai
from depthai_nodes.utils import AnnotationHelper


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
                        # Parse and draw the annotations using AnnotationHelper
                        bgr_frame = AnnotationHelper.draw(bgr_frame, self.latest_annotations)
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
