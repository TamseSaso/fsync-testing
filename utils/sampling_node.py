import time
import threading
import depthai as dai
from typing import Optional


class FrameSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    Input: dai.ImgFrame
    Output: dai.ImgFrame (sampled at specified interval)
    """

    def __init__(self, sample_interval_seconds: float = 20.0) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.sample_interval = sample_interval_seconds
        self.last_sample_time = 0.0
        self.latest_frame: Optional[dai.ImgFrame] = None
        self.frame_lock = threading.Lock()

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def run(self) -> None:
        print(f"FrameSamplingNode started with {self.sample_interval}s interval")
        
        # Start the sampling timer thread
        sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        sampling_thread.start()
        
        # Main loop: continuously update latest frame
        while self.isRunning():
            try:
                frame_msg: dai.ImgFrame = self.input.get()
                if frame_msg is not None:
                    with self.frame_lock:
                        self.latest_frame = frame_msg
            except Exception as e:
                print(f"FrameSamplingNode input error: {e}")
                continue

    def _sampling_loop(self) -> None:
        """Separate thread that samples frames at the specified interval."""
        while self.isRunning():
            current_time = time.time()
            
            # Check if it's time to sample
            if current_time - self.last_sample_time >= self.sample_interval:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        # Send the latest frame
                        self.out.send(self.latest_frame)
                        print(f"Frame sampled at {current_time:.2f}s")
                        self.last_sample_time = current_time
            
            # Sleep for a short duration to avoid busy waiting
            time.sleep(0.1)
