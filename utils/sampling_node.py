import time
import threading
import depthai as dai
from typing import Optional

class FrameSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    Input: dai.ImgFrame
    Output: dai.ImgFrame (sampled at specified interval)
    """

    def __init__(self, sample_interval_seconds: float = 20.0, shared_ticker: Optional[SharedTicker] = None, ptp_slot_period_sec: Optional[float] = None, ptp_slot_phase: float = 0.0) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.sample_interval = sample_interval_seconds
        self.shared_ticker = shared_ticker
        self._last_tick_idx = 0
        self.ptp_slot_period = ptp_slot_period_sec
        self.ptp_slot_phase = float(ptp_slot_phase)
        self._last_slot_idx = -1
        self.last_sample_time = 0.0
        self.latest_frame: Optional[dai.ImgFrame] = None
        self.frame_lock = threading.Lock()

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def run(self) -> None:
        if self.shared_ticker is not None:
            print("FrameSamplingNode started with shared global ticker")
        elif self.ptp_slot_period is not None:
            print(f"FrameSamplingNode started with PTP slotting at period {self.ptp_slot_period}s (phase {self.ptp_slot_phase})")
        else:
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
        while self.isRunning():
            # 1) Shared global ticker mode
            if self.shared_ticker is not None:
                self._last_tick_idx = self.shared_ticker.wait_next_tick(self._last_tick_idx)
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self.out.send(self.latest_frame)
                        print(f"Frame sampled on global tick #{self._last_tick_idx} at {time.monotonic():.3f}s")
                continue

            # 2) PTP-slotted mode (device timestamps aligned by PTP)
            if self.ptp_slot_period is not None:
                with self.frame_lock:
                    frame = self.latest_frame
                if frame is not None:
                    try:
                        ts = frame.getTimestamp(dai.CameraExposureOffset.END).total_seconds()
                    except Exception:
                        # Fallback: if offset not supported, use default timestamp
                        ts = frame.getTimestamp().total_seconds()
                    slot_idx = int((ts + self.ptp_slot_phase) / self.ptp_slot_period)
                    if slot_idx > self._last_slot_idx:
                        with self.frame_lock:
                            if self.latest_frame is not None:
                                self.out.send(self.latest_frame)
                        self._last_slot_idx = slot_idx
                        print(f"Frame sampled on PTP slot #{slot_idx} (ts={ts:.6f})")
                # Avoid busy spin if no new frame yet
                time.sleep(0.001)
                continue

            # 3) Local timer fallback
            time.sleep(self.sample_interval)
            with self.frame_lock:
                if self.latest_frame is not None:
                    self.out.send(self.latest_frame)
                    now = time.time()
                    print(f"Frame sampled at {now:.2f}s")
                    self.last_sample_time = now
