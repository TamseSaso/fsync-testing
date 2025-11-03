import time
import threading
import depthai as dai
from typing import Optional

class SharedTicker:
    """A simple cross-node wall-clock ticker.
    All samplers that share the same instance will "tick" at the same time.
    Call .start() once to begin ticking.
    """
    def __init__(self, period_sec: float, start_delay_sec: float = 0.0):
        self.period_sec = float(period_sec)
        self.start_delay_sec = float(start_delay_sec)
        self._start_time = None
        self._tick_idx = 0
        self._cond = threading.Condition()
        self._running = False

    def start(self):
        with self._cond:
            if self._running:
                return
            self._running = True
            self._start_time = time.monotonic() + self.start_delay_sec
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        next_fire = self._start_time
        while True:
            now = time.monotonic()
            sleep = max(0.0, next_fire - now)
            if sleep:
                time.sleep(sleep)
            with self._cond:
                self._tick_idx += 1
                self._cond.notify_all()
            next_fire += self.period_sec

    def wait_next_tick(self, last_seen_idx: int = 0) -> int:
        """Blocks until a new tick is published. Returns the new tick index."""
        with self._cond:
            while self._tick_idx <= last_seen_idx:
                self._cond.wait()
            return self._tick_idx


class FrameSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    Input: dai.ImgFrame
    Output: dai.ImgFrame (sampled at specified interval)
    """

    def __init__(self, sample_interval_seconds: Optional[float] = 5.0, shared_ticker: Optional[SharedTicker] = None, ptp_slot_period_sec: Optional[float] = None, ptp_slot_phase: float = 0.0) -> None:
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
        self._bootstrapped = False
        self._target_start_slot: Optional[int] = None
        self._every_frame_mode = (self.shared_ticker is None and self.ptp_slot_period is None and self.sample_interval is None)

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def wait_first_frame(self, timeout: float = 2.0) -> bool:
        """Block until the first frame has been received (or timeout). Returns True if a frame arrived."""
        deadline = time.monotonic() + (timeout if timeout is not None else 0.0)
        while True:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return True
            if timeout is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.005)

    def run(self) -> None:
        if self.shared_ticker is not None:
            print("FrameSamplingNode started with shared global ticker")
        elif self.ptp_slot_period is not None:
            print(f"FrameSamplingNode started with PTP slotting at period {self.ptp_slot_period}s (phase {self.ptp_slot_phase})")
        else:
            mode = "EVERY-FRAME mode" if self.sample_interval is None else f"{self.sample_interval}s interval"
            print(f"FrameSamplingNode started with {mode}")

        # Start the sampling timer thread only if not in every-frame mode
        if not self._every_frame_mode:
            sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
            sampling_thread.start()

        # Main loop: continuously update latest frame
        while self.isRunning():
            try:
                frame_msg: dai.ImgFrame = self.input.get()
                if frame_msg is not None:
                    with self.frame_lock:
                        self.latest_frame = frame_msg

                # Forward every frame immediately in every-frame mode
                if self._every_frame_mode and frame_msg is not None:
                    self.out.send(frame_msg)

                # Bootstrap: in tick-only mode do NOT emit; just mark bootstrapped so first tick can forward latest_frame
                if frame_msg is not None and not self._bootstrapped and (self.shared_ticker is not None and self.ptp_slot_period is None):
                    self._bootstrapped = True
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

                    # If no rendezvous is set, align the first emission to the next slot boundary
                    if self._target_start_slot is None:
                        self._target_start_slot = slot_idx + 1
                        print(f"FrameSamplingNode PTP: arming for next slot >= {self._target_start_slot}")
                        time.sleep(0.0005)
                        continue

                    # If armed for a specific slot, hold until that slot is reached
                    if slot_idx < self._target_start_slot:
                        time.sleep(0.0005)
                        continue

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
