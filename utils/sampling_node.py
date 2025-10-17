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

    def __init__(self, sample_interval_seconds: float = 20.0, shared_ticker: Optional[SharedTicker] = None) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.sample_interval = sample_interval_seconds
        self.shared_ticker = shared_ticker
        self._last_tick_idx = 0
        self.last_sample_time = 0.0
        self.latest_frame: Optional[dai.ImgFrame] = None
        self.frame_lock = threading.Lock()

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def run(self) -> None:
        if self.shared_ticker is not None:
            print("FrameSamplingNode started with shared global ticker")
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
        """Separate thread that samples frames either on a global shared ticker or a local interval."""
        while self.isRunning():
            if self.shared_ticker is not None:
                # Block until the next global tick shared by all samplers
                self._last_tick_idx = self.shared_ticker.wait_next_tick(self._last_tick_idx)
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self.out.send(self.latest_frame)
                        print(f"Frame sampled on global tick #{self._last_tick_idx} at {time.monotonic():.3f}s")
            else:
                # Local timer fallback
                time.sleep(self.sample_interval)
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self.out.send(self.latest_frame)
                        now = time.time()
                        print(f"Frame sampled at {now:.2f}s")
                        self.last_sample_time = now
