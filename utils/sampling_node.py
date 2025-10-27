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

    def __init__(self, sample_interval_seconds: float = 5.0, shared_ticker: Optional[SharedTicker] = None, ptp_slot_period_sec: Optional[float] = None, ptp_slot_phase: float = 0.0, emit_first_frame_immediately: bool = True) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        # Do not block upstream/downstream; keep small buffers
        try:
            self.input.setBlocking(False)
        except AttributeError:
            pass
        try:
            self.input.setQueueSize(4)
        except AttributeError:
            pass
        try:
            self.out.setBlocking(False)
        except AttributeError:
            pass
        try:
            self.out.setQueueSize(4)
        except AttributeError:
            pass
        
        # Signal when the very first frame is seen (used to avoid missing the first shared tick)
        self._first_frame_ev = threading.Event()
        
        self.sample_interval = float(sample_interval_seconds) if sample_interval_seconds else 5.0
        self.shared_ticker = shared_ticker
        self._last_tick_idx = 0
        self.latest_frame: Optional[dai.ImgFrame] = None
        self.frame_lock = threading.Lock()
        self._bootstrapped = False
        # Always emit only on synchronized ticks; no early/extra emits
        self.emit_first_frame_immediately = False

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def wait_first_frame(self, timeout: float | None = None) -> bool:
        """Block until the first frame arrives (used by main to barrier before starting the ticker)."""
        return self._first_frame_ev.wait(timeout)

    def run(self) -> None:
        if self.shared_ticker is not None:
            print(f"FrameSamplingNode: synchronized sampling every {self.sample_interval}s via shared ticker")
        else:
            print(f"FrameSamplingNode: synchronized sampling every {self.sample_interval}s (internal ticker)")
        
        # Start the sampling timer thread
        sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        sampling_thread.start()
        
        # Main loop: continuously update latest frame
        while self.isRunning():
            try:
                # Drain to latest frame (non-blocking)
                frame_msg: dai.ImgFrame | None = None
                try:
                    while True:
                        try:
                            m = self.input.tryGet()
                        except AttributeError:
                            if not self.input.has():
                                break
                            m = self.input.get()
                        if m is None:
                            break
                        frame_msg = m
                except Exception:
                    frame_msg = None

                if frame_msg is None:
                    time.sleep(0.001)
                    continue

                with self.frame_lock:
                    self.latest_frame = frame_msg
                # Mark that we've seen the very first frame
                self._first_frame_ev.set()

            except Exception as e:
                print(f"FrameSamplingNode input error: {e}")
                continue

    def _sampling_loop(self) -> None:
        """Emit the most recent frame exactly every self.sample_interval seconds,
        synchronized across all nodes via a shared ticker.
        """
        # Ensure we've actually seen a frame before we start ticking
        while self.isRunning() and not self._first_frame_ev.is_set():
            self._first_frame_ev.wait(timeout=0.05)

        # Ensure a ticker exists; if none provided, create an internal one
        if self.shared_ticker is None:
            self.shared_ticker = SharedTicker(period_sec=self.sample_interval, start_delay_sec=0.0)
            self.shared_ticker.start()

        # If the shared ticker's period differs, compute how many ticks correspond to our interval
        try:
            stride = max(1, int(round(self.sample_interval / float(self.shared_ticker.period_sec))))
        except Exception:
            stride = 1

        last_sent_idx = -1
        while self.isRunning():
            # Block until the next tick
            self._last_tick_idx = self.shared_ticker.wait_next_tick(self._last_tick_idx)
            # Only emit every `stride` ticks so we hit exactly self.sample_interval seconds
            if (self._last_tick_idx % stride) != 0:
                continue
            # Avoid double-send on the same tick index in case of spurious wakeups
            if self._last_tick_idx == last_sent_idx:
                continue

            with self.frame_lock:
                frame = self.latest_frame
            if frame is None:
                # No frame available yet; try again on the next tick
                continue

            try:
                self.out.send(frame)
                boundary_t = getattr(self.shared_ticker, 'scheduled_time', lambda i: float('nan'))(self._last_tick_idx)
                print(f"Frame sampled on global tick #{self._last_tick_idx} at {boundary_t:.3f}s (every {self.sample_interval:.1f}s)")
                last_sent_idx = self._last_tick_idx
            except Exception as e:
                print(f"FrameSamplingNode send error: {e}")
