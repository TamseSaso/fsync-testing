from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

import depthai as dai  # for type hints only; messages pass through untouched


__all__ = ["SharedTicker", "FrameSamplingNode"]


class SharedTicker:
    """
    Emits a tick callback at the same wall-clock times for all subscribers.
    Example with period=5.0: ticks land near ...:00, :05, :10, :15, etc.
    """

    def __init__(self, period_sec: float, start_delay_sec: float = 0.0, fast_start: bool = True):
        assert period_sec > 0, "period_sec must be > 0"
        self._period = float(period_sec)
        self._start_delay = float(start_delay_sec)
        self._subs: list[Callable[[float], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fast_start = bool(fast_start)

    def subscribe(self, cb: Callable[[float], None]) -> None:
        """Register a callback receiving the planned tick_time (epoch seconds)."""
        self._subs.append(cb)

    def start(self) -> None:
        """Begin ticking on the next aligned boundary."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SharedTicker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- internals ---------------------------------------------------------
    def _next_aligned_time(self, now: float) -> float:
        # Align to wall-clock multiples of period, then apply start delay.
        # Example: for 5s period -> boundaries at ..., T-10, T-5, T, T+5, ...
        period = self._period
        # Next multiple of period (ceil) in epoch seconds
        n = int((now + 1e-9) // period)
        candidate = (n + 1) * period
        return candidate + self._start_delay

    def _run(self) -> None:
        # Compute first aligned wall-clock tick
        next_wall = self._next_aligned_time(time.time())

        # Optional: emit a priming tick immediately so the first sample shows up
        # right away (still synchronized across all subscribers because it's the
        # same shared ticker instance).
        if self._fast_start:
            tick_now = time.time()
            for cb in list(self._subs):
                try:
                    cb(tick_now)
                except Exception:
                    # Never let a bad subscriber stall the ticker
                    pass

        # Main loop: wait for the aligned boundary using monotonic deadlines
        # so NTP/RTC wall-clock adjustments don't create multi-minute delays.
        while not self._stop.is_set():
            remaining = max(0.0, next_wall - time.time())
            deadline_mono = time.monotonic() + remaining

            # Sleep in short chunks so .stop() is responsive and we tolerate
            # spurious wakeups or time adjustments.
            while not self._stop.is_set():
                to_wait = deadline_mono - time.monotonic()
                if to_wait <= 0:
                    break
                self._stop.wait(min(0.2, max(0.0, to_wait)))

            if self._stop.is_set():
                break

            tick_time = next_wall
            for cb in list(self._subs):
                try:
                    cb(tick_time)
                except Exception:
                    pass

            next_wall += self._period


class _HostStream:
    """
    Tiny publish/subscribe stream used by the visualizer and other host nodes.

    Expected contract:
      - .subscribe(callback) to receive frames
      - .publish(message) to push frames (dai.ImgFrame or compatible)
    """
    def __init__(self) -> None:
        self._subs: list[Callable[[object], None]] = []

    def subscribe(self, cb: Callable[[object], None]) -> None:
        self._subs.append(cb)

    def publish(self, msg: object) -> None:
        for cb in list(self._subs):
            cb(msg)


class FrameSamplingNode:
    """
    Buffers incoming frames and, on each SharedTicker tick, publishes the frame
    closest to the tick timestamp (within a small drift tolerance).

    Interface:
      - build(upstream) -> self
      - .out : subscribable video stream (ImgFrame pass-through)
      - wait_first_frame(timeout) -> bool
    """

    def __init__(
        self,
        sample_interval_seconds: float,
        shared_ticker: SharedTicker,
        max_drift_sec: Optional[float] = None,
        buffer_seconds: float = 2.0,
    ) -> None:
        assert sample_interval_seconds > 0, "sample_interval_seconds must be > 0"
        self._period = float(sample_interval_seconds)
        self._ticker = shared_ticker
        # Default drift tolerance: ~2 frames at 25–30 FPS.
        self._max_drift = 0.04 if max_drift_sec is None else float(max_drift_sec)

        # Buffer ~FPS * seconds; we don't know FPS, so store by time window.
        self._buf: Deque[Tuple[float, object]] = deque()
        self._buf_lock = threading.Lock()
        self._buffer_seconds = float(buffer_seconds)

        self._got_input_event = threading.Event()
        self._out = _HostStream()

        # Keep reference to upstream for subscription
        self._upstream = None

        self._pull_thread: Optional[threading.Thread] = None
        self._pull_stop = threading.Event()

    # Public API --------------------------------------------------------------
    @property
    def out(self) -> _HostStream:
        return self._out

    def build(self, upstream) -> "FrameSamplingNode":
        """
        Connects to an upstream stream that must support .subscribe(callback).
        """
        self._upstream = upstream

        def on_frame(msg):
            # We prefer the host timestamp if available; fall back to receipt time.
            ts = None
            # DepthAI ImgFrame has getTimestamp() -> datetime
            try:
                ts_dt = msg.getTimestamp()  # type: ignore[attr-defined]
                if ts_dt is not None:
                    ts = ts_dt.timestamp()
            except Exception:
                ts = None
            if ts is None:
                ts = time.time()

            with self._buf_lock:
                self._buf.append((ts, msg))
                # Trim old frames beyond the time window
                cutoff = ts - self._buffer_seconds
                while self._buf and self._buf[0][0] < cutoff:
                    self._buf.popleft()
            self._got_input_event.set()

        # Subscribe to upstream (host or device). Support multiple APIs:
        #  - host-style: .subscribe(cb)
        #  - queue-style: .tryGet() / .get()
        #  - alt callback: .addCallback(cb)
        if not self._attach_upstream(upstream, on_frame):
            # Give a helpful error including available attributes
            attrs = ", ".join(sorted(a for a in dir(upstream) if not a.startswith("_"))[:40])
            raise RuntimeError(
                f"Upstream object of type {type(upstream).__name__} is not subscribable. "
                f"Tried subscribe/addCallback/polling; available attrs: {attrs}"
            )

        # Subscribe to shared ticker
        self._ticker.subscribe(self._on_tick)

        return self

    def wait_first_frame(self, timeout: Optional[float] = None) -> bool:
        """Block until at least one upstream frame is seen."""
        return self._got_input_event.wait(timeout=timeout)

    def _attach_upstream(self, upstream, on_frame: Callable[[object], None]) -> bool:
        # 1) Direct host-style subscription
        if hasattr(upstream, "subscribe"):
            try:
                upstream.subscribe(on_frame)
                return True
            except Exception:
                pass
        # 2) Alternate callback name used by some wrappers
        if hasattr(upstream, "addCallback"):
            try:
                upstream.addCallback(on_frame)
                return True
            except Exception:
                pass
        # 3) DepthAI Node Output: create a host queue and poll it
        if hasattr(upstream, "createOutputQueue"):
            try:
                oq = upstream.createOutputQueue(maxSize=4, blocking=False)
            except TypeError:
                # Older/newer API variants may be positional-only
                try:
                    oq = upstream.createOutputQueue(4, False)
                except Exception:
                    oq = None
            except Exception:
                oq = None
            if oq is not None:
                def _poll_q():
                    while not self._pull_stop.is_set():
                        msg = None
                        try:
                            if hasattr(oq, "tryGet"):
                                msg = oq.tryGet()
                            else:
                                try:
                                    msg = oq.get(timeout=0.05)
                                except TypeError:
                                    try:
                                        msg = oq.get()
                                    except Exception:
                                        msg = None
                        except Exception:
                            msg = None
                        if msg is not None:
                            try:
                                on_frame(msg)
                            except Exception:
                                pass
                        else:
                            time.sleep(0.005)
                self._pull_stop.clear()
                self._pull_thread = threading.Thread(target=_poll_q, name="FSN-PollQ", daemon=True)
                self._pull_thread.start()
                return True
        # 4) Queue-like interface: poll tryGet()/get() in a tiny thread
        if hasattr(upstream, "tryGet") or hasattr(upstream, "get"):
            def _poll():
                # Small sleep to avoid a busy loop; daemon thread exits with process.
                while not self._pull_stop.is_set():
                    msg = None
                    try:
                        if hasattr(upstream, "tryGet"):
                            msg = upstream.tryGet()
                        else:
                            # get() with short timeout if supported; otherwise non-blocking call
                            try:
                                msg = upstream.get(timeout=0.05)
                            except TypeError:
                                # Some get() don't take timeout; wrap in try/except
                                try:
                                    msg = upstream.get()
                                except Exception:
                                    msg = None
                    except Exception:
                        msg = None
                    if msg is not None:
                        try:
                            on_frame(msg)
                        except Exception:
                            pass
                    else:
                        time.sleep(0.005)
            self._pull_stop.clear()
            self._pull_thread = threading.Thread(target=_poll, name="FSN-Poll", daemon=True)
            self._pull_thread.start()
            return True
        # 5) Look for common host-bridge helpers on custom wrappers
        for name in ("asHostStream", "toHostStream", "asHost", "hostStream"):
            if hasattr(upstream, name):
                try:
                    hs = getattr(upstream, name)()
                    if hasattr(hs, "subscribe"):
                        hs.subscribe(on_frame)
                        return True
                except Exception:
                    continue
        return False

    # Internal ---------------------------------------------------------------
    def _on_tick(self, tick_time: float) -> None:
        # Pick the frame with timestamp closest to tick_time.
        best: Optional[Tuple[float, object]] = None
        with self._buf_lock:
            if not self._buf:
                return
            # Linear scan (buffers are tiny); avoids dependencies.
            best = min(self._buf, key=lambda it: abs(it[0] - tick_time))
            # Optional: drop everything older than the chosen frame to limit growth
            # and keep latency small.
            cutoff_idx = 0
            for i, (ts, _) in enumerate(self._buf):
                if ts <= best[0]:
                    cutoff_idx = i
                else:
                    break
            for _ in range(cutoff_idx):
                self._buf.popleft()

        if best is None:
            return

        ts, msg = best
        if abs(ts - tick_time) <= self._max_drift or True:
            # Even if drift is a bit larger due to clock noise, publish anyway.
            # The visual sync is governed by the common tick, so all devices
            # still land near the same wall-clock time.
            self._out.publish(msg)
