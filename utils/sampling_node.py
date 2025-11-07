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

        # Barrier/rendezvous support for cross-device tick alignment
        self._participants = 0
        self._rendezvous_acks = {}
        self._barrier_timeout_sec = 0.02

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

    def epoch_monotonic(self) -> Optional[float]:
        """Return the ticker start time (host monotonic seconds) or None if not started."""
        with self._cond:
            return self._start_time

    def tick_index_for_time(self, mono_time: float) -> int:
        """Map a host-monotonic timestamp to a tick index.
        Returns 0 if before the ticker epoch or the ticker hasn't started."""
        with self._cond:
            if self._start_time is None or mono_time < self._start_time:
                return 0
            delta = mono_time - self._start_time
            return int(delta // self.period_sec) + 1  # tick #1 at epoch

    def register_participant(self) -> None:
        """Register one sampler as a participant in rendezvous barriers."""
        with self._cond:
            self._participants += 1

    def set_barrier_timeout(self, seconds: float) -> None:
        with self._cond:
            self._barrier_timeout_sec = float(seconds)

    def rendezvous(self, tick_idx: int, timeout: Optional[float] = None) -> bool:
        """Barrier: wait until all registered participants acknowledge this tick.
        Returns True if all arrived before timeout, False otherwise.
        """
        with self._cond:
            # increment acks for this tick
            cur = self._rendezvous_acks.get(tick_idx, 0) + 1
            self._rendezvous_acks[tick_idx] = cur
            self._cond.notify_all()
            if self._participants <= 1:
                return True
            deadline = time.monotonic() + (self._barrier_timeout_sec if timeout is None else float(timeout))
            while self._rendezvous_acks.get(tick_idx, 0) < self._participants:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            ok = self._rendezvous_acks.get(tick_idx, 0) >= self._participants
            # cleanup old ticks to prevent growth
            for old in list(self._rendezvous_acks.keys()):
                if old < tick_idx - 2:
                    del self._rendezvous_acks[old]
            self._cond.notify_all()
            return ok


class FrameSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    Input: dai.ImgFrame
    Output: dai.ImgFrame (sampled at specified interval)
    """

    def __init__(self, sample_interval_seconds: Optional[float] = 5.0, shared_ticker: Optional[SharedTicker] = None, ptp_slot_period_sec: Optional[float] = None, ptp_slot_phase: float = 0.0, debug: bool = False, tick_grace_sec: float = 0.001, arrival_latency_correction_sec: float = 0.0, barrier: bool = True, barrier_timeout_sec: float = 0.02, wait_window_sec: float = 0.040, auto_calibrate_correction: bool = True) -> None:
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
        self.latest_arrival_mono: Optional[float] = None
        self.frame_lock = threading.Lock()
        self._bootstrapped = False
        self._target_start_slot: Optional[int] = None
        self._every_frame_mode = (self.shared_ticker is None and self.ptp_slot_period is None and self.sample_interval is None)
        self.debug = bool(debug)

        # Tick alignment tuning & barrier
        self.tick_grace_sec = float(tick_grace_sec) if tick_grace_sec is not None else 0.0
        self.arrival_latency_correction_sec = float(arrival_latency_correction_sec or 0.0)
        self.barrier_enabled = bool(barrier)
        self.barrier_timeout_sec = float(barrier_timeout_sec)

        # Wait-and-grab window & auto-correction of arrival offset
        self.wait_window_sec = float(wait_window_sec)
        self.auto_calibrate_correction = bool(auto_calibrate_correction)
        # dynamic correction we can adjust at runtime
        self._dyn_arrival_corr = float(self.arrival_latency_correction_sec)
        self._dyn_corr_step = 0.002  # 2 ms per adjustment
        self._dyn_corr_min = -0.050  # allow up to +50 ms push later (negative means add)
        self._dyn_corr_max = 0.050   # and -50 ms pull earlier

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        if self.shared_ticker is not None and self.barrier_enabled:
            self.shared_ticker.register_participant()
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
                        self.latest_arrival_mono = time.monotonic()

                # Forward every frame immediately in every-frame mode
                if self._every_frame_mode and frame_msg is not None:
                    self.out.send(frame_msg)

                # Bootstrap: in tick-only mode do NOT emit; just mark bootstrapped so first tick can forward latest_frame
                if frame_msg is not None and not self._bootstrapped and (self.shared_ticker is not None and self.ptp_slot_period is None):
                    self._bootstrapped = True
            except Exception as e:
                print(f"FrameSamplingNode input error: {e}")
                continue

    def _tick_start_time(self, tick_idx: int) -> Optional[float]:
        if self.shared_ticker is None:
            return None
        epoch = self.shared_ticker.epoch_monotonic()
        if epoch is None or tick_idx <= 0:
            return None
        return epoch + (tick_idx - 1) * self.shared_ticker.period_sec

    def _sampling_loop(self) -> None:
        while self.isRunning():
            # 1) Shared global ticker mode
            if self.shared_ticker is not None:
                # Wait for the next global tick number (shared across devices)
                self._last_tick_idx = self.shared_ticker.wait_next_tick(self._last_tick_idx)
                tick_start = self._tick_start_time(self._last_tick_idx) or time.monotonic()

                # Grace window lets frames land for this tick
                if self.tick_grace_sec > 0.0:
                    time.sleep(min(self.tick_grace_sec, max(0.0, self.shared_ticker.period_sec * 0.25)))

                deadline = tick_start + self.wait_window_sec
                sent = False
                while time.monotonic() < deadline:
                    with self.frame_lock:
                        frame = self.latest_frame
                        arrival = self.latest_arrival_mono
                    if frame is None or arrival is None:
                        time.sleep(0.0005)
                        continue

                    corrected = arrival - self._dyn_arrival_corr
                    slot = self.shared_ticker.tick_index_for_time(corrected)
                    if slot == self._last_tick_idx:
                        barrier_ok = True
                        if self.barrier_enabled:
                            barrier_ok = self.shared_ticker.rendezvous(self._last_tick_idx, timeout=self.barrier_timeout_sec)
                        self.out.send(frame)
                        sent = True
                        if self.debug:
                            print(
                                f"Frame sampled on global tick #{self._last_tick_idx} (arrival match, barrier={'ok' if barrier_ok else 'timeout'}) at {time.monotonic():.3f}s; corrected_arrival={corrected:.6f} grace={self.tick_grace_sec:.3f}s corr={self._dyn_arrival_corr:.3f}s"
                            )
                        break

                    # Auto-calibrate arrival correction towards the current tick
                    if self.auto_calibrate_correction:
                        if slot < self._last_tick_idx:
                            # frame mapped to previous tick -> push later (make correction more negative)
                            self._dyn_arrival_corr = max(self._dyn_corr_min, self._dyn_arrival_corr - self._dyn_corr_step)
                        elif slot > self._last_tick_idx:
                            # frame mapped to future tick -> pull earlier (increase correction)
                            self._dyn_arrival_corr = min(self._dyn_corr_max, self._dyn_arrival_corr + self._dyn_corr_step)
                    time.sleep(0.0005)

                if not sent:
                    # Final check once after window in case it landed exactly at the boundary
                    with self.frame_lock:
                        frame = self.latest_frame
                        arrival = self.latest_arrival_mono
                    if frame is not None and arrival is not None:
                        corrected = arrival - self._dyn_arrival_corr
                        slot = self.shared_ticker.tick_index_for_time(corrected)
                        if slot == self._last_tick_idx:
                            barrier_ok = True
                            if self.barrier_enabled:
                                barrier_ok = self.shared_ticker.rendezvous(self._last_tick_idx, timeout=self.barrier_timeout_sec)
                            self.out.send(frame)
                            sent = True
                            if self.debug:
                                print(
                                    f"Frame sampled on global tick #{self._last_tick_idx} (arrival match post-window, barrier={'ok' if barrier_ok else 'timeout'}) at {time.monotonic():.3f}s; corrected_arrival={corrected:.6f} window={self.wait_window_sec:.3f}s corr={self._dyn_arrival_corr:.3f}s"
                                )

                if not sent and self.debug:
                    print(
                        f"[SKIP] no matching frame for tick {self._last_tick_idx} (corr={self._dyn_arrival_corr:.3f}s, window={self.wait_window_sec:.3f}s, grace={self.tick_grace_sec:.3f}s)"
                    )
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
