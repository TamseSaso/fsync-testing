import logging
import time
import threading
import depthai as dai
from typing import Optional

from datetime import datetime, timedelta

from .health import NodeHealth

log = logging.getLogger(__name__)

class SharedTicker:
    """A simple cross-node ticker that can be periodic or manual.
    - Periodic mode: pass a positive `period_sec` to tick on a wall clock.
    - Manual mode: pass `period_sec=None` and call `.pulse()` from host code.
    All samplers that share the same instance will observe the same tick index.
    Call `.start()` once (ok in manual mode too) or just `.pulse()` in manual mode.
    """
    def __init__(self, period_sec: Optional[float] = None, start_delay_sec: float = 0.0):
        self.period_sec = float(period_sec) if period_sec is not None else None
        self._manual_mode = (period_sec is None)
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
            if not self._manual_mode:
                self._start_time = time.monotonic() + self.start_delay_sec
                threading.Thread(target=self._run, daemon=True).start()

    def pulse(self) -> int:
        """Manually advance the ticker by one and notify all waiters.
        Use this from a host loop (e.g., after draining device queues) to
        rendezvous samplers across nodes. Returns the new tick index.
        """
        with self._cond:
            self._tick_idx += 1
            self._cond.notify_all()
            return self._tick_idx

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def _run(self):
        if self._manual_mode:
            return
        next_fire = self._start_time
        while self._running:
            now = time.monotonic()
            sleep = max(0.0, next_fire - now)
            if sleep:
                time.sleep(sleep)
            with self._cond:
                self._tick_idx += 1
                self._cond.notify_all()
            next_fire += self.period_sec
            # Catch up if we fell behind (skip missed ticks instead of bunching)
            now2 = time.monotonic()
            if next_fire < now2 - self.period_sec:
                skipped = int((now2 - next_fire) / self.period_sec)
                next_fire += skipped * self.period_sec
                if skipped > 0:
                    log.warning("SharedTicker skipped %d ticks (load)", skipped)

    def wait_next_tick(self, last_seen_idx: int = 0) -> int:
        """Blocks until a new tick is published. Returns the new tick index.
        Raises RuntimeError if the ticker is stopped while waiting."""
        with self._cond:
            while self._tick_idx <= last_seen_idx:
                if not self._running:
                    raise RuntimeError("SharedTicker stopped")
                self._cond.wait(timeout=1.0)
            return self._tick_idx

class M8FsyncSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    """

    def __init__(self,
        num_cameras: int = 2,
        undersampling_factor: int = 10,
        frame_lost_threshold_sec: float = 0,
        recv_all_timeout_sec: int = 10,
        sync_threshold_sec: float = 0,
        debug: bool = False,
        device_ids: list[str] = None) -> None:
        super().__init__()

        self.num_cameras = num_cameras
        self.inputs = [self.createInput() for _ in range(num_cameras)]

        for i in self.inputs:
            i.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.outputs = [self.createOutput() for _ in range(num_cameras)]

        for o in self.outputs:
            o.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.latest_frames = {}
        self.receivedFrames = [False for _ in range(num_cameras)]
        self.previous_frame_rcv_timestamp = {}
        
        self.initially_locked = False
        self.undersampling_factor = undersampling_factor
        self.debug = debug

        self.frame_lost_threshold_sec = frame_lost_threshold_sec
        self.recv_all_timeout_sec = recv_all_timeout_sec
        self.sync_threshold_sec = sync_threshold_sec

        self.device_ids = device_ids
    
    def build(self, frames: list[dai.Node.Output]):
        assert(len(frames) == self.num_cameras)
        for i in range(len(frames)):
            frames[i].link(self.inputs[i])
        return self

    def run(self) -> None:
        self.start_time = datetime.now()
        count = 0
        while self.isRunning():
            self.all_received = True
            # -------------------------------------------------------------------
            # Collect the newest frame from each queue (non‑blocking)
            # -------------------------------------------------------------------
            for idx, i in enumerate(self.inputs):
                self.all_received &= self.receivedFrames[idx]

                if self.receivedFrames[idx]:
                    ts = self.previous_frame_rcv_timestamp[idx]

                    if datetime.now() - ts > timedelta(microseconds=round(self.frame_lost_threshold_sec*1e6)):
                        raise RuntimeError(f"[{self.device_ids[idx]}]: Frame lost")
                while i.has():
                    self.latest_frames[idx] = i.get()
                    self.previous_frame_rcv_timestamp[idx] = datetime.now()
                    if not self.receivedFrames[idx]:
                        log.info(f"=== Received frame from {self.device_ids[idx]}")
                        self.receivedFrames[idx] = True

            end_time = datetime.now()
            elapsed_sec = (end_time - self.start_time).total_seconds()
            if not self.all_received:
                if elapsed_sec >= self.recv_all_timeout_sec:
                    raise RuntimeError("Timeout: Didn't receive all frames in time")

            # -------------------------------------------------------------------
            # Synchronise: we need at least one frame from every camera and their
            # timestamps must align within sync_threshold_sec.
            # -------------------------------------------------------------------
            if len(self.latest_frames) == len(self.inputs):
                ts_values = [f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() for f in self.latest_frames.values()]

                delta = abs(max(ts_values) - min(ts_values))

                if not self.initially_locked and delta < self.sync_threshold_sec:
                    self.initially_locked = True
                
                if self.debug:
                    log.debug(f"M8FsyncSamplingNode: delta = {delta}")

                if self.initially_locked and delta > self.sync_threshold_sec:
                    raise RuntimeError("Synchronization lost")

                if count % self.undersampling_factor == 0:
                    for i in range(len(self.inputs)):
                        self.outputs[i].send(self.latest_frames[i])
                count += 1

                self.latest_frames.clear()  # Wait for next batch

class FrameSamplingNode(dai.node.ThreadedHostNode):
    """Samples frames from input at specified intervals and forwards them to output.
    
    Input: dai.ImgFrame
    Output: dai.ImgFrame (sampled at specified interval)
    """

    def __init__(self, sample_interval_seconds: Optional[float] = 5.0, shared_ticker: Optional[SharedTicker] = None, ptp_slot_period_sec: Optional[float] = None, ptp_slot_phase: float = 0.0, debug: bool = False) -> None:
        super().__init__()
        self.health = NodeHealth()

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
        self.debug = bool(debug)

    def build(self, frames: dai.Node.Output) -> "FrameSamplingNode":
        frames.link(self.input)
        return self

    def wait_first_frame(self, timeout: Optional[float] = 2.0) -> bool:
        """Block until the first frame has been received (or timeout). Returns True if a frame arrived."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)

    def run(self) -> None:
        if self.shared_ticker is not None:
            if self.shared_ticker._manual_mode:
                log.info("FrameSamplingNode started with MANUAL shared ticker")
            else:
                log.info("FrameSamplingNode started with PERIODIC shared ticker")
        elif self.ptp_slot_period is not None:
            log.info(f"FrameSamplingNode started with PTP slotting at period {self.ptp_slot_period}s (phase {self.ptp_slot_phase})")
        else:
            mode = "EVERY-FRAME mode" if self.sample_interval is None else f"{self.sample_interval}s interval"
            log.info(f"FrameSamplingNode started with {mode}")

        # Start the sampling timer thread only if not in every-frame mode
        if not self._every_frame_mode:
            sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
            sampling_thread.start()

        # Main loop: continuously update latest frame
        while self.isRunning():
            try:
                frame_msg: dai.ImgFrame = self.input.get()
                if frame_msg is not None:
                    self.health.record_received()
                    with self.frame_lock:
                        self.latest_frame = frame_msg

                # Forward every frame immediately in every-frame mode
                if self._every_frame_mode and frame_msg is not None:
                    self.out.send(frame_msg)
                    self.health.record_produced()

                # Bootstrap: in tick-only mode do NOT emit; just mark bootstrapped so first tick can forward latest_frame
                if frame_msg is not None and not self._bootstrapped and (self.shared_ticker is not None and self.ptp_slot_period is None):
                    self._bootstrapped = True
            except Exception as e:
                if "closed" in str(e).lower():
                    log.debug("FrameSamplingNode exiting: queue closed")
                    break
                log.error("FrameSamplingNode input error: %s", e)
                continue

    def _sampling_loop(self) -> None:
        try:
            self._sampling_loop_inner()
        except Exception as e:
            if "closed" in str(e).lower() or "stopped" in str(e).lower():
                log.debug("Sampling loop exiting: %s", e)
            else:
                log.error("Sampling loop error: %s", e)

    def _sampling_loop_inner(self) -> None:
        while self.isRunning():
            # 1) Shared global ticker mode
            if self.shared_ticker is not None:
                self._last_tick_idx = self.shared_ticker.wait_next_tick(self._last_tick_idx)
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self.out.send(self.latest_frame)
                        self.health.record_produced()
                        if self.debug:
                            log.debug(f"Frame sampled on global tick #{self._last_tick_idx} at {time.monotonic():.3f}s")
                continue

            # 2) PTP-slotted mode (device timestamps aligned by PTP)
            if self.ptp_slot_period is not None:
                with self.frame_lock:
                    frame = self.latest_frame
                if frame is not None:
                    try:
                        ts = frame.getTimestamp(dai.CameraExposureOffset.END).total_seconds()
                    except Exception:
                        ts = frame.getTimestamp().total_seconds()
                    slot_idx = int((ts + self.ptp_slot_phase) / self.ptp_slot_period)

                    if self._target_start_slot is None:
                        self._target_start_slot = slot_idx + 1
                        log.info(f"FrameSamplingNode PTP: arming for next slot >= {self._target_start_slot}")
                        time.sleep(0.0005)
                        continue

                    if slot_idx < self._target_start_slot:
                        time.sleep(0.0005)
                        continue

                    if slot_idx > self._last_slot_idx:
                        with self.frame_lock:
                            if self.latest_frame is not None:
                                self.out.send(self.latest_frame)
                                self.health.record_produced()
                        self._last_slot_idx = slot_idx
                        log.debug(f"Frame sampled on PTP slot #{slot_idx} (ts={ts:.6f})")
                time.sleep(0.001)
                continue

            # 3) Local timer fallback
            time.sleep(self.sample_interval)
            with self.frame_lock:
                if self.latest_frame is not None:
                    self.out.send(self.latest_frame)
                    self.health.record_produced()
                    now = time.time()
                    log.debug(f"Frame sampled at {now:.2f}s")
                    self.last_sample_time = now
