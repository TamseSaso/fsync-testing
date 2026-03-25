"""Health metrics mixin and monitoring utilities for pipeline nodes."""

import logging
import threading
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)


class NodeHealth:
    """Lightweight health tracker that any ThreadedHostNode can embed.

    Usage inside a node::

        def __init__(self):
            ...
            self.health = NodeHealth()

        def run(self):
            while self.isRunning():
                frame = self.input.get()
                self.health.record_received()
                ...
                self.out.send(result)
                self.health.record_produced()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames_received: int = 0
        self.frames_produced: int = 0
        self.last_received_time: float = 0.0
        self.last_produced_time: float = 0.0
        self.consecutive_errors: int = 0
        self._start_time: float = time.monotonic()

    def record_received(self) -> None:
        with self._lock:
            self.frames_received += 1
            self.last_received_time = time.monotonic()

    def record_produced(self) -> None:
        with self._lock:
            self.frames_produced += 1
            self.last_produced_time = time.monotonic()
            self.consecutive_errors = 0

    def record_error(self) -> None:
        with self._lock:
            self.consecutive_errors += 1

    def clear_errors(self) -> None:
        with self._lock:
            self.consecutive_errors = 0

    def snapshot(self) -> Dict:
        with self._lock:
            now = time.monotonic()
            return {
                "frames_received": self.frames_received,
                "frames_produced": self.frames_produced,
                "consecutive_errors": self.consecutive_errors,
                "sec_since_last_recv": now - self.last_received_time if self.last_received_time else None,
                "sec_since_last_prod": now - self.last_produced_time if self.last_produced_time else None,
                "uptime_sec": now - self._start_time,
            }


class HealthMonitor:
    """Background watchdog that polls ``NodeHealth`` instances.

    Escalation levels:
      - WARNING after ``warn_after_sec`` of no output from a node
      - ERROR   after ``error_after_sec``
      - Calls ``on_critical`` callback after ``critical_after_sec``
    """

    def __init__(
        self,
        poll_interval_sec: float = 2.0,
        warn_after_sec: float = 10.0,
        error_after_sec: float = 30.0,
        critical_after_sec: float = 60.0,
        on_critical=None,
    ) -> None:
        self._nodes: Dict[str, NodeHealth] = {}
        self._poll_interval = poll_interval_sec
        self._warn = warn_after_sec
        self._error = error_after_sec
        self._critical = critical_after_sec
        self._on_critical = on_critical
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register(self, name: str, health: NodeHealth) -> None:
        self._nodes[name] = health

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        log.info("HealthMonitor started, polling %d nodes every %.1fs",
                 len(self._nodes), self._poll_interval)
        while self._running:
            time.sleep(self._poll_interval)
            for name, health in self._nodes.items():
                snap = health.snapshot()
                since_prod = snap["sec_since_last_prod"]
                if since_prod is None:
                    continue
                errs = snap["consecutive_errors"]

                if since_prod >= self._critical or errs >= 50:
                    log.critical(
                        "CRITICAL: %s stalled %.1fs, %d consecutive errors",
                        name, since_prod, errs,
                    )
                    if self._on_critical:
                        self._on_critical(name, snap)
                elif since_prod >= self._error or errs >= 20:
                    log.error(
                        "%s stalled %.1fs, %d consecutive errors",
                        name, since_prod, errs,
                    )
                elif since_prod >= self._warn or errs >= 5:
                    log.warning(
                        "%s no output for %.1fs, %d consecutive errors",
                        name, since_prod, errs,
                    )
