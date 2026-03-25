"""Background watchdog that auto-tunes ``threshold_multiplier`` on LEDGridAnalyzer nodes.

Periodically collects grid brightness values, performs bimodal histogram
analysis (Otsu's method) to find the optimal ON/OFF split, and updates all
analyzers when the smoothed optimal multiplier diverges significantly from
the current value.
"""

import logging
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ThresholdWatchdog:
    """Continuously optimizes ``threshold_multiplier`` for a set of LEDGridAnalyzer nodes.

    Uses exponential moving average (EMA) smoothing to prevent oscillation.
    Only applies a change when the smoothed estimate has been stable for
    several consecutive polls.
    """

    def __init__(
        self,
        analyzers: List,
        poll_interval_sec: float = 5.0,
        ema_alpha: float = 0.2,
        min_change_pct: float = 0.15,
        min_separation: float = 0.05,
        stable_polls_required: int = 3,
    ) -> None:
        self._analyzers = analyzers
        self._poll_interval = poll_interval_sec
        self._ema_alpha = ema_alpha
        self._min_change_pct = min_change_pct
        self._min_separation = min_separation
        self._stable_polls_required = stable_polls_required
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_multiplier: Optional[float] = None
        self._ema_value: Optional[float] = None
        self._stable_count: int = 0

    @property
    def current_multiplier(self) -> Optional[float]:
        return self._current_multiplier

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._analyzers:
            self._current_multiplier = self._analyzers[0].threshold_multiplier
            self._ema_value = self._current_multiplier
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("ThresholdWatchdog started, polling every %.1fs", self._poll_interval)

    def stop(self) -> None:
        self._running = False

    def _collect_grid_values(self) -> Optional[np.ndarray]:
        all_values = []
        for analyzer in self._analyzers:
            grid = getattr(analyzer, "_last_grid_state", None)
            if grid is None:
                continue
            data_rows = grid[:-1, :]
            all_values.append(data_rows.flatten())

        if not all_values:
            return None
        return np.concatenate(all_values)

    def _find_optimal_multiplier(self, values: np.ndarray) -> Optional[float]:
        """Use Otsu's method to find the optimal threshold between OFF and ON LEDs."""
        if values.size < 32:
            return None

        vals_u8 = np.clip(values * 255, 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        split_point = float(threshold) / 255.0

        mean_off = float(values[values <= split_point].mean()) if np.any(values <= split_point) else 0.0
        mean_on = float(values[values > split_point].mean()) if np.any(values > split_point) else 0.0

        if abs(mean_on - mean_off) < self._min_separation:
            return None

        avg_brightness = float(values.mean())
        if avg_brightness < 1e-6:
            return None

        return split_point / avg_brightness

    def _run(self) -> None:
        while self._running:
            time.sleep(self._poll_interval)
            try:
                values = self._collect_grid_values()
                if values is None:
                    continue

                raw_mult = self._find_optimal_multiplier(values)
                if raw_mult is None:
                    continue

                # EMA smoothing
                if self._ema_value is None:
                    self._ema_value = raw_mult
                else:
                    self._ema_value = self._ema_alpha * raw_mult + (1 - self._ema_alpha) * self._ema_value

                # Check if smoothed value diverges enough from current applied value
                if self._current_multiplier is not None:
                    change = abs(self._ema_value - self._current_multiplier) / max(self._current_multiplier, 1e-6)
                    if change < self._min_change_pct:
                        self._stable_count = 0
                        continue

                # Require consecutive polls agreeing before applying
                self._stable_count += 1
                if self._stable_count < self._stable_polls_required:
                    log.debug(
                        "ThresholdWatchdog: EMA=%.4f, stable %d/%d",
                        self._ema_value, self._stable_count, self._stable_polls_required,
                    )
                    continue

                old = self._current_multiplier
                self._current_multiplier = self._ema_value
                for analyzer in self._analyzers:
                    analyzer.threshold_multiplier = self._ema_value
                self._stable_count = 0

                log.info(
                    "ThresholdWatchdog updated multiplier: %.4f -> %.4f",
                    old if old else 0.0, self._ema_value,
                )
            except Exception:
                log.error("ThresholdWatchdog error", exc_info=True)
