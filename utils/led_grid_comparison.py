

import cv2
import numpy as np
import depthai as dai
from typing import Optional, Tuple
import time as _t
from datetime import timedelta as _td


class LEDGridComparison(dai.node.ThreadedHostNode):
    """
    Compares two LED grid analyzer streams and determines if they are in sync.

    Inputs: (set via set_queues) two host-side OutputQueues that receive dai.Buffer from LEDGridAnalyzer
            Each buffer encodes:
              - 32x32 float grid_state (values in [0..1], bottom row pre-scaled by analyzer)
              - 4 floats metadata: [overall_avg_brightness, threshold_multiplier, speed, intervals]
    Outputs:
      - out_overlay: dai.ImgFrame (BGR) overlay of both LED masks (A-only, B-only, Both)
      - out_report:  dai.ImgFrame (BGR) textual report with pass/fail and metrics

    Logic:
      1) Check configuration on bottom row:
         - first 16 cells encode "speed" (count of lit LEDs) and is provided in metadata
         - last 16 cells encode "intervals" (binary) and is provided in metadata
         If speed OR intervals differ between the two panels, SKIP comparison (report only).
      2) If config matches, compare placement of "green" LEDs (mask = grid_state > dynamic_threshold).
         - dynamic_threshold = overall_avg_brightness * threshold_multiplier (same as visualizer)
         - Exclude bottom config row from the metric.
         - Use buffer timestamps to estimate temporal offset:
             shift_cols = round(|tA - tB| / led_period_us)
           Roll mask from the earlier frame forward by shift_cols along columns to align in time.
         - Compute overlap metrics (excluding the bottom row):
             recallA = overlap / max(1, onA)
             recallB = overlap / max(1, onB)
             iou     = overlap / max(1, (onA + onB - overlap))
           PASS if min(recallA, recallB) >= pass_ratio (default 0.90).
      3) Produce overlay image and textual report.
    """

    def __init__(
        self,
        grid_size: int = 32,
        output_size: Tuple[int, int] = (1024, 1024),
        led_period_us: float = 170.0,
        pass_ratio: float = 0.90
    ) -> None:
        super().__init__()

        # No inputs; comparison consumes host queues set via set_queues()
        self.out_overlay = self.createOutput()
        self.out_overlay.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.out_report = self.createOutput()
        self.out_report.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        # Lightweight tick input to attach/schedule this host node in a pipeline
        self._tickIn = self.createInput()
        self._tickIn.setPossibleDatatypes([
            (dai.DatatypeEnum.Buffer, True),
            (dai.DatatypeEnum.ImgFrame, True),
        ])

        self.grid_size = grid_size
        self.output_w, self.output_h = output_size
        self.cell_w = max(1, self.output_w // self.grid_size)
        self.cell_h = max(1, self.output_h // self.grid_size)
        self.led_period_us = float(led_period_us)
        self.pass_ratio = float(pass_ratio)
        self._last_placeholder_time = 0.0
        self._seq_counter = 1
        self._lastA = None  # (grid, avg, mult, speed, intervals, ts, seq)
        self._lastB = None  # (grid, avg, mult, speed, intervals, ts, seq)

        # Host-side queues are provided from analyzer node outputs
        self._qA = None
        self._qB = None

    # --- Public API -------------------------------------------------------
    def build(self, tick_source: dai.Node.Output) -> "LEDGridComparison":
        """
        Attach this host node to a pipeline by linking any stream as a lightweight 'tick'.
        We don't read payloads from this input for logic; it's only to bind/schedule the node.
        """
        tick_source.link(self._tickIn)
        return self

    def set_queues(self, qA, qB) -> None:
        """Provide the two host queues that output dai.Buffer from LEDGridAnalyzer."""
        self._qA = qA
        self._qB = qB

    # --- Utility ----------------------------------------------------------
    def _parse_buffer(self, buf: dai.Buffer):
        data = np.frombuffer(buf.getData(), dtype=np.float32)
        expected = self.grid_size * self.grid_size + 4
        if data.size != expected:
            raise ValueError(f"Invalid buffer size {data.size}, expected {expected}")

        gsz = self.grid_size * self.grid_size
        grid = data[:gsz].reshape((self.grid_size, self.grid_size))
        meta = data[-4:]
        overall_avg = float(meta[0])
        thr_mult = float(meta[1])
        speed = int(meta[2])
        intervals = int(meta[3])
        ts = buf.getTimestamp()
        seq = buf.getSequenceNum()
        return grid, overall_avg, thr_mult, speed, intervals, ts, seq

    def _dynamic_threshold(self, avg: float, mult: float) -> float:
        return float(avg) * float(mult)

    def _mask_from_grid(self, grid: np.ndarray, thr: float) -> np.ndarray:
        return (grid > thr)


    def _roll_columns(self, mask: np.ndarray, cols: int) -> np.ndarray:
        if cols == 0:
            return mask
        # positive cols -> shift right (later in time)
        return np.roll(mask, shift=cols, axis=1)

    def _create_imgframe(self, bgr: np.ndarray, ts, seq: int) -> dai.ImgFrame:
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888i)
        img.setWidth(bgr.shape[1])
        img.setHeight(bgr.shape[0])
        img.setData(bgr.tobytes())
        img.setSequenceNum(int(seq))
        img.setTimestamp(ts)
        return img

    def _send_placeholder(self, text_line1: str = "Waiting for LED analyzer streams...", text_line2: Optional[str] = "Connect two devices and ensure LEDGridAnalyzer is running.") -> None:
        # Create a neutral overlay (dark background with message)
        overlay = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        overlay[:] = (30, 30, 30)
        cv2.putText(overlay, text_line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
        if text_line2:
            cv2.putText(overlay, text_line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

        # Minimal report image too, so both topics are visible in the visualizer
        report = np.zeros((240, 960, 3), dtype=np.uint8)
        report[:] = (20, 20, 20)
        cv2.putText(report, "LED Sync Report", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(report, text_line1, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        if text_line2:
            cv2.putText(report, text_line2, (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

        # Use zero timestamp for placeholders and increment seq
        ts = _td(seconds=0)
        seq = self._seq_counter
        self._seq_counter += 1

        self.out_overlay.send(self._create_imgframe(overlay, ts, seq))
        self.out_report.send(self._create_imgframe(report, ts, seq))

    def _draw_waiting_report(self, missing_side: str, avail_speed: Optional[int], avail_intervals: Optional[int]) -> np.ndarray:
        w, h = 960, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)
        def put(y, text, color=(255, 255, 255), scale=0.8, thick=2):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        put(34, "LED Sync Report", (255, 255, 0))
        put(90, f"Waiting for stream {missing_side}…", (0, 165, 255))
        if avail_speed is not None and avail_intervals is not None:
            put(140, f"Seen config on other side → Speed={avail_speed}, Intervals=0b{avail_intervals:016b}")
        put(190, "Comparison paused until both streams are available.")
        return img

    # --- Visualization ----------------------------------------------------
    def _draw_overlay(self, maskA: np.ndarray, maskB: np.ndarray) -> np.ndarray:
        """
        Color code per cell:
          - both ON   : white
          - A only    : green
          - B only    : red
          - both OFF  : dark gray
        """
        h, w = self.grid_size, self.grid_size
        out = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)

        for r in range(h):
            for c in range(w):
                y1 = r * self.cell_h
                y2 = min((r + 1) * self.cell_h, self.output_h)
                x1 = c * self.cell_w
                x2 = min((c + 1) * self.cell_w, self.output_w)

                a = bool(maskA[r, c])
                b = bool(maskB[r, c])
                if a and b:
                    color = (255, 255, 255)  # white
                elif a and not b:
                    color = (0, 255, 0)      # green
                elif b and not a:
                    color = (0, 0, 255)      # red
                else:
                    color = (40, 40, 40)     # dark gray

                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, -1)
                # subtle grid
                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (90, 90, 90), 1)

        # Legend
        cv2.rectangle(out, (10, 10), (22, 22), (255, 255, 255), -1)
        cv2.putText(out, "Both ON", (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(out, (10, 36), (22, 48), (0, 255, 0), -1)
        cv2.putText(out, "A only", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(out, (10, 62), (22, 74), (0, 0, 255), -1)
        cv2.putText(out, "B only", (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return out

    def _draw_report(
        self,
        cfg_ok: bool,
        speed: int,
        intervals: int,
        dt_us_abs: int,
        shift_cols: int,
        intervals_offset: int,
        intervals_offset_real: float,
        lead_text: str,
        onA: int,
        onB: int,
        overlap: int,
        recallA: float,
        recallB: float,
        iou: float,
        passed: Optional[bool],
    ) -> np.ndarray:
        w, h = 960, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)

        def put(y, text, color=(255, 255, 255), scale=0.7, thick=2):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        # Titles
        put(30, "LED Sync Report", (255, 255, 0))

        # Config
        cfg_text = f"Config check (Speed={speed}, Intervals=0b{intervals:016b})"
        # Constants for report
        num_leds_no_config = self.grid_size * (self.grid_size - 1)
        put(55, f"LEDs considered (no config row) = {num_leds_no_config} | Period = {int(self.led_period_us)} \u00B5s", (200, 200, 200))
        if cfg_ok:
            put(80, cfg_text + " -> MATCH", (0, 255, 0))
        else:
            put(80, cfg_text + " -> MISMATCH", (0, 165, 255))
            put(100, "Skipping LED placement comparison due to config mismatch.", (0, 165, 255))

        # Timing / shift
        put(110, f"Δt ≈ {dt_us_abs} us   |   Column shift ≈ {shift_cols}")
        put(127, f"Intervals offset (int) ≈ {intervals_offset}  ({lead_text})")
        dt_seconds = intervals_offset_real * (self.led_period_us / 1e6)
        put(145, f"Intervals offset (real) ≈ {intervals_offset_real:.3f}   |   Δt ≈ {dt_seconds:.6f} s")

        # Metrics
        put(175, f"ON A={onA}, ON B={onB}, Overlap={overlap}")
        put(200, f"RecallA={recallA:.3f}, RecallB={recallB:.3f}, IoU={iou:.3f}")
        if passed is None:
            put(230, "Verdict: N/A (config mismatch)", (0, 165, 255), scale=0.9)
        else:
            verdict = "PASS" if passed else "FAIL"
            color = (0, 255, 0) if passed else (0, 0, 255)
            put(230, f"Verdict: {verdict}  (threshold {self.pass_ratio:.0%} on both recalls)", color, scale=0.9)

        return img

    # --- Main loop --------------------------------------------------------
    def run(self) -> None:
        print(
            f"LEDGridComparison started: {self.grid_size}x{self.grid_size}, out={self.output_w}x{self.output_h}, "
            f"period={self.led_period_us}us, pass_ratio={self.pass_ratio:.2f}"
        )
        if self._qA is None or self._qB is None:
            print("LEDGridComparison warning: queues not set yet. Waiting...")
            now = _t.time()
            if now - self._last_placeholder_time > 0.5:
                self._send_placeholder()
                self._last_placeholder_time = now

        last_seqA = -1
        last_seqB = -1

        while self.isRunning():
            try:
                # Drain tick input (non-blocking) to avoid pipeline backpressure
                try:
                    while True:
                        try:
                            m = self._tickIn.tryGet()
                        except AttributeError:
                            if not self._tickIn.has():
                                break
                            m = self._tickIn.get()
                        if m is None:
                            break
                except Exception:
                    pass

                # Ensure queues available
                if self._qA is None or self._qB is None:
                    now = _t.time()
                    if now - self._last_placeholder_time > 0.5:
                        self._send_placeholder()
                        self._last_placeholder_time = now
                    _t.sleep(0.005)
                    continue

                # Drain both queues to the latest elements (non-blocking preferred)
                bufA: Optional[dai.Buffer] = None
                bufB: Optional[dai.Buffer] = None
                try:
                    while True:
                        m = getattr(self._qA, "tryGet", None)
                        m = m() if m is not None else self._qA.get()  # tryGet() if exists, else blocking get()
                        if m is None:
                            break
                        bufA = m
                except Exception:
                    bufA = None

                try:
                    while True:
                        m = getattr(self._qB, "tryGet", None)
                        m = m() if m is not None else self._qB.get()
                        if m is None:
                            break
                        bufB = m
                except Exception:
                    bufB = None

                # Parse incoming packets and update last states
                parsedA = parsedB = False
                try:
                    if bufA is not None:
                        self._lastA = self._parse_buffer(bufA)
                        parsedA = True
                    if bufB is not None:
                        self._lastB = self._parse_buffer(bufB)
                        parsedB = True
                except Exception as e:
                    print(f"Comparison parse error: {e}")
                    # If parsing failed for this iteration, continue (we may still have old state)
                    pass

                # If neither side has any state yet → keep placeholders and wait
                if self._lastA is None and self._lastB is None:
                    now = _t.time()
                    if now - self._last_placeholder_time > 0.5:
                        self._send_placeholder("Waiting for LED analyzer streams…", "No packets received yet from either side.")
                        self._last_placeholder_time = now
                    _t.sleep(0.001)
                    continue

                # If only one side available → render degraded overlay (one mask vs empty)
                if (self._lastA is None) ^ (self._lastB is None):
                    side_have = 'A' if self._lastA is not None else 'B'
                    side_miss = 'B' if side_have == 'A' else 'A'
                    if self._lastA is not None:
                        grid, avg, mult, speed, intervals, ts, seq = self._lastA
                        thr = self._dynamic_threshold(avg, mult)
                        mask_have = self._mask_from_grid(grid, thr)
                        mask_missing = np.zeros_like(mask_have)
                        ts_use, seq_use = ts, seq
                        speed_use, intervals_use = speed, intervals
                    else:
                        grid, avg, mult, speed, intervals, ts, seq = self._lastB
                        thr = self._dynamic_threshold(avg, mult)
                        mask_have = self._mask_from_grid(grid, thr)
                        mask_missing = np.zeros_like(mask_have)
                        ts_use, seq_use = ts, seq
                        speed_use, intervals_use = speed, intervals

                    # Full-size overlay (include config row) showing one side vs empty
                    overlay_img = self._draw_overlay(mask_have if side_have=='A' else mask_missing,
                                                     mask_have if side_have=='B' else mask_missing)
                    overlay_frame = self._create_imgframe(overlay_img, ts_use, seq_use)
                    self.out_overlay.send(overlay_frame)

                    report_img = self._draw_waiting_report(side_miss, speed_use, intervals_use)
                    report_frame = self._create_imgframe(report_img, ts_use, seq_use)
                    self.out_report.send(report_frame)
                    _t.sleep(0.001)
                    continue

                # From here on, both sides have some state (may be updated this tick)
                gridA, avgA, multA, speedA, intervalsA, tsA, seqA = self._lastA
                gridB, avgB, multB, speedB, intervalsB, tsB, seqB = self._lastB

                # Avoid re-processing identical seq pairs
                if seqA == last_seqA and seqB == last_seqB:
                    _t.sleep(0.0005)
                    continue
                last_seqA, last_seqB = seqA, seqB

                # Config check (strict match)
                cfg_ok = (speedA == speedB) and (intervalsA == intervalsB)

                # Build masks using each stream's dynamic threshold
                thrA = self._dynamic_threshold(avgA, multA)
                thrB = self._dynamic_threshold(avgB, multB)
                maskA_full = self._mask_from_grid(gridA, thrA)
                maskB_full = self._mask_from_grid(gridB, thrB)

                # Estimate best column alignment by scoring IoU across shifts (robust to unsynced clocks)
                A_eval = maskA_full[:-1, :]
                max_shift = self.grid_size  # search full width
                best_s = 0
                best_iou_tmp = -1.0
                scores = {}
                for s in range(-max_shift, max_shift + 1):
                    B_eval = np.roll(maskB_full[:-1, :], s, axis=1)
                    overlap_tmp = int(np.logical_and(A_eval, B_eval).sum())
                    union_tmp = int(np.logical_or(A_eval, B_eval).sum())
                    iou_tmp = (overlap_tmp / union_tmp) if union_tmp > 0 else 0.0
                    scores[s] = iou_tmp
                    if iou_tmp > best_iou_tmp:
                        best_iou_tmp = iou_tmp
                        best_s = s
                shift_cols_signed = int(best_s)
                # Quadratic interpolation for sub-column (real) shift if neighbors exist
                left = scores.get(shift_cols_signed - 1, None)
                center = scores.get(shift_cols_signed, None)
                right = scores.get(shift_cols_signed + 1, None)
                shift_cols_real = float(shift_cols_signed)
                if left is not None and center is not None and right is not None:
                    denom = (left - 2 * center + right)
                    if abs(denom) > 1e-9:
                        shift_cols_real = shift_cols_signed + 0.5 * (left - right) / denom
                shiftedB_full = self._roll_columns(maskB_full, int(round(shift_cols_real)))
                # Δt from *real* offset and configured LED period (in microseconds)
                dt_us_abs = int(abs(shift_cols_real) * self.led_period_us)
                intervals_offset_real = abs(shift_cols_real)

                # Prepare overlay image from full masks (includes bottom row)
                overlay_img = self._draw_overlay(maskA_full, shiftedB_full)
                overlay_frame = self._create_imgframe(overlay_img, tsA, max(seqA, seqB))
                self.out_overlay.send(overlay_frame)

                # Ensure lead_text uses sign of shift_cols_real
                if shift_cols_real > 0:
                    lead_text = "B lags A"
                elif shift_cols_real < 0:
                    lead_text = "A lags B"
                else:
                    lead_text = "aligned"

                intervals_offset = abs(shift_cols_signed)
                # If config mismatched -> report SKIP and continue
                if not cfg_ok:
                    report_img = self._draw_report(
                        cfg_ok=False,
                        speed=max(speedA, speedB),
                        intervals=max(intervalsA, intervalsB),
                        dt_us_abs=dt_us_abs,
                        shift_cols=abs(shift_cols_signed),
                        intervals_offset=intervals_offset,
                        intervals_offset_real=intervals_offset_real,
                        lead_text=lead_text,
                        onA=0, onB=0, overlap=0,
                        recallA=0.0, recallB=0.0, iou=0.0,
                        passed=None
                    )
                    report_frame = self._create_imgframe(report_img, tsA, max(seqA, seqB))
                    self.out_report.send(report_frame)
                    continue

                # Exclude bottom configuration row for metrics
                maskA = maskA_full[:-1, :]
                shiftedB = shiftedB_full[:-1, :]

                # Overlap metrics
                onA = int(maskA.sum())
                onB = int(shiftedB.sum())
                overlap = int(np.logical_and(maskA, shiftedB).sum())
                union = onA + onB - overlap

                recallA = (overlap / onA) if onA > 0 else 0.0
                recallB = (overlap / onB) if onB > 0 else 0.0
                iou = (overlap / union) if union > 0 else 0.0

                passed = (min(recallA, recallB) >= self.pass_ratio)

                # Report
                report_img = self._draw_report(
                    cfg_ok=True,
                    speed=speedA,
                    intervals=intervalsA,
                    dt_us_abs=dt_us_abs,
                    shift_cols=abs(shift_cols_signed),
                    intervals_offset=intervals_offset,
                    intervals_offset_real=intervals_offset_real,
                    lead_text=lead_text,
                    onA=onA, onB=onB, overlap=overlap,
                    recallA=recallA, recallB=recallB, iou=iou,
                    passed=passed
                )
                report_frame = self._create_imgframe(report_img, tsA, max(seqA, seqB))
                self.out_report.send(report_frame)

            except Exception as e:
                print(f"LEDGridComparison error: {e}")
                continue