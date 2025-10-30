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
         If intervals differ between the two panels, SKIP comparison (report only). Speed is ignored.
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
        pass_ratio: float = 0.90,
        sync_threshold_sec: Optional[float] = None,
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

        # Analyzer inputs (optional). If set_queues receives Node.Output, we'll link here.
        self._inA = self.createInput()
        self._inA.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        try:
            self._inA.setBlocking(False)
            self._inA.setQueueSize(1)
        except AttributeError:
            pass

        self._inB = self.createInput()
        self._inB.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        try:
            self._inB.setBlocking(False)
            self._inB.setQueueSize(1)
        except AttributeError:
            pass

        self.grid_size = grid_size
        self.output_w, self.output_h = output_size
        self.cell_w = max(1, self.output_w // self.grid_size)
        self.cell_h = max(1, self.output_h // self.grid_size)
        self.led_period_us = float(led_period_us)
        self.pass_ratio = float(pass_ratio)
        self.sync_threshold_sec = sync_threshold_sec
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
        """Wire analyzer outputs. Accepts either Node.Output (host-to-host link) or OutputQueue."""
        # If objects have `link`, treat them as Node.Output and link into our inputs.
        if hasattr(qA, "link") and hasattr(qB, "link"):
            qA.link(self._inA)
            qB.link(self._inB)
            self._qA = None
            self._qB = None
        else:
            # Assume host-side OutputQueues
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

    def _unwrap16(self, delta: int) -> int:
        d = int(delta) & 0xFFFF
        return d - 0x10000 if d >= 0x8000 else d


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
        report = np.zeros((240, 1440, 3), dtype=np.uint8)
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
        w, h = 1440, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)
        def put(y, text, color=(255, 255, 255), scale=0.8, thick=2):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        put(34, "LED Sync Report", (255, 255, 0))
        put(90, f"Waiting for stream {missing_side}...", (0, 165, 255))
        if avail_speed is not None and avail_intervals is not None:
            put(140, f"Seen config on other side -> Speed={avail_speed}, Intervals={int(avail_intervals) if avail_intervals is not None else '-'}")
        put(190, "Comparison paused until both streams are available.")
        return img

    # --- Visualization ----------------------------------------------------

    def _active_index(self, mask_full: np.ndarray):
        """Return (idx_int, idx_real) along the raster scan (row-major),
        excluding the bottom config row. If no ON cells, return None.
        idx spans [0, (grid_size-1)*grid_size - 1]."""
        if mask_full.ndim != 2:
            return None
        if mask_full.shape[0] < 2 or mask_full.shape[1] < 1:
            return None
        eval_mask = mask_full[:-1, :]  # exclude bottom config row
        ys, xs = np.nonzero(eval_mask)
        if ys.size == 0:
            return None
        # centroid for robustness (handles slight blurs or multiple pixels)
        r_mean = float(ys.mean())
        c_mean = float(xs.mean())
        W = self.grid_size
        H = self.grid_size - 1
        N = W * H
        idx_real = r_mean * W + c_mean
        idx_int = int(round(idx_real))
        idx_int = max(0, min(idx_int, N - 1))
        return idx_int, float(idx_real)
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
        dt_squares_sec: float,
        shift_cols: int,
        shift_cols_real: float,
        squares_forward_int: int,
        squares_forward_real: float,
        intervals_diff_signed: int,
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
        passed_by_time: bool = False,
    ) -> np.ndarray:
        # Canvas
        W, H = 1440, 240
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (18, 18, 18)

        # Helpers
        def put(x, y, text, color=(230, 230, 230), scale=0.8, thick=2):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def row(y, label, value, value_color=(255, 255, 255)):
            put(16, y, label, (180, 180, 180))
            put(260, y, value, value_color)

        # Header
        put(16, 32, "LED Sync Report", (255, 255, 0), 0.95, 2)
        cv2.line(img, (16, 40), (W - 16, 40), (70, 70, 70), 1)

        # Config
        cfg_str = f"Speed={speed}, Intervals={int(intervals)}"
        if cfg_ok:
            row(68, "Config:", cfg_str + "  -> MATCH", (0, 255, 0))
        else:
            row(68, "Config:", cfg_str + "  -> MISMATCH", (0, 165, 255))
            row(92, "Note:", "Skipping placement comparison due to config mismatch.", (0, 165, 255))

        # Timing block
        row(118, "Timing:", f"dT (from squares) = {dt_squares_sec:.6f} s   |   LED period = {int(self.led_period_us)} us")
        row(142, "Shift:", f"{squares_forward_real:.3f} squares A->B (int {squares_forward_int}) | cols: {shift_cols_real:+.3f} (int {shift_cols}) | Delta intervals (B - A) = {int(intervals_diff_signed)}")
        row(166, "Order:", lead_text)

        # Metrics block (excludes config row)
        row(190, "Metrics:", f"onA={onA}, onB={onB}, overlap={overlap}  |  recallA={recallA:.3f}, recallB={recallB:.3f}, IoU={iou:.3f}")

        # Verdict banner
        if passed is None:
            banner_text = "VERDICT: N/A (config mismatch)"
            banner_color = (0, 165, 255)
            sub_text = ""
        else:
            if passed:
                banner_text = "VERDICT: PASS"
                banner_color = (0, 180, 0)
                if passed_by_time and self.sync_threshold_sec is not None:
                    sub_text = f"time threshold: dT <= {self.sync_threshold_sec:.3f}s"
                else:
                    sub_text = f"recall threshold: min(recallA, recallB) >= {self.pass_ratio:.0%}"
            else:
                banner_text = "VERDICT: FAIL"
                banner_color = (0, 0, 220)
                sub_text = f"recall threshold not met ({self.pass_ratio:.0%})"

        # Draw banner as a rounded rectangle substitute
        x1, y1, x2, y2 = 16, H - 44, W - 16, H - 12
        cv2.rectangle(img, (x1, y1), (x2, y2), banner_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        put(26, H - 20, banner_text, (255, 255, 255), 0.8, 2)
        if sub_text:
            put(330, H - 20, f"({sub_text})", (240, 240, 240), 0.7, 2)

        return img

    # --- Main loop --------------------------------------------------------
    def run(self) -> None:
        print(
            f"LEDGridComparison started: {self.grid_size}x{self.grid_size}, out={self.output_w}x{self.output_h}, "
            f"period={self.led_period_us}us, pass_ratio={self.pass_ratio:.2f}"
        )

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


                # Drain latest packets from either queues (if provided) or from inputs (if linked)
                bufA: Optional[dai.Buffer] = None
                bufB: Optional[dai.Buffer] = None

                if self._qA is not None:
                    try:
                        while True:
                            m = getattr(self._qA, "tryGet", None)
                            m = m() if m is not None else self._qA.get()
                            if m is None:
                                break
                            bufA = m
                    except Exception:
                        bufA = None
                else:
                    try:
                        while True:
                            try:
                                m = self._inA.tryGet()
                            except AttributeError:
                                if not self._inA.has():
                                    break
                                m = self._inA.get()
                            if m is None:
                                break
                            bufA = m
                    except Exception:
                        bufA = None

                if self._qB is not None:
                    try:
                        while True:
                            m = getattr(self._qB, "tryGet", None)
                            m = m() if m is not None else self._qB.get()
                            if m is None:
                                break
                            bufB = m
                    except Exception:
                        bufB = None
                else:
                    try:
                        while True:
                            try:
                                m = self._inB.tryGet()
                            except AttributeError:
                                if not self._inB.has():
                                    break
                                m = self._inB.get()
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
                        self._send_placeholder("Waiting for LED analyzer streams...", "No packets received yet from either side.")
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
                cfg_ok = (intervalsA == intervalsB)

                # Build masks using each stream's dynamic threshold
                thrA = self._dynamic_threshold(avgA, multA)
                thrB = self._dynamic_threshold(avgB, multB)
                maskA_full = self._mask_from_grid(gridA, thrA)
                maskB_full = self._mask_from_grid(gridB, thrB)

                # Squares + intervals-based shift (A->B). Combine centroid index difference with 16-bit cycle delta.
                idxA = self._active_index(maskA_full)
                idxB = self._active_index(maskB_full)
                W = self.grid_size
                H = self.grid_size - 1
                N = W * H
                if idxA is not None and idxB is not None and N > 0:
                    idxA_int, idxA_real = idxA
                    idxB_int, idxB_real = idxB
                    cycles_diff_signed = self._unwrap16(int(intervalsB) - int(intervalsA))
                    squares_forward_real = (cycles_diff_signed * N + (idxB_real - idxA_real)) % N
                    squares_forward_int = int(round(squares_forward_real)) % N
                else:
                    squares_forward_real = 0.0
                    squares_forward_int = 0

                # Time difference from LED dwell time (each square lasts self.led_period_us microseconds)
                dt_squares_sec = (squares_forward_real * self.led_period_us) / 1e6

                # Determine which stream is in front/back using the shorter forward path (ASCII text, A->B / B->A)
                if squares_forward_int == 0:
                    lead_text = "Aligned (A==B)"
                else:
                    if squares_forward_real <= (N / 2.0):
                        lead_text = f"Front: B | Back: A (A->B = {squares_forward_real:.3f} squares, {dt_squares_sec:.6f} s)"
                    else:
                        comp_squares = N - squares_forward_real
                        comp_time = (comp_squares * self.led_period_us) / 1e6
                        lead_text = f"Front: A | Back: B (B->A = {comp_squares:.3f} squares, {comp_time:.6f} s)"

                # Compute signed intervals difference (B - A)
                intervals_diff_signed = self._unwrap16(int(intervalsB) - int(intervalsA))

                # --- Timestamp-based timing (fallback) and degenerate detection ---
                # Exclude bottom config row for activity check
                _A_eval = maskA_full[:-1, :]
                _B_eval = maskB_full[:-1, :]
                onA_full_raw = int(_A_eval.sum())
                onB_full_raw = int(_B_eval.sum())
                # Absolute timestamp delta (microseconds) and equivalent intervals at configured LED period
                ts_delta_sec = (tsA - tsB).total_seconds()
                ts_delta_us = int(abs(ts_delta_sec) * 1e6)
                intervals_from_ts_real = (ts_delta_us / self.led_period_us) if self.led_period_us > 0 else 0.0
                # Signed real intervals based on which side lags by timestamp
                signed_intervals_from_ts_real = (
                    intervals_from_ts_real if ts_delta_sec > 0 else
                    (-intervals_from_ts_real if ts_delta_sec < 0 else 0.0)
                )
                # Decide whether IoU-based shift is meaningful: require both sides have some LEDs and config match
                use_iou_alignment = (onA_full_raw > 0 and onB_full_raw > 0 and (intervalsA == intervalsB))

                if use_iou_alignment:
                    # Estimate best column alignment by scoring IoU across shifts (robust to unsynced clocks)
                    A_eval = _A_eval
                    max_shift = self.grid_size  # search full width
                    best_s = 0
                    best_iou_tmp = -1.0
                    scores = {}
                    for s in range(-max_shift, max_shift + 1):
                        B_eval = np.roll(_B_eval, s, axis=1)
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
                else:
                    # Fallback: use host timestamps to estimate real interval offset and direction
                    shift_cols_real = signed_intervals_from_ts_real
                    shift_cols_signed = int(round(shift_cols_real))
                    shiftedB_full = self._roll_columns(maskB_full, shift_cols_signed)
                    dt_us_abs = ts_delta_us
                    intervals_offset_real = abs(shift_cols_real)

                # Prepare overlay image from full masks (includes bottom row)
                overlay_img = self._draw_overlay(maskA_full, shiftedB_full)
                overlay_frame = self._create_imgframe(overlay_img, tsA, max(seqA, seqB))
                self.out_overlay.send(overlay_frame)

                intervals_offset = abs(shift_cols_signed)
                # If config mismatched -> report SKIP and continue
                if not cfg_ok:
                    report_img = self._draw_report(
                        cfg_ok=False,
                        speed=max(speedA, speedB),
                        intervals=max(intervalsA, intervalsB),
                        dt_us_abs=dt_us_abs,
                        dt_squares_sec=dt_squares_sec,
                        shift_cols=shift_cols_signed,
                        shift_cols_real=shift_cols_real,
                        squares_forward_int=squares_forward_int,
                        squares_forward_real=squares_forward_real,
                        intervals_diff_signed=intervals_diff_signed,
                        intervals_offset=intervals_offset,
                        intervals_offset_real=intervals_offset_real,
                        lead_text=lead_text + (" (from TS)" if not use_iou_alignment else ""),
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

                dt_seconds = dt_squares_sec
                time_pass = (self.sync_threshold_sec is not None and dt_squares_sec <= self.sync_threshold_sec)
                passed = time_pass or (min(recallA, recallB) >= self.pass_ratio)

                # Report
                report_img = self._draw_report(
                    cfg_ok=True,
                    speed=speedA,
                    intervals=intervalsA,
                    dt_us_abs=dt_us_abs,
                    dt_squares_sec=dt_squares_sec,
                    shift_cols=shift_cols_signed,
                    shift_cols_real=shift_cols_real,
                    squares_forward_int=squares_forward_int,
                    squares_forward_real=squares_forward_real,
                    intervals_diff_signed=intervals_diff_signed,
                    intervals_offset=intervals_offset,
                    intervals_offset_real=intervals_offset_real,
                    lead_text=lead_text,
                    onA=onA, onB=onB, overlap=overlap,
                    recallA=recallA, recallB=recallB, iou=iou,
                    passed=passed,
                    passed_by_time=time_pass,
                )
                report_frame = self._create_imgframe(report_img, tsA, max(seqA, seqB))
                self.out_report.send(report_frame)

            except Exception as e:
                print(f"LEDGridComparison error: {e}")
                continue