

import cv2
import numpy as np
import depthai as dai
from typing import Optional, Tuple
from datetime import timedelta


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
        led_period_us: float = 160.0,
        pass_ratio: float = 0.90
    ) -> None:
        super().__init__()

        # No inputs; comparison consumes host queues set via set_queues()
        self.out_overlay = self.createOutput()
        self.out_overlay.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.out_report = self.createOutput()
        self.out_report.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

        self.grid_size = grid_size
        self.output_w, self.output_h = output_size
        self.cell_w = max(1, self.output_w // self.grid_size)
        self.cell_h = max(1, self.output_h // self.grid_size)
        self.led_period_us = float(led_period_us)
        self.pass_ratio = float(pass_ratio)

        # Host-side queues are provided from analyzer node outputs
        self._qA = None
        self._qB = None
        # Keep last tick for timestamps in placeholder frames
        self._last_tick_ts = None
        self._last_tick_seq = 0

    # --- Public API -------------------------------------------------------
    def build(self, tick_source: Optional[dai.Node.Output] = None) -> "LEDGridComparison":
        """
        Optionally attach this host node to a pipeline by linking any stream as a lightweight 'tick'.
        If no tick_source is provided, the node will still run but won't be bound to a specific stream.
        """
        if tick_source is not None:
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
        if cfg_ok:
            put(70, cfg_text + " -> MATCH", (0, 255, 0))
        else:
            put(70, cfg_text + " -> MISMATCH", (0, 165, 255))
            put(100, "Skipping LED placement comparison due to config mismatch.", (0, 165, 255))
            return img

        # Timing / shift
        put(110, f"Δt ≈ {dt_us_abs} us   |   Column shift ≈ {shift_cols}")

        # Metrics
        put(145, f"ON A={onA}, ON B={onB}, Overlap={overlap}")
        put(175, f"RecallA={recallA:.3f}, RecallB={recallB:.3f}, IoU={iou:.3f}")

        verdict = "PASS" if passed else "FAIL"
        color = (0, 255, 0) if passed else (0, 0, 255)
        put(210, f"Verdict: {verdict}  (threshold {self.pass_ratio:.0%} on both recalls)", color, scale=0.9)

        return img

    def _now_ts(self):
        # Fallback timestamp if no tick available
        return self._last_tick_ts if self._last_tick_ts is not None else timedelta(seconds=0)

    def _draw_placeholders(self, msgA: str, msgB: str) -> Tuple[np.ndarray, np.ndarray]:
        # Overlay placeholder
        overlay = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        cv2.putText(overlay, "LED Overlay [comparison]", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(overlay, msgA, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(overlay, msgB, (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # Report placeholder
        report = np.zeros((240, 960, 3), dtype=np.uint8)
        cv2.putText(report, "LED Sync Report", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(report, msgA, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(report, msgB, (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(report, "Streams will update automatically once inputs are ready.", (16, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        return overlay, report

    # --- Main loop --------------------------------------------------------
    def run(self) -> None:
        print(
            f"LEDGridComparison started: {self.grid_size}x{self.grid_size}, out={self.output_w}x{self.output_h}, "
            f"period={self.led_period_us}us, pass_ratio={self.pass_ratio:.2f}"
        )
        if self._qA is None or self._qB is None:
            print("LEDGridComparison warning: queues not set yet. Waiting...")

        last_seqA = -1
        last_seqB = -1

        while self.isRunning():
            try:
                # Ensure queues available
                if self._qA is None or self._qB is None:
                    overlay_img, report_img = self._draw_placeholders(
                        "Waiting for analyzer queues...",
                        "Provide two LEDGridAnalyzer outputs to compare."
                    )
                    ts = self._now_ts()
                    seq = self._last_tick_seq
                    self.out_overlay.send(self._create_imgframe(overlay_img, ts, seq))
                    self.out_report.send(self._create_imgframe(report_img, ts, seq))
                    import time as _t
                    _t.sleep(0.03)
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
                        try:
                            self._last_tick_ts = m.getTimestamp()
                            self._last_tick_seq = m.getSequenceNum()
                        except Exception:
                            pass
                        bufA = m
                except Exception:
                    bufA = None

                try:
                    while True:
                        m = getattr(self._qB, "tryGet", None)
                        m = m() if m is not None else self._qB.get()
                        if m is None:
                            break
                        try:
                            self._last_tick_ts = m.getTimestamp()
                            self._last_tick_seq = m.getSequenceNum()
                        except Exception:
                            pass
                        bufB = m
                except Exception:
                    bufB = None

                if bufA is None or bufB is None:
                    needA = "OK" if bufA is not None else "Waiting for stream A..."
                    needB = "OK" if bufB is not None else "Waiting for stream B..."
                    overlay_img, report_img = self._draw_placeholders(needA, needB)
                    ts = self._now_ts()
                    seq = self._last_tick_seq
                    self.out_overlay.send(self._create_imgframe(overlay_img, ts, seq))
                    self.out_report.send(self._create_imgframe(report_img, ts, seq))
                    import time as _t
                    _t.sleep(0.03)
                    continue

                # Parse
                try:
                    gridA, avgA, multA, speedA, intervalsA, tsA, seqA = self._parse_buffer(bufA)
                    gridB, avgB, multB, speedB, intervalsB, tsB, seqB = self._parse_buffer(bufB)
                except Exception as e:
                    print(f"Comparison parse error: {e}")
                    continue

                # Avoid re-processing identical seq pairs
                if seqA == last_seqA and seqB == last_seqB:
                    import time as _t
                    _t.sleep(0.0005)
                    continue
                last_seqA, last_seqB = seqA, seqB

                # Config check
                cfg_ok = (speedA == speedB) and (intervalsA == intervalsB)

                # Build masks using each stream's dynamic threshold
                thrA = self._dynamic_threshold(avgA, multA)
                thrB = self._dynamic_threshold(avgB, multB)
                maskA_full = self._mask_from_grid(gridA, thrA)
                maskB_full = self._mask_from_grid(gridB, thrB)

                # Compute time delta and column shift (use signed shift to roll earlier stream forward)
                dt = tsA - tsB
                dt_us = int(round(dt.total_seconds() * 1e6))
                dt_us_abs = abs(dt_us)
                # Positive dt_us -> A is newer; roll B forward by +shift_cols
                shift_cols_signed = int(round(dt_us / self.led_period_us))
                shiftedB_full = self._roll_columns(maskB_full, shift_cols_signed)

                # Prepare overlay image from full masks (includes bottom row)
                overlay_img = self._draw_overlay(maskA_full, shiftedB_full)
                overlay_frame = self._create_imgframe(overlay_img, tsA if dt_us >= 0 else tsB, max(seqA, seqB))
                self.out_overlay.send(overlay_frame)

                # If config mismatched -> report SKIP and continue
                if not cfg_ok:
                    report_img = self._draw_report(
                        cfg_ok=False,
                        speed=max(speedA, speedB),
                        intervals=max(intervalsA, intervalsB),
                        dt_us_abs=dt_us_abs,
                        shift_cols=abs(shift_cols_signed),
                        onA=0, onB=0, overlap=0,
                        recallA=0.0, recallB=0.0, iou=0.0,
                        passed=None
                    )
                    report_frame = self._create_imgframe(report_img, tsA if dt_us >= 0 else tsB, max(seqA, seqB))
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
                    onA=onA, onB=onB, overlap=overlap,
                    recallA=recallA, recallB=recallB, iou=iou,
                    passed=passed
                )
                report_frame = self._create_imgframe(report_img, tsA if dt_us >= 0 else tsB, max(seqA, seqB))
                self.out_report.send(report_frame)

            except Exception as e:
                print(f"LEDGridComparison error: {e}")
                continue