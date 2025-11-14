import cv2
import numpy as np
import depthai as dai
from typing import Optional, Tuple
import time as _t
import colorsys

from datetime import timedelta as _td
from collections import deque


class LEDGridComparison(dai.node.ThreadedHostNode):
    """
    Compares N LED grid analyzer streams and determines if they are in sync.

    Inputs: (set via set_queues) list of host-side OutputQueues that receive dai.Buffer from LEDGridAnalyzer
            Each buffer encodes:
              - 32x32 float grid_state (values in [0..1], bottom row pre-scaled by analyzer)
              - 4 floats metadata: [overall_avg_brightness, threshold_multiplier, speed, intervals]
    Outputs:
      - out_overlay: dai.ImgFrame (BGR) overlay of LED masks (shows all inputs)
      - out_report:  dai.ImgFrame (BGR) textual report with pass/fail and metrics including average dT

    Logic:
      1) Check configuration on bottom row:
         - first 16 cells encode "speed" (count of lit LEDs) and is provided in metadata
         - last 16 cells encode "intervals" (binary) and is provided in metadata
         If intervals differ between panels, SKIP comparison (report only). Speed is ignored.
      2) If config matches, compare placement of "green" LEDs (mask = grid_state > dynamic_threshold).
         - dynamic_threshold = overall_avg_brightness * threshold_multiplier (same as visualizer)
         - Exclude bottom config row from the metric.
         - For each pair of inputs, calculate square dT (time delta in seconds squared)
         - Compute average square dT across all pairs
         - Use buffer timestamps to estimate temporal offset for alignment
         - Compute overlap metrics (excluding the bottom row) for all pairs
           PASS if average metrics meet thresholds.
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
        # We'll create inputs dynamically as needed
        self._inputs = []  # List of input nodes, created dynamically

        self.grid_size = grid_size
        self.output_w, self.output_h = output_size
        self.cell_w = max(1, self.output_w // self.grid_size)
        self.cell_h = max(1, self.output_h // self.grid_size)
        self.led_period_us = float(led_period_us)
        self.pass_ratio = float(pass_ratio)
        self.sync_threshold_sec = sync_threshold_sec
        self._last_placeholder_time = 0.0
        self._seq_counter = 1
        self._last_states = []  # List of (grid, avg, mult, speed, intervals, ts, seq) tuples
        # Track last compared sequence numbers for each input
        self._compared_seqs = []  # List of sequence numbers
        # Small buffers to tolerate skipped/misaligned frames and always pair latest-to-latest
        self._buffers = []  # List of deques, one per input
        # Throttle for "waiting" overlay/report updates when inputs are missing
        self._last_waiting_time = 0.0

        # Host-side queues are provided from analyzer node outputs
        self._queues = []  # List of queues

    # --- Public API -------------------------------------------------------
    def build(self, tick_source: dai.Node.Output) -> "LEDGridComparison":
        """
        Attach this host node to a pipeline by linking any stream as a lightweight 'tick'.
        We don't read payloads from this input for logic; it's only to bind/schedule the node.
        """
        tick_source.link(self._tickIn)
        return self

    def set_queues(self, *queues) -> None:
        """
        Wire analyzer outputs. Accepts either Node.Output (host-to-host link) or OutputQueue.
        Can accept multiple queues as separate arguments or a single list.
        """
        # Handle both list and multiple arguments
        if len(queues) == 1 and isinstance(queues[0], (list, tuple)):
            queues = queues[0]
        
        # Clear existing state
        self._queues = []
        self._inputs = []
        self._last_states = []
        self._compared_seqs = []
        self._buffers = []
        
        # Check if we have Node.Output objects (have 'link' method) or OutputQueues
        has_link = all(hasattr(q, "link") for q in queues) if queues else False
        
        if has_link:
            # Create inputs dynamically and link
            for q in queues:
                inp = self.createInput()
                inp.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
                try:
                    inp.setBlocking(False)
                    inp.setQueueSize(1)
                except AttributeError:
                    pass
                q.link(inp)
                self._inputs.append(inp)
            self._queues = [None] * len(queues)
        else:
            # Assume host-side OutputQueues
            self._queues = list(queues)
            self._inputs = [None] * len(queues)
        
        # Initialize state for each input
        self._last_states = [None] * len(queues)
        self._compared_seqs = [-1] * len(queues)
        self._buffers = [deque(maxlen=8) for _ in queues]

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

    def _pop_latest_frames(self):
        """
        Return the latest parsed tuples for all inputs if all have at least one item.
        This gets the most recent frames from all inputs, dropping any older ones.
        Each tuple layout: (grid, avg, mult, speed, intervals, ts, seq)
        Returns list of tuples, or None if not all inputs have data.
        """
        if len(self._buffers) == 0:
            return None
        if any(len(buf) == 0 for buf in self._buffers):
            return None
        frames = [buf[-1] for buf in self._buffers]
        # Clear all buffers after extracting latest
        for buf in self._buffers:
            buf.clear()
        return frames

    def _create_imgframe(self, bgr: np.ndarray, ts, seq: int) -> dai.ImgFrame:
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888i)
        img.setWidth(bgr.shape[1])
        img.setHeight(bgr.shape[0])
        img.setData(bgr.tobytes())
        img.setSequenceNum(int(seq))
        img.setTimestamp(ts)
        return img

    def _send_placeholder(self, text_line1: str = "Waiting for LED analyzer streams...", text_line2: Optional[str] = "Connect devices and ensure LEDGridAnalyzer is running.") -> None:
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

    def _draw_waiting_report(self, missing_indices: list, avail_speed: Optional[int], avail_intervals: Optional[int]) -> np.ndarray:
        w, h = 1440, 240
        img = np.zeros((h, w, 3), dtype=np.uint8)
        def put(y, text, color=(255, 255, 255), scale=0.8, thick=2):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        put(34, "LED Sync Report", (255, 255, 0))
        if len(missing_indices) == 1:
            put(90, f"Waiting for stream {missing_indices[0]}...", (0, 165, 255))
        else:
            put(90, f"Waiting for streams: {', '.join(map(str, missing_indices))}...", (0, 165, 255))
        if avail_speed is not None and avail_intervals is not None:
            put(140, f"Seen config on available streams -> Speed={avail_speed}, Intervals={int(avail_intervals) if avail_intervals is not None else '-'}")
        put(190, f"Comparison paused until all {len(self._queues)} streams are available.")
        return img

    # --- Visualization ----------------------------------------------------

    def _active_index(self, mask_full: np.ndarray):
        """Return (idx_int, idx_real) for the **most-advanced ON cell** along the
        row-major raster (left→right, top→bottom), excluding the bottom config row.
        If no ON cells, return None.
        This is more stable than centroid when the ON area spans across rows.
        """
        if mask_full.ndim != 2:
            return None
        if mask_full.shape[0] < 2 or mask_full.shape[1] < 1:
            return None
        eval_mask = mask_full[:-1, :]  # exclude bottom config row
        H, W = eval_mask.shape
        best_idx = None
        for r in range(H):
            row = eval_mask[r, :]
            if not np.any(row):
                continue
            cols = np.flatnonzero(row)
            c_right = int(cols[-1])  # rightmost ON in this row (scan is left->right)
            idx = r * W + c_right
            if (best_idx is None) or (idx > best_idx[0]):
                best_idx = (idx, float(idx))
        if best_idx is None:
            return None
        idx_int, idx_real = best_idx
        return int(idx_int), float(idx_real)

    def _last_lit_index_top_row(self, mask_full: np.ndarray) -> Optional[int]:
        """
        Return the flattened row-major index (excluding bottom config row) of the
        rightmost ON pixel on the top-most active row. If no ON pixels exist,
        returns None.
        """
        if mask_full.ndim != 2:
            return None
        if mask_full.shape[0] < 2 or mask_full.shape[1] < 1:
            return None
        eval_mask = mask_full[:-1, :]
        rows = np.flatnonzero(np.any(eval_mask, axis=1))
        if rows.size == 0:
            return None
        r = int(rows.min())
        cols = np.flatnonzero(eval_mask[r, :])
        if cols.size == 0:
            return None
        c_right = int(cols[-1])
        W = eval_mask.shape[1]
        return int(r * W + c_right)
    def _draw_overlay(self, masks: list) -> np.ndarray:
        """
        Color code per cell for N inputs:
          - All ON: white
          - Some ON: blend of colors (weighted by count)
          - All OFF: dark gray
        Uses distinct colors for each input index.
        """
        h, w = self.grid_size, self.grid_size
        out = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        
        # Generate distinct colors for each input
        n_inputs = len(masks)
        colors = []
        for i in range(n_inputs):
            hue = int(180 * i / max(1, n_inputs - 1))  # 0-180 for BGR
            if n_inputs == 1:
                colors.append((255, 255, 255))
            elif n_inputs == 2:
                colors.append((0, 255, 0) if i == 0 else (0, 0, 255))
            else:
                # Use HSV to BGR conversion for more distinct colors
                rgb = colorsys.hsv_to_rgb(hue / 180.0, 1.0, 1.0)
                colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))

        for r in range(h):
            for c in range(w):
                y1 = r * self.cell_h
                y2 = min((r + 1) * self.cell_h, self.output_h)
                x1 = c * self.cell_w
                x2 = min((c + 1) * self.cell_w, self.output_w)

                # Count how many inputs have this cell ON
                on_count = sum(1 for mask in masks if mask[r, c])
                
                if on_count == 0:
                    color = (40, 40, 40)  # dark gray
                elif on_count == n_inputs:
                    color = (255, 255, 255)  # white (all ON)
                else:
                    # Blend colors of ON inputs
                    blend = np.zeros(3, dtype=np.float32)
                    for i, mask in enumerate(masks):
                        if mask[r, c]:
                            blend += np.array(colors[i], dtype=np.float32)
                    blend /= on_count
                    color = tuple(int(x) for x in blend)

                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, -1)
                # subtle grid
                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (90, 90, 90), 1)

        # Legend - show first few inputs
        legend_y = 10
        for i in range(min(n_inputs, 5)):
            cv2.rectangle(out, (10, legend_y), (22, legend_y + 12), colors[i], -1)
            label = f"Input {i}" if n_inputs > 2 else ("A" if i == 0 else "B")
            cv2.putText(out, label, (28, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 26
        if n_inputs > 5:
            cv2.putText(out, f"... ({n_inputs} total)", (28, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return out

    def _draw_report(
        self,
        n_inputs: int,
        cfg_ok: bool,
        speed: int,
        intervals: int,
        avg_dt_squares_sec: float,
        avg_dt_squares_squared: float,
        passed: Optional[bool],
        passed_by_time: bool = False,
        pair_count: int = 0,
    ) -> np.ndarray:
        # Canvas - make it taller for more info
        W, H = 1440, 280
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (18, 18, 18)

        # Helpers
        def put(x, y, text, color=(230, 230, 230), scale=0.8, thick=2):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def row(y, label, value, value_color=(255, 255, 255)):
            put(16, y, label, (180, 180, 180))
            put(260, y, value, value_color)

        # Header
        put(16, 32, f"LED Sync Report ({n_inputs} inputs)", (255, 255, 0), 0.95, 2)
        cv2.line(img, (16, 40), (W - 16, 40), (70, 70, 70), 1)

        # Config
        cfg_str = f"Speed={speed}, Intervals={int(intervals)}"
        if cfg_ok:
            row(68, "Config:", cfg_str + "  -> MATCH", (0, 255, 0))
        else:
            row(68, "Config:", cfg_str + "  -> MISMATCH", (0, 165, 255))
            row(92, "Note:", "Skipping placement comparison due to config mismatch.", (0, 165, 255))

        # Timing block - show average square dT
        row(118, "Avg dT:", f"{avg_dt_squares_sec:.6f} s   |   LED period = {int(self.led_period_us)} us")
        row(142, "Avg dT²:", f"{avg_dt_squares_squared:.12f} s²   |   Pairs compared: {pair_count}")
        
        # Additional info
        if pair_count > 0:
            row(166, "Stats:", f"Comparing {n_inputs} inputs ({pair_count} pairs)")

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
                    sub_text = f"time threshold: avg dT <= {self.sync_threshold_sec:.3f}s"
                else:
                    sub_text = f"sync threshold met"
            else:
                banner_text = "VERDICT: FAIL"
                banner_color = (0, 0, 220)
                if self.sync_threshold_sec is not None:
                    sub_text = f"time threshold not met (avg dT > {self.sync_threshold_sec:.3f}s)"
                else:
                    sub_text = "sync threshold not met"

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

                # Check if we have any queues configured
                if len(self._queues) == 0:
                    now = _t.time()
                    if now - self._last_placeholder_time > 0.5:
                        self._send_placeholder("No queues configured", "Call set_queues() with analyzer outputs.")
                        self._last_placeholder_time = now
                    _t.sleep(0.001)
                    continue

                # Drain latest packets from all queues/inputs
                buffers = [None] * len(self._queues)
                for i, (q, inp) in enumerate(zip(self._queues, self._inputs)):
                    if q is not None:
                        # Host-side queue
                        try:
                            while True:
                                m = getattr(q, "tryGet", None)
                                m = m() if m is not None else q.get()
                                if m is None:
                                    break
                                buffers[i] = m
                        except Exception:
                            buffers[i] = None
                    elif inp is not None:
                        # Linked input
                        try:
                            while True:
                                try:
                                    m = inp.tryGet()
                                except AttributeError:
                                    if not inp.has():
                                        break
                                    m = inp.get()
                                if m is None:
                                    break
                                buffers[i] = m
                        except Exception:
                            buffers[i] = None

                # Parse incoming packets and update last states
                for i, buf in enumerate(buffers):
                    if buf is not None:
                        try:
                            parsed = self._parse_buffer(buf)
                            self._last_states[i] = parsed
                            # Append only if this sequence is new (avoid duplicates)
                            if len(self._buffers[i]) == 0 or parsed[6] != self._buffers[i][-1][6]:
                                self._buffers[i].append(parsed)
                        except Exception as e:
                            print(f"Comparison parse error for input {i}: {e}")

                # Check how many inputs have data
                n_available = sum(1 for state in self._last_states if state is not None)
                n_expected = len(self._queues)

                # If no inputs have data yet → keep placeholders and wait
                if n_available == 0:
                    now = _t.time()
                    if now - self._last_placeholder_time > 0.5:
                        self._send_placeholder("Waiting for LED analyzer streams...", f"No packets received yet from any of {n_expected} inputs.")
                        self._last_placeholder_time = now
                    _t.sleep(0.001)
                    continue

                # If not all inputs available → render waiting state
                if n_available < n_expected:
                    missing_indices = [i for i, state in enumerate(self._last_states) if state is None]
                    # Get available config info
                    avail_speed = None
                    avail_intervals = None
                    for state in self._last_states:
                        if state is not None:
                            _, _, _, speed, intervals, _, _ = state
                            avail_speed = speed
                            avail_intervals = intervals
                            break

                    # Throttle UI updates while waiting
                    now = _t.time()
                    if now - self._last_waiting_time <= 0.5:
                        _t.sleep(0.001)
                        continue
                    self._last_waiting_time = now

                    # Create overlay with available masks
                    masks = []
                    for state in self._last_states:
                        if state is not None:
                            grid, avg, mult, _, _, _, _ = state
                            thr = self._dynamic_threshold(avg, mult)
                            masks.append(self._mask_from_grid(grid, thr))
                        else:
                            masks.append(np.zeros((self.grid_size, self.grid_size), dtype=bool))

                    # Use timestamp from first available input
                    first_available = next((s for s in self._last_states if s is not None), None)
                    ts_use = first_available[5] if first_available else _td(seconds=0)
                    seq_use = first_available[6] if first_available else self._seq_counter

                    overlay_img = self._draw_overlay(masks)
                    overlay_frame = self._create_imgframe(overlay_img, ts_use, seq_use)
                    self.out_overlay.send(overlay_frame)

                    report_img = self._draw_waiting_report(missing_indices, avail_speed, avail_intervals)
                    report_frame = self._create_imgframe(report_img, ts_use, seq_use)
                    self.out_report.send(report_frame)
                    _t.sleep(0.001)
                    continue

                # All inputs available - get latest frames from all
                frames = self._pop_latest_frames()
                if frames is None:
                    _t.sleep(0.0005)
                    continue

                # Extract data from all frames
                n_inputs = len(frames)
                grids = []
                avgs = []
                mults = []
                speeds = []
                intervals_list = []
                timestamps = []
                seqs = []

                for grid, avg, mult, speed, intervals, ts, seq in frames:
                    grids.append(grid)
                    avgs.append(avg)
                    mults.append(mult)
                    speeds.append(speed)
                    intervals_list.append(intervals)
                    timestamps.append(ts)
                    seqs.append(seq)

                # Config check: all intervals should match (with ±1 tolerance)
                cfg_ok = True
                base_intervals = intervals_list[0]
                for intervals in intervals_list[1:]:
                    diff = self._unwrap16(int(intervals) - int(base_intervals))
                    if abs(diff) > 1:
                        cfg_ok = False
                        break

                # Build masks for all inputs
                masks_full = []
                for i in range(n_inputs):
                    thr = self._dynamic_threshold(avgs[i], mults[i])
                    masks_full.append(self._mask_from_grid(grids[i], thr))

                W = self.grid_size
                H = self.grid_size - 1
                N = W * H

                # Calculate dT for all pairs and compute average square dT
                dt_squares_list = []
                dt_squares_squared_list = []

                for i in range(n_inputs):
                    for j in range(i + 1, n_inputs):
                        # Get last lit indices for both inputs
                        idx_i = self._last_lit_index_top_row(masks_full[i])
                        idx_j = self._last_lit_index_top_row(masks_full[j])

                        if idx_i is not None and idx_j is not None:
                            # Calculate square distance
                            diff_ij = (idx_j - idx_i) % N
                            diff_ji = (idx_i - idx_j) % N
                            squares_diff = min(diff_ij, diff_ji)
                            dt_sec = (squares_diff * self.led_period_us) / 1e6
                            dt_squares_list.append(dt_sec)
                            dt_squares_squared_list.append(dt_sec * dt_sec)
                        else:
                            # Fallback to timestamp-based dT
                            ts_delta_sec = abs((timestamps[i] - timestamps[j]).total_seconds())
                            dt_squares_list.append(ts_delta_sec)
                            dt_squares_squared_list.append(ts_delta_sec * ts_delta_sec)

                # Calculate averages
                pair_count = len(dt_squares_list)
                avg_dt_squares_sec = sum(dt_squares_list) / pair_count if pair_count > 0 else 0.0
                avg_dt_squares_squared = sum(dt_squares_squared_list) / pair_count if pair_count > 0 else 0.0

                # Prepare overlay image from all masks
                overlay_img = self._draw_overlay(masks_full)
                max_seq = max(seqs) if seqs else self._seq_counter
                avg_ts = timestamps[0] if timestamps else _td(seconds=0)
                overlay_frame = self._create_imgframe(overlay_img, avg_ts, max_seq)
                self.out_overlay.send(overlay_frame)

                # Console log
                try:
                    print(
                        f"LEDGridComparison ({n_inputs} inputs): avg dT={avg_dt_squares_sec:.6f}s, avg dT²={avg_dt_squares_squared:.12f}s²",
                        flush=True,
                    )
                except Exception:
                    pass

                # Determine pass/fail
                passed = None
                passed_by_time = False
                if cfg_ok:
                    if self.sync_threshold_sec is not None:
                        passed = avg_dt_squares_sec <= self.sync_threshold_sec
                        passed_by_time = passed
                    else:
                        # Without threshold, consider it passed if average dT is reasonable
                        passed = avg_dt_squares_sec < 0.1  # 100ms default threshold
                        passed_by_time = passed

                # Report
                report_img = self._draw_report(
                    n_inputs=n_inputs,
                    cfg_ok=cfg_ok,
                    speed=max(speeds) if speeds else 0,
                    intervals=base_intervals,
                    avg_dt_squares_sec=avg_dt_squares_sec,
                    avg_dt_squares_squared=avg_dt_squares_squared,
                    passed=passed,
                    passed_by_time=passed_by_time,
                    pair_count=pair_count,
                )
                report_frame = self._create_imgframe(report_img, avg_ts, max_seq)
                self.out_report.send(report_frame)

                # Mark all as compared
                for i, seq in enumerate(seqs):
                    if i < len(self._compared_seqs):
                        self._compared_seqs[i] = seq

            except Exception as e:
                print(f"LEDGridComparison error: {e}")
                import traceback
                traceback.print_exc()
                continue