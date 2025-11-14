import cv2
import numpy as np
import depthai as dai
from typing import Optional, Tuple, List, Union
import time as _t

from datetime import timedelta as _td
from collections import deque


class LEDGridComparison(dai.node.ThreadedHostNode):
    """
    Compares N LED grid analyzer streams and determines if they are in sync.

    Inputs: (set via set_queues) N host-side OutputQueues that receive dai.Buffer from LEDGridAnalyzer
            Each buffer encodes:
              - 32x32 float grid_state (values in [0..1], bottom row pre-scaled by analyzer)
              - 4 floats metadata: [overall_avg_brightness, threshold_multiplier, speed, intervals]
    Outputs:
      - out_overlay: dai.ImgFrame (BGR) overlay of all LED masks with color coding
      - out_report:  dai.ImgFrame (BGR) textual report with pass/fail and metrics

    Logic:
      1) Check configuration on bottom row:
         - first 16 cells encode "speed" (count of lit LEDs) and is provided in metadata
         - last 16 cells encode "intervals" (binary) and is provided in metadata
         If intervals differ between panels (beyond ±1 tolerance), SKIP comparison (report only). Speed is ignored.
      2) If config matches, compare placement of "green" LEDs (mask = grid_state > dynamic_threshold).
         - dynamic_threshold = overall_avg_brightness * threshold_multiplier (same as visualizer)
         - Exclude bottom config row from the metric.
         - Align all streams to a reference (first stream) using IoU-based column shifting
         - Compute overlap metrics (excluding the bottom row) across all pairs:
             For each pair (i, j): recall_i, recall_j, iou
           PASS if min recall across all pairs >= pass_ratio (default 0.90).
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

        # Dynamic analyzer inputs - created when set_queues is called
        self._inputs: List[dai.Node.Input] = []
        self._queues: List[Optional[Union[dai.OutputQueue, dai.Node.Output]]] = []
        self._num_streams = 0

        self.grid_size = grid_size
        self.output_w, self.output_h = output_size
        self.cell_w = max(1, self.output_w // self.grid_size)
        self.cell_h = max(1, self.output_h // self.grid_size)
        self.led_period_us = float(led_period_us)
        self.pass_ratio = float(pass_ratio)
        self.sync_threshold_sec = sync_threshold_sec
        self._last_placeholder_time = 0.0
        self._seq_counter = 1
        
        # Per-stream state: list of (grid, avg, mult, speed, intervals, ts, seq) tuples
        self._last_states: List[Optional[Tuple]] = []
        # Track last compared sequence numbers per stream
        self._compared_seqs: List[int] = []
        # Small buffers to tolerate skipped/misaligned frames
        self._buffers: List[deque] = []
        # Throttle for "waiting" overlay/report updates
        self._last_waiting_time = 0.0
        
        # Color palette for N streams (BGR format)
        self._color_palette = [
            (0, 255, 0),    # Green (stream 0)
            (0, 0, 255),    # Red (stream 1)
            (255, 0, 255),  # Magenta (stream 2)
            (0, 255, 255),  # Cyan (stream 3)
            (255, 255, 0),  # Yellow (stream 4)
            (255, 128, 0),  # Orange (stream 5)
            (128, 0, 255),  # Purple (stream 6)
            (0, 128, 255),  # Light Blue (stream 7)
        ]

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
        Wire analyzer outputs. Accepts variable number of queues/inputs.
        Each can be either Node.Output (host-to-host link) or OutputQueue.
        """
        if len(queues) == 0:
            raise ValueError("At least one queue/input must be provided")
        
        self._num_streams = len(queues)
        self._queues = list(queues)
        
        # Check if all are Node.Output (have link method) or all are queues
        all_have_link = all(hasattr(q, "link") for q in queues)
        
        if all_have_link:
            # Create inputs dynamically and link
            self._inputs = []
            for i, q in enumerate(queues):
                inp = self.createInput()
                inp.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
                try:
                    inp.setBlocking(False)
                    inp.setQueueSize(1)
                except AttributeError:
                    pass
                q.link(inp)
                self._inputs.append(inp)
            # Clear queue references since we're using linked inputs
            self._queues = [None] * len(queues)
        else:
            # Assume host-side OutputQueues
            self._inputs = []
            for _ in queues:
                inp = self.createInput()
                inp.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
                try:
                    inp.setBlocking(False)
                    inp.setQueueSize(1)
                except AttributeError:
                    pass
                self._inputs.append(inp)
        
        # Initialize per-stream state
        self._last_states = [None] * self._num_streams
        self._compared_seqs = [-1] * self._num_streams
        self._buffers = [deque(maxlen=8) for _ in range(self._num_streams)]

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
        Return the latest parsed tuples from all streams if all have at least one item.
        Returns list of tuples, one per stream, or None if any stream is empty.
        Each tuple layout: (grid, avg, mult, speed, intervals, ts, seq)
        """
        if self._num_streams == 0:
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

    def _send_placeholder(self, text_line1: str = "Waiting for LED analyzer streams...", text_line2: Optional[str] = None) -> None:
        if text_line2 is None:
            if self._num_streams > 0:
                text_line2 = f"Connect {self._num_streams} devices and ensure LEDGridAnalyzer is running."
            else:
                text_line2 = "Connect devices and ensure LEDGridAnalyzer is running."
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

    def _draw_waiting_report(self, missing_streams: List[int], avail_configs: List[Tuple[Optional[int], Optional[int]]]) -> np.ndarray:
        w, h = 1440, max(240, 60 + len(missing_streams) * 30 + 40)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        def put(y, text, color=(255, 255, 255), scale=0.8, thick=2):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        put(34, "LED Sync Report", (255, 255, 0))
        if missing_streams:
            put(90, f"Waiting for streams: {', '.join(f'#{i}' for i in missing_streams)}...", (0, 165, 255))
            y = 130
            for i, (speed, intervals) in enumerate(avail_configs):
                if speed is not None and intervals is not None:
                    put(y, f"Stream #{i}: Speed={speed}, Intervals={int(intervals)}", (200, 200, 200), 0.7)
                    y += 30
        else:
            put(90, "Waiting for all streams...", (0, 165, 255))
        put(h - 30, "Comparison paused until all streams are available.", (180, 180, 180), 0.7)
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
    def _draw_overlay(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Color code per cell based on which streams are ON:
          - All ON: white
          - Single stream ON: that stream's color
          - Multiple (but not all) ON: blended color
          - All OFF: dark gray
        """
        h, w = self.grid_size, self.grid_size
        out = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        num_masks = len(masks)

        for r in range(h):
            for c in range(w):
                y1 = r * self.cell_h
                y2 = min((r + 1) * self.cell_h, self.output_h)
                x1 = c * self.cell_w
                x2 = min((c + 1) * self.cell_w, self.output_w)

                # Count how many streams have this cell ON
                on_count = sum(1 for mask in masks if bool(mask[r, c]))
                
                if on_count == 0:
                    color = (40, 40, 40)  # dark gray
                elif on_count == num_masks:
                    color = (255, 255, 255)  # white - all agree
                elif on_count == 1:
                    # Single stream ON - use its color
                    for i, mask in enumerate(masks):
                        if bool(mask[r, c]):
                            color = self._color_palette[i % len(self._color_palette)]
                            break
                else:
                    # Multiple (but not all) - blend colors
                    colors = []
                    for i, mask in enumerate(masks):
                        if bool(mask[r, c]):
                            colors.append(self._color_palette[i % len(self._color_palette)])
                    # Average the colors
                    color = tuple(int(sum(c[i] for c in colors) / len(colors)) for i in range(3))

                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, -1)
                # subtle grid
                cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (90, 90, 90), 1)

        # Legend - show all stream colors (always show all configured streams, not just active ones)
        legend_y = 10
        legend_spacing = 28
        num_streams_to_show = max(num_masks, self._num_streams) if hasattr(self, '_num_streams') else num_masks
        for i in range(min(num_streams_to_show, len(self._color_palette))):
            color = self._color_palette[i]
            cv2.rectangle(out, (10, legend_y), (22, legend_y + 12), color, -1)
            # Show if this stream has data
            has_data = i < num_masks
            label = f"Stream #{i}" + (" (active)" if has_data else " (waiting)")
            cv2.putText(out, label, (28, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += legend_spacing
        
        if num_masks > 1:
            cv2.rectangle(out, (10, legend_y), (22, legend_y + 12), (255, 255, 255), -1)
            cv2.putText(out, f"All {num_masks} ON", (28, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return out

    def _draw_report(
        self,
        cfg_ok: bool,
        speeds: List[int],
        intervals: List[int],
        max_dt_squares_sec: float,
        pair_metrics: List[Tuple[int, int, float, float, float]],  # (i, j, recall_i, recall_j, iou)
        passed: Optional[bool],
        passed_by_time: bool = False,
    ) -> np.ndarray:
        """
        Draw report for N streams.
        pair_metrics: list of (stream_i, stream_j, recall_i, recall_j, iou) for each pair
        """
        # Canvas - dynamically sized based on number of streams
        W = 1440
        base_h = 180
        metrics_lines = len(pair_metrics) if len(pair_metrics) <= 5 else 6  # Show max 5 pairs + "and N more"
        H = base_h + max(0, (metrics_lines - 2) * 24)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (18, 18, 18)

        # Helpers
        def put(x, y, text, color=(230, 230, 230), scale=0.8, thick=2):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

        def row(y, label, value, value_color=(255, 255, 255)):
            put(16, y, label, (180, 180, 180))
            put(260, y, value, value_color)

        # Header
        put(16, 32, f"LED Sync Report ({self._num_streams} streams)", (255, 255, 0), 0.95, 2)
        cv2.line(img, (16, 40), (W - 16, 40), (70, 70, 70), 1)

        # Config - show range if multiple streams
        if len(speeds) > 0:
            speed_str = f"{min(speeds)}-{max(speeds)}" if min(speeds) != max(speeds) else str(speeds[0])
            intervals_str = f"{min(intervals)}-{max(intervals)}" if min(intervals) != max(intervals) else str(int(intervals[0]))
            cfg_str = f"Speed={speed_str}, Intervals={intervals_str}"
            if cfg_ok:
                # Check if all intervals match exactly or within ±1
                intervals_ok = all(abs(int(intervals[i]) - int(intervals[0])) <= 1 for i in range(len(intervals)))
                tol_tag = " (±1 tol)" if not all(int(intervals[i]) == int(intervals[0]) for i in range(len(intervals))) else ""
                row(68, "Config:", cfg_str + f"  -> MATCH{tol_tag}", (0, 255, 0))
            else:
                row(68, "Config:", cfg_str + "  -> MISMATCH", (0, 165, 255))
                row(92, "Note:", "Skipping placement comparison due to config mismatch.", (0, 165, 255))

        # Timing block
        row(118, "Max dT:", f"{max_dt_squares_sec:.6f} s   |   LED period = {int(self.led_period_us)} us")

        # Metrics block - show pairwise comparisons
        if len(pair_metrics) > 0:
            min_recall = min(min(rec_i, rec_j) for _, _, rec_i, rec_j, _ in pair_metrics)
            avg_iou = sum(iou for _, _, _, _, iou in pair_metrics) / len(pair_metrics)
            row(142, "Metrics:", f"Min recall: {min_recall:.3f}  |  Avg IoU: {avg_iou:.3f}  |  Pairs: {len(pair_metrics)}")
            
            # Show individual pair metrics (up to 5)
            y = 166
            for idx, (i, j, rec_i, rec_j, iou_val) in enumerate(pair_metrics[:5]):
                put(16, y, f"  Pair #{i}-#{j}: recall=({rec_i:.2f}, {rec_j:.2f}), IoU={iou_val:.3f}", (200, 200, 200), 0.7)
                y += 24
            if len(pair_metrics) > 5:
                put(16, y, f"  ... and {len(pair_metrics) - 5} more pairs", (150, 150, 150), 0.65)
        else:
            row(142, "Metrics:", "No valid comparisons available", (150, 150, 150))

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
                    sub_text = f"time threshold: max dT <= {self.sync_threshold_sec:.3f}s"
                else:
                    sub_text = f"recall threshold: min recall >= {self.pass_ratio:.0%}"
            else:
                banner_text = "VERDICT: FAIL"
                banner_color = (0, 0, 220)
                sub_text = f"recall threshold not met ({self.pass_ratio:.0%})"

        # Draw banner
        x1, y1, x2, y2 = 16, H - 44, W - 16, H - 12
        cv2.rectangle(img, (x1, y1), (x2, y2), banner_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        put(26, H - 20, banner_text, (255, 255, 255), 0.8, 2)
        if sub_text:
            put(330, H - 20, f"({sub_text})", (240, 240, 240), 0.7, 2)

        return img

    # --- Main loop --------------------------------------------------------
    def run(self) -> None:
        if self._num_streams == 0:
            print("LEDGridComparison: No streams configured. Call set_queues() first.")
            return
        
        print(
            f"LEDGridComparison started: {self._num_streams} streams, {self.grid_size}x{self.grid_size}, "
            f"out={self.output_w}x{self.output_h}, period={self.led_period_us}us, pass_ratio={self.pass_ratio:.2f}"
        )
        num_host_queues = len([q for q in self._queues if q is not None])
        num_linked_inputs = len(self._inputs)
        print(f"LEDGridComparison: Using {num_host_queues} host queues, {num_linked_inputs} linked inputs")
        if self._num_streams != num_host_queues + num_linked_inputs:
            print(f"WARNING: Stream count mismatch! _num_streams={self._num_streams}, queues={num_host_queues}, inputs={num_linked_inputs}")

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


                # Drain latest packets from all streams
                buffers: List[Optional[dai.Buffer]] = [None] * self._num_streams
                
                for i in range(self._num_streams):
                    if self._queues[i] is not None:
                        # Host-side queue
                        try:
                            while True:
                                m = getattr(self._queues[i], "tryGet", None)
                                m = m() if m is not None else self._queues[i].get()
                                if m is None:
                                    break
                                buffers[i] = m
                        except Exception:
                            buffers[i] = None
                    else:
                        # Linked input
                        try:
                            while True:
                                try:
                                    m = self._inputs[i].tryGet()
                                except AttributeError:
                                    if not self._inputs[i].has():
                                        break
                                    m = self._inputs[i].get()
                                if m is None:
                                    break
                                buffers[i] = m
                        except Exception:
                            buffers[i] = None

                # Parse incoming packets and update last states
                for i in range(self._num_streams):
                    if buffers[i] is not None:
                        try:
                            parsed = self._parse_buffer(buffers[i])
                            self._last_states[i] = parsed
                            # Append only if this sequence is new (avoid duplicates)
                            if len(self._buffers[i]) == 0 or parsed[6] != self._buffers[i][-1][6]:
                                self._buffers[i].append(parsed)
                        except Exception as e:
                            print(f"Comparison parse error for stream {i}: {e}")

                # Check how many streams have data
                available_streams = [i for i in range(self._num_streams) if self._last_states[i] is not None]
                
                # Debug: periodically log stream status
                if _t.time() % 5.0 < 0.1:  # Log roughly every 5 seconds
                    print(f"LEDGridComparison: {len(available_streams)}/{self._num_streams} streams have data. "
                          f"Streams with data: {available_streams}")
                
                # If no streams have data yet
                if len(available_streams) == 0:
                    now = _t.time()
                    if now - self._last_placeholder_time > 0.5:
                        self._send_placeholder("Waiting for LED analyzer streams...", 
                                             f"No packets received yet from any of {self._num_streams} streams.")
                        self._last_placeholder_time = now
                    _t.sleep(0.001)
                    continue

                # If not all streams available → render waiting state
                if len(available_streams) < self._num_streams:
                    missing_streams = [i for i in range(self._num_streams) if i not in available_streams]
                    avail_configs = []
                    for i in range(self._num_streams):
                        if self._last_states[i] is not None:
                            _, _, _, speed, intervals, _, _ = self._last_states[i]
                            avail_configs.append((speed, intervals))
                        else:
                            avail_configs.append((None, None))
                    
                    # Throttle UI updates while waiting
                    now = _t.time()
                    if now - self._last_waiting_time <= 0.5:
                        _t.sleep(0.001)
                        continue
                    self._last_waiting_time = now
                    
                    # Create partial overlay with available streams
                    masks = []
                    for i in range(self._num_streams):
                        if self._last_states[i] is not None:
                            grid, avg, mult, _, _, _, _ = self._last_states[i]
                            thr = self._dynamic_threshold(avg, mult)
                            mask = self._mask_from_grid(grid, thr)
                        else:
                            # Create empty mask with same shape
                            if len(masks) > 0:
                                mask = np.zeros_like(masks[0])
                            else:
                                mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
                        masks.append(mask)
                    
                    overlay_img = self._draw_overlay(masks)
                    # Use timestamp from first available stream
                    _, _, _, _, _, ts_use, seq_use = self._last_states[available_streams[0]]
                    overlay_frame = self._create_imgframe(overlay_img, ts_use, seq_use)
                    self.out_overlay.send(overlay_frame)
                    
                    report_img = self._draw_waiting_report(missing_streams, avail_configs)
                    report_frame = self._create_imgframe(report_img, ts_use, seq_use)
                    self.out_report.send(report_frame)
                    _t.sleep(0.001)
                    continue

                # All streams available - get latest frames from all
                frames = self._pop_latest_frames()
                if frames is None:
                    _t.sleep(0.0005)
                    continue

                # Unpack all frames: (grid, avg, mult, speed, intervals, ts, seq)
                grids = []
                avgs = []
                mults = []
                speeds = []
                intervals_list = []
                timestamps = []
                seqs = []
                
                for frame in frames:
                    grid, avg, mult, speed, intervals, ts, seq = frame
                    grids.append(grid)
                    avgs.append(avg)
                    mults.append(mult)
                    speeds.append(speed)
                    intervals_list.append(intervals)
                    timestamps.append(ts)
                    seqs.append(seq)

                # Config check: all intervals must match within ±1 tolerance
                intervals_ok = all(abs(int(intervals_list[i]) - int(intervals_list[0])) <= 1 
                                 for i in range(1, len(intervals_list)))
                cfg_ok = intervals_ok

                # Build masks using each stream's dynamic threshold
                masks_full = []
                for i in range(self._num_streams):
                    thr = self._dynamic_threshold(avgs[i], mults[i])
                    mask_full = self._mask_from_grid(grids[i], thr)
                    masks_full.append(mask_full)

                W = self.grid_size
                H = self.grid_size - 1
                N = W * H

                # Align all streams to reference (stream 0) using IoU-based column shifting
                aligned_masks = [masks_full[0]]  # Reference stream, no shift
                shift_cols_list = [0]
                
                for i in range(1, self._num_streams):
                    ref_eval = masks_full[0][:-1, :]  # Exclude config row
                    stream_eval = masks_full[i][:-1, :]
                    
                    ref_on = int(ref_eval.sum())
                    stream_on = int(stream_eval.sum())
                    
                    # Use IoU alignment if both have LEDs and config matches
                    if ref_on > 0 and stream_on > 0 and intervals_list[0] == intervals_list[i]:
                        max_shift = self.grid_size
                        best_s = 0
                        best_iou = -1.0
                        for s in range(-max_shift, max_shift + 1):
                            shifted = np.roll(stream_eval, s, axis=1)
                            overlap = int(np.logical_and(ref_eval, shifted).sum())
                            union = int(np.logical_or(ref_eval, shifted).sum())
                            iou = (overlap / union) if union > 0 else 0.0
                            if iou > best_iou:
                                best_iou = iou
                                best_s = s
                        shift_cols = int(best_s)
                    else:
                        # Fallback: align by last-lit column
                        idx_ref = self._last_lit_index_top_row(masks_full[0])
                        idx_stream = self._last_lit_index_top_row(masks_full[i])
                        if idx_ref is not None and idx_stream is not None:
                            col_ref = int(idx_ref % W)
                            col_stream = int(idx_stream % W)
                            x = float(col_ref - col_stream)
                            Wf = float(W)
                            x_mod = ((x % Wf) + Wf) % Wf
                            shift_cols_real = x_mod - Wf if x_mod > (Wf / 2.0) else x_mod
                            shift_cols = int(round(shift_cols_real))
                        else:
                            # Final fallback: use timestamp delta
                            ts_delta_sec = (timestamps[0] - timestamps[i]).total_seconds()
                            ts_delta_us = abs(ts_delta_sec) * 1e6
                            shift_cols_real = (ts_delta_us / self.led_period_us) if self.led_period_us > 0 else 0.0
                            Wf = float(W)
                            x_mod = ((shift_cols_real % Wf) + Wf) % Wf
                            shift_cols_real = x_mod - Wf if x_mod > (Wf / 2.0) else x_mod
                            shift_cols = int(round(shift_cols_real))
                    
                    aligned_mask = self._roll_columns(masks_full[i], shift_cols)
                    aligned_masks.append(aligned_mask)
                    shift_cols_list.append(shift_cols)

                # Prepare overlay image from all aligned masks
                overlay_img = self._draw_overlay(aligned_masks)
                max_seq = max(seqs)
                ref_ts = timestamps[0]
                overlay_frame = self._create_imgframe(overlay_img, ref_ts, max_seq)
                self.out_overlay.send(overlay_frame)

                # Compute pairwise metrics (excluding config row)
                pair_metrics = []
                max_dt_squares_sec = 0.0
                
                for i in range(self._num_streams):
                    mask_i = aligned_masks[i][:-1, :]
                    for j in range(i + 1, self._num_streams):
                        mask_j = aligned_masks[j][:-1, :]
                        
                        on_i = int(mask_i.sum())
                        on_j = int(mask_j.sum())
                        overlap = int(np.logical_and(mask_i, mask_j).sum())
                        union = on_i + on_j - overlap
                        
                        recall_i = (overlap / on_i) if on_i > 0 else 0.0
                        recall_j = (overlap / on_j) if on_j > 0 else 0.0
                        iou = (overlap / union) if union > 0 else 0.0
                        
                        pair_metrics.append((i, j, recall_i, recall_j, iou))
                        
                        # Compute timing delta between this pair
                        idx_i = self._last_lit_index_top_row(aligned_masks[i])
                        idx_j = self._last_lit_index_top_row(aligned_masks[j])
                        if idx_i is not None and idx_j is not None:
                            diff = abs((idx_j - idx_i) % N)
                            diff_alt = abs((idx_i - idx_j) % N)
                            diff_min = min(diff, diff_alt)
                            dt_squares = (diff_min * self.led_period_us) / 1e6
                            max_dt_squares_sec = max(max_dt_squares_sec, dt_squares)

                # Determine pass/fail
                passed = None
                passed_by_time = False
                
                if cfg_ok and len(pair_metrics) > 0:
                    min_recall = min(min(rec_i, rec_j) for _, _, rec_i, rec_j, _ in pair_metrics)
                    
                    # Check time threshold first
                    if self.sync_threshold_sec is not None:
                        passed_by_time = (max_dt_squares_sec <= self.sync_threshold_sec)
                    
                    # Check recall threshold
                    passed_by_recall = (min_recall >= self.pass_ratio)
                    
                    passed = passed_by_time or passed_by_recall
                elif not cfg_ok:
                    passed = None

                # Report
                report_img = self._draw_report(
                    cfg_ok=cfg_ok,
                    speeds=speeds,
                    intervals=intervals_list,
                    max_dt_squares_sec=max_dt_squares_sec,
                    pair_metrics=pair_metrics,
                    passed=passed,
                    passed_by_time=passed_by_time,
                )
                report_frame = self._create_imgframe(report_img, ref_ts, max_seq)
                self.out_report.send(report_frame)
                
                # Mark all streams as compared
                for i in range(self._num_streams):
                    self._compared_seqs[i] = seqs[i]

                # Console log
                try:
                    print(f"LEDGridComparison: {self._num_streams} streams, max dT={max_dt_squares_sec:.6f}s", flush=True)
                except Exception:
                    pass

            except Exception as e:
                print(f"LEDGridComparison error: {e}")
                import traceback
                traceback.print_exc()
                continue