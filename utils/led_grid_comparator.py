import json
import struct
import zlib
import base64
try:
    import msgpack  # type: ignore
except Exception:
    msgpack = None
import time
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import depthai as dai


class LEDGridComparator(dai.node.ThreadedHostNode):
    """
    Consumes two LEDGridAnalyzer Buffer outputs, compares decoded panels cell-wise,
    and emits a visualization ImgFrame with overlay + PASS/FAIL based on a threshold.

    Inputs:  left_input  (dai.Buffer)
             right_input (dai.Buffer)
    Output:  out (dai.ImgFrame)
    """

    def __init__(self, pass_threshold: float = 0.90, output_size: Tuple[int, int] = (1024, 512)) -> None:
        super().__init__()
        self.pass_threshold = float(pass_threshold)
        self.out_w, self.out_h = map(int, output_size)

        # Inputs (latest-only, non-blocking if supported by this SDK)
        self.left_input = self.createInput()
        self.left_input.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        try:
            self.left_input.setQueueSize(1)
            self.left_input.setBlocking(False)
        except AttributeError:
            pass

        self.right_input = self.createInput()
        self.right_input.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        try:
            self.right_input.setQueueSize(1)
            self.right_input.setBlocking(False)
        except AttributeError:
            pass

        # Output
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        try:
            self.out.setQueueSize(1)
            self.out.setBlocking(False)
        except AttributeError:
            pass

        # State
        self._latest_left: Optional[dai.Buffer] = None
        self._latest_right: Optional[dai.Buffer] = None
        self._logged_decode_error_once = False

    # ---------------------------- Builder API ---------------------------- #
    def build(self, left: dai.Node.Output, right: dai.Node.Output) -> "LEDGridComparator":
        left.link(self.left_input)
        right.link(self.right_input)
        return self

    # ----------------------------- Runtime ------------------------------- #
    def run(self) -> None:
        while self.isRunning():
            try:
                # Drain to latest left/right buffers
                self._latest_left = self._drain_latest(self.left_input, self._latest_left)
                self._latest_right = self._drain_latest(self.right_input, self._latest_right)

                if self._latest_left is None or self._latest_right is None:
                    time.sleep(0.001)
                    continue

                # Decode to 2D uint8 arrays in {0,1}
                grid_l, meta_l = self._decode_grid(self._latest_left)
                grid_r, meta_r = self._decode_grid(self._latest_right)
                if grid_l is None or grid_r is None:
                    # Emit placeholder so the visualizer shows a stream even before decode succeeds
                    vis = self._compose_placeholder("Waiting for analyzer decode…")
                    msg = dai.ImgFrame()
                    msg.setType(dai.ImgFrame.Type.BGR888i)
                    msg.setWidth(vis.shape[1])
                    msg.setHeight(vis.shape[0])
                    msg.setData(vis.flatten())
                    try:
                        msg.setTimestamp(self._latest_left.getTimestamp())
                    except Exception:
                        pass
                    self.out.send(msg)
                    time.sleep(0.02)
                    continue

                # Align shapes (nearest-neighbor)
                grid_l, grid_r = self._align_shapes(grid_l, grid_r)

                # Interval gating: compare bottom-right 4x4 (or smaller if grid is tiny)
                H, W = grid_l.shape[:2]
                k = int(min(4, H, W))
                y0, x0 = H - k, W - k
                interval_l = grid_l[y0:H, x0:W]
                interval_r = grid_r[y0:H, x0:W]
                interval_match = bool(np.array_equal(interval_l, interval_r))

                if not interval_match:
                    # Build a SKIP visualization highlighting interval mismatch and emit
                    vis = self._compose_vis_interval(grid_l, grid_r, None, 0.0, False, interval_match, k)
                    msg = dai.ImgFrame()
                    msg.setType(dai.ImgFrame.Type.BGR888i)
                    msg.setWidth(vis.shape[1])
                    msg.setHeight(vis.shape[0])
                    msg.setData(vis.flatten())
                    try:
                        msg.setTimestamp(self._latest_left.getTimestamp())
                    except Exception:
                        pass
                    self.out.send(msg)
                    continue

                # If interval matches, compare MAIN LEDs (exclude the interval block and bottom-left speed block from scoring)
                matches_full = (grid_l == grid_r)
                main_mask = np.ones_like(matches_full, dtype=bool)
                # Exclude bottom-right interval block
                main_mask[y0:H, x0:W] = False
                # Exclude bottom-left speed block (k x k)
                y0_speed = H - k
                x0_speed = 0
                main_mask[y0_speed:H, x0_speed:x0_speed + k] = False

                main_total = int(np.count_nonzero(main_mask))
                main_correct = int(np.count_nonzero(matches_full & main_mask))
                ratio = (main_correct / main_total) if main_total else 0.0
                passed = ratio >= self.pass_threshold

                # Compose visualization (with interval annotated)
                vis = self._compose_vis_interval(grid_l, grid_r, matches_full, ratio, passed, interval_match=True, k=k)

                # Emit ImgFrame
                msg = dai.ImgFrame()
                # Use interleaved BGR to match our buffer layout
                msg.setType(dai.ImgFrame.Type.BGR888i)
                msg.setWidth(vis.shape[1])
                msg.setHeight(vis.shape[0])
                msg.setData(vis.flatten())

                # Timestamp: prefer left buffer's timestamp if available
                try:
                    msg.setTimestamp(self._latest_left.getTimestamp())
                except Exception:
                    pass

                self.out.send(msg)

            except Exception as e:
                print(f"LEDGridComparator error: {e}")
                time.sleep(0.002)
                continue

    # --------------------------- Helper methods -------------------------- #
    @staticmethod
    def _drain_latest(inp: Any, last: Optional[dai.Buffer]) -> Optional[dai.Buffer]:
        latest = last
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
                latest = m
        except Exception:
            pass
        return latest

    @staticmethod
    def _try_json_to_array(obj: dict) -> Optional[np.ndarray]:
        # Accept a variety of key names
        arr = None
        keys_array = ["grid", "cells", "values", "data", "bits", "panel"]
        for k in keys_array:
            if k in obj:
                arr = obj[k]
                break
        if arr is None:
            # base64 bitmap?
            for k in ["b64", "b64data", "bitmap_b64"]:
                if k in obj:
                    try:
                        raw = base64.b64decode(obj[k])
                        arr = list(raw)
                    except Exception:
                        pass
        if arr is None:
            return None
        a = np.array(arr)
        # Resolve width/height
        w = obj.get("w") or obj.get("width") or obj.get("cols")
        h = obj.get("h") or obj.get("height") or obj.get("rows")
        if a.ndim == 1:
            if w and h:
                try:
                    a = a.reshape(int(h), int(w))
                except Exception:
                    return None
            else:
                side = int(np.sqrt(a.size))
                if side * side == a.size and side > 0:
                    a = a.reshape(side, side)
                else:
                    return None
        elif a.ndim == 2:
            pass
        else:
            return None
        # Normalize to {0,1}
        if a.dtype != np.uint8:
            a = (a > 0.5).astype(np.uint8)
        else:
            a = (a > 0).astype(np.uint8)
        return a

    def _decode_grid(self, buf: dai.Buffer) -> Tuple[Optional[np.ndarray], dict]:
        data = buf.getData()
        by = bytes(data)
        # (A) Try JSON (plain or with leading text)
        try:
            s = by.decode('utf-8', errors='ignore')
            # Extract JSON substring if there is noise around
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                obj = json.loads(s[start:end+1])
                arr = self._try_json_to_array(obj)
                if arr is not None:
                    return arr, {"fmt": "json"}
        except Exception:
            pass
        # (B) zlib-compressed JSON
        try:
            dec = zlib.decompress(by)
            s = dec.decode('utf-8', errors='ignore')
            obj = json.loads(s)
            arr = self._try_json_to_array(obj)
            if arr is not None:
                return arr, {"fmt": "json+zlib"}
        except Exception:
            pass
        # (C) msgpack
        try:
            if msgpack is not None:
                obj = msgpack.loads(by, raw=False)
                if isinstance(obj, dict):
                    arr = self._try_json_to_array(obj)
                    if arr is not None:
                        return arr, {"fmt": "msgpack"}
        except Exception:
            pass
        # (D) Binary headers: <uint32,uint32> or <uint16,uint16>
        try:
            if len(by) >= 8:
                w, h = struct.unpack('<II', by[:8])
                if w > 0 and h > 0 and len(by) >= 8 + w*h:
                    raw = np.frombuffer(by[8:8+w*h], dtype=np.uint8)
                    arr = (raw.reshape(h, w) > 0).astype(np.uint8)
                    return arr, {"fmt": "bin_wh32"}
        except Exception:
            pass
        try:
            if len(by) >= 4:
                w16, h16 = struct.unpack('<HH', by[:4])
                if w16 > 0 and h16 > 0 and len(by) >= 4 + w16*h16:
                    raw = np.frombuffer(by[4:4+w16*h16], dtype=np.uint8)
                    arr = (raw.reshape(h16, w16) > 0).astype(np.uint8)
                    return arr, {"fmt": "bin_wh16"}
        except Exception:
            pass
        # (E) Heuristic: 32x32 common grid (1024 bytes)
        try:
            if len(by) == 1024:
                raw = np.frombuffer(by, dtype=np.uint8)
                arr = (raw.reshape(32, 32) > 0).astype(np.uint8)
                return arr, {"fmt": "raw_32x32"}
        except Exception:
            pass
        # (F) Fallback: auto-square
        try:
            raw = np.frombuffer(by, dtype=np.uint8)
            side = int(np.sqrt(raw.size))
            if side * side == raw.size and side > 0:
                arr = (raw.reshape(side, side) > 127).astype(np.uint8)
                return arr, {"fmt": "square"}
        except Exception:
            pass
        # One-time debug to help diagnose
        if not self._logged_decode_error_once:
            self._logged_decode_error_once = True
            preview = ' '.join(f"{b:02x}" for b in by[:32])
            print(f"[LEDGridComparator] Failed to decode analyzer Buffer. len={len(by)} head32={preview}")
        return None, {}

    @staticmethod
    def _align_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if a.shape == b.shape:
            return a, b
        H = max(a.shape[0], b.shape[0])
        W = max(a.shape[1], b.shape[1])
        a_rs = cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        b_rs = cv2.resize(b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return a_rs, b_rs

    def _compose_vis(self, grid_l: np.ndarray, grid_r: np.ndarray, matches: np.ndarray, ratio: float, passed: bool) -> np.ndarray:
        # Convert 0/1 grids to BGR visuals (black/white)
        def to_vis(g: np.ndarray) -> np.ndarray:
            vis = (g * 255).astype(np.uint8)
            return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        left_vis = to_vis(grid_l)
        right_vis = to_vis(grid_r)
        diff_vis = np.zeros_like(left_vis)
        # Green where match, red where mismatch
        mask = matches.astype(bool)
        diff_vis[mask] = (0, 255, 0)
        diff_vis[~mask] = (0, 0, 255)

        # Stack horizontally and resize to target output size
        tile = np.hstack([left_vis, right_vis, diff_vis])
        tile = cv2.resize(tile, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)

        # Overlay text
        status = "PASS" if passed else "FAIL"
        color = (0, 255, 0) if passed else (0, 0, 255)
        text = f"Match: {ratio*100:.1f}%  |  Threshold: {self.pass_threshold*100:.0f}%  |  {status}"
        cv2.putText(tile, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(tile, "Left", (20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(tile, "Right", (self.out_w // 3 + 20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(tile, "Diff", (2 * self.out_w // 3 + 20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        return tile

    def _compose_vis_interval(self, grid_l: np.ndarray, grid_r: np.ndarray, matches: Optional[np.ndarray], ratio: float, passed: bool, interval_match: bool, k: int) -> np.ndarray:
        # Base visuals
        def to_vis(g: np.ndarray) -> np.ndarray:
            vis = (g * 255).astype(np.uint8)
            return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        left_vis = to_vis(grid_l)
        right_vis = to_vis(grid_r)
        if matches is None:
            diff_vis = np.zeros_like(left_vis)
            # Paint interval block blue to highlight mismatch
            H, W = grid_l.shape[:2]
            y0, x0 = H - k, W - k
            diff_vis[y0:H, x0:W] = (255, 0, 0)  # Blue in BGR
        else:
            diff_vis = np.zeros_like(left_vis)
            mask = matches.astype(bool)
            diff_vis[mask] = (0, 255, 0)
            diff_vis[~mask] = (0, 0, 255)

        # Stack and resize
        tile = np.hstack([left_vis, right_vis, diff_vis])
        tile = cv2.resize(tile, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)

        # Draw interval ROI rectangles on each panel for clarity
        H, W = grid_l.shape[:2]
        # scale factors from grid space to output space
        sx = self.out_w / float(W * 3)
        sy = self.out_h / float(H)
        y0, x0 = H - k, W - k
        color_rect = (0, 255, 255) if interval_match else (255, 128, 0)  # cyan if OK, orange if mismatch

        def draw_roi(panel_idx: int):
            x1 = int((panel_idx * W + x0) * sx)
            y1 = int(y0 * sy)
            x2 = int((panel_idx * W + W) * sx)
            y2 = int((H) * sy)
            cv2.rectangle(tile, (x1, y1), (x2 - 1, y2 - 1), color_rect, 2)

        draw_roi(0)  # left panel
        draw_roi(1)  # right panel
        draw_roi(2)  # diff panel

        # Overlay text
        if not interval_match:
            status = "SKIP (interval mismatch)"
            color = (0, 165, 255)  # orange
            cv2.putText(tile, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        else:
            status = "PASS" if passed else "FAIL"
            color = (0, 255, 0) if passed else (0, 0, 255)
            text = f"Main match: {ratio*100:.1f}%  |  Threshold: {self.pass_threshold*100:.0f}%  |  {status}"
            cv2.putText(tile, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.putText(tile, "Left", (20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(tile, "Right", (self.out_w // 3 + 20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(tile, "Diff", (2 * self.out_w // 3 + 20, self.out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        return tile

    def _compose_placeholder(self, text: str) -> np.ndarray:
        tile = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        cv2.putText(tile, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(tile, "Left/Right analyzers not decoded yet", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2, cv2.LINE_AA)
        return tile