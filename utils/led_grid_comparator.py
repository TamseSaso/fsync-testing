

import json
import struct
import time
from typing import Optional, Tuple

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
                    time.sleep(0.001)
                    continue

                # Align shapes (nearest-neighbor)
                grid_l, grid_r = self._align_shapes(grid_l, grid_r)

                # Compare
                matches = (grid_l == grid_r)
                total = matches.size
                correct = int(matches.sum())
                ratio = (correct / total) if total else 0.0
                passed = ratio >= self.pass_threshold

                # Compose visualization
                vis = self._compose_vis(grid_l, grid_r, matches, ratio, passed)

                # Emit ImgFrame
                msg = dai.ImgFrame()
                # Fill data (BGR interleaved); choose a type supported by platform
                try:
                    msg.setType(dai.ImgFrame.Type.BGR888p)
                except Exception:
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
    def _drain_latest(inp: dai.Input, last: Optional[dai.Buffer]) -> Optional[dai.Buffer]:
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
    def _decode_grid(buf: dai.Buffer) -> Tuple[Optional[np.ndarray], dict]:
        data = buf.getData()
        # 1) JSON with key 'grid' (and optional w/h)
        try:
            s = bytes(data).decode('utf-8')
            obj = json.loads(s)
            if 'grid' in obj:
                arr = np.array(obj['grid'])
                if arr.ndim == 1 and 'w' in obj and 'h' in obj:
                    arr = arr.reshape(int(obj['h']), int(obj['w']))
                elif arr.ndim == 2:
                    pass
                else:
                    side = int(np.sqrt(arr.size))
                    if side * side == arr.size:
                        arr = arr.reshape(side, side)
                    else:
                        return None, {}
                # Normalize to 0/1
                arr = (arr > 0.5).astype(np.uint8)
                return arr, {"fmt": "json"}
        except Exception:
            pass

        # 2) Binary header: <uint32 w><uint32 h><uint8 data...>
        try:
            if len(data) >= 8:
                w, h = struct.unpack('<II', bytes(data[:8]))
                raw = np.frombuffer(bytes(data[8:8 + w*h]), dtype=np.uint8)
                if raw.size == w * h:
                    arr = (raw.reshape(h, w) > 0).astype(np.uint8)
                    return arr, {"fmt": "bin_wh"}
        except Exception:
            pass

        # 3) Fallback: assume square u8 image; threshold at 128
        try:
            raw = np.frombuffer(bytes(data), dtype=np.uint8)
            side = int(np.sqrt(raw.size))
            if side * side == raw.size and side > 0:
                arr = (raw.reshape(side, side) > 127).astype(np.uint8)
                return arr, {"fmt": "square"}
        except Exception:
            pass

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