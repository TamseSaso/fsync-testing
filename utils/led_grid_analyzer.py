import cv2
import numpy as np
import depthai as dai
from typing import Optional


class LEDGridAnalyzer(dai.node.ThreadedHostNode):
    """Analyzes input images to detect LED states in a grid pattern.
    
    Input: dai.ImgFrame (BGR)
    Output: dai.Buffer containing grid state data
    """

    def __init__(self, grid_size: int = 32, threshold_multiplier: float = 1.47, bottom_row_threshold_scale: float = 0.8, debug: bool = False) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        
        self.grid_size = grid_size
        self.threshold_multiplier = threshold_multiplier
        self.bottom_row_threshold_scale = bottom_row_threshold_scale
        self.debug = bool(debug)
        
    def build(self, frames: dai.Node.Output) -> "LEDGridAnalyzer":
        frames.link(self.input)
        return self

    def _analyze_grid(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Analyze the image to detect LED states in a grid pattern.
        
        Returns:
            tuple: (grid_state, overall_avg_brightness_excluding_bottom_row)
        """
        h, w = image.shape[:2]
        
        # Convert entire image to grayscale for overall brightness calculation
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        # Calculate cell dimensions to determine bottom row boundary
        cell_h = h // self.grid_size
        bottom_row_start = (self.grid_size - 1) * cell_h
        
        # Calculate overall average brightness excluding the bottom row
        image_without_bottom = image_gray[:bottom_row_start, :]
        overall_avg_brightness = np.mean(image_without_bottom) / 255.0
        
        # Calculate dynamic threshold based on overall brightness
        dynamic_threshold = overall_avg_brightness * self.threshold_multiplier
        
        # Calculate cell dimensions
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Initialize grid state array
        grid_state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Analyze each cell in the grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate cell boundaries
                y1 = row * cell_h
                y2 = min((row + 1) * cell_h, h)
                x1 = col * cell_w
                x2 = min((col + 1) * cell_w, w)
                
                # Extract cell region
                cell = image_gray[y1:y2, x1:x2]
                
                if cell.size > 0:
                    # Calculate average brightness (normalized to 0-1)
                    brightness = np.mean(cell) / 255.0
                    grid_state[row, col] = brightness
        
        return grid_state, overall_avg_brightness
    

    def _decode_bottom_row(self, grid_state: np.ndarray, dynamic_threshold: float) -> tuple[int, int]:
        """Decode bottom row for speed and intervals.
        
        Speed (first 16 LEDs): Count of green LEDs = speed value
        Intervals (last 16 LEDs): Binary encoding where green=1, red/off=0 (MSB first)
        Example: LED16=bit15, LED17=bit14, ..., LED31=bit0
        
        Returns:
            tuple: (speed, intervals)
        """
        bottom_row = self.grid_size - 1
        threshold = dynamic_threshold * self.bottom_row_threshold_scale
        
        # Decode speed from first 16 LEDs (count of green LEDs)
        speed = 0
        for col in range(16):
            if col < self.grid_size and grid_state[bottom_row, col] > threshold:
                speed += 1
        
        # Decode intervals from last 16 LEDs (binary encoding - MSB first)
        intervals = 0
        for col in range(16, 32):
            if col < self.grid_size:
                bit_position = 15 - (col - 16)  # 15-0 (MSB first)
                if grid_state[bottom_row, col] > threshold:
                    intervals |= (1 << bit_position)  # Set bit at position
        
        return speed, intervals

    def _create_buffer(self, grid_state: np.ndarray, overall_avg_brightness: float, speed: int, intervals: int, src: dai.ImgFrame) -> dai.Buffer:
        """Create a DAI buffer containing the grid state data, decoded values, and metadata."""
        # Create a combined array: [grid_state_flattened, overall_avg_brightness, threshold_multiplier, speed, intervals]
        metadata = np.array([overall_avg_brightness, self.threshold_multiplier, float(speed), float(intervals)], dtype=np.float32)
        combined_data = np.concatenate([grid_state.flatten(), metadata])
        
        buffer = dai.Buffer()
        buffer.setData(combined_data.tobytes())
        buffer.setSequenceNum(src.getSequenceNum())
        buffer.setTimestamp(src.getTimestamp())
        return buffer

    def run(self) -> None:
        print(f"LEDGridAnalyzer started with {self.grid_size}x{self.grid_size} grid")
        
        while self.isRunning():
            try:
                # Drain to latest frame (non-blocking)
                frame_msg: Optional[dai.ImgFrame] = None
                try:
                    while True:
                        try:
                            m = self.input.tryGet()
                        except AttributeError:
                            if not self.input.has():
                                break
                            m = self.input.get()
                        if m is None:
                            break
                        frame_msg = m
                except Exception:
                    frame_msg = None

                if frame_msg is None:
                    # No new frame; yield briefly to avoid busy spin
                    import time as _t
                    _t.sleep(0.001)
                    continue

                image = frame_msg.getCvFrame()
                if image is None:
                    continue
                
                # Analyze the grid
                grid_state, overall_avg_brightness = self._analyze_grid(image)
                
                # Calculate dynamic threshold
                dynamic_threshold = overall_avg_brightness * self.threshold_multiplier
                bottom_dynamic_threshold = dynamic_threshold * self.bottom_row_threshold_scale
                
                # Build per-row thresholds (bottom row uses lowered threshold)
                thresholds = np.full(grid_state.shape, dynamic_threshold, dtype=np.float32)
                thresholds[-1, :] = bottom_dynamic_threshold
                
                # Decode bottom row for speed and intervals
                speed, intervals = self._decode_bottom_row(grid_state, dynamic_threshold)
                
                # Prepare grid_state for output: scale bottom row so consumers using a single global threshold
                # effectively apply the lowered bottom-row threshold (divide by scale => lower effective threshold)
                grid_state_out = grid_state.copy()
                scale = max(self.bottom_row_threshold_scale, 1e-6)
                grid_state_out[-1, :] = grid_state_out[-1, :] / scale
                
                # Create output buffer with grid data, decoded values, and metadata
                buffer_msg = self._create_buffer(grid_state_out, overall_avg_brightness, speed, intervals, frame_msg)
                self.out.send(buffer_msg)
                
                # For logging, match the consumer's simple global thresholding
                num_leds_on = np.sum(grid_state_out > dynamic_threshold)
                
                if getattr(self, "debug", False):
                    print(f"Grid analyzed: {num_leds_on} LEDs above threshold | Speed: {speed}, Intervals: {intervals} (0b{intervals:016b}) | (avg={overall_avg_brightness:.3f} excl. bottom row, global_thresh={dynamic_threshold:.3f}, bottom_thresh={bottom_dynamic_threshold:.3f})")
                
            except Exception as e:
                print(f"LEDGridAnalyzer error: {e}")
                continue
