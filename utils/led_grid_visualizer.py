import cv2
import numpy as np
import depthai as dai
from typing import Tuple


class LEDGridVisualizer(dai.node.ThreadedHostNode):
    """Visualizes LED grid state data as a color-coded grid image.
    
    Input: dai.Buffer containing grid state data (32x32 float array)
    Output: dai.ImgFrame (BGR visualization of the grid)
    """

    def __init__(self, output_size: Tuple[int, int] = (512, 512), grid_size: int = 32) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.Buffer, True)])
        self.input.setQueueSize(1)
        self.input.setBlocking(False)
        
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        self.out.setQueueSize(1)
        self.out.setBlocking(False)
        
        self.output_w, self.output_h = output_size
        self.grid_size = grid_size
        self.cell_w = self.output_w // self.grid_size
        self.cell_h = self.output_h // self.grid_size
        
    def build(self, buffers: dai.Node.Output) -> "LEDGridVisualizer":
        buffers.link(self.input)
        return self

    def _create_grid_visualization(self, grid_state: np.ndarray, dynamic_threshold: float, overall_avg_brightness: float, speed: int, intervals: int) -> np.ndarray:
        """Create a visual representation of the LED grid state using dynamic threshold."""
        # Create output image
        output = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
        
        # Draw grid cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate cell boundaries
                y1 = row * self.cell_h
                y2 = min((row + 1) * self.cell_h, self.output_h)
                x1 = col * self.cell_w
                x2 = min((col + 1) * self.cell_w, self.output_w)
                
                # Get brightness value
                brightness = grid_state[row, col]
                
                # Determine color based on position and brightness
                if row == self.grid_size - 1:  # Bottom row - special coloring for decoding zones
                    if col < 16:  # Speed zone (first 16 LEDs) - ORANGE
                        if brightness > dynamic_threshold:  # Bright orange for "on" LEDs
                            color = (0, 165, 255)  # Bright orange
                        else:  # Dim orange for "off" LEDs
                            color = (0, 100, 180)  # Dim orange
                    else:  # Intervals zone (last 16 LEDs) - BLUE
                        if brightness > dynamic_threshold:  # Bright blue for "on" LEDs
                            color = (255, 100, 0)  # Bright blue
                        else:  # Dim blue for "off" LEDs
                            color = (180, 60, 0)  # Dim blue
                elif brightness > dynamic_threshold:  # Above dynamic threshold - GREEN (LED ON)
                    # Scale green intensity based on how much it exceeds the threshold
                    excess = brightness - dynamic_threshold
                    max_excess = 1.0 - dynamic_threshold  # Maximum possible excess
                    if max_excess > 0:
                        green_intensity = int(128 + (excess / max_excess) * 127)  # Scale from 128 to 255
                    else:
                        green_intensity = 255
                    color = (0, min(255, green_intensity), 0)
                elif brightness > overall_avg_brightness * 0.5:  # Detectable but below threshold
                    # Below threshold but detectable - dim red
                    color = (0, 0, 128)
                else:
                    # Very low/off - dark blue/black
                    color = (50, 0, 0)
                
                # Fill the cell with the color
                cv2.rectangle(output, (x1, y1), (x2-1, y2-1), color, -1)
                
                # Add grid lines
                cv2.rectangle(output, (x1, y1), (x2-1, y2-1), (128, 128, 128), 1)
                
                # Add brightness value as text (for debugging)
                if self.cell_w > 15 and self.cell_h > 15:
                    text = f"{brightness:.2f}"
                    font_scale = 0.3
                    # Use white text for better visibility
                    text_color = (255, 255, 255)
                    cv2.putText(output, text, (x1 + 2, y1 + self.cell_h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                

                # Add special markers for bottom row decoding zones
                if row == self.grid_size - 1:  # Bottom row
                    if col < 16:  # Speed zone (first 16 LEDs)
                        if self.cell_w > 12 and self.cell_h > 12:
                            cv2.putText(output, "S", (x1 + 2, y1 + 12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    elif col >= 16:  # Intervals zone (last 16 LEDs)
                        if self.cell_w > 12 and self.cell_h > 12:
                            bit_pos = 15 - (col - 16)  # Show MSB first (15-0)
                            cv2.putText(output, str(bit_pos), (x1 + 2, y1 + 12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # Add title with dynamic threshold info
        title = f"LED Grid {self.grid_size}x{self.grid_size} (Green > {dynamic_threshold:.3f}, Avg: {overall_avg_brightness:.3f} excl. bottom)"
        cv2.putText(output, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add decoded bottom row information
        decoded_info = f"Bottom Row - Speed: {speed}, Intervals: {intervals} (0b{intervals:016b})"
        cv2.putText(output, decoded_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        

        return output

    def _create_imgframe(self, bgr: np.ndarray, src: dai.Buffer) -> dai.ImgFrame:
        """Create a DAI ImgFrame from the visualization."""
        img = dai.ImgFrame()
        img.setType(dai.ImgFrame.Type.BGR888i)
        img.setWidth(self.output_w)
        img.setHeight(self.output_h)
        img.setData(bgr.tobytes())
        img.setSequenceNum(src.getSequenceNum())
        img.setTimestamp(src.getTimestamp())
        return img

    def run(self) -> None:
        print(f"LEDGridVisualizer started with {self.grid_size}x{self.grid_size} grid, output size {self.output_w}x{self.output_h}")
        
        while self.isRunning():
            try:
                # Drain to latest buffer (non-blocking)
                buffer_msg: dai.Buffer | None = None
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
                        buffer_msg = m
                except Exception:
                    buffer_msg = None

                if buffer_msg is None:
                    # No new buffer yet; yield briefly to avoid busy spin
                    import time as _t
                    _t.sleep(0.001)
                    continue
                
                # Extract grid state data, decoded values, and metadata from buffer
                data = np.frombuffer(buffer_msg.getData(), dtype=np.float32)
                
                # Expected size: grid_size^2 + 4 metadata values (grid_state, overall_avg_brightness, threshold_multiplier, speed, intervals)
                expected_size = self.grid_size * self.grid_size + 4
                if data.size == expected_size:
                    # Extract grid state and metadata
                    grid_size_squared = self.grid_size * self.grid_size
                    grid_data = data[:grid_size_squared]
                    metadata = data[-4:]
                    
                    grid_state = grid_data.reshape((self.grid_size, self.grid_size))
                    overall_avg_brightness = metadata[0]
                    threshold_multiplier = metadata[1]
                    speed = int(metadata[2])
                    intervals = int(metadata[3])
                    
                    # Calculate dynamic threshold
                    dynamic_threshold = overall_avg_brightness * threshold_multiplier
                    
                    # Create visualization with dynamic threshold and decoded values
                    visualization = self._create_grid_visualization(grid_state, dynamic_threshold, overall_avg_brightness, speed, intervals)
                    
                    # Send as ImgFrame
                    output_msg = self._create_imgframe(visualization, buffer_msg)
                    self.out.send(output_msg)
                else:
                    print(f"Invalid grid data size: {data.size}, expected {expected_size}")
                    
            except Exception as e:
                print(f"LEDGridVisualizer error: {e}")
                continue
