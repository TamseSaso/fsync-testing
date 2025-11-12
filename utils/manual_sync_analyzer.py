from .apriltag_warp_node import AprilTagWarpNode
from .led_grid_analyzer import LEDGridAnalyzer
from .led_grid_comparison import LEDGridComparison
from .led_grid_visualizer import LEDGridVisualizer
import depthai as dai
import threading
from typing import Optional

samplers = []
analyzers = []
warp_nodes = []
visualizers = []

class ManualFrameInjector(dai.node.ThreadedHostNode):
    """A node that accepts manual frame injection from host code.
    
    Input: None (frames are injected manually via inject_frame method)
    Output: dai.ImgFrame (frames injected from host)
    """
    def __init__(self):
        super().__init__()
        self.out = self.createOutput()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        self._frame_queue = []
        self._lock = threading.Lock()
    
    def inject_frame(self, frame: dai.ImgFrame):
        """Manually inject a frame from host code. The frame will be cloned to avoid reuse issues."""
        # Clone the frame to avoid issues with queue frame reuse
        cloned_frame = dai.ImgFrame()
        cloned_frame.setType(frame.getType())
        cloned_frame.setWidth(frame.getWidth())
        cloned_frame.setHeight(frame.getHeight())
        cloned_frame.setData(frame.getData())
        cloned_frame.setSequenceNum(frame.getSequenceNum())
        cloned_frame.setTimestamp(frame.getTimestamp())
        try:
            cloned_frame.setTimestampDevice(frame.getTimestampDevice())
        except Exception:
            pass
        try:
            cloned_frame.setCameraSocket(frame.getCameraSocket())
        except Exception:
            pass
        
        with self._lock:
            self._frame_queue.append(cloned_frame)
    
    def build(self, _=None) -> "ManualFrameInjector":
        # No input linking needed - frames are injected manually
        return self
    
    def run(self) -> None:
        while self.isRunning():
            with self._lock:
                if self._frame_queue:
                    frame = self._frame_queue.pop(0)
                    self.out.send(frame)
            import time
            time.sleep(0.001)  # Small sleep to avoid busy waiting

def deviceAnalyzer(
    frame_injector: ManualFrameInjector,
    threshold_multiplier: float = 1.47,
    visualizer = None,
    device = None,
    warp_size: tuple[int, int] = (1024, 1024),
    debug: bool = False,
):
    """
    Build the per-device chain: ManualFrameInjector -> AprilTagWarpNode -> LEDGridAnalyzer.
    
    Bypasses FrameSamplingNode - frames are manually injected via the injector.

    Args:
        frame_injector: ManualFrameInjector node that receives manually injected frames.
        threshold_multiplier: Analyzer threshold multiplier.
        warp_size: (width, height) for the perspective-warped crop produced by AprilTagWarpNode.

    Returns:
        (frame_injector, warp_node, analyzer): the created nodes.
    """
    # AprilTag warp (explicit output size is required by AprilTagWarpNode)
    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    warp_node = AprilTagWarpNode(out_width=warp_w, out_height=warp_h).build(frame_injector.out)
    warp_nodes.append(warp_node)

    # LED grid analysis
    led_analyzer = LEDGridAnalyzer(
        threshold_multiplier=threshold_multiplier,
        debug=debug,
    ).build(warp_node.out)
    analyzers.append(led_analyzer)

    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(led_analyzer.out)
    visualizers.append(led_visualizer)

    if debug == True and device is not None:
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Warped Sample" + suffix, warp_node.out, "images")
        visualizer.addTopic("LED Grid" + suffix, led_visualizer.out, "images")

    return frame_injector, warp_node, led_analyzer

def deviceComparison(analyzers, warp_nodes, comparisons, sync_threshold_sec, grid_size=32, output_size=(1024, 1024), visualizer=None, debug: bool = False):
    """
    If at least two analyzers exist, create an LEDGridComparison node, bind it to the
    first pipeline via a lightweight tick (warp_nodes[0].out), wire its inputs to the
    first two analyzers, append it to `comparisons`, and return the created node.
    Otherwise return None.
    """
    if len(analyzers) >= 2 and len(warp_nodes) >= 1:
        led_cmp = LEDGridComparison(
            grid_size=grid_size,
            output_size=output_size,
            sync_threshold_sec=sync_threshold_sec,
        ).build(warp_nodes[0].out)
        comparisons.append(led_cmp)
        # Wire analyzers directly to the new comparison node we just created
        led_cmp.set_queues(analyzers[0].out, analyzers[1].out)

        if debug == True:
            visualizer.addTopic("LED Sync Overlay", led_cmp.out_overlay, "images")
            visualizer.addTopic("LED Sync Report", led_cmp.out_report, "images")
        return None
    return None