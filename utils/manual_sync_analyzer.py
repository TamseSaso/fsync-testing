from .apriltag_warp_node import AprilTagWarpNode
from .led_grid_analyzer import LEDGridAnalyzer
from .led_grid_comparison import LEDGridComparison
from .led_grid_visualizer import LEDGridVisualizer

analyzers = []
warp_nodes = []
visualizers = []

def deviceAnalyzer(
    threshold_multiplier: float = 1.47,
    visualizer = None,
    device = None,
    warp_size: tuple[int, int] = (1024, 1024),
    debug: bool = False,
):
    """
    Build the per-device chain: AprilTagWarpNode -> LEDGridAnalyzer (without FrameSamplingNode).
    Frames will be fed manually from the main loop.

    Appends created nodes to the module-level collections and returns them.

    Args:
        threshold_multiplier: Analyzer threshold multiplier.
        visualizer: Optional visualizer for debug topics.
        device: Device instance for debug labeling.
        warp_size: (width, height) for the perspective-warped crop produced by AprilTagWarpNode.
        debug: Enable debug visualization topics.

    Returns:
        (warp_node, led_analyzer, led_visualizer): the created nodes (not linked yet).
    """
    # AprilTag warp (explicit output size is required by AprilTagWarpNode)
    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    warp_node = AprilTagWarpNode(out_width=warp_w, out_height=warp_h)
    warp_nodes.append(warp_node)

    # LED grid analysis
    led_analyzer = LEDGridAnalyzer(
        threshold_multiplier=threshold_multiplier,
        debug=debug,
    ).build(warp_node.out)
    analyzers.append(led_analyzer)

    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(led_analyzer.out)
    visualizers.append(led_visualizer)

    # Debug topics will be added in main after we have the output queue
    
    return warp_node, led_analyzer, led_visualizer

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