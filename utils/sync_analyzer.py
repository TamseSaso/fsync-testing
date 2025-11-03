from .sampling_node import FrameSamplingNode, SharedTicker
from .apriltag_warp_node import AprilTagWarpNode
from .led_grid_analyzer import LEDGridAnalyzer
from .led_grid_comparison import LEDGridComparison

samplers = []
analyzers = []
warp_nodes = []

def deviceAnalyzer(
    node_out,
    shared_ticker,
    sample_interval_seconds: float = 10.0,
    threshold_multiplier: float = 1.75,
    warp_size: tuple[int, int] = (1024, 1024),
):
    """
    Build the per-device chain: FrameSamplingNode -> AprilTagWarpNode -> LEDGridAnalyzer.

    Appends created nodes to the module-level collections and returns the same
    collections so the caller can wait on samplers etc.

    Args:
        node_out: Upstream Node.Output to sample from.
        shared_ticker: SharedTicker used by all samplers to align sampling moments.
        sample_interval_seconds: Fallback sample interval if not using ticker.
        threshold_multiplier: Analyzer threshold multiplier.
        warp_size: (width, height) for the perspective-warped crop produced by AprilTagWarpNode.

    Returns:
        (samplers, warp_nodes, analyzers): the module-level lists containing all created nodes.
    """
    # Sample frames from the live stream
    sampler = FrameSamplingNode(
        sample_interval_seconds=sample_interval_seconds,
        shared_ticker=shared_ticker,
    ).build(node_out)
    samplers.append(sampler)

    # AprilTag warp (explicit output size is required by AprilTagWarpNode)
    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    warp_node = AprilTagWarpNode(out_width=warp_w, out_height=warp_h).build(sampler.out)
    warp_nodes.append(warp_node)

    # LED grid analysis
    led_analyzer = LEDGridAnalyzer(
        threshold_multiplier=threshold_multiplier,
    ).build(warp_node.out)
    analyzers.append(led_analyzer)

    return samplers, warp_nodes, analyzers

def deviceComparison(analyzers, warp_nodes, comparisons, sync_threshold_sec, grid_size=32, output_size=(1024, 1024)):
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
        return led_cmp
    return None