import logging
from typing import Optional

from .sampling_node import FrameSamplingNode
from .sampling_node import M8FsyncSamplingNode
from .apriltag_warp_node import AprilTagWarpNode
from .led_grid_analyzer import LEDGridAnalyzer
from .led_grid_comparison import LEDGridComparison
from .led_grid_visualizer import LEDGridVisualizer

log = logging.getLogger(__name__)


def deviceAnalyzerM8Fsync(
    nodes_out,
    undersampling_factor: int = 10,
    frame_lost_threshold_sec: float = 0,
    recv_all_timeout_sec: int = 10,
    sync_threshold_sec: float = 0,
    threshold_multiplier: float = 1.47,
    visualizer=None,
    deviceIds=None,
    warp_size: tuple[int, int] = (1024, 1024),
    debug: bool = False,
    *,
    samplers: Optional[list] = None,
    warp_nodes: Optional[list] = None,
    analyzers: Optional[list] = None,
    visualizers: Optional[list] = None,
):
    """Build the per-device chain for M8Fsync cameras.

    All output lists are caller-owned: pass them in via keyword arguments.
    If omitted, fresh lists are created internally.

    Returns:
        (warp_nodes, analyzers, sampler)
    """
    if samplers is None:
        samplers = []
    if warp_nodes is None:
        warp_nodes = []
    if analyzers is None:
        analyzers = []
    if visualizers is None:
        visualizers = []

    sampler = M8FsyncSamplingNode(
        num_cameras=len(nodes_out),
        undersampling_factor=undersampling_factor,
        frame_lost_threshold_sec=frame_lost_threshold_sec,
        recv_all_timeout_sec=recv_all_timeout_sec,
        sync_threshold_sec=sync_threshold_sec,
        debug=False,
        device_ids=deviceIds,
    ).build(nodes_out)

    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    for i in range(len(nodes_out)):
        warp_node = AprilTagWarpNode(out_width=warp_w, out_height=warp_h).build(
            sampler.outputs[i]
        )
        warp_nodes.append(warp_node)

        led_analyzer = LEDGridAnalyzer(
            threshold_multiplier=threshold_multiplier,
            debug=debug,
        ).build(warp_node.out)
        analyzers.append(led_analyzer)

        led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(
            led_analyzer.out
        )
        visualizers.append(led_visualizer)

        if debug:
            assert len(nodes_out) == len(deviceIds)
            suffix = f" [{deviceIds[i]}]"
            visualizer.addTopic("Input Stream" + suffix, nodes_out[i], "images")
            visualizer.addTopic("Sample" + suffix, sampler.outputs[i], "images")
            visualizer.addTopic("Warped Sample" + suffix, warp_node.out, "images")
            visualizer.addTopic("LED Grid" + suffix, led_visualizer.out, "images")

    return warp_nodes, analyzers, sampler


def deviceAnalyzer(
    node_out,
    shared_ticker,
    sample_interval_seconds: float = 10.0,
    threshold_multiplier: float = 1.47,
    visualizer=None,
    device=None,
    warp_size: tuple[int, int] = (1024, 1024),
    debug: bool = False,
    *,
    samplers: Optional[list] = None,
    warp_nodes: Optional[list] = None,
    analyzers: Optional[list] = None,
    visualizers: Optional[list] = None,
):
    """Build the per-device chain: FrameSamplingNode -> AprilTagWarpNode -> LEDGridAnalyzer.

    All output lists are caller-owned: pass them in via keyword arguments.
    If omitted, fresh lists are created internally.

    Returns:
        (samplers, warp_nodes, analyzers)
    """
    if samplers is None:
        samplers = []
    if warp_nodes is None:
        warp_nodes = []
    if analyzers is None:
        analyzers = []
    if visualizers is None:
        visualizers = []

    sampler = FrameSamplingNode(
        sample_interval_seconds=sample_interval_seconds,
        shared_ticker=shared_ticker,
        debug=debug,
    ).build(node_out)
    samplers.append(sampler)

    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    warp_node = AprilTagWarpNode(out_width=warp_w, out_height=warp_h).build(
        sampler.out
    )
    warp_nodes.append(warp_node)

    led_analyzer = LEDGridAnalyzer(
        threshold_multiplier=threshold_multiplier,
        debug=debug,
    ).build(warp_node.out)
    analyzers.append(led_analyzer)

    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(
        led_analyzer.out
    )
    visualizers.append(led_visualizer)

    if debug:
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Input Stream" + suffix, node_out, "images")
        visualizer.addTopic("Sample" + suffix, sampler.out, "images")
        visualizer.addTopic("Warped Sample" + suffix, warp_node.out, "images")
        visualizer.addTopic("LED Grid" + suffix, led_visualizer.out, "images")

    return samplers, warp_nodes, analyzers


def deviceComparison(
    analyzers,
    warp_nodes,
    comparisons,
    sync_threshold_sec,
    grid_size=32,
    output_size=(1024, 1024),
    visualizer=None,
    debug: bool = False,
):
    """Create an LEDGridComparison wired to ALL analyzers (not just the first two).

    Returns the created node, or None if fewer than 2 analyzers exist.
    """
    if len(analyzers) < 2 or len(warp_nodes) < 1:
        return None

    led_cmp = LEDGridComparison(
        grid_size=grid_size,
        output_size=output_size,
        sync_threshold_sec=sync_threshold_sec,
    ).build(warp_nodes[0].out)
    comparisons.append(led_cmp)

    led_cmp.set_queues(*(a.out for a in analyzers))

    if debug:
        visualizer.addTopic("LED Sync Overlay", led_cmp.out_overlay, "images")
        visualizer.addTopic("LED Sync Report", led_cmp.out_report, "images")
    return led_cmp
