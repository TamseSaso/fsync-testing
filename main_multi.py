import contextlib
import depthai as dai

from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer


# Use the same per-device construction style as multi_devices.py
# Define two devices here; put master first. Update to your IPs/IDs.
DEVICE_INFOS = [
    dai.DeviceInfo("10.12.211.82"),
    dai.DeviceInfo("10.12.211.84"),
]

# Parse arguments used by the processing nodes (panel, apriltag, etc.)
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))

# Visualizer connection
visualizer = dai.RemoteConnection(httpPort=8082)


def build_nodes_on_pipeline(pipeline: dai.Pipeline, device: dai.Device, socket: dai.CameraBoardSocket):
    """Build the same processing nodes as main.py on the given pipeline/socket.

    Returns a list of (title, node_out, topic_type) for visualizer registration.
    """
    platform = device.getPlatform().name
    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )
    fps_limit = args.fps_limit if args.fps_limit else (10 if platform == "RVC2" else 30)

    cam = pipeline.create(dai.node.Camera).build(socket)
    cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=200)
    source_out = cam.requestOutput((1280, 720), frame_type, fps=fps_limit)

    # AprilTag detection and annotations
    apriltag_node = AprilTagAnnotationNode(
        families=args.apriltag_families,
        max_tags=args.apriltag_max,
        quad_decimate=args.apriltag_decimate,
    ).build(source_out)

    # Perspective-rectified panel crop
    warp_node = AprilTagWarpNode(
        panel_width,
        panel_height,
        families=args.apriltag_families,
        quad_decimate=args.apriltag_decimate,
        tag_size=args.apriltag_size,
        z_offset=args.z_offset,
    ).build(source_out)

    # Sample every 2 seconds from the panel crop
    sampling_node = FrameSamplingNode(sample_interval_seconds=2.0).build(warp_node.out)

    # Analyze LED grid, then visualize it
    led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5).build(sampling_node.out)
    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(led_analyzer.out)

    # Compose video + apriltag annotations
    video_composer = VideoAnnotationComposer().build(source_out, apriltag_node.out)

    topics = [
        ("Video with AprilTags", video_composer.out, "video"),
        ("Panel Crop", warp_node.out, "panel"),
        ("Sampled Panel (2s)", sampling_node.out, "panel"),
        ("LED Grid (32x32)", led_visualizer.out, "led"),
    ]
    # Return topics and strong references to nodes to prevent premature GC
    nodes = [cam, apriltag_node, warp_node, sampling_node, led_analyzer, led_visualizer, video_composer]
    return topics, nodes


with contextlib.ExitStack() as stack:
    queues = []  # reserved for future sync needs
    pipelines = []
    liveness_refs = []  # keep strong references to host nodes per pipeline

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        topics, nodes = build_nodes_on_pipeline(pipeline, device, socket)

        suffix = f" [{device.getDeviceId()}]"
        for title, out, topic_type in topics:
            visualizer.addTopic(title + suffix, out, topic_type)

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        pipelines.append(pipeline)
        liveness_refs.append(nodes)

    # Unified visualizer loop; press 'q' to exit
    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break


