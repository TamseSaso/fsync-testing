import contextlib
import depthai as dai
import time

from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer


# Define two devices here; put master first. Update to your IPs/IDs.
DEVICE_INFOS = [
    dai.DeviceInfo("10.12.211.82"),
    dai.DeviceInfo("10.12.211.84"),
]

# Synchronization settings
TARGET_FPS = 10  # Will be adjusted based on device platform
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"

# Parse arguments used by the processing nodes (panel, apriltag, etc.)
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))

# Visualizer connection
visualizer = dai.RemoteConnection(httpPort=8082)


def build_nodes_on_pipeline(pipeline: dai.Pipeline, device: dai.Device, socket: dai.CameraBoardSocket):
    """Build the same processing nodes as main.py on the given pipeline/socket.

    Returns topics for visualizer and a sync queue for timestamp monitoring.
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

    # Topics for visualizer (using Node.Output objects)
    topics = [
        ("Video with AprilTags", video_composer.out, "video"),
        ("Panel Crop", warp_node.out, "panel"),
        ("Sampled Panel (2s)", sampling_node.out, "panel"),
        ("LED Grid (32x32)", led_visualizer.out, "led"),
    ]
    
    # Create a separate sync queue for timestamp monitoring (non-blocking)
    sync_queue = video_composer.out.createOutputQueue()
    
    # Return topics, sync queue, and strong references to nodes to prevent premature GC
    nodes = [cam, apriltag_node, warp_node, sampling_node, led_analyzer, led_visualizer, video_composer]
    return topics, sync_queue, nodes


with contextlib.ExitStack() as stack:
    sync_queues = []  # Separate queues for timestamp monitoring
    device_ids = []
    pipelines = []
    liveness_refs = []  # keep strong references to host nodes per pipeline

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        topics, sync_queue, nodes = build_nodes_on_pipeline(pipeline, device, socket)

        # Register topics with visualizer (using Node.Output objects)
        suffix = f" [{device.getDeviceId()}]"
        for title, output, topic_type in topics:
            visualizer.addTopic(title + suffix, output, topic_type)

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        
        # Store sync queue and device info for timestamp monitoring
        sync_queues.append(sync_queue)
        device_ids.append(device.getDeviceId())
        pipelines.append(pipeline)
        liveness_refs.append(nodes)

    # Synchronization state
    latest_sync_frames = {}  # key = device_idx, value = sync frame
    receivedFrames = [False] * len(sync_queues)
    anyFrameEver = False
    allDevicesReported = False
    last_sync_report_time = time.time()
    sync_stats = {"in_sync": 0, "out_of_sync": 0}
    
    print(f"=== Starting synchronized visualization (threshold: {SYNC_THRESHOLD_SEC*1000:.2f}ms)")
    
    # Unified visualizer loop with synchronization monitoring
    while True:
        # Collect newest frame from each device's sync queue (non-blocking)
        frameReceivedThisIter = False
        for idx, sync_queue in enumerate(sync_queues):
            while sync_queue.has():
                latest_sync_frames[idx] = sync_queue.get()
                if not receivedFrames[idx]:
                    print(f"=== Received frame from {device_ids[idx]}")
                    receivedFrames[idx] = True
                frameReceivedThisIter = True

        if frameReceivedThisIter and not anyFrameEver:
            print("=== At least one device is sending frames")
            anyFrameEver = True
        if not allDevicesReported and all(receivedFrames):
            print("=== All devices are sending frames - synchronization active")
            allDevicesReported = True

        # Check synchronization: need at least one frame from every device
        if len(latest_sync_frames) == len(sync_queues):
            ts_values = [
                f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() 
                for f in latest_sync_frames.values()
            ]
            delta = max(ts_values) - min(ts_values)
            
            # Track sync status
            if delta <= SYNC_THRESHOLD_SEC:
                sync_stats["in_sync"] += 1
                latest_sync_frames.clear()
            else:
                sync_stats["out_of_sync"] += 1
            
            # Report sync status every 5 seconds
            if time.time() - last_sync_report_time > 5.0:
                total = sync_stats["in_sync"] + sync_stats["out_of_sync"]
                if total > 0:
                    sync_rate = (sync_stats["in_sync"] / total) * 100
                    print(f"=== Sync status: {sync_rate:.1f}% in sync "
                          f"({sync_stats['in_sync']} in / {sync_stats['out_of_sync']} out), "
                          f"current delta: {delta*1000:.2f}ms")
                sync_stats = {"in_sync": 0, "out_of_sync": 0}
                last_sync_report_time = time.time()
        
        # Visualizer update
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break


