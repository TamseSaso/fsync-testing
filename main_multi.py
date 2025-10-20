import contextlib
import depthai as dai
import time
import cv2

from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer
from utils.led_grid_comparison import LEDGridComparison

# --- FPS counter (latest 100 samples) ---
class FPSCounter:
    def __init__(self):
        self.frame_times = []

    def tick(self):
        now = time.time()
        self.frame_times.append(now)
        self.frame_times = self.frame_times[-100:]

    def getFps(self):
        if len(self.frame_times) <= 1:
            return 0.0
        return (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

# Define two devices here; put master first. Update to your IPs/IDs.
DEVICE_INFOS = [
    dai.DeviceInfo("10.12.211.82"),
    dai.DeviceInfo("10.12.211.84"),
]

# Synchronization settings
TARGET_FPS = 25  # Must match sensorFps in Camera
SET_MANUAL_EXPOSURE = True  # Toggle manual exposure like in multi_devices.py

# Parse arguments used by the processing nodes (panel, apriltag, etc.)
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))

EFFECTIVE_FPS = TARGET_FPS
# Sample once every 5 seconds using PTP-slotted timestamps
SAMPLING_PERIOD_SEC = 5.0
# Tolerance for cross-device alignment (match reference script: half-frame at current FPS)
SYNC_THRESHOLD_SEC = 1.0 / (2 * EFFECTIVE_FPS)

# Visualizer toggles: only publish the comparison to the visualizer
ENABLE_VISUALIZER_PIPELINES = True   # per-device/topic streams OFF
ENABLE_VISUALIZER_COMPARISON = True  # enable comparison topics
# Disable local OpenCV window (we only use the comparison visualizer topics)
SHOW_LOCAL_WINDOW = False
visualizer = None
if ENABLE_VISUALIZER_PIPELINES or ENABLE_VISUALIZER_COMPARISON:
    visualizer = dai.RemoteConnection(httpPort=8082)



def build_nodes_on_pipeline(pipeline: dai.Pipeline, device: dai.Device, socket: dai.CameraBoardSocket):
    """Build the same processing nodes as main.py on the given pipeline/socket.

    Returns topics for visualizer and a continuous host frame queue for sync/display.
    """
    platform = device.getPlatform().name
    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )
    fps_limit = EFFECTIVE_FPS

    cam = pipeline.create(dai.node.Camera).build(socket, sensorFps=fps_limit)
    if SET_MANUAL_EXPOSURE:
        cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=100)
    source_out = cam.requestOutput((1920, 1080), frame_type, fps=fps_limit)
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(8 * 1024 * 1024)
    manip.initialConfig.addRotateDeg(180)
    source_out.link(manip.inputImage)
    source_out = manip.out

    # AprilTag detection and annotations
    apriltag_node = AprilTagAnnotationNode(
        families=args.apriltag_families,
        max_tags=args.apriltag_max,
        quad_decimate=args.apriltag_decimate,
        quad_sigma=args.apriltag_sigma,
        decode_sharpening=args.apriltag_sharpening,
        decision_margin=args.apriltag_decision_margin,
        persistence_seconds=args.apriltag_persistence,
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

    # Latest-only sampler using PTP-slotted timestamps; now one sample every 5 seconds
    sampling_node = FrameSamplingNode(ptp_slot_period_sec=SAMPLING_PERIOD_SEC).build(warp_node.out)

    # Analyze LED grid on the sampled (latest) crop
    led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.3).build(sampling_node.out)
    analyzer_queue = led_analyzer.out.createOutputQueue(1, False)
    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(led_analyzer.out)

    # Compose video + apriltag annotations
    video_composer = VideoAnnotationComposer().build(source_out, apriltag_node.out)

    # Topics for visualizer (using Node.Output objects)
    topics = [
        ("Video with AprilTags", video_composer.out, "video"),
        ("Panel Crop", warp_node.out, "panel"),
        ("Sampled Panel (PTP slots)", sampling_node.out, "panel"),
        ("LED Grid (32x32)", led_visualizer.out, "led"),
    ]
    
    # Create a continuous host frame queue for sync/display (matches reference script behavior)
    sync_queue = source_out.createOutputQueue(1, False)
    
    # Return topics, sync queue, analyzer queue, and strong references to nodes to prevent premature GC
    nodes = [cam, apriltag_node, warp_node, sampling_node, led_analyzer, led_visualizer, video_composer]


    return topics, sync_queue, analyzer_queue, nodes


with contextlib.ExitStack() as stack:
    queues = []
    pipelines = []
    device_ids = []
    analyzer_queues = []
    topics_by_device = []

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        topics, sync_queue, analyzer_queue, _ = build_nodes_on_pipeline(pipeline, device, socket)

        # Register pipeline and add topics BEFORE starting the pipeline
        if (ENABLE_VISUALIZER_PIPELINES or ENABLE_VISUALIZER_COMPARISON) and visualizer is not None:
            visualizer.registerPipeline(pipeline)
        if ENABLE_VISUALIZER_PIPELINES and visualizer is not None:
            suffix = f" [{device.getDeviceId()}]"
            for title, output, topic_type in topics:
                visualizer.addTopic(title + suffix, output, topic_type)

        queues.append(sync_queue)
        device_ids.append(device.getDeviceId())
        pipelines.append(pipeline)
        analyzer_queues.append(analyzer_queue)
        topics_by_device.append(topics)

    # Create LED grid comparison once we have at least two analyzer queues
    if ENABLE_VISUALIZER_COMPARISON and visualizer is not None and len(analyzer_queues) >= 2:
        # Try to use the LED Grid output from the first device as the visual tick/overlay source
        led_vis_out = None
        for title, output, topic_type in topics_by_device[0]:
            if topic_type == "led" and "LED Grid" in title:
                led_vis_out = output
                break
        if led_vis_out is None:
            # Fallback to sampled panel
            for title, output, topic_type in topics_by_device[0]:
                if topic_type == "panel" and "Sampled Panel" in title:
                    led_vis_out = output
                    break
        if led_vis_out is None:
            # Final fallback to the first topic output
            led_vis_out = topics_by_device[0][0][1]
        comparison_node = LEDGridComparison(
            grid_size=32,
            output_size=(1024, 1024)
        ).build(led_vis_out)
        # Provide the two analyzer queues (master first)
        comparison_node.set_queues(analyzer_queues[0], analyzer_queues[1])
        # Expose comparison topics in the visualizer
        visualizer.addTopic("LED Overlay [comparison]", comparison_node.out_overlay, "led")
        visualizer.addTopic("LED Sync Report [comparison]", comparison_node.out_report, "video")

    # Start all pipelines after all topics (including comparison) are registered
    for p in pipelines:
        p.start()
        visualizer.registerPipeline(p)


    # Minimal loop: keep queues flowing; no PTP sync-gating or OpenCV windows
    receivedFrames = [False for _ in queues]
    while True:
        for idx, q in enumerate(queues):
            while q.has():
                _ = q.get()
                if not receivedFrames[idx]:
                    print("=== Received frame from", device_ids[idx])
                    receivedFrames[idx] = True

        key = visualizer.waitKey(1) if visualizer is not None else -1
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
