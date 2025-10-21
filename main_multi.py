import contextlib
import depthai as dai
import time
import cv2

import os
import sys
import locale

# --- Ensure UTF-8 I/O to avoid decode errors in some environments ---
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

print("ENC:", {
    "fs": sys.getfilesystemencoding(),
    "preferred": locale.getpreferredencoding(False)
})

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

#
# Visualizer toggles: only publish the comparison to the visualizer
ENABLE_VISUALIZER_PIPELINES = True   # per-device/topic streams OFF
ENABLE_VISUALIZER_COMPARISON = True  # enable comparison topics
# Disable local OpenCV window (set to True to see a host-side, sync-gated composite for debugging)
SHOW_LOCAL_WINDOW = False

# Image rotation options
ENABLE_ROTATE_180 = True          # you said you need rotation
USE_CAMERA_ORIENTATION = True     # prefer hardware rotation over ImageManip

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
    # Try to reduce device logs to avoid odd non-UTF8 output paths
    with contextlib.suppress(Exception):
        dev = pipeline.getDefaultDevice()
        dev.setLogLevel(dai.LogLevel.OFF)
        dev.setLogOutputLevel(dai.LogLevel.OFF)
    if SET_MANUAL_EXPOSURE:
        cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=100)
    # Prefer hardware rotation if requested
    if ENABLE_ROTATE_180 and USE_CAMERA_ORIENTATION:
        with contextlib.suppress(Exception):
            cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180)
        with contextlib.suppress(Exception):
            cam.initialControl.setImageOrientation(dai.CameraImageOrientation.ROTATE_180)

    source_out = cam.requestOutput((1920, 1080), frame_type, fps=fps_limit)

    # Optional rotate via ImageManip ONLY if we are not using camera orientation
    if ENABLE_ROTATE_180 and not USE_CAMERA_ORIENTATION:
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(8 * 1024 * 1024)
        manip.initialConfig.addRotateDeg(180)
        source_out.link(manip.inputImage)
        source_out = manip.out
        # Ensure ImageManip.out is linked to a HostNode to satisfy pipeline validation
        try:
            manip_host_q = manip.out.createOutputQueue(1, False, "manip_probe")
        except TypeError:
            manip_host_q = manip.out.createOutputQueue(1, False)
    else:
        manip_host_q = None

    # Device-side sync gate: quantize frames to PTP slots at camera FPS so all devices publish the same timestamps
    sync_gate_node = FrameSamplingNode(ptp_slot_period_sec=1.0 / fps_limit).build(source_out)
    # Optional host queue on gated stream (depth=1, non-blocking) to avoid backlog drift
    sync_q = sync_gate_node.out.createOutputQueue(1, False)

    # AprilTag detection and annotations
    apriltag_node = AprilTagAnnotationNode(
        families=args.apriltag_families,
        max_tags=args.apriltag_max,
        quad_decimate=args.apriltag_decimate,
        quad_sigma=args.apriltag_sigma,
        decode_sharpening=args.apriltag_sharpening,
        decision_margin=args.apriltag_decision_margin,
        persistence_seconds=args.apriltag_persistence,
    ).build(sync_gate_node.out)

    # Perspective-rectified panel crop
    warp_node = AprilTagWarpNode(
        panel_width,
        panel_height,
        families=args.apriltag_families,
        quad_decimate=args.apriltag_decimate,
        tag_size=args.apriltag_size,
        z_offset=args.z_offset,
    ).build(sync_gate_node.out)

    # Latest-only sampler using PTP-slotted timestamps; now one sample every 5 seconds
    sampling_node = FrameSamplingNode(ptp_slot_period_sec=SAMPLING_PERIOD_SEC).build(warp_node.out)
    sample_q = sampling_node.out.createOutputQueue(2, False)

    # Analyze LED grid on the sampled (latest) crop
    led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.3).build(sampling_node.out)
    analyzer_out = led_analyzer.out  # Defer queue creation until after pipeline is started
    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024)).build(led_analyzer.out)

    # Compose video + apriltag annotations
    video_composer = VideoAnnotationComposer().build(sync_gate_node.out, apriltag_node.out)
    video_q = video_composer.out.createOutputQueue(1, False)

    # Topics for visualizer (using Node.Output objects)
    topics = [
        ("Video with AprilTags", video_composer.out, "video"),
        ("Panel Crop", warp_node.out, "panel"),
        ("Sampled Panel (PTP slots)", sampling_node.out, "panel"),
        ("LED Grid (32x32)", led_visualizer.out, "led"),
    ]

    # Pre-create host queues for all topics to force queueDepth=1 and non-blocking (prevents backlog drift)
    _ = [out.createOutputQueue(1, False) for _, out, _ in topics]

    # Create a host queue on the base stream to ensure a HostNode link exists pre-build
    sync_queue = sync_gate_node.out.createOutputQueue(1, False)

    # Return topics, sync queue, analyzer queue, and strong references to nodes to prevent premature GC
    nodes = [cam, apriltag_node, warp_node, sampling_node, led_analyzer, led_visualizer, video_composer, manip_host_q]

    return topics, sync_queue, analyzer_out, nodes, sample_q, video_q


with contextlib.ExitStack() as stack:
    pipelines = []
    device_ids = []
    analyzer_queues = []
    topics_by_device = []
    liveness_refs = []  # keep strong references to nodes to prevent GC
    sample_queues = []
    video_queues = []

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        topics, sync_queue, analyzer_out, nodes, sample_q, video_q = build_nodes_on_pipeline(pipeline, device, socket)
        # Create analyzer host queue PRE-BUILD so DepthAI sees the HostNode link
        analyzer_q = analyzer_out.createOutputQueue(1, False)
        sample_queues.append(sample_q)
        video_queues.append(video_q)

        # Add topics BEFORE starting the pipeline (queues must be created pre-build)
        if ENABLE_VISUALIZER_PIPELINES and visualizer is not None:
            suffix = f" [{device.getDeviceId()}]"
            for title, output, topic_type in topics:
                visualizer.addTopic(title + suffix, output, topic_type)

        device_ids.append(device.getDeviceId())
        pipelines.append(pipeline)
        analyzer_queues.append(analyzer_q)
        topics_by_device.append(topics)
        liveness_refs.append(nodes)

    # Create LED grid comparison once we have at least two analyzer queues
    comparison_node = None
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
        # comparison_node.set_queues(analyzer_queues[0], analyzer_queues[1])
        # Expose comparison topics in the visualizer
        visualizer.addTopic("LED Overlay [comparison]", comparison_node.out_overlay, "led")
        visualizer.addTopic("LED Sync Report [comparison]", comparison_node.out_report, "video")

    # Start all pipelines after all topics (including comparison) are registered
    for p in pipelines:
        try:
            p.start()
        except UnicodeDecodeError as e:
            print("[ERROR] UnicodeDecodeError while starting pipeline. Details:")
            print("  args:", e.args)
            print("  preferred encoding:", locale.getpreferredencoding(False))
            print("  fs encoding:", sys.getfilesystemencoding())
            print("  TIP: ensure your shell has UTF-8 locale (e.g., LANG=C.UTF-8).")
            # Re-raise so the caller sees the failure with the extra context
            raise
        if visualizer is not None:
            visualizer.registerPipeline(p)

    # Connect pre-built analyzer queues to comparison node once pipelines are running
    if comparison_node is not None and len(analyzer_queues) >= 2:
        comparison_node.set_queues(analyzer_queues[0], analyzer_queues[1])

    # Buffer for latest sampled frames from each device (used for host-side sync gating / optional local view)
    latest_samples = {}
    latest_videos = {}

    # Idle loop – Visualizer handles display; press 'q' to quit
    while True:
        # --- Host-side synchronisation gate for sampled panel frames (optional local view) ---
        # Collect newest sample from each device (non-blocking)
        for idx, q in enumerate(sample_queues):
            while q.has():
                latest_samples[idx] = q.get()

        # When we have a fresh sample from every device, check timestamp skew
        if len(sample_queues) > 0 and len(latest_samples) == len(sample_queues):
            ts_vals = [f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() for f in latest_samples.values()]
            if max(ts_vals) - min(ts_vals) <= SYNC_THRESHOLD_SEC:
                # If you want a quick visual check, enable the local OpenCV window at the top via SHOW_LOCAL_WINDOW=True.
                if SHOW_LOCAL_WINDOW:
                    # Convert frames to BGR and stack side-by-side for a debug view
                    frames = [latest_samples[i].getCvFrame() for i in range(len(sample_queues))]
                    # Make heights equal for hconcat
                    min_h = min(fr.shape[0] for fr in frames)
                    frames = [cv2.resize(fr, (int(fr.shape[1] * (min_h / fr.shape[0])), min_h)) for fr in frames]
                    composite = cv2.hconcat(frames)
                    cv2.imshow("SYNC: sampled panels", composite)
                    cv2.waitKey(1)
                # Clear after a synchronised tick
                latest_samples.clear()

        # --- Host-side synchronisation gate for annotated video frames (optional local view) ---
        for idx, q in enumerate(video_queues):
            while q.has():
                latest_videos[idx] = q.get()

        if len(video_queues) > 0 and len(latest_videos) == len(video_queues):
            ts_vals = [f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() for f in latest_videos.values()]
            if max(ts_vals) - min(ts_vals) <= SYNC_THRESHOLD_SEC:
                if SHOW_LOCAL_WINDOW:
                    frames = [latest_videos[i].getCvFrame() for i in range(len(video_queues))]
                    min_h = min(fr.shape[0] for fr in frames)
                    frames = [cv2.resize(fr, (int(fr.shape[1] * (min_h / fr.shape[0])), min_h)) for fr in frames]
                    composite = cv2.hconcat(frames)
                    cv2.imshow("SYNC: video with apriltags", composite)
                    cv2.waitKey(1)
                latest_videos.clear()

        key = visualizer.waitKey(1) if visualizer is not None else -1
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
