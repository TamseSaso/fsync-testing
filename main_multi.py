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
TARGET_FPS = 30  # Will be adjusted based on device platform

# Parse arguments used by the processing nodes (panel, apriltag, etc.)
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))

EFFECTIVE_FPS = int(args.fps_limit) if args.fps_limit else TARGET_FPS
# Sample once every 5 seconds using PTP-slotted timestamps
SAMPLING_PERIOD_SEC = 5.0
# Tolerance for cross-device alignment when we latch the first frame after each 5s boundary
SYNC_THRESHOLD_SEC = 0.050  # 50 ms tolerance
# Keep snapshot/print cadence equal to the sampling period
SNAPSHOT_INTERVAL_SEC = SAMPLING_PERIOD_SEC

# Visualizer toggles: only publish the comparison to the visualizer
ENABLE_VISUALIZER_PIPELINES = False   # per-device/topic streams OFF
ENABLE_VISUALIZER_COMPARISON = True   # comparison overlay/report ONLY
# Disable local OpenCV window (we only use the comparison visualizer topics)
SHOW_LOCAL_WINDOW = False
visualizer = None
if ENABLE_VISUALIZER_PIPELINES or ENABLE_VISUALIZER_COMPARISON:
    visualizer = dai.RemoteConnection(httpPort=8082)



def build_nodes_on_pipeline(pipeline: dai.Pipeline, device: dai.Device, socket: dai.CameraBoardSocket):
    """Build the same processing nodes as main.py on the given pipeline/socket.

    Returns topics for visualizer and a sync queue for timestamp monitoring.
    """
    platform = device.getPlatform().name
    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )
    fps_limit = EFFECTIVE_FPS

    cam = pipeline.create(dai.node.Camera).build(socket, sensorFps=fps_limit)
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
    
    # Create a separate sync queue for timestamp monitoring (non-blocking)
    sync_queue = sampling_node.out.createOutputQueue(1, False)
    
    # Return topics, sync queue, analyzer queue, and strong references to nodes to prevent premature GC
    nodes = [cam, apriltag_node, warp_node, sampling_node, led_analyzer, led_visualizer, video_composer]

    # Register topics with visualizer (per-device streams) — disabled by default
    if ENABLE_VISUALIZER_PIPELINES and visualizer is not None:
        suffix = f" [{device.getDeviceId()}]"
        for title, output, topic_type in topics:
            visualizer.addTopic(title + suffix, output, topic_type)

    return topics, sync_queue, analyzer_queue, nodes


with contextlib.ExitStack() as stack:
    sync_queues = []  # Separate queues for timestamp monitoring
    device_ids = []
    pipelines = []
    liveness_refs = []  # keep strong references to host nodes per pipeline
    analyzer_queues = []  # LED analyzer output queues for comparison
    comparison_node = None

    for idx, deviceInfo in enumerate(DEVICE_INFOS):
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        topics, sync_queue, analyzer_queue, nodes = build_nodes_on_pipeline(pipeline, device, socket)

        # Capture LED grid visualization output for tick source
        led_vis_out = None
        for title, output, topic_type in topics:
            if topic_type == "led" and "LED Grid" in title:
                led_vis_out = output
                break

        if idx == 0 and comparison_node is None:
            if led_vis_out is None:
                # Fallback: use the sampled panel stream as tick if LED Grid wasn't found
                for title, output, topic_type in topics:
                    if topic_type == "panel" and "Sampled Panel" in title:
                        led_vis_out = output
                        break
            comparison_node = LEDGridComparison(
                grid_size=32,
                output_size=(1024, 1024),
                led_period_us=160.0,
                pass_ratio=0.90
            ).build(led_vis_out if led_vis_out is not None else topics[0][1])
            if ENABLE_VISUALIZER_COMPARISON and visualizer is not None:
                visualizer.addTopic("LED Overlay [comparison]", comparison_node.out_overlay, "led")
                visualizer.addTopic("LED Sync Report [comparison]", comparison_node.out_report, "video")

        analyzer_queues.append(analyzer_queue)

        # Removed pipeline.start() and visualizer.registerPipeline(pipeline) here

        # Store sync queue and device info for timestamp monitoring
        sync_queues.append(sync_queue)
        device_ids.append(device.getDeviceId())
        pipelines.append(pipeline)
        liveness_refs.append(nodes)

    if comparison_node is not None and len(analyzer_queues) >= 2:
        comparison_node.set_queues(analyzer_queues[0], analyzer_queues[1])

    # Start all pipelines together after building everything
    for p in pipelines:
        p.start()
    if (ENABLE_VISUALIZER_PIPELINES or ENABLE_VISUALIZER_COMPARISON) and visualizer is not None:
        for p in pipelines:
            visualizer.registerPipeline(p)

    # Per-queue FPS counters (to mirror reference script behavior)
    fpsCounters = [FPSCounter() for _ in sync_queues]


    # Synchronization state
    latest_sync_frames = {}  # key = device_idx, value = sync frame
    receivedFrames = [False] * len(sync_queues)
    anyFrameEver = False
    allDevicesReported = False
    last_sync_report_time = time.monotonic()
    sync_stats = {"in_sync": 0, "out_of_sync": 0}
    
    print("=== Waiting for all devices to be ready (first frame from each)...")
    # Block until each device has produced at least one frame
    while not all(receivedFrames):
        progressed = False
        for idx, sync_queue in enumerate(sync_queues):
            while sync_queue.has():
                latest_sync_frames[idx] = sync_queue.get()
                fpsCounters[idx].tick()
                if not receivedFrames[idx]:
                    print(f"=== Device ready: {device_ids[idx]}")
                    receivedFrames[idx] = True
                progressed = True
        if not progressed:
            time.sleep(0.005)
    # Clear any buffered frames to start in lockstep
    latest_sync_frames.clear()
    print(f"=== All devices ready — starting synchronized visualization (threshold: {SYNC_THRESHOLD_SEC*1000:.2f}ms)")
    if SHOW_LOCAL_WINDOW:
        cv2.namedWindow("synced_view", cv2.WINDOW_NORMAL)

    next_snapshot_time = time.monotonic() + SNAPSHOT_INTERVAL_SEC  # wait before first capture
    
    # Unified visualizer loop with synchronization monitoring
    while True:
        break_main = False
        # Collect newest frame from each device's sync queue (non-blocking)
        frameReceivedThisIter = False
        for idx, sync_queue in enumerate(sync_queues):
            while sync_queue.has():
                latest_sync_frames[idx] = sync_queue.get()
                fpsCounters[idx].tick()
                frameReceivedThisIter = True

        if frameReceivedThisIter and not anyFrameEver:
            print("=== At least one device is sending frames")
            anyFrameEver = True
        if not allDevicesReported and all(receivedFrames):
            print("=== All devices are sending frames - synchronization active")
            allDevicesReported = True

        # Check synchronization: need at least one frame from every device
        if len(latest_sync_frames) == len(sync_queues):
            ordered_idxs = list(range(len(sync_queues)))
            ts_map = {
                i: latest_sync_frames[i].getTimestamp(dai.CameraExposureOffset.END).total_seconds()
                for i in ordered_idxs
            }
            delta = max(ts_map.values()) - min(ts_map.values())

            # Track sync status and only clear when frames are aligned
            if delta <= SYNC_THRESHOLD_SEC:
                sync_stats["in_sync"] += 1

                # --- Build side-by-side composite of aligned frames ---
                imgs = []
                for i in ordered_idxs:
                    msg = latest_sync_frames[i]
                    frame = msg.getCvFrame()
                    fps = fpsCounters[i].getFps()
                    cv2.putText(
                        frame,
                        f"{device_ids[i]} | ts: {ts_map[i]:.6f}s | FPS:{fps:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 50),
                        2,
                        cv2.LINE_AA,
                    )
                    imgs.append(frame)

                # Only take/display a PTP snapshot every SNAPSHOT_INTERVAL_SEC seconds
                if time.monotonic() >= next_snapshot_time:
                    # Match reference script banner semantics (1 ms tightness indicator)
                    delta_ms = delta * 1e3
                    sync_status = "in sync" if abs(delta) < 0.001 else "out of sync"
                    color = (0, 255, 0) if sync_status == "in sync" else (0, 0, 255)
                    cv2.putText(
                        imgs[0],
                        f"{sync_status} | Δ = {delta_ms:.3f} ms",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    if SHOW_LOCAL_WINDOW:
                        cv2.imshow("synced_view", cv2.hconcat(imgs))
                    print(f"=== Captured PTP snapshot; Δ={delta_ms:.3f} ms — waiting {SNAPSHOT_INTERVAL_SEC:.1f}s before next")
                    next_snapshot_time = time.monotonic() + SNAPSHOT_INTERVAL_SEC
                    latest_sync_frames.clear()  # wait for next aligned batch
            else:
                sync_stats["out_of_sync"] += 1
                # Do not clear when out of sync; keep latest frames until they align

            # Report sync status every 5 seconds
            if time.monotonic() - last_sync_report_time > 5.0:
                total = sync_stats["in_sync"] + sync_stats["out_of_sync"]
                if total > 0:
                    sync_rate = (sync_stats["in_sync"] / total) * 100
                    print(
                        f"=== Sync status: {sync_rate:.1f}% in sync "
                        f"({sync_stats['in_sync']} in / {sync_stats['out_of_sync']} out), "
                        f"current delta: {delta*1000:.2f}ms"
                    )
                sync_stats = {"in_sync": 0, "out_of_sync": 0}
                last_sync_report_time = time.monotonic()
        
        # Visualizer + OpenCV key handling (non-blocking)
        if SHOW_LOCAL_WINDOW:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Got q key. Exiting...")
                break

    if SHOW_LOCAL_WINDOW:
        cv2.destroyAllWindows()
