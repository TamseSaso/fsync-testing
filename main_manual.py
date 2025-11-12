import contextlib
import depthai as dai
import time
from utils.manual_sync_analyzer import deviceAnalyzer, deviceComparison, ManualFrameInjector
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"
SET_MANUAL_EXPOSURE = True  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.220", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
# Sampling interval in seconds - frames will be polled and analyzed at this interval
SAMPLE = 5.0
DEBUG = False

# ---------------------------------------------------------------------------
# Pipeline creation - creates output queue for manual frame polling
# ---------------------------------------------------------------------------
def createPipeline(pipeline: dai.Pipeline, socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A):
    camRgb = (
        pipeline.create(dai.node.Camera)
        .build(socket, sensorFps=TARGET_FPS)
    )
    node_out = camRgb.requestOutput(
        (1920, 1080), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.STRETCH
    )
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(4 * 1024 * 1024)
    manip.initialConfig.addRotateDeg(180)
    node_out.link(manip.inputImage)
    node_out = manip.out
    # Create output queue for manual frame polling (like in multi_devices.py)
    output_queue = node_out.createOutputQueue(maxSize=1, blocking=False)
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(6000, 100)
        camRgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT)
        camRgb.initialControl.setSharpness(0)
    return pipeline, output_queue, node_out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
visualizer = dai.RemoteConnection(httpPort=8082)

with contextlib.ExitStack() as stack:
    queues = []
    pipelines = []
    device_ids = []
    frame_injectors = []
    analyzers = []
    warp_nodes = []
    comparisons = []
    latest_frames = {}  # Buffer for latest frames from each queue

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        pipeline, out_q, node_out = createPipeline(pipeline, socket)

        # Create ManualFrameInjector node for this device
        frame_injector = ManualFrameInjector().build()
        frame_injectors.append(frame_injector)

        # Build analyzer chain (bypasses FrameSamplingNode)
        _, warp_node, analyzer = deviceAnalyzer(
            frame_injector,
            threshold_multiplier=1.40,
            visualizer=visualizer,
            device=device,
            debug=DEBUG
        )
        warp_nodes.append(warp_node)
        analyzers.append(analyzer)

        pipelines.append(pipeline)
        queues.append(out_q)
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)

    # Register comparison topics before starting pipelines (required by RemoteConnection)
    deviceComparison(analyzers, warp_nodes, comparisons, SYNC_THRESHOLD_SEC, visualizer=visualizer, debug=DEBUG)

    # Register pipelines with the visualizer before starting them, so topics can be created.
    for p in pipelines:
        p.start()
        visualizer.registerPipeline(p)

    print("=== Waiting for first frames from all devices...")
    received_frames = [False] * len(queues)
    while not all(received_frames):
        for idx, q in enumerate(queues):
            if q.has():
                latest_frames[idx] = q.get()
                if not received_frames[idx]:
                    print(f"=== Received frame from {device_ids[idx]}")
                    received_frames[idx] = True

    print("=== All devices are sending frames. Starting manual sampling...")
    last_sample_time = time.time()

    # Main loop: manually poll frames and feed to analyzers at intervals
    while True:
        # Continuously update latest frames from all queues (like in multi_devices.py)
        for idx, q in enumerate(queues):
            while q.has():
                latest_frames[idx] = q.get()

        # Sample frames at intervals and inject into analyzers
        current_time = time.time()
        if SAMPLE is None or (current_time - last_sample_time) >= SAMPLE:
            # Inject the latest frame from each device into its frame injector
            for idx, frame_injector in enumerate(frame_injectors):
                if idx in latest_frames:
                    frame = latest_frames[idx]
                    # Convert to BGR if needed (frame is NV12 from camera)
                    frame_injector.inject_frame(frame)
                    if DEBUG:
                        print(f"Injected frame from device {idx} at {current_time:.3f}s")
            last_sample_time = current_time

        # Visualizer drives display and sync
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break