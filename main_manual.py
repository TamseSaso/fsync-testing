import contextlib
import depthai as dai
import time
from utils.manual_sync_analyzer import deviceAnalyzer, deviceComparison
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
SAMPLE_INTERVAL = 5.0
DEBUG = True

# ---------------------------------------------------------------------------
# Pipeline creation - creates output queue instead of feeding to host nodes
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
    
    # Create output queue for manual frame polling
    output_queue = node_out.createOutputQueue()
    
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
    warp_nodes = []
    analyzers = []
    visualizers_list = []
    comparisons = []
    latest_frames = {}  # Buffer for latest frames from each device
    node_outputs = []  # Store node outputs for debug visualization

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        pipeline, out_q, node_out = createPipeline(pipeline, socket)

        # Create analyzer chain (without FrameSamplingNode)
        warp_node, led_analyzer, led_visualizer = deviceAnalyzer(
            threshold_multiplier=1.40, 
            visualizer=visualizer, 
            device=device, 
            debug=DEBUG
        )
        
        queues.append(out_q)
        pipelines.append(pipeline)
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)
        node_outputs.append(node_out)
        visualizers_list.append(led_visualizer)

    # Register comparison topics before starting pipelines (required by RemoteConnection)
    deviceComparison(analyzers, warp_nodes, comparisons, SYNC_THRESHOLD_SEC, visualizer=visualizer, debug=DEBUG)

    # Add debug visualization topics if enabled
    if DEBUG:
        for idx, (node_out, device_id, led_vis) in enumerate(zip(node_outputs, device_ids, visualizers_list)):
            suffix = f" [{device_id}]"
            visualizer.addTopic("Input Stream" + suffix, node_out, "images")
            visualizer.addTopic("LED Grid" + suffix, led_vis.out, "images")

    # Register pipelines with the visualizer and start them
    for p in pipelines:
        p.start()
        visualizer.registerPipeline(p)

    print("=== Waiting for first frames from all devices...")
    receivedFrames = [False] * len(queues)
    
    # Wait for first frame from each device
    while not all(receivedFrames):
        for idx, q in enumerate(queues):
            if not receivedFrames[idx] and q.has():
                latest_frames[idx] = q.get()
                receivedFrames[idx] = True
                print(f"=== Received first frame from {device_ids[idx]}")

    print("=== All devices sending frames. Starting manual sampling...")
    last_sample_time = time.time()

    # Main loop: manually poll frames and feed to analyzers at intervals
    while True:
        # Continuously update latest frames from all queues
        for idx, q in enumerate(queues):
            while q.has():
                latest_frames[idx] = q.get()

        # Check if it's time to sample
        current_time = time.time()
        if current_time - last_sample_time >= SAMPLE_INTERVAL:
            # Send the latest frame from each device to its warp node
            for idx, warp_node in enumerate(warp_nodes):
                if idx in latest_frames:
                    frame = latest_frames[idx]
                    # Send frame to the warp node's input
                    warp_node.input.send(frame)
                    if DEBUG:
                        ts = frame.getTimestamp(dai.CameraExposureOffset.END).total_seconds()
                        print(f"Frame sent to analyzer {idx} at {current_time:.3f}s (ts={ts:.6f})")
            
            last_sample_time = current_time

        key = visualizer.waitKey(1)
        if key == ord("q"):
            break