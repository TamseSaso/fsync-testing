import contextlib
import depthai as dai
from utils.sync_analyzer import deviceAnalyzer, deviceComparison
from utils.sampling_node import SharedTicker
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"
SET_MANUAL_EXPOSURE = True  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
# If SAMPLE is None then it checks every frame for synchronization
SAMPLE = None
DEBUG = False

# ---------------------------------------------------------------------------
# Pipeline creation (unchanged API – only uses TARGET_FPS constant)
#  - Extended to also return the node output for visualizer registration.
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
    # No host output queue here; host nodes consume the stream
    output = None
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(6000, 200)
    # Backwards-compatible return plus node output for visualizer usage
    return pipeline, output, node_out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
visualizer = dai.RemoteConnection(httpPort=8082)

with contextlib.ExitStack() as stack:
    # deviceInfos = dai.Device.getAllAvailableDevices()
    # print("=== Found devices: ", deviceInfos)

    queues = []
    pipelines = []
    device_ids = []
    samplers = []
    analyzers = []
    warp_nodes = []
    comparisons = []

    # Create one global ticker so all devices sample at the same wall-clock time
    shared_ticker = None
    if SAMPLE is not None:
        shared_ticker = SharedTicker(period_sec=float(SAMPLE), start_delay_sec=0.3)

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        pipeline, out_q, node_out = createPipeline(pipeline, socket)

        samplers, warp_nodes, analyzers = deviceAnalyzer(node_out, shared_ticker, sample_interval_seconds = SAMPLE, threshold_multiplier = 1.75, visualizer = visualizer, device = device, debug = DEBUG)

        pipelines.append(pipeline)
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)


    # Register comparison topics before starting pipelines (required by RemoteConnection)
    deviceComparison(analyzers, warp_nodes, comparisons, SYNC_THRESHOLD_SEC, visualizer=visualizer, debug = DEBUG)

    # Register pipelines with the visualizer before starting them, so topics can be created.
    for p in pipelines:
        p.start()
        visualizer.registerPipeline(p)

    # Wait until every sampler has received at least one frame; start the global ticker only if enabled
    if shared_ticker is not None:
        shared_ticker.start()
    for s in samplers:
        s.wait_first_frame(timeout=None)

    # Visualizer drives display and sync; no host queue consumption here
    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break