#!/usr/bin/env python3

import contextlib
import datetime
import time
import cv2
import depthai as dai
from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.video_annotation_composer import VideoAnnotationComposer
from utils.sampling_node import FrameSamplingNode, SharedTicker
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"
SET_MANUAL_EXPOSURE = True  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
# Parse CLI arguments
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))
# ---------------------------------------------------------------------------
# Helpers (identical to multi_devices.py)
# ---------------------------------------------------------------------------
class FPSCounter:
    def __init__(self):
        self.frameTimes = []

    def tick(self):
        now = time.time()
        self.frameTimes.append(now)
        self.frameTimes = self.frameTimes[-100:]

    def getFps(self):
        if len(self.frameTimes) <= 1:
            return 0
        # Calculate the FPS
        return (len(self.frameTimes) - 1) / (self.frameTimes[-1] - self.frameTimes[0])


def format_time(td: datetime.timedelta) -> str:
    hours, remainder_seconds = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder_seconds, 60)
    milliseconds, microseconds_remainder = divmod(td.microseconds, 1000)
    days_prefix = f"{td.days} day{'s' if td.days != 1 else ''}, " if td.days else ""
    return (
        f"{days_prefix}{hours:02d}:{minutes:02d}:{seconds:02d}."
        f"{milliseconds:03d}.{microseconds_remainder:03d}"
    )

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
        camRgb.initialControl.setManualExposure(6000, 100)
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

    # Create one global ticker so all devices sample at the same wall-clock time
    shared_ticker = SharedTicker(period_sec=5.0, start_delay_sec=0.0)

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        pipeline, out_q, node_out = createPipeline(pipeline, socket)

        # Sample a frame every 5 seconds from the live stream, synchronized via a shared ticker
        sampler = FrameSamplingNode(sample_interval_seconds=2.0, shared_ticker=shared_ticker, emit_first_frame_immediately=True).build(node_out)
        samplers.append(sampler)

        apriltag_node = AprilTagAnnotationNode(
                families=args.apriltag_families,
                max_tags=args.apriltag_max,
                quad_decimate=args.apriltag_decimate,
                quad_sigma=args.apriltag_sigma,
                decode_sharpening=args.apriltag_sharpening,
                decision_margin=args.apriltag_decision_margin,
                persistence_seconds=args.apriltag_persistence,
                wait_for_n_tags=None,
            ).build(sampler.out)

        composer = VideoAnnotationComposer().build(sampler.out, apriltag_node.out)

        # Register topic per device without any annotations (raw stream)
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Camera" + suffix, node_out, "video")
        visualizer.addTopic("Sample" + suffix, sampler.out, "video")
        visualizer.addTopic("Sample+AprilTags" + suffix, composer.out, "video")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        pipelines.append(pipeline)
        queues.append(out_q)
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)

    # Wait until every sampler has received at least one frame, then start the global ticker
    for s in samplers:
        s.wait_first_frame(timeout=2.0)
    shared_ticker.start()

    # Visualizer drives display and sync; no host queue consumption here
    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break