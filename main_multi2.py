#!/usr/bin/env python3

import contextlib
import datetime
import time
import cv2
import depthai as dai

from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.video_annotation_composer import VideoAnnotationComposer
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_, args = initialize_argparser()
panel_width, panel_height = map(int, args.panel_size.split(","))
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"
SET_MANUAL_EXPOSURE = True  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
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
    
    # --- START FIX ---
    # This is the raw node output for the visualizer and downstream nodes
    raw_node_out = manip.out 
    
    # This creates the explicit host-side queue for the timestamp sync loop
    sync_queue_out = raw_node_out.createOutputQueue()
    # --- END FIX ---

    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(6000, 100)

    # AprilTag annotation node
    # Build subsequent nodes from the raw_node_out
    apriltag_node = AprilTagAnnotationNode(
        families=args.apriltag_families,
        max_tags=args.apriltag_max,
        quad_decimate=args.apriltag_decimate,
        quad_sigma=args.apriltag_sigma,
        decode_sharpening=args.apriltag_sharpening,
        decision_margin=args.apriltag_decision_margin,
        persistence_seconds=args.apriltag_persistence,
    ).build(raw_node_out) # <-- Use raw_node_out

    video_composer = VideoAnnotationComposer().build(raw_node_out, apriltag_node.out) # <-- Use raw_node_out
    composed_out = video_composer.out

    # Return node outputs for visualizer AND the queue for the sync loop
    return pipeline, raw_node_out, composed_out, sync_queue_out

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

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        
        # --- START FIX ---
        # Unpack the new return values
        pipeline, raw_node_out, composed_out, sync_queue = createPipeline(pipeline, socket)

        # Register topics per device: Give visualizer the NODE OUTPUTS
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Camera" + suffix, raw_node_out, "video")
        visualizer.addTopic("Camera+Tags" + suffix, composed_out, "video")
        
        # FIX 1: Call registerPipeline BEFORE starting
        visualizer.registerPipeline(pipeline)
        pipeline.start()
        # --- END FIX ---

        pipelines.append(pipeline)
        
        # FIX 2: Append the actual QUEUE object for the sync loop
        queues.append(sync_queue)
        
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)

    # Buffer for latest frames; key = queue index
    latest_frames = {}
    fpsCounters = [FPSCounter() for _ in queues]
    receivedFrames = [False for _ in queues]
    while True:
        # -------------------------------------------------------------------
        # Collect the newest frame from each queue (non‑blocking)
        # (This loop will now work correctly)
        # -------------------------------------------------------------------
        for idx, q in enumerate(queues):
            while q.has():
                latest_frames[idx] = q.get()
                if not receivedFrames[idx]:
                    print("=== Received frame from", device_ids[idx])
                    receivedFrames[idx] = True
                fpsCounters[idx].tick()

        # -------------------------------------------------------------------
        # Synchronise gate (no OpenCV visualization in this version)
        # (This will also work correctly)
        # -------------------------------------------------------------------
        if len(latest_frames) == len(queues):
            ts_values = [f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() for f in latest_frames.values()]
            if max(ts_values) - min(ts_values) <= SYNC_THRESHOLD_SEC:
                # In the OpenCV version, we would composite here.
                # With the visualizer, raw topics are already displayed.
                latest_frames.clear()

        key = visualizer.waitKey(1)
        if key == ord("q"):
            break