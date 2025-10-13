#!/usr/bin/env python3

"""
Minimal changes to original script:
  * Adds simple timestamp-based synchronisation across multiple devices.
  * Presents frames side‑by‑side when they are within 1 / FPS seconds.
  * Keeps v3 API usage and overall code structure intact.
  * Visualization is handled by the DepthAI visualizer (no annotations).
"""

import contextlib
import datetime
import time
import cv2
import depthai as dai
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)  # Max drift to accept as "in sync"
SET_MANUAL_EXPOSURE = False  # Set to True to use manual exposure settings
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
# Pipeline creation for raw camera stream
# ---------------------------------------------------------------------------
def createPipeline(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
):
    camRgb = pipeline.create(dai.node.Camera).build(socket, sensorFps=TARGET_FPS)
    output = camRgb.requestOutput(
        (640, 480), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.STRETCH
    )
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(6000, 200)
    return pipeline, output


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
        (640, 480), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.STRETCH
    )
    output = node_out.createOutputQueue()
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(1000, 100)
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

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]
        pipeline, out_q, node_out = createPipeline(pipeline, socket)

        # Register topic per device without any annotations (raw stream)
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Camera" + suffix, node_out, "video")
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        pipelines.append(pipeline)
        queues.append(out_q)
        device_ids.append(deviceInfo.getXLinkDeviceDesc().name)

    # Buffer for latest frames; key = queue index
    latest_frames = {}
    fpsCounters = [FPSCounter() for _ in queues]
    receivedFrames = [False for _ in queues]
    while True:
        # -------------------------------------------------------------------
        # Collect the newest frame from each queue (non‑blocking)
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