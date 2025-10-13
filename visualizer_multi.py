#!/usr/bin/env python3

"""
Streams raw video from multiple devices to the DepthAI visualizer.

Changes from the OpenCV version:
  * Removes all OpenCV usage (no windows, no overlays, no annotations).
  * Registers each camera's video stream as a topic in the visualizer.
  * No inter-device frame synchronization – just live streams per device.
"""

import contextlib
import depthai as dai
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SET_MANUAL_EXPOSURE = False  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
# ---------------------------------------------------------------------------
# Pipeline creation for raw camera stream
# ---------------------------------------------------------------------------
def createPipeline(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    frame_type: dai.ImgFrame.Type = dai.ImgFrame.Type.BGR888p,
):
    camRgb = pipeline.create(dai.node.Camera).build(socket, sensorFps=TARGET_FPS)
    output = camRgb.requestOutput(
        (1920, 1080), frame_type, dai.ImgResizeMode.STRETCH
    )
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(6000, 200)
    return pipeline, output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
visualizer = dai.RemoteConnection(httpPort=8082)

with contextlib.ExitStack() as stack:
    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]

        # Choose frame type based on platform for best compatibility
        platform = device.getPlatform().name
        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
        )

        pipeline, cam_out = createPipeline(pipeline, socket, frame_type)

        # Register topic per device without any annotations
        suffix = f" [{device.getDeviceId()}]"
        visualizer.addTopic("Camera" + suffix, cam_out, "video")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break