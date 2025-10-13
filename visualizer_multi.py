#!/usr/bin/env python3

"""
Streams raw video from multiple devices to the DepthAI visualizer.

Changes from the OpenCV version:
  * Removes all OpenCV usage (no windows, no overlays, no annotations).
  * Registers each camera's video stream as a topic in the visualizer.
  * No inter-device frame synchronization – just live streams per device.
"""

import contextlib
import time
import depthai as dai
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps in createPipeline()
SET_MANUAL_EXPOSURE = False  # Set to True to use manual exposure settings
# DEVICE_INFOS: list[dai.DeviceInfo] = ["IP_MASTER", "IP_SLAVE_1"] # Insert the device IPs here, e.g.:
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]] # The master camera needs to be first here
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."
# Debugging
LOG_INTERVAL_SEC = 2.0
WARN_NO_FRAME_SEC = 3.0
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
# Debug: host-side logger to confirm frame flow
# ---------------------------------------------------------------------------
class StreamDebugLogger(dai.node.ThreadedHostNode):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.input = self.createInput()
        self.input.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])
        self.total_frames = 0
        self.window_frames = 0
        self.last_log_time = time.time()
        self.last_recv_time = None
        self.warned_no_frames = False

    def build(self, src: dai.Node.Output) -> "StreamDebugLogger":
        src.link(self.input)
        return self

    def run(self) -> None:
        print(f"[Debug {self.name}] logger started")
        while self.isRunning():
            try:
                msg = self.input.tryGet()
                now = time.time()
                if msg is None:
                    if self.last_recv_time is not None and not self.warned_no_frames:
                        if now - self.last_recv_time > WARN_NO_FRAME_SEC:
                            print(f"[Debug {self.name}] WARNING: no frames for {now - self.last_recv_time:.1f}s")
                            self.warned_no_frames = True
                    time.sleep(0.05)
                    continue

                # Got a frame
                self.total_frames += 1
                self.window_frames += 1
                self.last_recv_time = now
                if self.warned_no_frames:
                    print(f"[Debug {self.name}] frames resumed")
                    self.warned_no_frames = False

                if now - self.last_log_time >= LOG_INTERVAL_SEC:
                    interval = now - self.last_log_time
                    fps = self.window_frames / interval if interval > 0 else 0.0
                    ts = msg.getTimestamp()
                    print(
                        f"[Debug {self.name}] total={self.total_frames} fps={fps:.2f} last_ts={ts}"
                    )
                    self.window_frames = 0
                    self.last_log_time = now
            except Exception as e:
                print(f"[Debug {self.name}] error: {e}")
                time.sleep(0.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
visualizer = dai.RemoteConnection(httpPort=8082)

with contextlib.ExitStack() as stack:

    deviceInfos = dai.Device.getAllAvailableDevices()
    print("=== Found devices: ", deviceInfos)

    for deviceInfo in DEVICE_INFOS:
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        print("=== Connected to", deviceInfo.getDeviceId())
        print("    Device ID:", device.getDeviceId())
        print("    Num of cameras:", len(device.getConnectedCameras()))

        socket = device.getConnectedCameras()[0]

        pipeline, cam_out = createPipeline(pipeline, socket)

        # Attach debug logger to the same stream
        dbg = StreamDebugLogger(device.getDeviceId()).build(cam_out)

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