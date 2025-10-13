#!/usr/bin/env python3

"""
Multi-device viewer using DepthAI visualizer (no OpenCV, no annotations).
Mirrors multi_devices.py device handling; one topic per device stream.
"""

import contextlib
import depthai as dai


# ---------------------------------------------------------------------------
# Configuration (kept close to multi_devices.py)
# ---------------------------------------------------------------------------
TARGET_FPS = 25  # Must match sensorFps if using camera
SET_MANUAL_EXPOSURE = True
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]]  # master first
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."


def main() -> None:
    visualizer = dai.RemoteConnection(httpPort=8082)

    with contextlib.ExitStack() as stack:
        for deviceInfo in DEVICE_INFOS:
            pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
            device = pipeline.getDefaultDevice()

            print("=== Connected to", deviceInfo.getDeviceId())
            print("    Device ID:", device.getDeviceId())
            print("    Num of cameras:", len(device.getConnectedCameras()))

            platform = device.getPlatform().name
            frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

            # Camera
            cam = pipeline.create(dai.node.Camera).build()
            if SET_MANUAL_EXPOSURE:
                cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=200)
            out = cam.requestOutput((1920, 1080), frame_type, fps=TARGET_FPS)

            # Start and register
            pipeline.start()
            visualizer.registerPipeline(pipeline)

            # Register topic for this device (use unique topic type per device)
            suffix = f" [{device.getDeviceId()}]"
            topic_type = f"video_{device.getDeviceId()}"
            visualizer.addTopic("Camera" + suffix, out, topic_type)

        while True:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break


if __name__ == "__main__":
    main()


