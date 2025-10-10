#!/usr/bin/env python3

"""
Minimal changes to original script:
  * Adds simple timestamp-based synchronisation across multiple devices.
  * Presents frames side‑by‑side when they are within 1 / FPS seconds.
  * Keeps v3 API usage and overall code structure intact.
"""

import contextlib
import datetime

import cv2
import depthai as dai
import time
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
# Helpers
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
# ---------------------------------------------------------------------------
def createPipeline(pipeline: dai.Pipeline, socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A):
    camRgb = (
        pipeline.create(dai.node.Camera)
        .build(socket, sensorFps=TARGET_FPS)
    )
    output = (
        camRgb.requestOutput(
            (640, 480), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.STRETCH
        ).createOutputQueue()
    )
    if SET_MANUAL_EXPOSURE:
        camRgb.initialControl.setManualExposure(1000, 100)
    return pipeline, output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        pipeline, out_q = createPipeline(pipeline, socket)
        pipeline.start()

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
        # Synchronise: we need at least one frame from every camera and their
        # timestamps must align within SYNC_THRESHOLD_SEC.
        # -------------------------------------------------------------------
        if len(latest_frames) == len(queues):
            ts_values = [f.getTimestamp(dai.CameraExposureOffset.END).total_seconds() for f in latest_frames.values()]
            if max(ts_values) - min(ts_values) <= SYNC_THRESHOLD_SEC:
                # Build composite image side‑by‑side
                imgs = []
                for i in range(len(queues)):
                    msg = latest_frames[i]
                    frame = msg.getCvFrame()
                    fps = fpsCounters[i].getFps()
                    cv2.putText(
                        frame,
                        f"{device_ids[i]} | Timestamp: {ts_values[i]} | FPS:{fps:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 50),
                        2,
                        cv2.LINE_AA,
                    )
                    imgs.append(frame)

                sync_status = "in sync" if abs(max(ts_values) - min(ts_values)) < 0.001 else "out of sync"
                delta = max(ts_values) - min(ts_values)
                color = (0, 255, 0) if sync_status == "in sync" else (0, 0, 255)
                
                cv2.putText(
                    imgs[0],
                    f"{sync_status} | delta = {delta*1e3:.3f} ms",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("synced_view", cv2.hconcat(imgs))
                latest_frames.clear()  # Wait for next batch

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
from pathlib import Path
import contextlib
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer

_, args = initialize_argparser()

# Parse panel size from arguments
panel_width, panel_height = map(int, args.panel_size.split(','))

visualizer = dai.RemoteConnection(httpPort=8082)

# Determine multi-device list if applicable
device_names = None
if args.devices:
    print("Multi-device mode enabled")
    device_names = [name.strip() for name in args.devices.split(',') if name.strip()]
elif not args.device:
    # No explicit single device provided; default to two known IPs
    device_names = ["10.12.211.82", "10.12.211.84"]
    print("Multi-device mode enabled (default IPs)")

# Multi-device path if we have a list
if device_names is not None:
    if len(device_names) < 2:
        print("--devices should contain at least two entries. Falling back to default IPs.")
        device_names = ["10.12.211.82", "10.12.211.84"]

    if args.media_path:
        print("--media_path is ignored in multi-device mode; using live cameras for all devices.")

    with contextlib.ExitStack() as stack:
        pipelines = []
        for idx, dev_name in enumerate(device_names):
            print(f"Connecting to device {idx}: {dev_name}")
            pipeline = stack.enter_context(dai.Pipeline(dai.Device(dai.DeviceInfo(dev_name))))
            device = pipeline.getDefaultDevice()
            platform = device.getPlatform().name
            print(f"  Platform: {platform}")

            # Determine frame type per device
            frame_type = (
                dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
            )

            # Determine FPS limit per device if not explicitly set
            fps_limit = args.fps_limit
            if not fps_limit:
                fps_limit = 10 if platform == "RVC2" else 30
                print(
                    f"  FPS limit set to {fps_limit} for {platform} platform. Override via --fps_limit."
                )

            print("  Creating pipeline...")

            # Camera input per device
            cam = pipeline.create(dai.node.Camera).build()
            cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=200)
            source_out = cam.requestOutput((1920, 1080), frame_type, fps=fps_limit)

            # AprilTag detection and annotations
            apriltag_node = AprilTagAnnotationNode(
                families=args.apriltag_families,
                max_tags=args.apriltag_max,
                quad_decimate=args.apriltag_decimate,
            )
            apriltag_node.build(source_out)

            # Perspective-rectified panel crop
            warp_node = AprilTagWarpNode(
                panel_width,
                panel_height,
                families=args.apriltag_families,
                quad_decimate=args.apriltag_decimate,
                tag_size=args.apriltag_size,
                z_offset=args.z_offset,
            )
            warp_node.build(source_out)

            # Create sampling node that captures frames every 2 seconds from warp_node
            sampling_node = FrameSamplingNode(sample_interval_seconds=2.0)
            sampling_node.build(warp_node.out)

            # Create LED grid analyzer to detect 32x32 LED states from sampled frames
            led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5)
            led_analyzer.build(sampling_node.out)

            # Create LED grid visualizer to display the LED grid state
            led_visualizer = LEDGridVisualizer(output_size=(1024, 1024))
            led_visualizer.build(led_analyzer.out)

            # Create composite video with AprilTag annotations overlaid
            video_composer = VideoAnnotationComposer()
            video_composer.build(source_out, apriltag_node.out)

            # Name topics per-device and ensure unique stream keys per device
            prefix = f"{dev_name}"
            visualizer.addTopic(f"{prefix} | Video with AprilTags", video_composer.out, f"video_{idx}")
            visualizer.addTopic(f"{prefix} | Panel Crop", warp_node.out, f"panel_{idx}")
            visualizer.addTopic(f"{prefix} | Sampled Panel (2s)", sampling_node.out, f"panel_sampled_{idx}")
            visualizer.addTopic(f"{prefix} | LED Grid (32x32)", led_visualizer.out, f"led_{idx}")

            pipeline.start()
            visualizer.registerPipeline(pipeline)
            pipelines.append(pipeline)

        # UI loop
        while True:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break
else:
    # Original single-device behavior
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    print(f"Platform: {platform}")

    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )

    if not args.fps_limit:
        args.fps_limit = 10 if platform == "RVC2" else 30
        print(
            f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
        )

    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        # Create camera or video input
        if args.media_path:
            replay = pipeline.create(dai.node.ReplayVideo)
            replay.setReplayVideoFile(Path(args.media_path))
            replay.setOutFrameType(frame_type)
            replay.setLoop(True)
            if args.fps_limit:
                replay.setFps(args.fps_limit)
            source_out = replay.out
        else:
            cam = pipeline.create(dai.node.Camera).build()
            cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=200)
            source_out = cam.requestOutput((1920, 1080), frame_type, fps=args.fps_limit)

        # AprilTag detection and annotations
        apriltag_node = AprilTagAnnotationNode(
            families=args.apriltag_families,
            max_tags=args.apriltag_max,
            quad_decimate=args.apriltag_decimate,
        )
        apriltag_node.build(source_out)

        # Perspective-rectified panel crop
        warp_node = AprilTagWarpNode(
            panel_width,
            panel_height,
            families=args.apriltag_families,
            quad_decimate=args.apriltag_decimate,
            tag_size=args.apriltag_size,
            z_offset=args.z_offset,
        )
        warp_node.build(source_out)

        # Create sampling node that captures frames every 2 seconds from warp_node
        sampling_node = FrameSamplingNode(sample_interval_seconds=2.0)
        sampling_node.build(warp_node.out)

        # Create LED grid analyzer to detect 32x32 LED states from sampled frames
        led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5)
        led_analyzer.build(sampling_node.out)

        # Create LED grid visualizer to display the LED grid state
        led_visualizer = LEDGridVisualizer(output_size=(1024, 1024))
        led_visualizer.build(led_analyzer.out)

        # Create composite video with AprilTag annotations overlaid
        video_composer = VideoAnnotationComposer()
        video_composer.build(source_out, apriltag_node.out)

        # Add video with AprilTag annotations overlaid
        visualizer.addTopic("Video with AprilTags", video_composer.out, "video")
        visualizer.addTopic("Panel Crop", warp_node.out, "panel")
        visualizer.addTopic("Sampled Panel (2s)", sampling_node.out, "panel")
        visualizer.addTopic("LED Grid (32x32)", led_visualizer.out, "led")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while True:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break