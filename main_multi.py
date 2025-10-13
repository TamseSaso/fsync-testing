#!/usr/bin/env python3

"""
Multi-device pipeline with Remote Visualizer.

For each connected device, build the same processing graph as in main.py:
  - Camera/Replay source
  - AprilTag annotations
  - Perspective warp (panel crop)
  - Periodic sampling
  - LED grid analysis and visualization
  - Composited video with AprilTag overlays

All outputs are published to the Remote Visualizer with per-device topic names.
"""

from pathlib import Path
import contextlib

import depthai as dai

from utils.arguments import initialize_argparser
from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Provide device endpoints here (master first if you rely on external sync)
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]]
assert len(DEVICE_INFOS) > 1, "At least two devices are required for this example."


def per_device_frame_type(platform_name: str) -> dai.ImgFrame.Type:
    return dai.ImgFrame.Type.BGR888i if platform_name == "RVC4" else dai.ImgFrame.Type.BGR888p


def default_fps_for_platform(platform_name: str) -> int:
    return 10 if platform_name == "RVC2" else 30


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_, args = initialize_argparser()

# Parse panel size from arguments (e.g. "1024,1024")
panel_width, panel_height = map(int, args.panel_size.split(","))

# One Remote Visualizer instance for all devices/pipelines
visualizer = dai.RemoteConnection(httpPort=8082)

with contextlib.ExitStack() as stack:
    pipelines = []

    for deviceInfo in DEVICE_INFOS:
        # Build a dedicated pipeline per physical device
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
        device = pipeline.getDefaultDevice()

        device_name = deviceInfo.getXLinkDeviceDesc().name
        platform_name = device.getPlatform().name
        print(f"=== Connected to {device_name} | Platform: {platform_name}")

        frame_type = per_device_frame_type(platform_name)
        fps_limit = args.fps_limit if args.fps_limit else default_fps_for_platform(platform_name)
        if not args.fps_limit:
            print(f"FPS limit set to {fps_limit} for {platform_name} (override with --fps_limit)")

        with pipeline:
            # Source: camera (default) or replay
            if args.media_path:
                replay = pipeline.create(dai.node.ReplayVideo)
                replay.setReplayVideoFile(Path(args.media_path))
                replay.setOutFrameType(frame_type)
                replay.setLoop(True)
                if fps_limit:
                    replay.setFps(fps_limit)
                source_out = replay.out
            else:
                cam = pipeline.create(dai.node.Camera).build()
                # Match exposure used in main.py
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

            # Periodic sampling from warped output
            sampling_node = FrameSamplingNode(sample_interval_seconds=2.0)
            sampling_node.build(warp_node.out)

            # LED grid analysis and visualization
            led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5)
            led_analyzer.build(sampling_node.out)

            led_visualizer = LEDGridVisualizer(output_size=(1024, 1024))
            led_visualizer.build(led_analyzer.out)

            # Composite annotated video (raw + AprilTag overlays)
            video_composer = VideoAnnotationComposer()
            video_composer.build(source_out, apriltag_node.out)

            # Topic names are unique per device
            visualizer.addTopic(f"{device_name} | Video with AprilTags", video_composer.out, "video")
            visualizer.addTopic(f"{device_name} | Panel Crop", warp_node.out, "panel")
            visualizer.addTopic(f"{device_name} | Sampled Panel (2s)", sampling_node.out, "panel")
            visualizer.addTopic(f"{device_name} | LED Grid (32x32)", led_visualizer.out, "led")

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        pipelines.append(pipeline)

    # Simple visualizer-driven run loop; quit on 'q'
    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break