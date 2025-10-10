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

# Single viewer for all devices
visualizer = dai.RemoteConnection(httpPort=8082)

# Multi-device only; ignore --device, require at least two devices
if args.device and not args.devices:
    print("Ignoring --device. This application requires at least two devices. Use --devices ip1,ip2.")

# Build device list
if args.devices:
    print("Multi-device mode enabled")
    device_names = [name.strip() for name in args.devices.split(',') if name.strip()]
else:
    device_names = ["10.12.211.82", "10.12.211.84"]
    print("Multi-device mode enabled (default IPs)")

assert len(device_names) >= 2, "At least two devices are required. Provide them via --devices ip1,ip2"

if args.media_path:
    print("--media_path is ignored in multi-device mode; using live cameras for all devices.")

with contextlib.ExitStack() as stack:
    # Build all pipelines first
    built = []
    for idx, dev_name in enumerate(device_names):
        print(f"Connecting to device {idx}: {dev_name}")
        pipeline = stack.enter_context(dai.Pipeline(dai.Device(dai.DeviceInfo(dev_name))))
        device = pipeline.getDefaultDevice()
        platform = device.getPlatform().name
        print(f"  Platform: {platform}")

        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
        )

        fps_limit = args.fps_limit
        if not fps_limit:
            fps_limit = 10 if platform == "RVC2" else 30
            print(f"  FPS limit set to {fps_limit} for {platform} platform. Override via --fps_limit.")

        print("  Creating pipeline...")

        cam = pipeline.create(dai.node.Camera).build()
        cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=200)
        source_out = cam.requestOutput((1920, 1080), frame_type, fps=fps_limit)

        apriltag_node = AprilTagAnnotationNode(
            families=args.apriltag_families,
            max_tags=args.apriltag_max,
            quad_decimate=args.apriltag_decimate,
        )
        apriltag_node.build(source_out)

        warp_node = AprilTagWarpNode(
            panel_width,
            panel_height,
            families=args.apriltag_families,
            quad_decimate=args.apriltag_decimate,
            tag_size=args.apriltag_size,
            z_offset=args.z_offset,
        )
        warp_node.build(source_out)

        sampling_node = FrameSamplingNode(sample_interval_seconds=2.0)
        sampling_node.build(warp_node.out)

        led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5)
        led_analyzer.build(sampling_node.out)

        led_visualizer = LEDGridVisualizer(output_size=(1024, 1024))
        led_visualizer.build(led_analyzer.out)

        video_composer = VideoAnnotationComposer()
        video_composer.build(source_out, apriltag_node.out)

        built.append({
            "idx": idx,
            "dev_name": dev_name,
            "pipeline": pipeline,
            "source_out": source_out,
            "video_out": video_composer.out,
            "panel_out": warp_node.out,
            "sampled_out": sampling_node.out,
            "led_out": led_visualizer.out,
        })

    # Start all pipelines, then register topics for each
    for entry in built:
        idx = entry["idx"]
        dev_name = entry["dev_name"]
        pipeline = entry["pipeline"]
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        prefix = f"{dev_name}"
        visualizer.addTopic(f"{prefix} | Raw Camera", entry["source_out"], "video")
        visualizer.addTopic(f"{prefix} | Video with AprilTags", entry["video_out"], "video")
        visualizer.addTopic(f"{prefix} | Panel Crop", entry["panel_out"], "panel")
        visualizer.addTopic(f"{prefix} | Sampled Panel (2s)", entry["sampled_out"], "panel")
        visualizer.addTopic(f"{prefix} | LED Grid (32x32)", entry["led_out"], "led")

    # UI loop
    while True:
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break