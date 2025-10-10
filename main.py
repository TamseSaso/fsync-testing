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