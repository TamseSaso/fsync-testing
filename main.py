from pathlib import Path
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
        cam.initialControl.setManualExposure(exposureTimeUs=6000,
                           sensitivityIso=200)
        source_out = cam.requestOutput((1920, 1080), frame_type, fps=args.fps_limit)

    # AprilTag detection and annotations
    apriltag_node = AprilTagAnnotationNode(
        families=args.apriltag_families,
        max_tags=args.apriltag_max,
        quad_decimate=args.apriltag_decimate,
        quad_sigma=args.apriltag_sigma,
        decode_sharpening=args.apriltag_sharpening,
    )
    apriltag_node.build(source_out)

    # Perspective-rectified panel crop
    warp_node = AprilTagWarpNode(
        panel_width, 
        panel_height, 
        families=args.apriltag_families, 
        quad_decimate=args.apriltag_decimate,
        tag_size=args.apriltag_size,
        z_offset=args.z_offset
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