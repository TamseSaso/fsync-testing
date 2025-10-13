from pathlib import Path
import contextlib
import types
import depthai as dai
from utils.arguments import initialize_argparser

from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer

def createPipeline(pipeline: dai.Pipeline, args, frame_type: dai.ImgFrame.Type, panel_width: int, panel_height: int):
    # Source: camera or media replay
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

    # Sample every 2 seconds from warp_node
    sampling_node = FrameSamplingNode(sample_interval_seconds=2.0)
    sampling_node.build(warp_node.out)

    # LED grid analyzer and visualizer
    led_analyzer = LEDGridAnalyzer(grid_size=32, threshold_multiplier=1.5)
    led_analyzer.build(sampling_node.out)

    led_visualizer = LEDGridVisualizer(output_size=(1024, 1024))
    led_visualizer.build(led_analyzer.out)

    # Composite video with AprilTag annotations overlaid
    video_composer = VideoAnnotationComposer()
    video_composer.build(source_out, apriltag_node.out)

    return {
        "video_with_apriltags": video_composer.out,
        "panel_crop": warp_node.out,
        "sampled_panel": sampling_node.out,
        "led_grid": led_visualizer.out,
    }


def main():
    _, args = initialize_argparser()

    # Parse panel size from arguments
    panel_width, panel_height = map(int, args.panel_size.split(","))

    visualizer = dai.RemoteConnection(httpPort=8082)
    # Hardcoded devices (master first)
    device_infos: list[dai.DeviceInfo] = [
        dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]
    ]
    selected_infos = device_infos

    with contextlib.ExitStack() as stack:
        for deviceInfo in selected_infos:
            pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
            device = pipeline.getDefaultDevice()

            platform = device.getPlatform().name
            print(f"Platform: {platform} | Device: {device.getDeviceId()}")

            frame_type = (
                dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
            )

            # Per-device FPS defaulting if not set
            per_device_args = types.SimpleNamespace(**vars(args))
            if not per_device_args.fps_limit:
                per_device_args.fps_limit = 10 if platform == "RVC2" else 30
                print(
                    f"\nFPS limit set to {per_device_args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
                )

            print("Creating pipeline...")
            outputs = createPipeline(
                pipeline, per_device_args, frame_type, panel_width, panel_height
            )
            # Suffix topics with device id for clarity
            suffix = f" [{device.getDeviceId()}]"
            visualizer.addTopic(
                "Video with AprilTags" + suffix, outputs["video_with_apriltags"], "video"
            )
            visualizer.addTopic("Panel Crop" + suffix, outputs["panel_crop"], "panel")
            visualizer.addTopic(
                "Sampled Panel (2s)" + suffix, outputs["sampled_panel"], "panel"
            )
            visualizer.addTopic("LED Grid (32x32)" + suffix, outputs["led_grid"], "led")

            # Start pipeline after topics are linked
            pipeline.start()
            visualizer.registerPipeline(pipeline)

        while True:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break


if __name__ == "__main__":
    main()