from pathlib import Path
import contextlib
import datetime
import time
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

from utils.apriltag_node import AprilTagAnnotationNode
from utils.apriltag_warp_node import AprilTagWarpNode
from utils.sampling_node import FrameSamplingNode
from utils.led_grid_analyzer import LEDGridAnalyzer
from utils.led_grid_visualizer import LEDGridVisualizer
from utils.video_annotation_composer import VideoAnnotationComposer

# ---------------------------------------------------------------------------
# Multi-Device Configuration
# ---------------------------------------------------------------------------
SYNC_THRESHOLD_SEC = 0.05  # Max drift to accept as "in sync" (50ms)
SET_MANUAL_EXPOSURE = True  # Use manual exposure for consistent sync

# Configure device IPs here - add your device IPs to enable multi-device mode
# To enable multi-device mode:
# 1. Uncomment the line below and add your device IP addresses
# 2. Ensure all devices are connected to the same network
# 3. The first device in the list will be treated as the master
# Example: DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["192.168.0.146", "192.168.0.149"]]
#DEVICE_INFOS = []  # Leave empty for single device mode
DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in ["10.12.211.82", "10.12.211.84"]]

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

_, args = initialize_argparser()

# Parse panel size from arguments
panel_width, panel_height = map(int, args.panel_size.split(','))

visualizer = dai.RemoteConnection(httpPort=8082)

# Determine which devices to use
devices_to_use = DEVICE_INFOS if DEVICE_INFOS else [None]  # None means use default device
is_multi_device = len(devices_to_use) > 1

# Initialize first device to determine platform and frame type
first_device = dai.Device(devices_to_use[0]) if devices_to_use[0] else (dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device())
platform = first_device.getPlatform().name
print(f"Platform: {platform}")
print(f"Multi-device mode: {'Enabled' if is_multi_device else 'Disabled'}")
if is_multi_device:
    print(f"Number of devices: {len(devices_to_use)}")

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if not args.fps_limit:
    args.fps_limit = 10 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

# Close the first device to free it for multi-device mode
if is_multi_device:
    first_device.close()
    first_device = None

# ---------------------------------------------------------------------------
# Pipeline creation function for reuse across devices
# ---------------------------------------------------------------------------
def setup_device_nodes(pipeline, device_id="device"):
    """Set up nodes within an active pipeline context."""
    
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
        if SET_MANUAL_EXPOSURE:
            cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=100)
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

    # Return node references
    return {
        'video_composer': video_composer,
        'sampling_node': sampling_node, 
        'led_visualizer': led_visualizer,
        'source_out': source_out
    }

# ---------------------------------------------------------------------------
# Main execution - supports both single and multi-device modes
# ---------------------------------------------------------------------------
if is_multi_device:
    # Multi-device mode with synchronization
    with contextlib.ExitStack() as stack:
        pipelines = []
        device_ids = []
        
        print("Creating pipelines for multiple devices...")
        
        # Initialize devices and pipelines
        for i, device_info in enumerate(devices_to_use):
            device = stack.enter_context(dai.Device(device_info))
            device_id = device_info.getXLinkDeviceDesc().name
            device_ids.append(device_id)
            
            print(f"=== Connected to device {i+1}: {device_id}")
            print(f"    Device ID: {device.getDeviceId()}")
            print(f"    Num of cameras: {len(device.getConnectedCameras())}")
            
            # Create pipeline using context manager for this device
            pipeline = stack.enter_context(dai.Pipeline(device))
            pipelines.append(pipeline)
            
            # Set up nodes within the pipeline context
            nodes = setup_device_nodes(pipeline, device_id)
            
            # Register this device's topics with visualizer
            device_prefix = f"Device_{i+1}"
            visualizer.addTopic(f"{device_prefix} - Video with AprilTags", nodes['video_composer'].out, "video")
            visualizer.addTopic(f"{device_prefix} - Sampled Panel (2s)", nodes['sampling_node'].out, "panel")
            visualizer.addTopic(f"{device_prefix} - LED Grid (32x32)", nodes['led_visualizer'].out, "led")
            visualizer.registerPipeline(pipeline)
            
            # Start the pipeline
            pipeline.start()
        
        # Main loop
        print("Starting multi-device synchronized processing...")
        
        while True:
            # In multi-device mode, the visualizer manages the display
            # Each device pipeline runs independently and feeds the visualizer
            
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting multi-device mode...")
                break

else:
    # Single device mode (original behavior)
    with dai.Pipeline(first_device) as pipeline:
        print("Creating single device pipeline...")
        
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
            if SET_MANUAL_EXPOSURE:
                cam.initialControl.setManualExposure(exposureTimeUs=6000, sensitivityIso=100)
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

        # Add topics to visualizer
        visualizer.addTopic("Video with AprilTags", video_composer.out, "video")
        visualizer.addTopic("Sampled Panel (2s)", sampling_node.out, "panel")
        visualizer.addTopic("LED Grid (32x32)", led_visualizer.out, "led")

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        
        while True:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got q key. Exiting...")
                break