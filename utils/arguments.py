import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Pose model to run the inference on.",
        required=False,
        default="luxonis/lite-hrnet:18-coco-192x256",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=None,
        type=int,
    )

    parser.add_argument(
        "-media",
        "--media_path",
        help="Path to the media file you aim to run the model on. If not set, the model will run on the camera input.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--panel_size",
        help="Output size for the LED panel crop (width,height). Default is 512,512.",
        required=False,
        default="1024,1024",
        type=str,
    )

    parser.add_argument(
        "--apriltag_families",
        help="Comma-separated AprilTag families (e.g. tag36h11,tag25h9).",
        required=False,
        default="tag36h11",
        type=str,
    )

    parser.add_argument(
        "--apriltag_max",
        help="Maximum number of AprilTags to annotate.",
        required=False,
        default=4,
        type=int,
    )

    parser.add_argument(
        "--apriltag_decimate",
        help="Decimation factor for AprilTag detector (lower = more sensitive, 0.5-4.0).",
        required=False,
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--apriltag_sigma",
        help="Gaussian blur sigma for AprilTag edge detection (0.0 = no blur, higher = more blur).",
        required=False,
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--apriltag_sharpening",
        help="Sharpening factor during AprilTag decoding (0.0-1.0, higher = more sharpening).",
        required=False,
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--apriltag_decision_margin",
        help="Minimum decision margin for accepting AprilTag detections (lower = more detections, 0-100).",
        required=False,
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--apriltag_persistence",
        help="How long to remember AprilTag positions after last detection in seconds (0 = no persistence).",
        required=False,
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--apriltag_size",
        help="AprilTag size in meters (e.g., 0.1 for 10cm tags).",
        required=False,
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--z_offset",
        help="Z-axis offset from crop in meters (e.g., 0.01 for 1cm offset).",
        required=False,
        default=0.01,
        type=float,
    )
    args = parser.parse_args()

    return parser, args
