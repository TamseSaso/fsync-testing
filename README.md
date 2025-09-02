# DIY LED Pattern Analysis with Luxonis OAK Devices

Real-time LED pattern analysis and frame synchronization testing using a deterministic LED pattern generator and OAK cameras.

## Demo

![LED Pattern Analysis Demo](example/demo.gif)

*Real-time LED pattern analysis showing 32×32 grid detection, AprilTag-based perspective correction, and color-coded visualization (Green=LED ON, Red=LED OFF)*

## What It Does

- **Single Camera Analysis**: Monitor and analyze 32×32 LED grid states with precise timing
- **Frame Sync Testing**: Verify synchronization between multiple OAK cameras (advanced)
- **AprilTag-based Rectification**: Automatic perspective correction regardless of camera angle
- **Real-time Visualization**: Color-coded LED patterns via web interface

## Hardware Requirements

- **Luxonis DepthAI camera** (OAK series)
- **Adafruit Matrix Portal S3** (ESP32-S3 based) + 32×32 RGB LED matrix panel (HUB75)
- **AprilTags** (tag36h11 family) printed and mounted around panel
- **3D printed stand** (STL files included) + adequate power supply

## Quick Start

1. **Install:**
```bash
git clone <repository-url>
cd fsync-testing
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2. **Hardware Setup:**
   - 3D print stand (`3d stand/Pixelrunner.stl` + `Pixelrunner_legs.stl`)
   - Upload Arduino code (`MatrixPortalS3/main.ino`) to Matrix Portal S3
   - Assemble: LED panel in frame, connect Matrix Portal, mount 4 AprilTags around panel

3. **Run:**
```bash
python main.py
```
View results at http://localhost:8082

## Usage Examples

```bash
# Basic usage
python main.py

# Specific camera/settings
python main.py --device 192.168.1.100 --panel_size 512,512

# Video file input
python main.py --media_path /path/to/video.mp4

# OAK4 container deployment
oakctl app run .



## Configuration

### Command Line Options
- `-d, --device` - Camera device ID or IP address
- `--media_path` - Video file input instead of live camera
- `--panel_size` - LED panel crop size (default: 1024,1024)
- `--fps_limit` - FPS limit (auto: 10 for RVC2, 30 for RVC4)
- `--apriltag_size` - AprilTag size in meters (default: 0.1)

### AprilTag Padding/Margin (Advanced)
Edit `utils/apriltag_warp_node.py` lines 73-75:
```python
self.margin = 0.008          # Top/bottom margin (0.8% of height)
self.padding_left = -0.01    # Left padding (-1% of width)
self.padding_right = -0.012  # Right padding (-1.2% of width)
```

## Visualization Streams (http://localhost:8082)

1. **Video** - Original camera feed
2. **AprilTags** - Feed with AprilTag annotations
3. **Panel Crop** - Perspective-corrected LED panel view
4. **LED Grid (32×32)** - Color-coded LED state analysis

**LED Grid Key:**
- 🟢 Green = LED ON, 🔴 Red = LED OFF
- Bottom row: Speed indicator (cols 1-16) + frame counter (cols 17-32)

## Troubleshooting

- **No AprilTags detected**: Check lighting and tag visibility
- **Poor perspective correction**: Verify AprilTag positioning
- **Low FPS**: Reduce resolution or increase decimation factor
- **Connection issues**: Check camera connection and device ID

## Project Structure

```
fsync-testing/
├── main.py                    # Main application
├── requirements.txt           # Dependencies
├── utils/                     # Processing modules
├── MatrixPortalS3/main.ino   # Arduino LED controller
├── 3d stand/                  # 3D printable components
└── example/                   # Demo assets
```

## Dependencies

- `depthai` - DepthAI SDK for OAK cameras
- `opencv-contrib-python` - Computer vision library
- `pupil-apriltags` - AprilTag detection
- `numpy` - Numerical computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## Support

- Check troubleshooting section above
- Review [DepthAI documentation](https://docs.luxonis.com/)
- Open an issue in this repository