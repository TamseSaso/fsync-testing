# PixelRunner — Multi-Camera LED Panel Analysis

Real-time LED pattern analysis and multi-camera frame synchronization using deterministic LED patterns, Luxonis OAK devices, and AprilTag-based perspective correction.

## Overview

PixelRunner captures frames from multiple synchronized OAK cameras pointed at a 32x32 HUB75 LED matrix panel. Each frame is perspective-corrected using four AprilTags mounted around the panel, then the LED grid state is extracted and compared across cameras to verify frame synchronization.

### Pipeline Architecture

```
┌─────────────┐
│  OAK Camera  │  x N devices (auto-discovered)
│  (mono ISP)  │
└──────┬───────┘
       │  FSYNC or PTP synchronized
       ▼
┌──────────────┐
│ ImageManip   │  270° rotation
│ (on-device)  │
└──────┬───────┘
       │  host-side Sync node groups frames by timestamp
       ▼
┌──────────────────┐
│ FrameSamplingNode │  SharedTicker-driven periodic sampling
└──────┬───────────┘
       ▼
┌──────────────────┐
│ AprilTagWarpNode │  Detects 4 AprilTags, computes homography,
│                  │  outputs perspective-rectified panel crop
└──────┬───────────┘
       ▼
┌────────────────┐      ┌───────────────────┐
│ LEDGridAnalyzer│─────▶│ LEDGridVisualizer │  Color-coded grid view
│ (32x32 grid)   │      └───────────────────┘
└──────┬─────────┘
       │  x N streams
       ▼
┌───────────────────┐
│ LEDGridComparison │  TimestampRendezvous matches frames across
│                   │  N streams, compares grid states (O(N))
└───────────────────┘
```

### Background Watchdogs

| Watchdog | Purpose |
|----------|---------|
| **ThresholdWatchdog** | Continuously auto-tunes `threshold_multiplier` on all `LEDGridAnalyzer` instances using Otsu's method + EMA smoothing |
| **HealthMonitor** | Polls `NodeHealth` on every processing node; escalates warnings → errors → critical based on stall duration |
| **Warp-loss watchdog** | If any warp node stops producing frames for 30s, tears down the pipeline and restarts the ISO calibration sweep |

## Hardware Requirements

- **Luxonis OAK cameras** (2+) with mono sensors (OV9282), connected via FSYNC or PTP
- **Adafruit Matrix Portal S3** (ESP32-S3) + 32x32 RGB LED matrix panel (HUB75)
- **4x AprilTags** (tag36h11 family) printed and mounted around the panel corners
- **3D printed stand** (STL files in `3d stand/` if present)
- Adequate 5V power supply for the LED panel

## Installation

```bash
git clone <repository-url>
cd pixelrunner
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `depthai` | 3.4.0 | DepthAI SDK (v3 API) for OAK cameras |
| `depthai-nodes` | 0.4.0 | Pre-built DepthAI host nodes |
| `opencv-contrib-python` | >= 4.7.0 | Computer vision (warp, threshold, drawing) |
| `numpy` | latest | Numerical operations |
| `pupil-apriltags` | >= 1.0.4 | AprilTag detection |

## Usage

```bash
# Run the main multi-camera pipeline
python main_short.py
```

View visualization streams at **http://localhost:8082**

### OAK4 Container Deployment

```bash
oakctl app run .
```

## How It Works

### Startup — ISO Calibration Sweep

On launch, `main()` iterates through ISO values `[100, 200, 400, 600, 800, 1200, 1600]` (low to high). For each ISO:

1. Builds the full pipeline on all discovered devices
2. Waits for all pipelines to report `isBuilt()` and `isRunning()` (30s timeout)
3. Waits for each `FrameSamplingNode` to receive its first frame (15s timeout)
4. Checks warp node output consistency across 3 x 10s windows — every warp node must produce frames in every window (meaning all 4 AprilTags are detected)
5. If calibration **passes**, transitions directly into the main loop on the same pipeline (no teardown/rebuild)
6. If calibration **fails**, tears down and retries with the next ISO

### Runtime — Main Loop

Once calibrated, the pipeline runs continuously:

- **Frame sync monitoring**: The `dai.node.Sync` node groups master/slave frames by timestamp; deltas exceeding `1/(2*FPS)` are logged as warnings
- **LED grid comparison**: `LEDGridComparison` uses `TimestampRendezvous` to match grid states from N cameras by capture timestamp, then compares each against camera 0 (reference-based, O(N))
- **Threshold auto-tuning**: `ThresholdWatchdog` polls LED brightness histograms every sample interval, applies Otsu's method, and smoothly adjusts `threshold_multiplier` via EMA
- **Health monitoring**: `HealthMonitor` polls all nodes every 2s, logs warnings at 15s stall, errors at 45s, critical at 90s
- **Warp-loss recovery**: If any warp node goes silent for 30s (AprilTags lost), the entire pipeline tears down and the ISO sweep restarts from scratch

### Camera Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sensor | Mono (socket index 2) | OV9282 on OAK devices |
| Resolution | 640x480 | With ISP undistortion enabled |
| Frame rate | 30 FPS | Master sets FPS; slaves follow FSYNC |
| Exposure | 6000 us (fixed) | Cannot be changed at runtime |
| ISO | Auto-calibrated | Swept at startup |
| White balance | Daylight (fixed) | |
| Sharpness | 0 (disabled) | |
| Rotation | 270° via ImageManip | Applied on-device |
| Undistortion | Enabled | ISP applies factory calibration on-device |

### Synchronization

Two sync modes are supported (configured via `SYNC_TYPE`):

- **`EXTERNAL`** (default): Hardware FSYNC — one device is master (generates strobe), others are slaves. The master's `setExternalStrobeEnable(True)` drives all cameras.
- **`PTP`**: Precision Time Protocol — each camera syncs to PTP clock via `FrameSyncMode.TIME_PTP`.

Host-side, a `dai.node.Sync` node groups frames from all devices within `1/(2*FPS)` seconds. Slave frames are forwarded to the sync node via dedicated Python threads with adaptive sleep.

### AprilTag Warp Node

`AprilTagWarpNode` is the most complex processing node. Key features:

- **Multi-strategy detection**: Primary detector, high-res fallback (decimate=1.0), ultra fallback (decimate=0.5), blur-mode detector with extra sharpening
- **Preprocessing variants**: CLAHE, bilateral filter, top-hat, adaptive threshold, inversion fallback
- **Tag persistence**: Remembers tag positions for a configurable duration to bridge brief dropouts
- **Last-good warp hold**: Continues outputting the last successful warp for up to 1s if tags blink out
- **Fast-path caching**: After 3 consecutive successful detections, reuses the cached homography matrix and only re-detects every 5th frame
- **Inner corner selection**: For each of the 4 tags, selects the corner closest to the panel center to minimize sensitivity to tag edge detection errors
- **Tunable padding/margins**: `margin`, `padding_left`, `padding_right`, `bottom_y_offset`, `bottom_right_y_offset` adjust the destination quad

### LED Grid Analyzer

`LEDGridAnalyzer` divides the warped panel image into a 32x32 grid and determines each LED's on/off state using adaptive thresholding with a configurable `threshold_multiplier`. The bottom row is treated specially (different threshold scale) as it contains speed/frame-counter metadata from the Matrix Portal.

### LED Grid Comparison

`LEDGridComparison` uses `TimestampRendezvous` to match grid-state buffers from N analyzer streams by capture timestamp. It computes IoU (Intersection over Union) and shift alignment metrics between camera 0 (reference) and all others, producing overlay and report images.

## Project Structure

```tree
pixelrunner/
├── main_short.py                    # Main application (multi-camera, ISO sweep, watchdogs)
├── visualizer_multi.py              # Lightweight multi-device raw camera viewer
├── requirements.txt                 # Python dependencies
├── oakapp.toml                      # OAK app deployment config
├── MatrixPortalS3/
│   └── main.ino                     # Arduino LED pattern generator
└── utils/
    ├── __init__.py
    ├── sync_analyzer.py             # Factory functions: deviceAnalyzer(), deviceComparison()
    ├── sampling_node.py             # SharedTicker, FrameSamplingNode, M8FsyncSamplingNode
    ├── apriltag_node.py             # AprilTagAnnotationNode (annotation-only, for viewer)
    ├── apriltag_warp_node.py        # AprilTagWarpNode (detect + perspective rectify)
    ├── led_grid_analyzer.py         # LEDGridAnalyzer (32x32 grid state extraction)
    ├── led_grid_visualizer.py       # LEDGridVisualizer (color-coded grid rendering)
    ├── led_grid_comparison.py       # LEDGridComparison (cross-camera grid comparison)
    ├── video_annotation_composer.py # VideoAnnotationComposer (overlay compositing)
    ├── health.py                    # NodeHealth + HealthMonitor
    ├── threshold_watchdog.py        # ThresholdWatchdog (auto-tune threshold_multiplier)
    └── timestamp_rendezvous.py      # TimestampRendezvous (N-stream timestamp matching)
```

## Visualization Streams (http://localhost:8082)

When `DEBUG = True`, the following topics are registered per device:

| Topic | Content |
|-------|---------|
| **Input Stream [device_id]** | Raw camera output after rotation |
| **Sample [device_id]** | Periodically sampled frame |
| **Warped Sample [device_id]** | Perspective-corrected panel crop |
| **LED Grid [device_id]** | Color-coded 32x32 grid (green=ON, red=OFF) |
| **LED Sync Overlay** | Side-by-side overlay of all camera grids |
| **LED Sync Report** | Comparison metrics (IoU, shift, pass/fail) |

## Configuration Reference

Key constants in `main_short.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TARGET_FPS` | 30 | Camera frame rate |
| `SYNC_THRESHOLD_SEC` | 1/(2*FPS) | Max acceptable sync delta |
| `SAMPLE` | 2.0 | Sampling interval in seconds |
| `EXPOSURE_US` | 6000 | Fixed exposure in microseconds |
| `ISO_VALUES` | [100..1600] | ISO sweep order (low to high) |
| `CALIBRATION_CHECK_SEC` | 30.0 | Total calibration check duration |
| `WARP_LOSS_TIMEOUT_SEC` | 30.0 | Warp silence before recalibration |
| `DEBUG` | True | Enable visualization topics |
| `SYNC_TYPE` | EXTERNAL | EXTERNAL (FSYNC) or PTP |

## Troubleshooting

- **No AprilTags detected**: Check lighting, tag visibility, and ISO. The calibration sweep will try higher ISO values automatically.
- **Warp node keeps failing**: Ensure all 4 tags are visible and not occluded. Check `decision_margin` threshold in `AprilTagWarpNode`.
- **Bottom of panel warped**: Enable `enableUndistortion=True` on camera output to correct lens distortion.
- **ISO calibration exhausted**: All ISO values failed. Check physical setup — tags may be too small, too far, or poorly lit.
- **"mutex lock failed" errors**: Ensure `SharedTicker.stop()` is called before pipeline teardown.
- **Device crash / FW error**: ISO value may exceed sensor limits (OV9282 max is typically 1600).
- **LEDGridVisualizer garbage collected**: Ensure the `visualizers` list is passed to `deviceAnalyzer()` and kept alive.

## TODO

- [ ] **Sync all topics and streams** — ensure all visualization topics and data streams are synchronized end-to-end across the pipeline, from capture timestamp through sampling, warping, analysis, comparison, and visualization
- [ ] **Update pixelrunner code** - when available update main.ino for MatrixPortalS3
