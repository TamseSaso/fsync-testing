import contextlib
import datetime
import logging
import depthai as dai
import numpy as np
import time
import threading
import signal
from typing import Optional, Dict, List
from enum import Enum

from utils.sync_analyzer import deviceAnalyzer, deviceComparison
from utils.sampling_node import SharedTicker
from utils.health import HealthMonitor
from utils.threshold_watchdog import ThresholdWatchdog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pixelrunner")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FPS = 30
SYNC_THRESHOLD_SEC = 1.0 / (2 * TARGET_FPS)
SET_MANUAL_EXPOSURE = True
DEVICE_IPS: List[str] = []  # Leave empty to auto-discover
SAMPLE = 2.0
DEBUG = True

EXPOSURE_US = 6000
ISO_VALUES = [100, 200, 400, 600, 800, 1200, 1600]
DEFAULT_ISO = 800
CALIBRATION_CHECK_SEC = 30.0
WARP_LOSS_TIMEOUT_SEC = 30.0


class SyncType(Enum):
    EXTERNAL = 0
    PTP = 1


SYNC_TYPE = SyncType.EXTERNAL


# ---------------------------------------------------------------------------
# Camera creation – respects master/slave sync roles
# ---------------------------------------------------------------------------
def createCameraOutputs(
    pipeline: dai.Pipeline,
    socket: dai.CameraBoardSocket,
    sensorFps: float,
    role,
    iso: int = DEFAULT_ISO,
):
    if role == dai.ExternalFrameSyncRole.MASTER or SYNC_TYPE == SyncType.PTP:
        cam = pipeline.create(dai.node.Camera).build(socket, sensorFps=sensorFps)
    else:
        cam = pipeline.create(dai.node.Camera).build(socket)

    node_out = cam.requestOutput(
        (640, 480), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.STRETCH,
        enableUndistortion=True,
    )

    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(2 * 1280 * 800)
    manip.initialConfig.addRotateDeg(270)
    node_out.link(manip.inputImage)
    node_out = manip.out

    if SET_MANUAL_EXPOSURE:
        cam.initialControl.setManualExposure(EXPOSURE_US, iso)
        cam.initialControl.setAutoWhiteBalanceMode(
            dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT
        )
        cam.initialControl.setSharpness(0)

    if SYNC_TYPE == SyncType.PTP:
        cam.initialControl.setFrameSyncMode(
            dai.CameraControl.FrameSyncMode.TIME_PTP
        )
        log.info("Setting PTP for %s", socket.name)

    return node_out


def getDeviceName(device: dai.Device) -> str:
    info = device.getDeviceInfo()
    name = info.deviceId
    if info.name is not None and info.name != "":
        name += "[" + info.name + "]"
    return name


# ---------------------------------------------------------------------------
# Sync node creation
# ---------------------------------------------------------------------------
def createSyncNode(masterPipeline, masterNode, masterName, slaveQueues, syncThreshold):
    outputNames = []
    inputQueues = {}

    sync = masterPipeline.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setSyncThreshold(syncThreshold)

    for socketName, camOutput in masterNode.items():
        name = f"master_{masterName}_{socketName}"
        camOutput.link(sync.inputs[name])
        outputNames.append(name)

    for deviceName, sockets in slaveQueues.items():
        for socketName, _ in sockets.items():
            name = f"slave_{deviceName}_{socketName}"
            outputNames.append(name)
            inputQueues[name] = sync.inputs[name].createInputQueue()

    return sync, outputNames, inputQueues


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
running = True


def interruptHandler(sig, frame):
    global running
    if running:
        log.info("Interrupted! Exiting...")
        running = False
    else:
        log.info("Exiting now!")
        exit(0)


signal.signal(signal.SIGINT, interruptHandler)


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
if DEVICE_IPS:
    DEVICE_INFOS = [dai.DeviceInfo(ip) for ip in DEVICE_IPS]
else:
    DEVICE_INFOS = dai.Device.getAllAvailableDevices()
assert len(DEVICE_INFOS) > 1, "At least two devices are required."
log.info("Discovered %d devices", len(DEVICE_INFOS))


# ---------------------------------------------------------------------------
# Calibration check: returns True if all warp nodes produce frames consistently
# ---------------------------------------------------------------------------
def run_calibration(samplers, warp_nodes, all_pipelines) -> bool:
    # Step 1: Wait for all pipelines to be built and running on device
    log.info("Calibrating: waiting for pipelines to be built...")
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if all(p.isBuilt() and p.isRunning() for p in all_pipelines):
            break
        time.sleep(0.1)
    else:
        log.warning("Timeout waiting for pipelines to build/start")
        return False

    log.info("All pipelines built and running")

    # Step 2: Wait for actual frames to arrive through the chain
    for i, s in enumerate(samplers):
        if not s.wait_first_frame(timeout=15.0):
            log.warning("Sampler %d never received a frame", i)
            return False
        log.info("Sampler %d received first frame", i)

    # Step 3: Check warp node output consistency across multiple windows
    n_checks = 3
    check_interval = CALIBRATION_CHECK_SEC / n_checks
    log.info(
        "Calibrating: checking warp node consistency (%d x %.1fs windows)...",
        n_checks, check_interval,
    )

    for check_round in range(n_checks):
        baseline_counts = [wn.health.frames_produced for wn in warp_nodes]
        time.sleep(check_interval)
        final_counts = [wn.health.frames_produced for wn in warp_nodes]

        for i, (before, after) in enumerate(zip(baseline_counts, final_counts)):
            produced = after - before
            if produced == 0:
                log.warning(
                    "Warp node %d produced 0 frames in check %d/%d — calibration FAILED",
                    i, check_round + 1, n_checks,
                )
                return False
            log.info(
                "Warp node %d produced %d frames in check %d/%d",
                i, produced, check_round + 1, n_checks,
            )

    log.info("All warp nodes produced frames consistently across %d checks", n_checks)
    return True


# ---------------------------------------------------------------------------
# Build pipeline, calibrate, and if calibration passes run the main loop
# without tearing down.
# Returns: "failed" | "recalibrate" | "done"
# ---------------------------------------------------------------------------
def build_and_run(iso: int) -> str:
    global running
    log.info("Building pipeline with ISO=%d", iso)

    visualizer = dai.RemoteConnection(httpPort=8082)

    with contextlib.ExitStack() as stack:
        masterPipeline: Optional[dai.Pipeline] = None
        masterNode: Optional[Dict[str, dai.Node.Output]] = None
        masterName: Optional[str] = None
        slavePipelines: Dict[str, dai.Pipeline] = {}
        slaveQueues: Dict[str, Dict[str, dai.MessageQueue]] = {}

        samplers: List = []
        analyzers: List = []
        warp_nodes: List = []
        visualizers: List = []
        comparisons: List = []

        shared_ticker = None
        if SAMPLE is not None:
            shared_ticker = SharedTicker(period_sec=float(SAMPLE), start_delay_sec=0.3)

        for deviceInfo in DEVICE_INFOS:
            pipeline = stack.enter_context(dai.Pipeline(dai.Device(deviceInfo)))
            device = pipeline.getDefaultDevice()
            name = getDeviceName(device)

            role = None
            if SYNC_TYPE == SyncType.EXTERNAL:
                role = device.getExternalFrameSyncRole()

            log.info("Connected to %s", deviceInfo.getDeviceId())
            log.info("  Device ID: %s", device.getDeviceId())
            log.info("  Num of cameras: %d", len(device.getConnectedCameras()))

            socket = device.getConnectedCameras()[2]
            node_out = createCameraOutputs(pipeline, socket, TARGET_FPS, role, iso=iso)

            samplers, warp_nodes, analyzers = deviceAnalyzer(
                node_out,
                shared_ticker,
                sample_interval_seconds=SAMPLE,
                threshold_multiplier=1.40,
                visualizer=visualizer,
                device=device,
                debug=DEBUG,
                samplers=samplers,
                warp_nodes=warp_nodes,
                analyzers=analyzers,
                visualizers=visualizers,
            )

            if SYNC_TYPE == SyncType.EXTERNAL:
                if role == dai.ExternalFrameSyncRole.MASTER:
                    device.setExternalStrobeEnable(True)
                    log.info("%s is master", device.getDeviceId())
                    if masterPipeline is not None:
                        raise RuntimeError("Only one master pipeline is supported")
                    masterPipeline = pipeline
                    masterName = name
                    masterNode = {socket.name: node_out}
                elif role == dai.ExternalFrameSyncRole.SLAVE:
                    slavePipelines[name] = pipeline
                    if slaveQueues.get(name) is None:
                        slaveQueues[name] = {}
                    slaveQueues[name][socket.name] = node_out.createOutputQueue()
                    log.info("%s is slave", device.getDeviceId())
                else:
                    raise RuntimeError(f"Unknown role {role}")
            elif SYNC_TYPE == SyncType.PTP:
                if masterPipeline is None:
                    masterPipeline = pipeline
                    masterName = name
                    masterNode = {socket.name: node_out}
                else:
                    slavePipelines[name] = pipeline
                    if slaveQueues.get(name) is None:
                        slaveQueues[name] = {}
                    slaveQueues[name][socket.name] = node_out.createOutputQueue()

        if masterPipeline is None or masterNode is None:
            raise RuntimeError("No master detected!")
        if len(slavePipelines) < 1:
            raise RuntimeError("No slaves detected!")

        # Create sync node
        sync, outputNames, inputQueues = createSyncNode(
            masterPipeline,
            masterNode,
            masterName,
            slaveQueues,
            datetime.timedelta(milliseconds=1000 / (2 * TARGET_FPS)),
        )
        syncQueue = sync.out.createOutputQueue()

        # Comparison node
        led_cmp = deviceComparison(
            analyzers,
            warp_nodes,
            comparisons,
            SYNC_THRESHOLD_SEC,
            visualizer=visualizer,
            debug=DEBUG,
        )

        timedelta_queue = None
        if led_cmp is not None:
            timedelta_queue = led_cmp.out_timedelta.createOutputQueue(
                maxSize=4, blocking=False
            )

        # Start pipelines
        masterPipeline.start()
        visualizer.registerPipeline(masterPipeline)
        for _, slavePipeline in slavePipelines.items():
            slavePipeline.start()
            visualizer.registerPipeline(slavePipeline)

        if shared_ticker is not None:
            shared_ticker.start()

        # Forward slave frames to the sync node (adaptive sleep)
        def data_collector(deviceName, socketName):
            camOutputQueue = slaveQueues[deviceName][socketName]
            sleep_sec = 0.001
            while running:
                if camOutputQueue.has():
                    inputQueues[f"slave_{deviceName}_{socketName}"].send(
                        camOutputQueue.get()
                    )
                    sleep_sec = 0.0001
                else:
                    time.sleep(sleep_sec)
                    sleep_sec = min(sleep_sec * 1.5, 0.005)

        threads = {}
        for deviceName, sockets in slaveQueues.items():
            for socketName in sockets:
                key = f"slave_{deviceName}_{socketName}"
                threads[key] = threading.Thread(
                    target=data_collector,
                    args=(deviceName, socketName),
                    daemon=True,
                )
                threads[key].start()

        # ---------------------------------------------------------------
        # CALIBRATION — verify all warp nodes produce frames
        # ---------------------------------------------------------------
        all_pipelines = [masterPipeline] + list(slavePipelines.values())
        if not run_calibration(samplers, warp_nodes, all_pipelines):
            if shared_ticker is not None:
                shared_ticker.stop()
            return "failed"

        log.info("Calibration passed — transitioning to main loop")

        # ---------------------------------------------------------------
        # RUNNING — main processing loop (same pipeline, no rebuild)
        # ---------------------------------------------------------------
        threshold_watchdog = ThresholdWatchdog(analyzers=analyzers, poll_interval_sec=SAMPLE)
        threshold_watchdog.start()

        health_monitor = HealthMonitor(
            poll_interval_sec=2.0,
            warn_after_sec=15.0,
            error_after_sec=45.0,
            critical_after_sec=90.0,
        )
        for i, s in enumerate(samplers):
            health_monitor.register(f"sampler_{i}", s.health)
        for i, wn in enumerate(warp_nodes):
            health_monitor.register(f"warp_{i}", wn.health)
        for i, a in enumerate(analyzers):
            health_monitor.register(f"analyzer_{i}", a.health)
        health_monitor.start()

        log.info("Entering main loop")
        recalibrate = False
        while running:
            key = visualizer.waitKey(1)
            if key == ord("q"):
                running = False
                break

            # Warp-loss watchdog: if any warp node hasn't produced in 30s
            now = time.monotonic()
            for i, wn in enumerate(warp_nodes):
                lpt = wn.health.last_produced_time
                if lpt > 0 and (now - lpt) >= WARP_LOSS_TIMEOUT_SEC:
                    log.warning(
                        "Warp node %d silent for %.1fs — triggering recalibration",
                        i, now - lpt,
                    )
                    recalibrate = True
                    break
            if recalibrate:
                break

            while syncQueue.has():
                frameGroup = syncQueue.get()
                if frameGroup.getNumMessages() == len(outputNames):
                    tsValues = {
                        n: frameGroup[n]
                        .getTimestamp(dai.CameraExposureOffset.END)
                        .total_seconds()
                        for n in outputNames
                    }
                    delta = max(tsValues.values()) - min(tsValues.values())
                    if delta > SYNC_THRESHOLD_SEC:
                        log.warning("Sync warning: delta = %.3f ms", delta * 1e3)

            if timedelta_queue is not None:
                try:
                    dt_buffer = timedelta_queue.tryGet()
                    if dt_buffer is not None:
                        dt_sec = np.frombuffer(dt_buffer.getData(), dtype=np.float32)[0]
                        if dt_sec > SYNC_THRESHOLD_SEC:
                            log.warning(
                                "dT=%.6fs exceeds threshold %.6fs",
                                dt_sec, SYNC_THRESHOLD_SEC,
                            )
                except Exception as e:
                    log.debug("timedelta queue error: %s", e)

        # Cleanup
        threshold_watchdog.stop()
        health_monitor.stop()
        if shared_ticker is not None:
            shared_ticker.stop()
        for t in threads.values():
            t.join(timeout=2.0)

    return "recalibrate" if recalibrate else "done"


# ---------------------------------------------------------------------------
# Main entry point — ISO calibration sweep, re-sweep on warp loss
# ---------------------------------------------------------------------------
def main():
    global running

    while running:
        sweep_result = None
        for iso in ISO_VALUES:
            if not running:
                break
            log.info("=== Attempting ISO=%d ===", iso)
            try:
                result = build_and_run(iso)
                if result == "done":
                    running = False
                    break
                elif result == "recalibrate":
                    sweep_result = "recalibrate"
                    break
                # "failed" → try next ISO
                log.warning("ISO=%d failed calibration, trying next", iso)
            except Exception:
                log.error("Error at ISO=%d", iso, exc_info=True)
        else:
            if running:
                log.error("ISO calibration exhausted all values without success")
            break

        if sweep_result == "recalibrate":
            log.warning("Warp node lost — restarting ISO calibration sweep")
            continue
        break

    log.info("Shutdown complete")


if __name__ == "__main__":
    main()
