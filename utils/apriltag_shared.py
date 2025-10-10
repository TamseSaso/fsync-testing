import threading

try:
    from pupil_apriltags import Detector as AprilTagDetector
except Exception:
    AprilTagDetector = None

_lock = threading.RLock()
_detectors: dict[tuple[str, float], AprilTagDetector] = {}


def get_detector(families: str = "tag36h11", quad_decimate: float = 1.0) -> tuple[AprilTagDetector, threading.RLock]:
    if AprilTagDetector is None:
        raise RuntimeError("pupil-apriltags is not installed. Add 'pupil-apriltags' to requirements.txt.")
    key = (families, float(quad_decimate if quad_decimate and quad_decimate >= 0.5 else 1.0))
    with _lock:
        if key not in _detectors:
            _detectors[key] = AprilTagDetector(
                families=key[0],
                nthreads=2,
                quad_decimate=key[1],
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
            )
        return _detectors[key], _lock


