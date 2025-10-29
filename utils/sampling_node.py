import time
from threading import Lock, Thread

class SharedTicker:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self._interval = 5.0  # seconds
        self._last_tick = time.time()
        self._tick = False
        self._running = True
        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        while self._running:
            now = time.time()
            if now - self._last_tick >= self._interval:
                self._tick = True
                self._last_tick = now
            else:
                self._tick = False
            time.sleep(0.1)

    def tick(self):
        return self._tick

    @staticmethod
    def get_instance():
        with SharedTicker._lock:
            if SharedTicker._instance is None:
                SharedTicker._instance = SharedTicker()
            return SharedTicker._instance


class FrameSamplingNode:
    def __init__(self):
        self.out = None
        self._last_emitted_tick = 0
        self._ticker = SharedTicker.get_instance()

    def build(self, input_queue):
        class SampledStream:
            def __init__(self, q, ticker, interval=5):
                self.q = q
                self.ticker = ticker
                self.interval = interval
                self._last_emitted_tick = 0

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    frame = self.q.get()
                    if frame.timestamp >= self._last_emitted_tick + self.interval:
                        self._last_emitted_tick = frame.timestamp
                        return frame

        self.out = SampledStream(input_queue, self._ticker)
        return self
