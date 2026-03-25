"""Timestamp-based rendezvous for synchronizing N data streams.

Items submitted from different streams are matched when their timestamps
fall within a configurable threshold.  Once all N streams agree on a
timestamp, the matched group is returned and consumed.
"""

import logging
from collections import deque
from typing import Any, List, Optional

log = logging.getLogger(__name__)


class TimestampRendezvous:
    """Matches items from N streams by capture timestamp.

    Usage::

        rv = TimestampRendezvous(n_streams=3, match_threshold_sec=0.02)
        group = rv.submit(stream_idx=0, timestamp_sec=1.000, data=frame_a)
        group = rv.submit(stream_idx=1, timestamp_sec=1.001, data=frame_b)
        group = rv.submit(stream_idx=2, timestamp_sec=0.999, data=frame_c)
        # group is now [frame_a, frame_b, frame_c] (matched!)
    """

    def __init__(
        self,
        n_streams: int,
        match_threshold_sec: float,
        max_buffered: int = 16,
    ) -> None:
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        self.n = n_streams
        self.threshold = float(match_threshold_sec)
        self.max_buffered = int(max_buffered)
        self._buffers: List[deque] = [
            deque(maxlen=self.max_buffered) for _ in range(self.n)
        ]
        self._match_count = 0

    @property
    def match_count(self) -> int:
        return self._match_count

    def submit(
        self, stream_idx: int, timestamp_sec: float, data: Any
    ) -> Optional[List[Any]]:
        """Submit a result from *stream_idx*.

        Returns a list of N matched items (one per stream, ordered by
        stream index) when a full match is found, otherwise ``None``.
        """
        self._buffers[stream_idx].append((timestamp_sec, data))
        return self._try_match()

    def _try_match(self) -> Optional[List[Any]]:
        if any(len(b) == 0 for b in self._buffers):
            return None

        for ref_ts, ref_data in self._buffers[0]:
            group = [ref_data]
            matched = True
            for i in range(1, self.n):
                best = None
                best_delta = float("inf")
                for ts, data in self._buffers[i]:
                    delta = abs(ts - ref_ts)
                    if delta < best_delta:
                        best_delta = delta
                        best = (ts, data)
                if best is None or best_delta > self.threshold:
                    matched = False
                    break
                group.append(best[1])

            if matched:
                matched_ts = ref_ts
                for i in range(self.n):
                    while (
                        self._buffers[i]
                        and self._buffers[i][0][0] <= matched_ts + self.threshold
                    ):
                        self._buffers[i].popleft()
                self._match_count += 1
                return group

        self._prune_stale()
        return None

    def _prune_stale(self) -> None:
        """Drop items that can never match because all streams have newer data."""
        if any(len(b) == 0 for b in self._buffers):
            return
        min_newest = min(b[-1][0] for b in self._buffers)
        cutoff = min_newest - self.threshold
        for b in self._buffers:
            while b and b[0][0] < cutoff:
                b.popleft()
