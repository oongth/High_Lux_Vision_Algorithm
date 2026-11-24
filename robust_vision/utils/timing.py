# 처리 시간 통계와 FPS 측정

from __future__ import annotations

from collections import deque
import math
from threading import Lock
import time

import numpy as np

MAX_STORED_INTERVALS = 100000
FPS_STALE_SECONDS = 1.0


def _finite_floats(values):
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


# 밀리초 표본의 개수, 평균, 표준편차, 분위수
def sample_summary(values):
    data = np.asarray(_finite_floats(list(values)), dtype=np.float64)
    if data.size == 0:
        return {'count': 0, 'mean': None, 'std': None, 'min': None,
                'p50': None, 'p95': None, 'p99': None, 'max': None}

    p50, p95, p99 = np.percentile(data, [50, 95, 99])
    return {
        'count': int(data.size),
        'mean': float(data.mean()),
        'std': float(data.std()),
        'min': float(data.min()),
        'p50': float(p50),
        'p95': float(p95),
        'p99': float(p99),
        'max': float(data.max()),
    }


# 프레임 완료 시각으로 FPS 측정. break_interval() 직후 프레임은 간격에서 제외
class StreamTiming:
    def __init__(self, recent=30):
        self._lock = Lock()
        self.count = 0
        self.last = None
        self.last_add = None
        self.intervals = deque(maxlen=MAX_STORED_INTERVALS)
        self.recent = deque(maxlen=recent)

    def add(self, timestamp=None):
        stamp = time.perf_counter() if timestamp is None else float(timestamp)
        with self._lock:
            if self.last is not None and stamp > self.last:
                interval = stamp - self.last
                self.intervals.append(interval)
                self.recent.append(interval)
            self.last = stamp
            self.last_add = stamp
            self.count += 1

    def break_interval(self):
        with self._lock:
            self.last = None

    def recent_fps(self):
        with self._lock:
            is_stale = (self.last_add is None
                        or time.perf_counter() - self.last_add > FPS_STALE_SECONDS)
            if is_stale:
                return 0.

            total = sum(self.recent)
            if total > 0:
                return len(self.recent) / total
            return 0.

    def summary(self):
        with self._lock:
            values = list(self.intervals)
            count = self.count

        total = sum(values)
        fps = len(values) / total if total > 0 else None
        return {
            'frames': count,
            'intervals': len(values),
            'fps': fps,
            'interval_ms': sample_summary([value * 1000 for value in values]),
        }
