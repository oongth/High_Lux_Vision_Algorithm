# 알고리즘 설정(시작할 때 고정)과 컨트롤(실행 중 트랙바로 변경)
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

MODES = ('highlight', 'paper')
VIEWS = ('result', 'reverse', 'difference')
SPATIAL_KERNELS = ('paper', 'gaussian')
TUNABLE_FIELDS = ('strength', 'highlights', 'shadows', 'detail', 'peak')
MIN_PEAK = 55


def _is_int_in(value, low, high):
    return type(value) is int and low <= value <= high


@dataclass(frozen=True)
class ProcessorConfig:
    gamma: float = 0.6
    reverse_gamma: float = 0.6
    smoothness: float = 0.15
    illumination_size: int = 64
    pyramid_levels: int = 6
    temporal_seconds: float = 0.05
    gap_reset_seconds: float = 0.5
    shoulder_start: float = 0.5
    spatial_kernel: str = 'paper'

    def validate(self):
        for value in (self.gamma, self.reverse_gamma):
            if not (math.isfinite(value) and 0 < value <= 1):
                raise ValueError('gamma/reverse-gamma must be in (0, 1].')

        if not math.isfinite(self.smoothness) or self.smoothness < 0:
            raise ValueError('smoothness must be nonnegative.')
        if not _is_int_in(self.illumination_size, 0, 1920):
            raise ValueError('illumination-size must be an integer in [0, 1920].')
        if not _is_int_in(self.pyramid_levels, 1, 12):
            raise ValueError('pyramid-levels must be in [1, 12].')
        if not math.isfinite(self.temporal_seconds) or self.temporal_seconds < 0:
            raise ValueError('temporal-seconds must be nonnegative.')
        if not math.isfinite(self.gap_reset_seconds) or self.gap_reset_seconds <= 0:
            raise ValueError('gap-reset-seconds must be positive.')
        if not math.isfinite(self.shoulder_start) or not 0.2 <= self.shoulder_start <= 0.5:
            raise ValueError('shoulder-start must be in [.2, .5].')
        if self.spatial_kernel not in SPATIAL_KERNELS:
            raise ValueError('spatial-kernel must be paper or gaussian.')
        return self


@dataclass(frozen=True)
class ProcessorControls:
    strength: int = 100
    highlights: int = 100
    shadows: int = 20
    detail: int = 40
    peak: int = 70
    enabled: bool = True
    temporal: bool = True
    mode: str = 'highlight'
    view: str = 'result'

    def validate(self):
        for name in ('strength', 'highlights', 'shadows', 'detail'):
            if not _is_int_in(getattr(self, name), 0, 100):
                raise ValueError(f'{name} must be an integer in [0, 100].')

        if not _is_int_in(self.peak, MIN_PEAK, 100):
            raise ValueError('peak must be an integer in [55, 100].')
        if type(self.enabled) is not bool or type(self.temporal) is not bool:
            raise ValueError('enabled/temporal must be bool.')
        if self.mode not in MODES:
            raise ValueError('Unsupported mode/view.')
        if self.view not in VIEWS:
            raise ValueError('Unsupported mode/view.')
        return self

    def to_dict(self):
        return asdict(self)
