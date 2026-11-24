# DUAL 조명 추정 + 하이라이트 압축 처리기

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import time

import cv2
import numpy as np

from .algorithm.fusion import fuse_exposures
from .algorithm.highlight import exposure_candidates, render_highlight_cpu, select_view
from .algorithm.illumination import IlluminationOptimizer, initial_illuminations
from .algorithm.metrics import difference_statistics
from .algorithm.settings import ProcessorConfig, ProcessorControls
from .gpu.fusion import CuPyExposureFusion
from .gpu.highlight import CudaHighlightRenderer
from .gpu.runtime import CudaRuntime

BACKENDS = ('cuda', 'cpu', 'auto')
CPU_BACKEND_NAME = 'scipy+opencv-cpu'
GPU_BACKEND_NAME = 'scipy-banded+cupy-highlight-render'


def _validate_bgr_image(image):
    is_valid = (isinstance(image, np.ndarray)
                and image.dtype == np.uint8
                and image.ndim == 3
                and image.shape[2] == 3
                and min(image.shape[:2]) >= 1)
    if not is_valid:
        raise ValueError('Input must be a nonempty uint8 HxWx3 BGR image.')


def _capture_timestamp(timestamp):
    stamp = time.perf_counter() if timestamp is None else float(timestamp)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError('Timestamp must be finite and nonnegative.')
    return stamp


def _overexposure_severity(initial):
    bright_fraction = float(np.mean((initial[0] + 1 - initial[1]) * .5 > .75))
    severity = float(np.clip((bright_fraction - .10) / .50, 0, 1))
    return bright_fraction, severity


def _resize_maps(maps, height, width):
    resized = []
    for illumination in maps:
        if illumination.shape != (height, width):
            illumination = cv2.resize(illumination, (width, height), interpolation=cv2.INTER_LINEAR)
        resized.append(illumination)
    return tuple(resized)


class DualIlluminationProcessor:
    def __init__(self, config=None, backend='cuda'):
        self.config = (config or ProcessorConfig()).validate()
        if backend not in BACKENDS:
            raise ValueError('Unsupported backend.')
        self.optimizer = IlluminationOptimizer(self.config.smoothness, self.config.spatial_kernel)

        self.runtime = None
        self.reference_fusion = None
        self.renderer = None
        self.backend = CPU_BACKEND_NAME
        self.device_name = None
        self.fallback_reason = None

        if backend in ('cuda', 'auto'):
            try:
                self._init_cuda()
            except Exception as exc:
                if backend == 'cuda':
                    raise RuntimeError(f'CUDA initialization failed: {exc}') from exc
                self.runtime = None
                self.reference_fusion = None
                self.renderer = None
                self.backend = CPU_BACKEND_NAME
                self.fallback_reason = str(exc)

        self.frames = 0
        self.solves = 0
        self.previews = 0
        self.resets = 0
        self.max_residual = 0.
        self.reset(count=False)

    def _init_cuda(self):
        self.runtime = CudaRuntime()
        self.reference_fusion = CuPyExposureFusion(self.runtime)
        self.renderer = CudaHighlightRenderer(self.runtime)
        self.backend = GPU_BACKEND_NAME
        self.device_name = self.runtime.device_name
        self._warm_up_gpu()

    # 작은 영상으로 커널을 미리 컴파일, 실행해 첫 프레임 지연을 없앰
    def _warm_up_gpu(self):
        dummy = np.full((16, 16, 3), 128, np.uint8)
        maps = (np.full((16, 16), .5, np.float32),) * 2
        self.renderer.render(dummy, maps, ProcessorControls(), self.config, False)
        self.reference_fusion(dummy, *maps, self.config.gamma, self.config.pyramid_levels, 'hard')

    def reset(self, *, count=True):
        self.previous_initial = None
        self.previous_maps = None
        self.frame = None
        self.last_key = None
        self.last_timestamp = None
        self.last_shape = None
        self.cached_output = None
        self.cached_info = None
        self.cached_controls = None
        if count:
            self.resets += 1

    # ------------------------------------------------------------------
    # 조명 맵 계산 + 시간 방향 안정화
    # ------------------------------------------------------------------
    def _solve(self, image, timestamp):
        initial = initial_illuminations(image, self.config.illumination_size)
        forward_map, forward_report = self.optimizer.solve(initial[0])
        reverse_map, reverse_report = self.optimizer.solve(initial[1])
        maps = (forward_map, reverse_map)

        self.solves += 2
        self.max_residual = max(self.max_residual,
                                forward_report['relative_linear_residual'],
                                reverse_report['relative_linear_residual'])

        stable, event, alpha = self._temporal_update(initial, maps, image.shape, timestamp)
        self.previous_initial = initial
        self.previous_maps = stable

        bright_fraction, severity = _overexposure_severity(initial)
        mean_previous_weight = float(alpha.mean()) if alpha is not None else 0.

        return {
            'raw': maps,
            'stable': stable,
            'forward': forward_report,
            'reverse': reverse_report,
            'bright_fraction_proxy': bright_fraction,
            'severity': severity,
            'temporal': {'event': event, 'mean_previous_weight': mean_previous_weight},
        }

    def _temporal_update(self, initial, maps, shape, timestamp):
        if self.previous_initial is None:
            return maps, 'first_frame', None
        if self.last_shape != shape:
            return maps, 'shape_reset', None

        dt = timestamp - self.last_timestamp
        if dt <= 0:
            return maps, 'timestamp_reset', None
        if dt > self.config.gap_reset_seconds:
            return maps, 'capture_gap_reset', None

        change = np.maximum(np.abs(initial[0] - self.previous_initial[0]),
                            np.abs(initial[1] - self.previous_initial[1]))
        is_scene_cut = float(np.mean(change)) > .12 or float(np.mean(change > .2)) > .35
        if is_scene_cut:
            return maps, 'scene_cut_reset', None
        if self.config.temporal_seconds <= 0:
            return maps, 'paper_frame_only', None

        alpha = np.exp(-dt / self.config.temporal_seconds) * np.exp(-(change / .03) ** 2)
        stable = tuple((current + alpha * (previous - current)).astype(np.float32)
                       for current, previous in zip(maps, self.previous_maps))
        return stable, 'motion_gated_smoothing', alpha

    # ------------------------------------------------------------------
    # paper 모드 렌더링
    # ------------------------------------------------------------------
    def _paper(self, image, maps, controls, debug):
        if self.runtime is not None and controls.view == 'result':
            return self._paper_gpu(image, maps, controls, debug)
        return self._paper_cpu(image, maps, controls, debug)

    def _paper_gpu(self, image, maps, controls, debug):
        cfg = self.config
        output, _, images = self.reference_fusion(image, *maps, cfg.gamma, cfg.pyramid_levels,
                                                  'hard', return_debug=debug)
        original = image.astype(np.float32) / 255.

        if controls.strength != 100:
            fused = output.astype(np.float32) / 255.
            blended = original + (controls.strength / 100.) * (fused - original)
            output = np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8)

        return output, difference_statistics(original, output.astype(np.float32) / 255.), images

    def _paper_cpu(self, image, maps, controls, debug):
        cfg = self.config
        original = image.astype(np.float32) / 255.
        # paper 모드는 reverse 쪽에도 gamma 사용
        bright, dark = exposure_candidates(original, maps[0], maps[1], cfg.gamma, cfg.gamma)

        fused, _ = fuse_exposures([original, bright, dark], cfg.pyramid_levels, 'hard')
        result = original + (controls.strength / 100.) * (fused - original)
        rendered = select_view(controls.view, result, original, dark)
        measured = np.rint(np.clip(result, 0, 1) * 255).astype(np.float32) / 255.
        output = np.rint(rendered * 255).astype(np.uint8)

        images = {}
        if debug:
            images = {'shadow_candidate': bright, 'highlight_candidate': dark, 'paper_fused': result}
        return output, difference_statistics(original, measured), images

    # ------------------------------------------------------------------
    # 프레임 처리
    # ------------------------------------------------------------------
    def __call__(self, image, controls=None, *, frame_key=None, timestamp=None, return_debug=False):
        start = time.perf_counter()
        _validate_bgr_image(image)
        controls = (controls or ProcessorControls()).validate()
        stamp = _capture_timestamp(timestamp)

        key = self.frames + 1 if frame_key is None else frame_key
        held = self.frame is not None and key == self.last_key

        if held:
            if stamp != self.last_timestamp or image.shape != self.last_shape:
                raise ValueError('Repeated frame key has different timestamp/shape.')
            self.previews += 1
            if controls == self.cached_controls and not return_debug:
                return self._cached_result(start)
        else:
            self.frame = self._solve(image, stamp)
            self.frames += 1
            self.last_key = key
            self.last_timestamp = stamp
            self.last_shape = image.shape

        small = self.frame['stable' if controls.temporal else 'raw']
        h, w = image.shape[:2]
        maps = _resize_maps(small, h, w)
        output, metrics, debug = self._render(image, maps, controls, return_debug)

        downsampled = small[0].shape != (h, w)
        info = self._build_info(controls, key, stamp, held, downsampled, metrics, start)
        self.cached_controls = controls
        self.cached_output = output.copy()
        self.cached_info = deepcopy(info)

        if return_debug:
            info['debug_images'] = {
                **debug,
                'forward_illumination': maps[0].copy(),
                'reverse_illumination': maps[1].copy(),
            }
        return output, info

    def _cached_result(self, start):
        info = deepcopy(self.cached_info)
        info.update(held_frame=True, reused_output=True, solves_this_call=0,
                    processing_ms=(time.perf_counter() - start) * 1000)
        return self.cached_output.copy(), info

    def _render(self, image, maps, controls, return_debug):
        if not controls.enabled or controls.strength == 0:
            normalized = image.astype(np.float32) / 255.
            return image.copy(), difference_statistics(normalized, normalized), {}

        if controls.mode == 'paper':
            return self._paper(image, maps, controls, return_debug)

        severity = self.frame['severity']
        if self.runtime is not None:
            return self.renderer.render(image, maps, controls, self.config, return_debug, severity)
        return render_highlight_cpu(image, *maps, controls, self.config,
                                    return_debug=return_debug, severity=severity)

    def _build_info(self, controls, key, stamp, held, downsampled, metrics, start):
        if controls.mode == 'highlight':
            algorithm = 'DUAL optimization + luminance shoulder compression'
        else:
            algorithm = 'DUAL paper fusion'

        return {
            'algorithm': algorithm,
            'backend': self.backend,
            'source_frame': key,
            'capture_timestamp': stamp,
            'held_frame': held,
            'reused_output': False,
            'solves_this_call': 0 if held else 2,
            'config': asdict(self.config),
            'controls': controls.to_dict(),
            'forward_illumination': deepcopy(self.frame['forward']),
            'reverse_illumination': deepcopy(self.frame['reverse']),
            'downsampled': downsampled,
            'temporal': {
                **self.frame['temporal'],
                'applied': controls.temporal and self.config.temporal_seconds > 0,
                'paper_equation': False,
            },
            'exposure_analysis': {
                'bright_fraction_proxy': self.frame['bright_fraction_proxy'],
                'broad_overexposure_severity': self.frame['severity'],
                'paper_equation': False,
            },
            'difference': metrics,
            'display_peak_is_texture_recovery': False,
            'processing_ms': (time.perf_counter() - start) * 1000,
        }

    def summary(self):
        return {
            'algorithm': 'DUAL Eq. (3) plus selectable highlight display extension',
            'backend': self.backend,
            'device': self.device_name,
            'fallback_reason': self.fallback_reason,
            'config': asdict(self.config),
            'unique_source_frames': self.frames,
            'illumination_solves': self.solves,
            'held_previews': self.previews,
            'manual_resets': self.resets,
            'maximum_relative_linear_residual': self.max_residual,
        }
