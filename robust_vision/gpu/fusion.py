# paper 모드 GPU 융합: 후보 생성, 품질 맵, 다중 해상도 융합 (algorithm/fusion.py의 GPU 버전)

from __future__ import annotations

import numpy as np

from ..algorithm.fusion import pyramid_depth
from .runtime import load_kernel_source

FUSION_KERNEL_NAMES = (
    'candidates', 'weights',
    'down', 'up', 'down_x', 'down_y', 'up_x', 'up_y',
    'accumulate', 'add', 'quantize',
)
THREADS_PER_BLOCK = 256
SEPARABLE_RESIZE_MIN_SIZE = 65536


def _int32s(*values):
    return [np.int32(value) for value in values]


class CuPyExposureFusion:
    def __init__(self, runtime):
        self.cp = runtime.cp
        self.device = runtime.device
        source = load_kernel_source('fusion.cu')
        self.kernels = {
            name: self.cp.RawKernel(source, name, options=('--fmad=false',))
            for name in FUSION_KERNEL_NAMES
        }

    def _launch(self, name, count, *args):
        blocks = (count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self.kernels[name]((blocks,), (THREADS_PER_BLOCK,), args)

    def _copy(self, value):
        value = np.ascontiguousarray(value)
        result = self.cp.empty(value.shape, dtype=value.dtype)
        result.set(value)
        return result

    def _resize_pyramid(self, value, shape, direction):
        h, w = value.shape[:2]
        oh, ow = shape[:2]
        channels = 3 if value.ndim == 3 else 1
        result = self.cp.empty(shape, dtype=self.cp.float32)

        if result.size >= SEPARABLE_RESIZE_MIN_SIZE:
            intermediate = self.cp.empty((h, ow) + value.shape[2:], dtype=self.cp.float32)
            self._launch(direction + '_x', intermediate.size, value, intermediate,
                         *_int32s(w, h, ow, channels))
            self._launch(direction + '_y', result.size, intermediate, result,
                         *_int32s(ow, h, oh, channels))
        else:
            self._launch(direction, result.size, value, result,
                         *_int32s(w, h, ow, oh, channels))
        return result

    def pyramid(self, value, depth):
        result = [value]
        for _ in range(1, depth):
            h, w = result[-1].shape[:2]
            half_shape = ((h + 1) // 2, (w + 1) // 2) + value.shape[2:]
            result.append(self._resize_pyramid(result[-1], half_shape, 'down'))
        return result

    def __call__(self, image, forward, reverse, gamma, levels, method, *, return_debug=False):
        with self.device:
            depth = pyramid_depth(image.shape, levels)

            candidates = self._make_candidates(image, forward, reverse, gamma)
            weights = self._make_weights(candidates, method)
            accumulators = self._blend_pyramids(candidates, weights, depth)
            result = self._collapse_pyramid(accumulators)
            host = self._quantize_to_host(result, image.shape)

            debug = {}
            if return_debug:
                debug = {
                    'shadow_candidate': self.cp.asnumpy(candidates[1]),
                    'highlight_candidate': self.cp.asnumpy(candidates[2]),
                }

            info = {'actual_pyramid_levels': depth, 'backend': 'cupy'}
            return host, info, debug

    def _make_candidates(self, image, forward, reverse, gamma):
        h, w = image.shape[:2]
        candidates = [self.cp.empty(image.shape, dtype=self.cp.float32) for _ in range(3)]
        self._launch('candidates', image.size,
                     self._copy(image), self._copy(forward), self._copy(reverse),
                     *candidates, np.int32(h * w), np.float32(gamma))
        return candidates

    def _make_weights(self, candidates, method):
        h, w = candidates[0].shape[:2]
        weights = [self.cp.empty((h, w), dtype=self.cp.float32) for _ in range(3)]
        self._launch('weights', h * w, *candidates, *weights,
                     np.int32(w), np.int32(h), np.int32(method == 'hard'))
        return weights

    def _blend_pyramids(self, candidates, weights, depth):
        accumulators = None
        for index, (candidate, weight) in enumerate(zip(candidates, weights)):
            image_pyramid = self.pyramid(candidate, depth)
            weight_pyramid = self.pyramid(weight, depth)

            if accumulators is None:
                accumulators = [self.cp.empty_like(level) for level in image_pyramid]

            is_first = index == 0
            for level in range(depth):
                current = image_pyramid[level]
                is_last = level == depth - 1
                if is_last:
                    expanded = current
                else:
                    expanded = self._resize_pyramid(image_pyramid[level + 1], current.shape, 'up')

                self._launch('accumulate', current.size,
                             current, expanded, weight_pyramid[level], accumulators[level],
                             np.int32(current.size), np.int32(is_last), np.int32(is_first))
        return accumulators

    def _collapse_pyramid(self, accumulators):
        result = accumulators[-1]
        for level in range(len(accumulators) - 2, -1, -1):
            expanded = self._resize_pyramid(result, accumulators[level].shape, 'up')
            result = self.cp.empty_like(expanded)
            self._launch('add', result.size, expanded, accumulators[level], result,
                         np.int32(result.size))
        return result

    def _quantize_to_host(self, result, shape):
        output = self.cp.empty(shape, dtype=self.cp.uint8)
        self._launch('quantize', output.size, result, output, np.int32(output.size))
        return self.cp.asnumpy(output)
