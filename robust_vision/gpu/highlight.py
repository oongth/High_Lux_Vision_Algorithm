# 하이라이트 압축 GPU 렌더링 (algorithm/highlight.py의 GPU 버전)

from __future__ import annotations

import numpy as np

from .runtime import load_kernel_source

BLOCK_SIZE = 256  # cuda/highlight.cu의 sums[8][256]과 같아야 함
KERNEL_CONTROLS = ('strength', 'highlights', 'shadows', 'detail', 'peak')  # 커널 인자 순서
VIEW_CODES = {'result': 0, 'reverse': 1, 'difference': 2}


def _difference_metrics(stats, pixel_count):
    (changed_count, total_change, highlight_count, highlight_source_sum,
     highlight_result_sum, white_count, source_sum, result_sum) = stats
    n = pixel_count

    return {
        'mean_absolute_change_0_255': total_change / (3 * n),
        'changed_gt_1_fraction': changed_count / n,
        'highlight_fraction': highlight_count / n,
        'highlight_source_mean_0_255': highlight_source_sum / highlight_count if highlight_count else None,
        'highlight_result_mean_0_255': highlight_result_sum / highlight_count if highlight_count else None,
        'white_core_fraction': white_count / n,
        'source_mean_0_255': source_sum / n,
        'result_mean_0_255': result_sum / n,
    }


class CudaHighlightRenderer:
    def __init__(self, runtime):
        self.cp = runtime.cp
        self.device = runtime.device
        self.kernel = self.cp.RawKernel(load_kernel_source('highlight.cu'), 'highlight_render',
                                        options=('--fmad=false',))
        self.shape = None

    def _ensure_buffers(self, shape):
        if self.shape == shape:
            return

        cp = self.cp
        self.shape = shape
        self.input = cp.empty(shape, cp.uint8)
        self.output = cp.empty_like(self.input)
        self.fwd = cp.empty(shape[:2], cp.float32)
        self.rev = cp.empty_like(self.fwd)
        self.stats = cp.empty(8, cp.uint64)
        self.dummy = cp.empty(1, cp.float32)

    def render(self, image, maps, controls, config, debug, severity=0.):
        cp = self.cp

        with self.device:
            self._ensure_buffers(image.shape)
            self.input.set(np.ascontiguousarray(image))
            self.fwd.set(np.ascontiguousarray(maps[0]))
            self.rev.set(np.ascontiguousarray(maps[1]))
            self.stats.fill(0)

            if debug:
                debug_y = cp.empty_like(self.fwd)
                debug_mask = cp.empty_like(self.fwd)
                debug_dark = cp.empty(image.shape, cp.float32)
            else:
                debug_y = debug_mask = debug_dark = self.dummy

            height, width = image.shape[:2]
            pixel_count = height * width
            args = (
                self.input, self.fwd, self.rev, self.output, self.stats,
                debug_y, debug_mask, debug_dark,
                np.int32(width), np.int32(height),
                np.float32(config.gamma), np.float32(config.reverse_gamma),
                *[np.float32(getattr(controls, name) / 100.) for name in KERNEL_CONTROLS],
                np.float32(config.shoulder_start), np.float32(severity),
                np.int32(VIEW_CODES[controls.view]), np.int32(debug),
            )
            blocks = (pixel_count + BLOCK_SIZE - 1) // BLOCK_SIZE
            self.kernel((blocks,), (BLOCK_SIZE,), args)

            host = cp.asnumpy(self.output)
            stats = cp.asnumpy(self.stats).tolist()
            metrics = _difference_metrics(stats, pixel_count)

            images = {}
            if debug:
                images = {
                    'target_luma': cp.asnumpy(debug_y),
                    'highlight_mask': cp.asnumpy(debug_mask),
                    'highlight_candidate': cp.asnumpy(debug_dark),
                }
            return host, metrics, images
