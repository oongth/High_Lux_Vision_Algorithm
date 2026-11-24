# 보정 전후 변화 통계 (하이라이트 비율, 평균 변화량 등)

from __future__ import annotations

import numpy as np

LUMA_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299], np.float32)
HIGHLIGHT_THRESHOLD = 0.75
WHITE_CORE_LEVEL = 250
CHANGE_THRESHOLD = 1


def _to_255_levels(image):
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.int16)


def _mean_or_none(values):
    if values.size == 0:
        return None
    return float(np.mean(values))


# 화면에 진단용 영상을 띄우고 있어도 보정 결과 자체의 변화를 측정
def difference_statistics(original, corrected):
    source = _to_255_levels(original)
    result = _to_255_levels(corrected)
    delta = np.abs(result - source)

    source_luma = original @ LUMA_WEIGHTS_BGR
    bright = source_luma >= HIGHLIGHT_THRESHOLD

    source_luma_255 = np.rint(source_luma * 255)
    result_luma_255 = np.rint((corrected @ LUMA_WEIGHTS_BGR) * 255)

    changed = np.max(delta, axis=2) > CHANGE_THRESHOLD
    white_core = np.all(source >= WHITE_CORE_LEVEL, axis=2)

    return {
        'mean_absolute_change_0_255': float(delta.mean()),
        'changed_gt_1_fraction': float(np.mean(changed)),
        'highlight_fraction': float(np.mean(bright)),
        'highlight_source_mean_0_255': _mean_or_none(source_luma_255[bright]),
        'highlight_result_mean_0_255': _mean_or_none(result_luma_255[bright]),
        'white_core_fraction': float(np.mean(white_core)),
        'source_mean_0_255': float(np.mean(source_luma_255)),
        'result_mean_0_255': float(np.mean(result_luma_255)),
    }
