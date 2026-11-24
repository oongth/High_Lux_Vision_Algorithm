# 하이라이트 압축 렌더링 (CPU) | 같은 연산의 GPU 버전은 gpu/cuda/highlight.cu.
from __future__ import annotations

import cv2
import numpy as np

from .metrics import LUMA_WEIGHTS_BGR, difference_statistics


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


# S(y) = t + (y-t) / (1 + a(y-t)/(1-t)),  a = ((1-t)/(peak-t) - 1) * amount
def shoulder(value, start, peak, amount):
    a = ((1 - start) / (peak - start) - 1) * amount
    delta = np.maximum(value - start, 0)
    compressed = start + delta / (1 + a * delta / (1 - start))
    return np.where(value > start, compressed, value)


# 조명 맵으로 밝게 보정한 후보(bright)와 어둡게 보정한 후보(dark)
def exposure_candidates(original, forward, reverse, gamma, reverse_gamma):
    forward_divisor = np.maximum(forward[..., None], 1e-6) ** gamma
    reverse_divisor = np.maximum(reverse[..., None], 1e-6) ** reverse_gamma
    bright = np.clip(original / forward_divisor, 0, 1)
    dark = 1. - np.clip((1. - original) / reverse_divisor, 0, 1)
    return bright, dark


def select_view(view, result, original, dark):
    if view == 'reverse':
        return dark
    if view == 'difference':
        return np.clip(np.abs(result - original) * 4, 0, 1)
    return result


def render_highlight_cpu(image, forward, reverse, controls, config, *, return_debug=False, severity=0.):
    original = image.astype(np.float32) / 255.
    bright, dark = exposure_candidates(original, forward, reverse, config.gamma, config.reverse_gamma)

    y = original @ LUMA_WEIGHTS_BGR
    yf = bright @ LUMA_WEIGHTS_BGR
    yd = dark @ LUMA_WEIGHTS_BGR
    max_channel = np.max(original, axis=2)
    min_channel = np.min(original, axis=2)

    h = controls.highlights / 100.
    peak = controls.peak / 100.
    high = smoothstep((y - .48) / .32)
    low = smoothstep((.50 - y) / .35)

    # 목표 휘도
    reverse_gain = np.clip(yd / np.maximum(y, 1e-6), .65 - .30 * severity, 1.)
    lift = np.minimum(np.maximum(yf - y, 0), y * .60)
    shadow_lift = (controls.shadows / 100.) * low * lift
    highlight_drop = .70 * h * high * y * (1 - reverse_gain)
    dual_y = y + shadow_lift - highlight_drop
    target = shoulder(dual_y, config.shoulder_start, peak, h)

    # 클리핑 안 된 디테일 유지
    base = cv2.GaussianBlur(y, (5, 5), 0, borderType=cv2.BORDER_REFLECT_101)
    texture = np.clip(y - base, -.03, .03)
    nonclipped = 1. - smoothstep((max_channel - .94) / .05)
    target += (controls.detail / 100.) * h * high * nonclipped * texture
    target = np.clip(target, 0, 1)

    # 넓은 과노출 톤 확장
    white_anchor = float(shoulder(np.float32(1.), config.shoulder_start, peak, h))
    expanded = white_anchor * np.power(np.maximum(target / white_anchor, 0), 1 + .75 * severity * h)
    target += high * (expanded - target)

    # 채도 (색상 유지, 색역 안으로 제한)
    reverse_strength = np.clip(1 / np.maximum(reverse, 1e-6) ** config.reverse_gamma - 1, 0, 2.5)
    chroma = 1 + .65 * h * high * reverse_strength
    chroma = np.minimum(chroma, target / np.maximum(y - min_channel, 1e-6))
    chroma = np.minimum(chroma, (1 - target) / np.maximum(max_channel - y, 1e-6))

    mapped = target[..., None] + chroma[..., None] * (original - y[..., None])
    corrected = original + (controls.strength / 100.) * (mapped - original)
    corrected = np.clip(corrected, 0, 1)

    if not controls.enabled or controls.strength == 0:
        corrected = original.copy()
        rendered = original
    else:
        rendered = select_view(controls.view, corrected, original, dark)

    output = np.rint(rendered * 255).astype(np.uint8)

    debug = {}
    if return_debug:
        debug = {
            'shadow_candidate': bright,
            'highlight_candidate': dark,
            'target_luma': target,
            'highlight_mask': high,
            'corrected_result': corrected,
        }
    return output, difference_statistics(original, corrected), debug
