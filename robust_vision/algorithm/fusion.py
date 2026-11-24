# paper 모드 CPU 융합: Eq. (7)-(8), Laplacian 피라미드 기반 다중 해상도 융합

from __future__ import annotations

import cv2
import numpy as np

EXPOSEDNESS_SIGMA = 0.2
QUALITY_FLOOR = 1e-6
SOFT_WEIGHT_EPS = 1e-12
FUSION_CANDIDATES = ('original', 'bright', 'dark')


# Eq. (7): C * S * E (각 지수 = 1, E의 sigma는 Mertens 값)
# C와 S가 0이 되면 곱 전체가 0이 되므로 작은 양수 하한을 둠
def _visual_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    saturation = np.std(image, axis=2)
    exposedness = np.exp(-np.sum((image - 0.5) ** 2, axis=2) / (2.0 * EXPOSEDNESS_SIGMA ** 2))

    contrast = np.maximum(contrast, QUALITY_FLOOR)
    saturation = np.maximum(saturation, QUALITY_FLOOR)
    return contrast * saturation * exposedness


# hard: Eq. (8), 위치마다 품질이 가장 높은 후보에 1, 나머지 0
#       동률이면 argmax 특성상 첫 후보(원본)가 선택됨
# soft: Eq. (7) 품질을 정규화한 비교용 가중치
def _fusion_weights(quality, method, count):
    if method == 'hard':
        winner = np.argmax(quality, axis=0)
        weights = [(winner == k).astype(np.float32) for k in range(count)]
        return weights, winner

    quality += SOFT_WEIGHT_EPS
    normalized = quality / np.sum(quality, axis=0, keepdims=True)
    winner = np.argmax(normalized, axis=0)
    return list(normalized), winner


# CPU와 GPU 융합이 같은 피라미드 깊이를 쓰도록 공유
def pyramid_depth(shape, levels):
    shortest_side = min(shape[:2])
    max_depth = int(np.floor(np.log2(shortest_side))) + 1
    return max(1, min(levels, max_depth))


def _gaussian_pyramid(array, levels):
    result = [array]
    for _ in range(1, levels):
        result.append(cv2.pyrDown(result[-1]))
    return result


def _laplacian_pyramid(image, depth):
    gaussian = _gaussian_pyramid(image, depth)
    laplacians = []
    for i in range(depth - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacians.append(gaussian[i] - upsampled)
    laplacians.append(gaussian[-1])
    return laplacians


def _collapse_pyramid(levels):
    result = levels[-1]
    for i in range(len(levels) - 2, -1, -1):
        size = (levels[i].shape[1], levels[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + levels[i]
    return result


# Gaussian 가중치 피라미드 + Laplacian 영상 피라미드를 이용한 다중 해상도 융합
def fuse_exposures(images, levels, method):
    quality = np.stack([_visual_quality(image) for image in images])
    weights, winner = _fusion_weights(quality, method, len(images))
    depth = pyramid_depth(images[0].shape, levels)

    accumulators = None
    for image, weight in zip(images, weights):
        image_pyramid = _laplacian_pyramid(image, depth)
        weight_pyramid = _gaussian_pyramid(weight, depth)

        if accumulators is None:
            accumulators = [np.zeros_like(level) for level in image_pyramid]

        for i, level in enumerate(image_pyramid):
            accumulators[i] += level * weight_pyramid[i][..., None]

    result = _collapse_pyramid(accumulators)

    winner_fraction = [float(np.mean(winner == k)) for k in range(len(FUSION_CANDIDATES))]
    info = {
        'actual_pyramid_levels': depth,
        'winner_fraction_original_bright_dark': winner_fraction,
    }
    return np.clip(result, 0.0, 1.0), info
