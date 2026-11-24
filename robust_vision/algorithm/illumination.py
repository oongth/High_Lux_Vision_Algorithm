# 조명 추정: Eq. (2) 초기 조명, Eq. (4)-(6) 가중치, Eq. (3) 풀이

from __future__ import annotations

from functools import lru_cache
import math
import time

import cv2
import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.sparse.linalg import spsolve

SPATIAL_KERNEL_RADIUS = 7
SPATIAL_KERNEL_SIGMA = 3.0
WEIGHT_EPS = 1e-3

BANDED_MAX_WIDTH = 96
BANDED_MAX_BYTES = 64 * 1024 ** 2

RESIDUAL_TOLERANCE = 1e-5
SOLUTION_MIN = -1e-5
SOLUTION_MAX = 1.00001


# Eq. (6). 'paper'는 논문 표기대로 거리 D, 'gaussian'은 거리 제곱을 사용
@lru_cache(maxsize=2)
def spatial_affinity(kind='paper'):
    if kind not in ('paper', 'gaussian'):
        raise ValueError('Unknown spatial kernel.')

    r = SPATIAL_KERNEL_RADIUS
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    distance_squared = (x * x + y * y).astype(np.float64)

    if kind == 'paper':
        distance = np.sqrt(distance_squared)
    else:
        distance = distance_squared

    kernel = np.exp(-distance / (2 * SPATIAL_KERNEL_SIGMA ** 2))
    kernel.setflags(write=False)
    return kernel


def forward_difference(image):
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    dx[:, :-1] = np.diff(image, axis=1)
    dy[:-1, :] = np.diff(image, axis=0)
    return dx, dy


def direction_weight(gradient, kernel, kernel_sum):
    local = cv2.filter2D(gradient, cv2.CV_64F, kernel,
                         borderType=cv2.BORDER_REFLECT_101)
    return kernel_sum / ((np.abs(local) + WEIGHT_EPS) * (np.abs(gradient) + WEIGHT_EPS))


# Eq. (4)-(5)
def illumination_weights(initial, kind='paper'):
    dx, dy = forward_difference(initial)
    kernel = spatial_affinity(kind)
    kernel_sum = float(kernel.sum())

    wx = direction_weight(dx, kernel, kernel_sum)
    wy = direction_weight(dy, kernel, kernel_sum)

    # 마지막 열에서 다음 행 첫 칸으로 이어지는 가짜 경계를 끊음
    wx[:, -1] = 0.0
    wy[-1, :] = 0.0
    return wx, wy


# Eq. (3)의 에너지 값과, 정규방정식 잔차를 계산
def objective(initial, solution, wx, wy, smoothness):
    difference = solution - initial
    dx = np.diff(solution, axis=1)
    dy = np.diff(solution, axis=0)
    ex = smoothness * wx[:, :-1] * dx
    ey = smoothness * wy[:-1, :] * dy

    normal_residual = difference.copy()
    normal_residual[:, :-1] -= ex
    normal_residual[:, 1:] += ex
    normal_residual[:-1, :] -= ey
    normal_residual[1:, :] += ey

    data_term = float(np.sum(difference * difference))
    smooth_term = float(np.sum(ex * dx) + np.sum(ey * dy))
    energy = data_term + smooth_term

    initial_norm = max(np.linalg.norm(initial.ravel()), 1e-12)
    residual = float(np.linalg.norm(normal_residual.ravel()) / initial_norm)
    return energy, residual


def validate_illumination(initial):
    initial = np.asarray(initial, dtype=np.float64)
    if initial.ndim != 2 or initial.size == 0 or not np.isfinite(initial).all():
        raise ValueError('Illumination must be a finite, nonempty 2D array.')
    if np.min(initial) < 0 or np.max(initial) > 1:
        raise ValueError('Illumination must be in [0, 1].')
    return initial


def system_diagonal(rhs, ax, ay):
    diagonal = np.ones_like(rhs)
    diagonal[:, :-1] += ax[:, :-1]
    diagonal[:, 1:] += ax[:, :-1]
    diagonal[:-1, :] += ay[:-1, :]
    diagonal[1:, :] += ay[:-1, :]
    return diagonal


def can_use_banded(width, count):
    band_bytes = (width + 1) * count * 8
    return width <= BANDED_MAX_WIDTH and band_bytes <= BANDED_MAX_BYTES


def solve_sparse(rhs, diagonal, ax, ay):
    count = rhs.size
    ids = np.arange(count).reshape(rhs.shape)
    diagonal_ids = ids.ravel()

    left = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
    right = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
    edge = np.concatenate((ax[:, :-1].ravel(), ay[:-1, :].ravel()))

    values = np.concatenate((diagonal.ravel(), -edge, -edge))
    rows = np.concatenate((diagonal_ids, left, right))
    cols = np.concatenate((diagonal_ids, right, left))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsc()

    solution = spsolve(matrix, rhs.ravel(), permc_spec='MMD_AT_PLUS_A')
    return solution.reshape(rhs.shape)


def check_solution(solution, residual):
    converged = (np.isfinite(solution).all()
                 and math.isfinite(residual)
                 and residual <= RESIDUAL_TOLERANCE)
    if not converged:
        raise RuntimeError(f'Illumination solve did not converge: relative residual={residual:.3g}')

    if np.min(solution) < SOLUTION_MIN or np.max(solution) > SOLUTION_MAX:
        raise RuntimeError('Illumination solve violated the maximum principle.')


# (Id + lambda D^T W D) L = L0 를 float64로 풀이
# 폭이 좁으면 banded Cholesky, 넓으면 메모리 때문에 sparse LU 사용
class IlluminationOptimizer:
    def __init__(self, smoothness=0.15, spatial_kernel='paper'):
        if not math.isfinite(smoothness) or smoothness < 0:
            raise ValueError('smoothness must be finite and nonnegative.')
        spatial_affinity(spatial_kernel)  # 잘못된 kernel 이름이면 여기서 ValueError

        self.smoothness = smoothness
        self.spatial_kernel = spatial_kernel
        self.band = None

    def solve(self, initial):
        start = time.perf_counter()
        initial = validate_illumination(initial)
        wx, wy = illumination_weights(initial, self.spatial_kernel)
        lam = self.smoothness

        if lam == 0 or float(np.ptp(initial)) < 1e-12:
            solution = initial.copy()
            solver = 'constant/identity exact solution'
        else:
            solution, solver = self._solve_linear_system(initial, wx, wy, lam)

        energy, residual = objective(initial, solution, wx, wy, lam)
        initial_energy, _ = objective(initial, initial, wx, wy, lam)
        check_solution(solution, residual)

        result = np.clip(solution, 0, 1).astype(np.float32)
        info = {
            'solve_shape_hw': list(initial.shape),
            'solver': solver,
            'relative_linear_residual': residual,
            'objective_initial': initial_energy,
            'objective_solution': energy,
            'processing_ms': (time.perf_counter() - start) * 1000,
            'residual_scope': 'float64 Eq. (3) solution on solve grid, before float32 storage, '
                              'temporal smoothing, and upsampling',
        }
        return result, info

    def _solve_linear_system(self, initial, wx, wy, lam):
        # 짧은 축이 행 안쪽으로 오도록 전치하면 band 폭이 좁아짐
        transpose = initial.shape[1] > initial.shape[0]
        if transpose:
            rhs = initial.T
            ax, ay = wy.T * lam, wx.T * lam
        else:
            rhs = initial
            ax, ay = wx * lam, wy * lam

        diagonal = system_diagonal(rhs, ax, ay)

        if can_use_banded(rhs.shape[1], rhs.size):
            solution = self._solve_banded(rhs, diagonal, ax, ay)
            solver = 'float64 banded Cholesky'
        else:
            solution = solve_sparse(rhs, diagonal, ax, ay)
            solver = 'float64 sparse LU'

        if transpose:
            solution = solution.T
        return solution, solver

    def _solve_banded(self, rhs, diagonal, ax, ay):
        w = rhs.shape[1]
        shape = (w + 1, rhs.size)

        if self.band is None or self.band.shape != shape:
            self.band = np.zeros(shape, dtype=np.float64, order='F')
        else:
            self.band.fill(0.0)

        self.band[0] = diagonal.ravel()
        self.band[1, :-1] = -ax.ravel()[:-1]
        # 폭이 1이면 두 band가 같은 행(1)을 쓰므로 대입이 아니라 빼기(-=)
        self.band[w, :-w] -= ay.ravel()[:-w]

        solution = solveh_banded(self.band, rhs.ravel(), lower=True,
                                 overwrite_ab=True, check_finite=False)
        return solution.reshape(rhs.shape)


def channel_max(image):
    return cv2.max(cv2.max(image[..., 0], image[..., 1]), image[..., 2])


def channel_min(image):
    return cv2.min(cv2.min(image[..., 0], image[..., 1]), image[..., 2])


# Eq. (2): I와 1-I의 초기 조명. 채널 max/min을 먼저 구한 뒤 축소
def initial_illuminations(image, limit):
    h, w = image.shape[:2]
    forward = channel_max(image).astype(np.float64) / 255.0
    reverse = 1.0 - channel_min(image).astype(np.float64) / 255.0

    if limit and max(h, w) > limit:
        ratio = limit / max(h, w)
        size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
        forward = cv2.resize(forward, size, interpolation=cv2.INTER_AREA)
        reverse = cv2.resize(reverse, size, interpolation=cv2.INTER_AREA)

    # INTER_AREA 계수 합이 1을 아주 조금 넘을 수 있어서 clip
    return np.clip(forward, 0, 1), np.clip(reverse, 0, 1)
