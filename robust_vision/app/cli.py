# Robust Vision Algorithm: 웹캠 영상의 고조도(강한 조명) 영역을 실시간으로 보정

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

import cv2

from ..algorithm.settings import (MODES, SPATIAL_KERNELS, TUNABLE_FIELDS, VIEWS,
                                  ProcessorConfig, ProcessorControls)
from ..camera.devices import camera_description, list_cameras
from ..processor import BACKENDS
from .runner import run

DEFAULT_OUTPUT_DIR = Path('results')


def _add_control_arguments(parser):
    defaults = ProcessorControls()
    for field in TUNABLE_FIELDS:
        parser.add_argument(f'--{field}', type=int, default=getattr(defaults, field))
    parser.add_argument('--backend', choices=BACKENDS, default='cuda')
    parser.add_argument('--mode', choices=MODES, default=defaults.mode)
    parser.add_argument('--view', choices=VIEWS, default=defaults.view)


def _add_camera_arguments(parser):
    parser.add_argument('--camera-name')
    parser.add_argument('--index', type=int)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30.)


def _add_algorithm_arguments(parser):
    defaults = ProcessorConfig()
    parser.add_argument('--gamma', type=float, default=defaults.gamma)
    parser.add_argument('--reverse-gamma', type=float, default=defaults.reverse_gamma)
    parser.add_argument('--smoothness', type=float, default=defaults.smoothness)
    parser.add_argument('--illumination-size', type=int, default=defaults.illumination_size)
    parser.add_argument('--pyramid-levels', type=int, default=defaults.pyramid_levels)
    parser.add_argument('--temporal-seconds', type=float, default=defaults.temporal_seconds)
    parser.add_argument('--gap-reset-seconds', type=float, default=defaults.gap_reset_seconds)
    parser.add_argument('--shoulder-start', type=float, default=defaults.shoulder_start)
    parser.add_argument('--spatial-kernel', choices=SPATIAL_KERNELS, default=defaults.spatial_kernel)


def _add_run_arguments(parser):
    parser.add_argument('--stall-seconds', type=float, default=3.)
    parser.add_argument('--warmup-seconds', type=float, default=2.)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-preview', action='store_true')
    parser.add_argument('--save-debug', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)


def build_parser():
    parser = argparse.ArgumentParser(prog='robust_vision', description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_control_arguments(parser)
    _add_camera_arguments(parser)
    _add_algorithm_arguments(parser)
    _add_run_arguments(parser)
    return parser


def _controls_from_args(args):
    values = {field: getattr(args, field) for field in TUNABLE_FIELDS}
    return ProcessorControls(**values, mode=args.mode, view=args.view).validate()


def _config_from_args(args):
    return ProcessorConfig(
        gamma=args.gamma,
        reverse_gamma=args.reverse_gamma,
        smoothness=args.smoothness,
        illumination_size=args.illumination_size,
        pyramid_levels=args.pyramid_levels,
        temporal_seconds=args.temporal_seconds,
        gap_reset_seconds=args.gap_reset_seconds,
        shoulder_start=args.shoulder_start,
        spatial_kernel=args.spatial_kernel,
    ).validate()


def _validate_run_args(args):
    if min(args.width, args.height) < 2 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('해상도/FPS는 양수여야 합니다.')
    if not math.isfinite(args.stall_seconds) or args.stall_seconds <= 0:
        raise ValueError('stall-seconds는 양수여야 합니다.')
    for value in (args.seconds, args.warmup_seconds):
        if not (math.isfinite(value) and value >= 0):
            raise ValueError('시간/프레임 수는 0 이상이어야 합니다.')
    if args.max_frames < 0:
        raise ValueError('시간/프레임 수는 0 이상이어야 합니다.')
    if args.no_show and not args.list and not (args.seconds or args.max_frames):
        raise ValueError('--no-show에는 --seconds 또는 --max-frames가 필요합니다.')


def _new_output_directory(output_dir):
    name = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8]
    directory = output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        initial = _controls_from_args(args)
        config = _config_from_args(args)
        _validate_run_args(args)
        cv2.setNumThreads(1)

        if args.list:
            cameras = [camera_description(camera) for camera in list_cameras()]
            print(json.dumps(cameras, ensure_ascii=False, indent=2))
            return 0

        directory = _new_output_directory(args.output_dir)
        print(f'Robust Vision | {initial.to_dict()} | 결과: {directory}', flush=True)
        return run(args, initial, config, directory)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f'Robust Vision 실행 실패: {exc}', file=sys.stderr)
        return 1
