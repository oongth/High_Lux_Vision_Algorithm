"""실행 환경 준비: 로컬 패키지 경로, 프로젝트 전용 Python 재실행, 필수 패키지 확인.

이 모듈은 표준 라이브러리만 사용합니다.
cv2, numpy 같은 외부 패키지를 import하기 전에 실행되어야 하기 때문입니다.
"""
from __future__ import annotations

import os
from pathlib import Path
import site
import subprocess
import sys

PACKAGE_NAME = 'robust_vision'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REEXEC_FLAG = 'ROBUST_VISION_PROJECT_PYTHON'

# .packages, .build가 있는 폴더. 환경변수를 설정하면 다른 폴더의 환경을 복사 없이 공유
ENV_ROOT_VARIABLE = 'ROBUST_VISION_ENV_ROOT'
ENV_ROOT = Path(os.environ.get(ENV_ROOT_VARIABLE) or PROJECT_ROOT).expanduser().resolve()
PACKAGE_DIR = ENV_ROOT / '.packages' / sys.implementation.cache_tag


def activate_local_packages():
    if not PACKAGE_DIR.is_dir():
        return

    package_dir = str(PACKAGE_DIR)
    site.addsitedir(package_dir)
    if package_dir in sys.path:
        sys.path.remove(package_dir)
    sys.path.insert(0, package_dir)


def is_help_requested(args):
    return '-h' in args or '--help' in args


def is_cpu_backend(args):
    if '--backend=cpu' in args:
        return True

    for i in range(len(args) - 1):
        if args[i] == '--backend' and args[i + 1] == 'cpu':
            return True

    return False


def find_bundled_python():
    python_dir = ENV_ROOT / '.build' / 'python'
    candidates = sorted(python_dir.glob('cpython-3.12.*/python.exe'))

    if not candidates:
        return None
    return candidates[0]


def is_same_executable(path_a, path_b):
    return Path(path_a).resolve() == Path(path_b).resolve()


# tools/setup_cupy.ps1이 만든 .build/python/cpython-3.12.*가 있으면 그 Python으로 다시 실행
def use_project_python(argv=None):
    if argv is None:
        args = list(sys.argv[1:])
    else:
        args = list(argv)

    if os.environ.get(REEXEC_FLAG) == '1':
        return None

    if is_help_requested(args) or is_cpu_backend(args):
        return None

    bundled_python = find_bundled_python()
    if bundled_python is None:
        return None
    if is_same_executable(sys.executable, bundled_python):
        return None

    command = [str(bundled_python), '-m', PACKAGE_NAME, *args]
    env = os.environ.copy()
    env[REEXEC_FLAG] = '1'
    # 현재 폴더와 상관없이 다시 실행된 Python이 이 패키지를 찾을 수 있게 함
    python_path = [str(PROJECT_ROOT), env.get('PYTHONPATH', '')]
    env['PYTHONPATH'] = os.pathsep.join(path for path in python_path if path)
    return subprocess.call(command, env=env)


def require_packages():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401
        import scipy.sparse  # noqa: F401
        import scipy.sparse.linalg  # noqa: F401
    except ImportError as error:
        message = f'필수 패키지 없음: {error}. pip install -r requirements.txt 를 실행하세요.'
        raise SystemExit(message) from error
