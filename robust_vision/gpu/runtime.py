"""CuPy 장치 초기화와 Windows용 CUDA DLL 경로 설정, CUDA 소스 불러오기."""
from __future__ import annotations

import os
from pathlib import Path

from ..bootstrap import ENV_ROOT, PACKAGE_DIR

CUDA_DLL_PACKAGES = ('cuda_runtime', 'cuda_nvrtc')
KERNEL_DIR = Path(__file__).resolve().parent / 'cuda'


def load_kernel_source(name):
    return (KERNEL_DIR / name).read_text(encoding='utf-8')


class CudaRuntime:
    def __init__(self):
        os.environ.setdefault('CUPY_CACHE_DIR', str(ENV_ROOT / '.build' / 'cupy-cache'))
        self.dll_handles = []

        dll_dirs = self._register_cuda_dlls()
        if dll_dirs:
            self._prepend_to_path(dll_dirs)
            os.environ.setdefault('CUDA_PATH', str(PACKAGE_DIR / 'nvidia' / 'cuda_runtime'))

        import cupy as cp
        self.cp = cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA GPU가 없습니다.')
        self.device = cp.cuda.Device(0)
        self.device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()

    def _register_cuda_dlls(self):
        if not hasattr(os, 'add_dll_directory'):
            return []

        dll_dirs = []
        for name in CUDA_DLL_PACKAGES:
            directory = PACKAGE_DIR / 'nvidia' / name / 'bin'
            if not directory.is_dir():
                continue

            self.dll_handles.append(os.add_dll_directory(str(directory)))
            dll_dirs.append(str(directory))
            if name == 'cuda_nvrtc':
                self._preload_nvrtc_builtins(directory)
        return dll_dirs

    def _preload_nvrtc_builtins(self, directory):
        # NVRTC는 내부에서 builtins DLL을 예전 방식의 DLL 탐색으로 불러오므로 미리 로드해 둠
        import ctypes
        for library in directory.glob('nvrtc-builtins*.dll'):
            self.dll_handles.append(ctypes.WinDLL(str(library)))

    @staticmethod
    def _prepend_to_path(directories):
        os.environ['PATH'] = os.pathsep.join([*directories, os.environ.get('PATH', '')])
