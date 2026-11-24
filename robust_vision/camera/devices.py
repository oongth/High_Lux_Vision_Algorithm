# 연결된 웹캠 찾기와 선택
from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import cv2

CAMERA_PROBE_COUNT = 8


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    backend: int
    path: str | None = None
    vid: int | None = None
    pid: int | None = None


def _capture_backend():
    if os.name == 'nt':
        return cv2.CAP_DSHOW
    if sys.platform.startswith('linux'):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


# 라이브러리 인덱스에 백엔드 번호가 더해져 있으면 장치 번호만 남김
def _device_index(camera):
    index = camera.index
    if camera.backend and camera.backend <= index < camera.backend + 100:
        index -= camera.backend
    return index


def _enumerate_named_cameras(backend):
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return []

    found = []
    for camera in enumerate_cameras(backend):
        index = _device_index(camera)
        found.append(CameraInfo(
            index=index,
            name=camera.name or f'camera {index}',
            backend=camera.backend or backend,
            path=getattr(camera, 'path', None),
            vid=getattr(camera, 'vid', None),
            pid=getattr(camera, 'pid', None),
        ))
    return found


def _probe_camera_indices(backend):
    found = []
    for index in range(CAMERA_PROBE_COUNT):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                found.append(CameraInfo(index, f'camera {index}', backend))
        finally:
            capture.release()
    return found


# cv2-enumerate-cameras가 있으면 장치 이름까지, 없으면 인덱스를 하나씩 열어 보고 찾음
def list_cameras():
    backend = _capture_backend()
    found = _enumerate_named_cameras(backend)
    if found:
        return found
    return _probe_camera_indices(backend)


# 이름(부분 일치), 인덱스로 선택. 둘 다 없으면 ZED가 아닌 첫 카메라
def select_webcam(cameras, name=None, index=None):
    if not cameras:
        raise RuntimeError('사용 가능한 웹캠이 없습니다.')

    matches = list(cameras)
    if name:
        keyword = name.lower()
        matches = [camera for camera in matches if keyword in camera.name.lower()]
    if index is not None:
        matches = [camera for camera in matches if camera.index == index]
    if not name and index is None:
        non_zed = [camera for camera in matches if 'zed' not in camera.name.lower()]
        matches = non_zed or matches

    if not matches:
        raise RuntimeError(f'조건에 맞는 웹캠이 없습니다: name={name!r}, index={index!r}')
    return matches[0]


def camera_description(camera):
    try:
        backend_name = cv2.videoio_registry.getBackendName(camera.backend)
    except Exception:
        backend_name = str(camera.backend)

    return {
        'index': camera.index,
        'name': camera.name,
        'backend': backend_name,
        'path': camera.path,
        'vid': camera.vid,
        'pid': camera.pid,
    }
