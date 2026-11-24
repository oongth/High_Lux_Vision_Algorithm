# 별도 스레드에서 웹캠을 계속 읽고 최신 프레임 하나만 보관

from __future__ import annotations

from threading import Condition, Event, Thread
import time

import cv2
import numpy as np

from ..utils.timing import StreamTiming

READ_RETRY_SECONDS = 0.005
NEW_FRAME_WAIT_SECONDS = 0.02


def _to_bgr_uint8(frame):
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.dtype != np.uint8:
        raise RuntimeError(f'지원하지 않는 프레임 형식: {frame.dtype}')
    return np.ascontiguousarray(frame)


class WebcamReader:
    def __init__(self, camera, width, height, fps):
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.condition = Condition()
        self.ready = Event()
        self.stop = Event()
        self.thread = Thread(target=self.run, name='webcam-reader', daemon=True)
        self.packet = None
        self.error = None
        self.actual = None
        self.timing = StreamTiming()

    def run(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.camera.index, self.camera.backend)
            self._configure(capture)
            self._read_loop(capture)
        except Exception as exc:
            self._report_error(exc)
        finally:
            if capture is not None:
                capture.release()

    def _configure(self, capture):
        if not capture.isOpened():
            raise RuntimeError(f'웹캠을 열 수 없습니다: {self.camera.name} (index {self.camera.index})')

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual = {
            'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': float(capture.get(cv2.CAP_PROP_FPS)),
            'backend': capture.getBackendName(),
        }

    def _read_loop(self, capture):
        frame_id = 0
        while not self.stop.is_set():
            ok, frame = capture.read()
            stamp = time.perf_counter()
            if not ok or frame is None:
                self.stop.wait(READ_RETRY_SECONDS)
                continue

            frame = _to_bgr_uint8(frame)
            frame_id += 1
            self._publish(frame_id, stamp, frame)

    def _publish(self, frame_id, stamp, frame):
        with self.condition:
            if frame_id == 1:
                self.actual['frame_shape_hwc'] = list(frame.shape)
            self.packet = (frame_id, stamp, frame)
            self.timing.add(stamp)
            self.condition.notify_all()
        self.ready.set()

    def _report_error(self, exc):
        self.error = exc
        self.ready.set()
        with self.condition:
            self.condition.notify_all()

    def _has_new_state(self, previous):
        if self.error is not None:
            return True
        return self.packet is not None and self.packet[0] != previous

    # 새 프레임을 최대 20 ms 기다린 뒤 (최신 packet, 새 프레임 여부) 반환
    def latest(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self._has_new_state(previous),
                                    timeout=NEW_FRAME_WAIT_SECONDS)
            if self.error is not None:
                raise RuntimeError(f'웹캠 읽기 실패: {self.error}')
            packet = self.packet

        is_new = packet is not None and packet[0] != previous
        return packet, is_new
