# 화면 스레드(트랙바)와 처리 스레드가 공유하는 컨트롤 상태
from __future__ import annotations

from dataclasses import replace
from threading import Lock


# 컨트롤이 바뀔 때마다 revision이 1씩 증가 -> 처리 스레드가 같은 프레임도 다시 계산
class SessionState:
    def __init__(self, controls):
        self._lock = Lock()
        self.controls = controls.validate()
        self.revision = 0

    def snapshot(self):
        with self._lock:
            return self.controls, self.revision

    def change(self, **changes):
        with self._lock:
            self.controls = replace(self.controls, **changes).validate()
            self.revision += 1
