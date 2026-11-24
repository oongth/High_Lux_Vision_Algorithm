# 화면 표시: 원본/보정 비교 창, 트랙바 창, 결과 이미지 저장

from __future__ import annotations

from uuid import uuid4

import cv2
import numpy as np

from ..algorithm.settings import MIN_PEAK, TUNABLE_FIELDS
from ..utils.jsonio import write_json

WINDOW_TITLE = 'Robust Vision | Original / Enhanced'
PANEL_MAX_WIDTH = 1440
PANEL_MAX_HEIGHT = 650
PANEL_HEADER = 30
TEXT_COLOR = (240, 240, 240)


def without_debug_images(info):
    return {key: value for key, value in info.items() if key != 'debug_images'}


class ControlPanel:
    TITLE = 'Robust Vision | Controls'
    PEAK_LABEL = 'Light peak (%)'
    LABELS = {
        'strength': 'Strength',
        'highlights': 'Highlights',
        'shadows': 'Shadows',
        'detail': 'Detail',
        'peak': PEAK_LABEL,
    }
    BANNER_LINES = (
        ('Lower Light peak = darker lamps | Default 70% = white 255 -> 179', 23),
        ('Display compression; clipped textures and physical flare are not restored.', 47),
    )

    def __init__(self, state):
        self.state = state
        self.initializing = True
        self.error = None

        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        cv2.imshow(self.TITLE, self._make_banner())
        try:
            self._create_trackbars()
            cv2.resizeWindow(self.TITLE, 650, 330)
        finally:
            self.initializing = False

    def _make_banner(self):
        banner = np.full((75, 650, 3), 25, np.uint8)
        for text, y in self.BANNER_LINES:
            cv2.putText(banner, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .43, TEXT_COLOR, 1)
        return banner

    def _create_trackbars(self):
        for field in TUNABLE_FIELDS:
            initial_value = getattr(self.state.snapshot()[0], field)
            cv2.createTrackbar(self.LABELS[field], self.TITLE, initial_value, 100,
                               lambda value, key=field: self.changed(key, value))
        if hasattr(cv2, 'setTrackbarMin'):
            cv2.setTrackbarMin(self.PEAK_LABEL, self.TITLE, MIN_PEAK)

    def changed(self, field, value):
        if self.initializing:
            return
        try:
            if field == 'peak' and value < MIN_PEAK:
                self.initializing = True
                cv2.setTrackbarPos(self.PEAK_LABEL, self.TITLE, MIN_PEAK)
                self.initializing = False
                value = MIN_PEAK
            self.state.change(**{field: value})
        except Exception as exc:
            self.error = exc


def view_label(controls):
    if not controls.enabled or controls.strength == 0:
        return 'BYPASS'
    if controls.view == 'reverse':
        return 'REVERSE CANDIDATE'
    if controls.view == 'difference':
        return 'ABS DIFFERENCE x4'
    if controls.mode == 'highlight':
        return 'HIGHLIGHT COMPRESSION'
    return 'PAPER REFERENCE'


def _side_by_side(images, size):
    resized = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]
    panel = np.hstack(resized)
    return cv2.copyMakeBorder(panel, PANEL_HEADER, 0, 0, 0, cv2.BORDER_CONSTANT)


def comparison(packet):
    h, w = packet.original.shape[:2]
    scale = min(1., PANEL_MAX_WIDTH / (2 * w), PANEL_MAX_HEIGHT / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    panel = _side_by_side((packet.original, packet.enhanced), size)

    labels = [
        ('ORIGINAL', (8, 21)),
        (view_label(packet.controls), (size[0] + 8, 21)),
    ]
    for text, position in labels:
        cv2.putText(panel, text, position, cv2.FONT_HERSHEY_SIMPLEX, .42, TEXT_COLOR, 1)
    return panel


def open_display(state):
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    return ControlPanel(state)


def show(packet):
    cv2.imshow(WINDOW_TITLE, comparison(packet))


# 창 갱신, 트랙바 이벤트 처리 (키 입력은 사용하지 않음)
def pump_events():
    cv2.waitKey(1)


def windows_visible():
    try:
        return all(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1
                   for title in (WINDOW_TITLE, ControlPanel.TITLE))
    except cv2.error:
        return False


def close_windows():
    for title in (WINDOW_TITLE, ControlPanel.TITLE):
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass


def save_frame(directory, packet):
    name = f'frame_{packet.frame_id:08d}_{uuid4().hex[:6]}'
    images = {
        'original': packet.original,
        'enhanced': packet.enhanced,
        'comparison': comparison(packet),
    }
    for label, value in packet.info.get('debug_images', {}).items():
        images[label] = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)

    for label, image in images.items():
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('PNG encoding failed.')
        encoded.tofile(str(directory / f'{name}_{label}.png'))  # 한글 경로 대응

    write_json(directory / f'{name}.json', {
        'frame': packet.frame_id,
        'controls': packet.controls.to_dict(),
        'diagnostics': without_debug_images(packet.info),
    })
    print(f'원본/보정 저장: {directory / name}', flush=True)
    return name
