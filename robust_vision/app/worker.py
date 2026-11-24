# 처리 스레드: 웹캠 최신 프레임을 받아 보정하고, 최신 결과 하나만 보관

from __future__ import annotations

from collections import deque, namedtuple
from threading import Condition, Event, Thread
import time

from ..camera.reader import NEW_FRAME_WAIT_SECONDS
from ..processor import DualIlluminationProcessor
from ..utils.timing import StreamTiming

ILLUMINATION_KEYS = ('forward_illumination', 'reverse_illumination')
MAX_TIMING_SAMPLES = 10000
MAX_CONTROL_HISTORY = 1000
IDLE_WAIT_SECONDS = 0.003

ResultPacket = namedtuple(
    'ResultPacket',
    'sequence frame_id captured_at original enhanced info controls',
)


# CUDA는 이 스레드 하나에서만 사용
class ProcessingWorker:
    def __init__(self, reader, state, config, *, backend, stall_seconds, max_frames, save_debug):
        self.reader = reader
        self.state = state
        self.config = config
        self.backend_request = backend
        self.stall_seconds = stall_seconds
        self.max_frames = max_frames
        self.save_debug = save_debug

        self.condition = Condition()
        self.ready = Event()
        self.stop = Event()
        self.thread = Thread(target=self.run, name='robust-vision-worker', daemon=True)

        self.packet = None
        self.error = None
        self.backend = None
        self.summary = None

        self.measure_at = float('inf')
        self.timing = StreamTiming()
        self.algorithm_ms = deque(maxlen=MAX_TIMING_SAMPLES)
        self.latency_ms = deque(maxlen=MAX_TIMING_SAMPLES)
        self.solve_ms = deque(maxlen=MAX_TIMING_SAMPLES)
        self.skipped_frames = 0
        self.history = deque(maxlen=MAX_CONTROL_HISTORY)

    def run(self):
        processor = None
        try:
            processor = DualIlluminationProcessor(self.config, self.backend_request)
            self.backend = processor.backend
            self._process_loop(processor)
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.ready.set()
                self.condition.notify_all()
        finally:
            if processor is not None:
                self.summary = processor.summary()

    def _process_loop(self, processor):
        seen = 0
        sequence = 0
        work_key = None
        last_measured = None
        last_revision = None

        while not self.stop.is_set():
            source, _ = self.reader.latest(seen)
            if source is None:
                continue
            self._check_stall(source)
            frame_id, captured_at, image = source
            seen = frame_id

            # 같은 프레임 + 같은 컨트롤이면 다시 계산하지 않음
            controls, revision = self.state.snapshot()
            key = (frame_id, revision)
            if key == work_key:
                self.stop.wait(IDLE_WAIT_SECONDS)
                continue

            result, info = processor(image, controls, frame_key=frame_id, timestamp=captured_at,
                                     return_debug=self.save_debug)
            finished = time.perf_counter()
            info.update(finished_at=finished, capture_to_output_ms=(finished - captured_at) * 1000)
            sequence += 1

            with self.condition:
                if finished >= self.measure_at and frame_id != last_measured:
                    self._record_measurement(finished, info, frame_id, last_measured)
                    last_measured = frame_id

                if revision != last_revision:
                    self.history.append({'frame': frame_id, 'controls': controls.to_dict()})
                    last_revision = revision

                self.packet = ResultPacket(sequence, frame_id, captured_at, image, result, info, controls)
                self.ready.set()
                self.condition.notify_all()

                if self.max_frames and self.timing.count >= self.max_frames:
                    self.stop.set()
            work_key = key

    def _check_stall(self, source):
        if time.perf_counter() - source[1] > self.stall_seconds:
            raise RuntimeError('웹캠에서 새 프레임을 받지 못했습니다.')

    def _record_measurement(self, finished, info, frame_id, last_measured):
        self.timing.add(finished)
        self.algorithm_ms.append(info['processing_ms'])
        self.latency_ms.append(info['capture_to_output_ms'])
        self.solve_ms.append(sum(info[key]['processing_ms'] for key in ILLUMINATION_KEYS))
        if last_measured is not None:
            self.skipped_frames += max(0, frame_id - last_measured - 1)

    def _has_new_state(self, previous):
        if self.error is not None:
            return True
        return self.packet is not None and self.packet.sequence != previous

    def latest(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self._has_new_state(previous),
                                    timeout=NEW_FRAME_WAIT_SECONDS)
            if self.error is not None:
                raise self.error
            return self.packet
