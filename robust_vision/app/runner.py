# ㄴ실시간 실행 흐름: 웹캠 - 처리 스레드 - 화면, 종료 후 metrics.json 기록

from __future__ import annotations

import sys
import time

from ..camera.devices import camera_description, list_cameras, select_webcam
from ..camera.reader import WebcamReader
from ..utils.jsonio import write_json
from ..utils.timing import StreamTiming, sample_summary
from .display import (close_windows, open_display, pump_events, save_frame, show,
                      without_debug_images, windows_visible)
from .state import SessionState
from .worker import ProcessingWorker

READER_READY_TIMEOUT = 20
WORKER_READY_TIMEOUT = 60
READER_JOIN_TIMEOUT = 5
WORKER_JOIN_TIMEOUT = 15
LOG_INTERVAL_SECONDS = 5

REPORT_INTERPRETATION = (
    'Actual live frames after initialization and warmup. '
    'Latest-frame mailboxes bound queue latency. '
    'Capture-to-output excludes sensor/driver buffering and display. '
    'Highlight mode deliberately lowers displayed white luminance; '
    'it does not reconstruct clipped detail or separate physical flare.'
)


def _wait_until_ready(component, timeout, timeout_message):
    if not component.ready.wait(timeout) or component.error:
        raise RuntimeError(str(component.error or timeout_message))


def _reached_limit(args, worker, now):
    if args.seconds and now - worker.measure_at >= args.seconds:
        return True
    return bool(args.max_frames and worker.timing.count >= args.max_frames)


def _stop_threads(reader, worker, reader_started, worker_started):
    if worker is not None:
        worker.stop.set()
    if reader is not None:
        reader.stop.set()
    if worker_started:
        worker.thread.join(timeout=WORKER_JOIN_TIMEOUT)
    if reader_started:
        reader.thread.join(timeout=READER_JOIN_TIMEOUT)


def _collect_thread_failure(failure, reader, worker, reader_started, worker_started):
    if reader is not None and reader.error:
        failure = failure or str(reader.error)
    if worker is not None and worker.error:
        failure = failure or str(worker.error)

    still_running = ((worker_started and worker.thread.is_alive())
                     or (reader_started and reader.thread.is_alive()))
    if still_running:
        failure = failure or '스레드 종료 시간 초과'
    return failure


def _print_run_result(report, failure, directory):
    if failure:
        print(f'Robust Vision 실행 실패: {failure}', file=sys.stderr, flush=True)
        return

    fps = report['processing']['fps']
    fps_text = f'{fps:.2f}' if fps is not None else 'n/a'
    mean_ms = report['algorithm_ms']['mean']
    print(f"완료: Robust Vision {fps_text} FPS | 평균 {mean_ms:.2f} ms | {directory / 'metrics.json'}", flush=True)


def run(args, initial, config, directory):
    state = SessionState(initial)
    reader = None
    worker = None
    panel = None
    packet = None
    reader_started = False
    worker_started = False
    saved = []
    failure = None
    previous = 0
    displayed = StreamTiming()
    previous_display = None

    try:
        camera = select_webcam(list_cameras(), args.camera_name, args.index)
        print(f'노트북 웹캠: {camera.name} / index {camera.index} | 전체 프레임', flush=True)

        reader = WebcamReader(camera, args.width, args.height, args.fps)
        reader.thread.start()
        reader_started = True
        _wait_until_ready(reader, READER_READY_TIMEOUT, '웹캠 첫 프레임 대기 시간 초과')

        worker = ProcessingWorker(reader, state, config, backend=args.backend,
                                  stall_seconds=args.stall_seconds, max_frames=args.max_frames,
                                  save_debug=args.save_debug)
        print('Robust Vision 준비 중: 조명 최적화와 고조도 보정 연산을 초기화합니다.', flush=True)
        worker.thread.start()
        worker_started = True
        _wait_until_ready(worker, WORKER_READY_TIMEOUT, '첫 보정 프레임 대기 시간 초과')
        print(f'연산: {worker.backend} | 실제 입력: {reader.actual}', flush=True)

        if not args.no_show:
            panel = open_display(state)

        worker.measure_at = time.perf_counter() + args.warmup_seconds
        last_log = 0.
        while True:
            current = worker.latest(previous)

            if current is not None and current.sequence != previous:
                packet = current
                previous = current.sequence
                measured = packet.info['finished_at'] >= worker.measure_at

                if panel is not None:
                    show(packet)
                    if measured and packet.frame_id != previous_display:
                        displayed.add(time.perf_counter())
                        previous_display = packet.frame_id

                if args.save_preview and not saved and measured:
                    saved.append(save_frame(directory, packet))

            if panel is not None:
                pump_events()
                if panel.error:
                    raise panel.error
                if not windows_visible():
                    break

            now = time.perf_counter()
            if _reached_limit(args, worker, now):
                break
            if now - last_log >= LOG_INTERVAL_SECONDS:
                print(f'획득 {reader.timing.recent_fps():.1f} / '
                      f'보정 {worker.timing.recent_fps():.1f} FPS', flush=True)
                last_log = now

    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or '사용자 중단'
    finally:
        _stop_threads(reader, worker, reader_started, worker_started)
        failure = _collect_thread_failure(failure, reader, worker, reader_started, worker_started)
        if not args.no_show:
            close_windows()

    if worker_started and not worker.timing.count and failure is None:
        failure = '준비 시간 이후 측정 프레임이 없습니다.'

    has_reader = reader is not None
    has_worker = worker is not None
    report = {
        'success': failure is None,
        'error': failure,
        'mode': 'live_webcam_robust_vision',
        'source': camera_description(reader.camera) if has_reader else None,
        'actual': reader.actual if has_reader else None,
        'config': vars(config),
        'initial_controls': initial.to_dict(),
        'final_controls': state.snapshot()[0].to_dict(),
        'capture': reader.timing.summary() if has_reader else None,
        'processing': worker.timing.summary() if has_worker else None,
        'display': displayed.summary() if panel is not None else None,
        'algorithm_ms': sample_summary(worker.algorithm_ms) if has_worker else None,
        'capture_to_output_ms': sample_summary(worker.latency_ms) if has_worker else None,
        'both_illumination_solves_ms': sample_summary(worker.solve_ms) if has_worker else None,
        'skipped_source_frames': worker.skipped_frames if has_worker else None,
        'processor': worker.summary if has_worker else None,
        'last_diagnostics': without_debug_images(packet.info) if packet is not None else None,
        'saved': saved,
        'controls_history': list(worker.history) if has_worker else [],
        'interpretation': REPORT_INTERPRETATION,
    }
    write_json(directory / 'metrics.json', report)
    _print_run_result(report, failure, directory)
    return 1 if failure else 0
