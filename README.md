# Robust Vision Algorithm (High Lux)

고조도(강한 조명) 환경을 위한 실시간 영상 보정 패키지
DUAL 조명 추정으로 하이라이트는 압축하고, 그림자 속 디테일은 살리는 방식을 적용

- Image -> Video -> Real Time Video 순서로 알고리즘을 적용하며 성능을 검증

- **highlight 모드 (기본)**: DUAL 조명 추정 + 휘도 shoulder 압축으로 밝은 영역을 낮춤
- **paper 모드**: 논문 방식의 원본/밝은/어두운 후보 다중 해상도 융합 (비교용)
- CUDA(CuPy)가 있으면 GPU로, 없으면 CPU로 동작

<img width="1276" height="462" alt="image" src="https://github.com/user-attachments/assets/d877c00c-8edf-4977-9f22-8ae8748d1861" />

<img width="1920" height="935" alt="image" src="https://github.com/user-attachments/assets/cf88f5ca-af2d-4cdc-ac82-887a8c0ecb64" />


## 폴더 구조

```
Robust_Vision_Algorithm/
├── pyproject.toml            패키지 설정 (설치·의존성·실행 명령)
├── requirements.txt          필수 패키지 목록
├── README.md
└── robust_vision/
    ├── __init__.py           .packages 경로 활성화, 버전
    ├── __main__.py           실행 진입점 (python -m robust_vision)
    ├── bootstrap.py          실행 환경 준비 (프로젝트 Python 재실행, 필수 패키지 확인)
    ├── processor.py          DualIlluminationProcessor: 알고리즘 전체를 묶은 처리기
    ├── algorithm/            순수 영상처리 연산 (화면·카메라와 무관)
    │   ├── settings.py       ProcessorConfig, ProcessorControls
    │   ├── illumination.py   조명 추정 Eq. (2)-(6)
    │   ├── fusion.py         paper 모드 CPU 융합 Eq. (7)-(8)
    │   ├── highlight.py      highlight 모드 CPU 렌더링
    │   └── metrics.py        보정 전후 변화 통계
    ├── gpu/                  CUDA 가속
    │   ├── runtime.py        CuPy 초기화, Windows CUDA DLL 경로
    │   ├── fusion.py         paper 모드 GPU 융합
    │   ├── highlight.py      highlight 모드 GPU 렌더링
    │   └── cuda/             CUDA 커널 소스 (fusion.cu, highlight.cu)
    ├── camera/
    │   ├── devices.py        웹캠 검색·선택
    │   └── reader.py         최신 프레임 읽기 스레드
    ├── utils/
    │   ├── timing.py         FPS·처리 시간 통계
    │   └── jsonio.py         JSON 저장
    └── app/                  실시간 실행 애플리케이션
        ├── state.py          트랙바 ↔ 처리 스레드 공유 상태
        ├── worker.py         처리 스레드
        ├── display.py        비교 화면, 트랙바, 결과 저장
        ├── runner.py         실행 흐름, metrics.json 기록
        └── cli.py            명령줄 인자
```

## 설치

```bash
pip install -r requirements.txt
# 또는 패키지로 설치 (robust-vision 명령이 생김)
pip install .
pip install .[cuda,camera]   # GPU 가속, 웹캠 이름 표시까지
```

## 실행

프로젝트 폴더(`Robust_Vision_Algorithm/`)에서 실행합니다.

```bash
python -m robust_vision                       # 기본 실행 (CUDA)
python -m robust_vision --backend auto        # GPU가 없으면 CPU로 전환
python -m robust_vision --mode paper          # 논문 방식 비교
python -m robust_vision --list                # 연결된 웹캠 목록
python -m robust_vision --no-show --seconds 30 --save-preview
python -m robust_vision --help                # 전체 옵션
```

- 종료: 영상 창 또는 트랙바 창의 X 버튼, 터미널에서 `Ctrl+C`
- 결과: 실행한 위치의 `results/<날짜_시간_ID>/metrics.json` (경로는 `--output-dir`로 변경)

## 코드에서 사용

```python
import cv2
from robust_vision.processor import DualIlluminationProcessor
from robust_vision.algorithm.settings import ProcessorControls

processor = DualIlluminationProcessor(backend='auto')
image = cv2.imread('input.png')
enhanced, info = processor(image, ProcessorControls(highlights=80, peak=65))
```

## 로컬 실행 환경 (선택)

`tools/setup_cupy.ps1`로 만든 아래 폴더가 프로젝트 폴더에 있으면 자동으로 사용합니다.

- `.packages/<python 버전>/` : 프로젝트 전용 패키지 (import 경로 맨 앞에 추가)
- `.build/python/cpython-3.12.*/` : 프로젝트 전용 Python (있으면 그 Python으로 다시 실행)
- `.build/cupy-cache/` : CUDA 커널 컴파일 캐시
