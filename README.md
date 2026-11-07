# KS Studio

한국어 교육 크리에이터 **Korean Summer** 채널을 위해 만든 Windows용 영상 제작 앱입니다 (macOS는 [실험적 지원](#설치--macos-apple-silicon-실험적)).
녹화부터 컷편집, 자막, AI 얼굴·몸매 보정, 쇼츠·릴스용 내보내기까지 한 프로그램에서 합니다.
단계마다 따로 쓸 수 있습니다 (휴대폰으로 찍은 영상을 가져와 편집만 해도 됩니다).

- 지금 버전: **0.2.0** — 바뀐 점은 [패치노트](docs/CHANGELOG.md), 개발 과정은 [개발일지](docs/devlog/)
- Python 3.12 · PySide6(Qt) · PyAV(FFmpeg) · ONNX Runtime(GPU)

## 무엇을 하나

| 단계 | 하는 일 |
|---|---|
| **녹화** | 웹캠·마이크 녹화, 박수 한 번으로 소리·화면 싱크 맞추기, 9:16 세로 가이드 |
| **컷편집** | 여러 줄 타임라인, 자르기·분할·디졸브·화면 속 화면·마커, 말 없는 구간 한 번에 자르기, 단축키 설정 |
| **자막** (컷편집 안) | 음성으로 자막 자동 생성 — **한국어 + 영어 두 줄**, 단어 시간으로 자동 줄 나누기, 타임라인에서 끌어 옮기기·나누기·합치기, 찾아 바꾸기, 용어집, 모양(글꼴·색·배경 상자·위치·숏폼 안전 영역), SRT |
| **AI 보정** | 얼굴 보정(피부·얼굴형·메이크업 20가지 남짓, 립 색·광택), 얼굴 작게·몸매 보정 [Beta], 사람이 비켜난 배경 메꾸기, **다른 사람 얼굴 흐리기**(주인공·등록한 사람 제외). 샘플 컷 확인 → 1분 미리보기 → 보정본 |
| **내보내기** | 비율(9:16·4:5·1:1·16:9)·해상도·H.264/H.265, 자막 굽기, 저해상도 미리보기 |

## 필요한 PC

| | 최소 | 권장 (테스트한 사양) |
|---|---|---|
| OS | Windows 10/11 64비트 (맥은 [아래](#설치--macos-apple-silicon-실험적)) | |
| 그래픽 카드 | 없어도 됨 (AI·인코딩이 CPU로 — 많이 느림) | NVIDIA RTX, VRAM 8GB 이상 (RTX 5060 8GB · RTX 5080 16GB) |
| 메모리 | 16GB | 32GB 이상 (4K 영상) |
| 저장 공간 | 프로그램 약 5GB | + AI 모델 약 4GB + 작업 영상 |

NVIDIA 그래픽 카드는 **드라이버만** 최신이면 됩니다. CUDA·cuDNN은 따로 설치하지 않아도 됩니다 (pip로 같이 들어옵니다).

## 설치 — 배포판(exe)을 받았다면

`KS_Studio-<버전>.zip`을 풀고 `KS Studio.exe`를 실행하면 끝입니다. Python이 없어도 됩니다.
AI 모델은 앱 안 **AI 보정 → 위쪽 '모델' 버튼**에서 받습니다. 자세한 것은 폴더 안 `사용법.txt`.

## 설치 — 소스 코드로 (배포판을 받지 못했다면)

아래 순서대로 **명령 프롬프트(cmd)** 에서 입력합니다. 처음 한 번 20~30분쯤 걸립니다 (내려받는 양이 수 GB).

### 1. 준비물 설치

1. **Git** — <https://git-scm.com/download/win> (설치 옵션은 기본값 그대로)
2. **Miniconda** (Python 3.12 환경을 따로 만들기 위해) — <https://docs.anaconda.com/miniconda/>
   - 설치 뒤 시작 메뉴의 **Anaconda Prompt**를 엽니다. 아래 명령은 모두 여기서 입력합니다.
   - Miniconda 대신 [python.org](https://www.python.org/downloads/)의 Python 3.12를 써도 됩니다 (아래 2단계에서 `venv` 사용).
3. **NVIDIA 드라이버** (NVIDIA 그래픽 카드가 있다면) — <https://www.nvidia.com/drivers>

### 2. 코드 받기 · Python 환경 만들기

```bat
git clone https://github.com/ksh0321-dev/Korean_Summer_Studio.git
cd Korean_Summer_Studio
conda create -n ks-studio python=3.12 -y
conda activate ks-studio
```

> Miniconda 없이 python.org Python을 쓴다면 마지막 두 줄 대신:
> `py -3.12 -m venv .venv` 그리고 `.venv\Scripts\activate`

⚠ 이 환경에는 **pip로만** 설치하세요. `conda install -c conda-forge ...`로 섞어 깔면 Qt(PySide6)가 깨질 수 있습니다.

### 3. 기본 설치 (녹화 · 컷편집 · 내보내기)

```bat
pip install -e ".[dev]"
python -m ks_studio
```

창이 뜨면 기본 기능은 준비된 것입니다. AI 보정·음성 자막이 필요 없으면 여기서 끝입니다.

### 4. AI 기능 설치 (AI 보정 · 음성 자막 · 몸매 보정)

```bat
pip install -e ".[ai]"
pip uninstall -y onnxruntime onnxruntime-gpu opencv-contrib-python opencv-python
pip install "onnxruntime-gpu[cuda,cudnn]" opencv-python
```

두 번째·세 번째 줄이 꼭 필요한 이유: AI 패키지들(insightface·faster-whisper·rtmlib)이 **CPU용 onnxruntime**과
**opencv-contrib**를 같이 깔아서 GPU용 onnxruntime과 opencv를 덮어씁니다. 둘 다 지우고 GPU판과 opencv만 다시 깝니다.

세 번째 줄 뒤에 `ERROR: pip's dependency resolver ... requires onnxruntime, which is not installed` 같은 글이 나오지만
**정상입니다** — GPU판(onnxruntime-gpu)이 같은 역할을 하는데 pip가 이름이 달라 알아보지 못할 뿐입니다. 5단계 확인만 통과하면 됩니다.

### 5. 잘 깔렸는지 확인

```bat
python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"
```

목록에 **`CUDAExecutionProvider`** 가 있으면 GPU로 돕니다 (없으면 CPU로 돌아 아주 느립니다 — 아래 '문제 해결').

```bat
python -m ks_studio --selftest
```

`user\logs\selftest.txt`에 글꼴·영상 재생·인코더·얼굴 모델·몸매 모델·배경 메꾸기·음성 인식이 각각 OK인지 적힙니다.

### 6. AI 모델 받기

앱을 켜고 **AI 보정** 탭 위쪽 **'모델'** 줄에서 필요한 것만 누릅니다 (인터넷 필요, `models\` 폴더에 저장).

| 모델 | 크기 | 쓰는 곳 | 출처 |
|---|---|---|---|
| 얼굴 | 약 290MB | AI 보정 전체 · 얼굴 흐리기 | [InsightFace](https://github.com/deepinsight/insightface) buffalo_l |
| 몸매 | 약 70MB | 몸매 보정 · 얼굴 작게 | [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (Apache-2.0) · [MediaPipe selfie multiclass](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) (Apache-2.0) |
| MI-GAN | 28MB | 배경 메꾸기 (기본) | [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (MIT) |
| LaMa | 208MB | 배경 메꾸기 (무늬까지) | [LaMa](https://github.com/advimman/lama) (Apache-2.0) |
| 자막 | 약 3.1GB | 음성으로 자막 만들기 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3 (MIT) |

자막 모델은 미리 받지 않아도 자막을 처음 만들 때 받습니다.

### 다음부터 실행

```bat
conda activate ks-studio
cd Korean_Summer_Studio
python -m ks_studio
```

새 버전으로 바꾸기: `git pull` 한 뒤 `pip install -e ".[dev]"`. 개인 설정·프리셋은 `user\`, 모델은 `models\`에 남아 있습니다.

## 설치 — macOS (Apple Silicon, 실험적)

맥에서도 같은 소스 코드로 돕니다. 배포판(exe)은 Windows 전용이라, 맥에서는 아래 순서대로 소스 코드로 설치합니다.
처음 한 번 20~30분쯤 걸립니다 (내려받는 양이 수 GB).

| | 필요 |
|---|---|
| 맥 | **Apple Silicon (M1 이상)**. Intel 맥은 지원하지 않습니다 (음성 자막을 쓸 수 없습니다) |
| macOS | 14 (Sonoma) 이상 |
| 메모리 | 16GB 이상 (음성 자막 모델이 약 3GB를 씁니다) |
| 저장 공간 | 프로그램 약 3GB + AI 모델 약 4GB + 작업 영상 |

맥에서는 이렇게 돕니다:
- AI 모델(얼굴·몸·배경 메꾸기)은 **Core ML**(Apple GPU·Neural Engine)로, 음성 자막은 **mlx-whisper**(Apple GPU)로
- 영상 인코딩은 **VideoToolbox**(맥 하드웨어 인코더), 안 되면 CPU로

> ⚠ 아직 실제 맥에서 끝까지 확인하지 못했습니다. 안 되는 곳이 있으면 [이슈](https://github.com/ksh0321-dev/Korean_Summer_Studio/issues)로 알려 주세요
> (아래 5단계의 `selftest.txt`를 붙여 주시면 빨리 찾을 수 있습니다). NVIDIA PC보다 AI 보정이 느립니다 (특히 4K).

아래 명령은 모두 **터미널**(응용 프로그램 → 유틸리티 → 터미널)에 한 줄씩 입력합니다.

### 1. 내 맥이 Apple Silicon인지 확인

화면 왼쪽 위 **🍎 메뉴 → 이 Mac에 관하여**에서 **칩**이 `Apple M1`·`M2`·`M3`·`M4`… 이면 됩니다.
`Intel`이라고 나오면 이 프로그램을 쓸 수 없습니다.

### 2. 준비물 설치

**① 개발 도구 (Xcode 명령줄 도구)** — 얼굴 인식 패키지(insightface)를 설치할 때 필요합니다.

```bash
xcode-select --install
```

창이 뜨면 **설치**를 누르고 끝날 때까지 기다립니다 (몇 분). `already installed`라고 나오면 이미 있는 것입니다.

**② Miniconda (Python 3.12 환경을 따로 만들기 위해)**

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

- 사용 조건이 나오면 스페이스로 넘기고 `yes`, 설치 위치는 Enter(기본값), 마지막 질문(터미널을 열 때 conda를 켤지)에도 `yes`.
- 설치가 끝나면 **터미널을 닫고 새로 엽니다.** 줄 맨 앞에 `(base)`가 보이면 된 것입니다.

### 3. 코드 받기 · Python 환경 만들기

```bash
git clone https://github.com/ksh0321-dev/Korean_Summer_Studio.git
cd Korean_Summer_Studio
conda create -n ks-studio python=3.12 -y
conda activate ks-studio
```

- `git`을 처음 쓰면 개발 도구 설치 창이 다시 뜰 수 있습니다 — 설치 후 같은 명령을 다시 입력합니다.
- 줄 맨 앞이 `(ks-studio)`로 바뀌었는지 확인합니다.

⚠ 이 환경에는 **pip로만** 설치하세요. `conda install -c conda-forge ...`로 섞어 깔면 Qt(PySide6)가 깨질 수 있습니다.

### 4. 설치하고 켜 보기

**① 기본 (녹화 · 컷편집 · 내보내기)**

```bash
pip install -e ".[dev]"
python -m ks_studio
```

창이 뜨면 기본 기능은 준비된 것입니다. 창을 닫고 계속합니다.

**② AI 기능 (AI 보정 · 음성 자막 · 몸매 보정)**

```bash
pip install -e ".[ai]"
pip uninstall -y opencv-contrib-python opencv-python
pip install opencv-python
```

- 첫 줄은 맥용으로 **onnxruntime 기본판(Core ML 포함)** 과 **mlx-whisper**를 깝니다 (NVIDIA 패키지는 빠집니다).
- 둘째·셋째 줄이 필요한 이유: 몸매 패키지(rtmlib)가 opencv-contrib를 같이 깔아 opencv와 겹칩니다. 둘 다 지우고 opencv만 다시 깝니다.
- 중간에 `dependency resolver ... requires ...` 같은 글이 나와도 5단계 확인만 통과하면 괜찮습니다.

### 5. 잘 깔렸는지 확인

```bash
python -c "import platform; print(platform.machine())"
```

`arm64`가 나와야 합니다. `x86_64`가 나오면 Intel용 Python이 깔린 것이니 2단계 ②를 `arm64` 파일로 다시 합니다.

```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```

목록에 **`CoreMLExecutionProvider`** 가 있으면 AI 모델이 Apple GPU로 돕니다.

```bash
python -m ks_studio --selftest
cat user/logs/selftest.txt
```

글꼴 · 영상 재생 · 영상 인코더 · 얼굴 모델 · 몸매 모델 · 배경 메꾸기 · 음성 인식이 줄마다 `OK` 또는 `FAIL`로 나옵니다.
모델을 아직 안 받았으면 '모델 없음'으로 나오는 게 정상입니다 (6단계에서 받습니다).
`음성 인식: mlx-whisper (Apple Silicon GPU)`, `영상 인코더: hevc_videotoolbox`가 보이면 맥 가속까지 된 것입니다.

### 6. AI 모델 받기

`python -m ks_studio`로 켜고 **AI 보정** 탭 위쪽 **'모델'** 줄에서 필요한 것만 누릅니다 (인터넷 필요, `models/` 폴더에 저장).
모델 종류는 Windows와 같습니다 (위 6단계 표). 다만 **자막 모델은 맥용(mlx 형식, 약 3GB)** 으로 받습니다.

### 7. 녹화 권한 (카메라 · 마이크)

처음 녹화 화면을 열면 "터미널이 카메라에 접근하려고 합니다" 창이 뜹니다 → **허용**.
놓쳤거나 거부했다면 **시스템 설정 → 개인정보 보호 및 보안 → 카메라**(그리고 **마이크**)에서 **터미널**을 켜고, 앱을 다시 켭니다.
(VS Code 터미널에서 실행했다면 터미널 대신 VS Code를 켭니다.)

### 다음부터 실행

```bash
conda activate ks-studio
cd Korean_Summer_Studio
python -m ks_studio
```

새 버전으로 바꾸기: `git pull` 한 뒤 `pip install -e ".[dev]"`. 개인 설정·프리셋은 `user/`, 모델은 `models/`에 남아 있습니다.

### 맥에서 다른 점

- 단축키의 **Ctrl은 ⌘(Command)** 입니다 (예: 되돌리기 ⌘Z).
- '파일 위치 열기'는 Finder로 열립니다.
- 배포용 exe 만들기는 Windows 전용입니다.

### 맥 문제 해결

| 증상 | 해결 |
|---|---|
| 4단계 ②에서 insightface 설치가 `error: command 'clang' failed` 등으로 실패 | 2단계 ①(`xcode-select --install`)을 하고 다시 |
| `CoreMLExecutionProvider`가 없다 | `pip uninstall -y onnxruntime onnxruntime-gpu` 후 `pip install onnxruntime` |
| `python -m ks_studio`가 `zsh: command not found: python` | `conda activate ks-studio`를 먼저 (줄 맨 앞이 `(ks-studio)`인지) |
| 음성 자막이 오류로 멈춘다 | mlx 대신 CPU로: `pip uninstall -y mlx-whisper` 후 `pip install faster-whisper` (훨씬 느립니다) |
| 영상 인코더가 `libx264`로 나온다 | VideoToolbox를 못 쓰는 상태입니다. 그대로 써도 되고 (CPU라 내보내기가 느릴 뿐) 이슈로 알려 주세요 |
| 카메라 화면이 검다 | 7단계 권한 확인. 다른 앱(FaceTime·Zoom 등)이 카메라를 쓰고 있으면 닫기 |
| 그 밖의 문제 | `user/logs/selftest.txt`와 터미널에 나온 오류 글을 이슈에 붙여 주세요 |

## 문제 해결

| 증상 | 해결 |
|---|---|
| 5단계 확인에 `CUDAExecutionProvider`가 없다 | 4단계의 두 번째·세 번째 줄을 다시 실행. 그래도 없으면 NVIDIA 드라이버를 최신으로 |
| `import cv2`가 안 된다 | `pip install --force-reinstall --no-deps opencv-python` |
| 앱이 Qt·DLL 오류로 안 켜진다 | conda-forge 패키지가 섞였을 가능성 — 환경을 지우고(`conda env remove -n ks-studio`) 2단계부터 다시 |
| 음성 자막이 아주 느리다 | `pip install nvidia-cublas-cu12` (음성 인식은 CUDA 12의 cuBLAS를 씁니다) |
| 모델 받기가 실패한다 | 인터넷 연결·방화벽 확인 후 다시 누르기 (받다 끊긴 파일은 남지 않습니다) |
| 그 밖의 문제 | `user\logs\ks_studio.log` 와 `user\logs\selftest.txt`를 이슈에 첨부해 주세요 |

## 배포용 exe 직접 만들기

```bat
pip install -e ".[build]"
python tools\build_exe.py --zip
```

`dist\KS_Studio-<버전>\` 폴더와 zip이 생깁니다. 폴더째 복사하면 Python·CUDA가 없는 PC에서도 `KS Studio.exe`로 실행됩니다
(CUDA·cuDNN DLL이 들어 있습니다). 인터넷이 없는 PC용으로 모델까지 넣으려면 `--with-models` (+약 4GB).

- 버전은 `src/ks_studio/__init__.py`의 `__version__` 하나 (폴더 이름·창 제목이 따릅니다)
- 배포한 PC에서 점검: `"KS Studio.exe" --selftest` → `user\logs\selftest.txt`

## 폴더

```
src/ks_studio/   코드 (core: 순수 로직 · services: 영상·AI 처리 · ui: 화면 · app: 시작·경로·글꼴)
tests/           pytest — python -m pytest
tools/           build_exe.py (배포용 exe), spikes/ (실험 스크립트)
docs/            CHANGELOG.md (패치노트) · devlog/ (개발일지)
user/            개인 설정·프리셋·용어집·등록한 사람·로그 — 저장소에 올리지 않음 (앱이 처음 켤 때 만든다)
models/          AI 모델 — 앱에서 내려받음, 저장소에 올리지 않음
```

모든 경로가 앱 폴더 기준이라 폴더째 옮겨도 됩니다. 다른 PC로 옮길 때는 `user\`(와 받아 둔 `models\`)만 같이 옮기면 됩니다.
