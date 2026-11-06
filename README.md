# KS Studio

Korean Summer 채널용 영상 제작 데스크톱 앱 (Windows, Python + PySide6).
촬영 → 컷편집 → AI 보정 → 자막 → 내보내기, 단계마다 독립적으로 쓸 수 있다.

| 단계 | 하는 일 |
|---|---|
| 녹화 | 웹캠·마이크 녹화, 박수로 싱크 보정 |
| 컷편집 | 여러 줄 타임라인, 자르기·디졸브·화면 속 화면·무음 자르기, 휴대폰 영상 편집용 사본 |
| AI 보정 | 얼굴 보정 17가지 (InsightFace), 샘플 컷 확인 → 1분 미리보기 영상 → 보정본, 프리셋 |
| 자막 | 음성 인식으로 영어 자막 자동 생성 (faster-whisper large-v3), 표에서 고치기, SRT |
| 내보내기 | 비율(9:16·4:5·1:1·16:9)·해상도·H.264/H.265, 자막 굽기, 800p 저해상도 미리보기 |

## 폴더

```
src/ks_studio/   코드 (core: 순수 로직 · services: 영상·AI 처리 · ui: 화면 · app: 시작·경로·글꼴)
tests/           pytest
tools/           build_exe.py (배포용 exe), spikes/ (실험 스크립트)
docs/devlog/     개발 일지
user/            개인 설정·프리셋·기본 프로젝트 — 저장소에 올리지 않는다 (앱이 처음 켤 때 만든다)
models/          AI 모델 — 처음 쓸 때 내려받는다 (얼굴 약 300MB, 음성 인식 약 3GB), 저장소에 올리지 않는다
```

모든 경로는 앱 폴더 기준 상대 경로라 폴더째 옮겨도 된다. 새 버전으로 바꿀 때는 `user/`(와 받아 둔 `models/`)만 옮기면 된다.

## 소스로 실행

Python 3.12 (conda 환경 권장, 패키지는 pip로만).

```bat
pip install -e .[dev]
pip install -e .[ai]
:: insightface·faster-whisper가 CPU용 onnxruntime을 같이 깔아 GPU판을 덮어쓴다 → GPU판만 남기기
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]"
python -m ks_studio
```

테스트: `python -m pytest`

## 배포용 exe 만들기

```bat
python tools\build_exe.py
```

`dist\KS_Studio-<버전>\` 폴더가 생긴다. 폴더째 복사하면 Python·CUDA가 없는 PC에서도 `KS Studio.exe`로 바로 실행된다
(NVIDIA 드라이버가 있으면 GPU로, 없으면 CPU로). CUDA·cuDNN DLL은 exe 폴더에 들어 있다.

- 버전: `src/ks_studio/__init__.py`의 `__version__` 하나 (폴더 이름·창 제목이 따른다)
- 인터넷이 없는 PC용: `python tools\build_exe.py --with-models` (모델까지 넣는다, +약 3GB)
- 배포한 PC에서 점검: `"KS Studio.exe" --selftest` → `user\logs\selftest.txt` (글꼴·영상 재생·인코더·얼굴 모델·음성 인식이 GPU로 도는지)
- 문제가 생기면 `user\logs\ks_studio.log`
