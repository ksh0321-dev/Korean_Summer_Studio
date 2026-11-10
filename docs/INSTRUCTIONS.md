# 작업 안내 (새 대화에서 먼저 읽기)

KS Studio를 이어서 작업할 때 알아야 할 것: 지금 상태, 정해 둔 방식, 코드 구조와 고칠 때 주의할 점, 실측으로 알게 된 것.
할 일 목록은 [TODO.md](TODO.md), 버전별 바뀐 점은 [CHANGELOG.md](CHANGELOG.md), 날마다의 기록은 [devlog/](devlog/)에 있다.

마지막 정리: 2026-11-10

---

## 1. 지금 상태

- **프로그램:** KS Studio — Windows용 Python 3.12 + PySide6 녹화·편집 프로그램 (맥은 실험적). 공개 저장소 `ksh0321-dev/Korean_Summer_Studio`.
- **단계(탭):** 녹화 → 컷편집 → 색 보정 → AI 보정 → 내보내기
- **버전:** `__version__ = "0.3.1"` (배포·푸시함). 그 뒤 커밋은 아직 버전이 없다 — CHANGELOG의 '다음 버전 (작업 중)'에 쌓여 있다. 판올림한다면 **0.4.0** 추천 (하이라이트 분할 · 내보낼 때 AI 보정 · 팔자주름).
- **0.3.1 뒤에 들어간 것 (2026-11-09~10):**
  - 컷편집 도구모음 두 줄
  - **✨ 하이라이트 분할** 1~5단계 (점수 매기기 → 하이라이트 앞뒤에서 V1 자르기 → 맨 아래 점수 줄 · 컷 뱃지 · 초록 띠 → ▾ 메뉴: 하이라이트만 남기기 · 점수 낮은 컷 지우기 · 모두 선택 · 분할 설정)
  - **AI 보정을 내보낼 때 입히기** (보정본 파일을 만들어 다시 자르지 않는다)
  - **콧볼 축소 100% = 콧볼 폭 약 −10%** (전 −6%) · **팔자주름 지우기** 새 칸
  - `docs/TODO.md` (할 일 목록), 이 문서
- **테스트:** 385개 (`python -m pytest -q`, 아래 4장). 커밋 전에 늘 전체를 돌린다.

## 2. 사용자와 일하는 방식 (정해 둔 것)

- **말투:** 한국어 반말, **결과 먼저**. 계획을 물으면 계획만(구현하지 않는다). 시작 전에 걸릴 시간을 알린다.
- **커밋·푸시는 요청할 때만.** 커밋과 푸시는 따로 실행한다.
- **CLAUDE.md는 일부러 지운 상태 — 절대 커밋하지 않는다.** 스테이지는 늘 `git add -A -- . ':!CLAUDE.md'`.
- **저장소에 올리지 않는 것:** `models/` · `user/` · 영상 파일 · `backup(s)/` · `dist/` · `build/` · `sample_*` 폴더 · `.claude/`. 올리기 전에 스테이지된 내용에 개인 정보(내 PC 경로 · 메일 · 사람 이름 · 촬영 위치)가 없는지 본다.
- **패키지:** pip만 (conda-forge 섞지 않는다).
- **하지 말 것:** 보안 설정 약하게 하기 · 시스템 설정 바꾸기 · 생성형(무거운) 영상 모델 넣기 (사용자가 "프로그램이 무거워진다"며 거절) · InsightFace 라이선스 문제를 다시 꺼내기 (사용자가 이미 검토함).
- **기록:**
  - 기능을 넣은 날은 `docs/devlog/YYYY-MM-DD.md`에 블로그처럼 (무엇을 · 왜 · 실측 · 확인).
  - `docs/CHANGELOG.md`의 '다음 버전 (작업 중)'을 그때그때 고친다.
  - **판올림:** 개발일지와 git을 보고 그 칸에 버전·날짜를 붙인다. `src/ks_studio/__init__.py`의 `__version__`과 README의 버전 줄을 고친다.
  - 미룬 일은 그 자리에서 `docs/TODO.md`에 더한다. "todo 진행해줘"라고 하면 TODO부터 읽는다.
- **목표 PC:** i5-14600K · 64GB · RTX 5060 8GB. 개발 PC(RTX 5080)에서 잰 속도는 낙관적인 값이다.
- 색 보정의 **자동 맞춤은 [Beta]** 표시를 유지한다.

## 3. 코드 구조

```
src/ks_studio/
  core/      순수 계산 — Qt·파일 접근 없이 (테스트하기 쉽게). Project/편집 연산은 새 객체를 돌려준다 (원본 불변)
  services/  영상·소리·모델·캐시를 다루는 일 (일부 QObject 작업자)
  ui/        화면 (PySide6)
  app/       경로·글꼴·시작
tests/       core · services · ui · app (tests 아래에 __init__.py 없음 → 테스트 파일 이름이 겹치면 안 된다)
```

자주 만지는 곳:

| 무엇 | 파일 |
|---|---|
| 프로젝트 · 클립 · 글자 | `core/project.py`, 편집 연산 `core/edits.py` (V1은 자석식, `follow_main` = 연동) |
| 타임라인 그리기(재생·내보내기 공통) | `services/render.py` `TimelineRenderer` |
| 내보내기 | `services/deliver.py` `export_timeline` (임시 내보내기 `services/export.py`도 있다) |
| 색 보정 | `core/grade.py` (보정값 → 33³ LUT), `services/grade.py` (.cube 캐시, FFmpeg lut3d) |
| YUV ↔ RGB | `services/yuv.py` — **모든 변환은 여기 함수로** (아래 5장) |
| AI 보정 | `core/beauty.py` (설정·얼굴 점 계산), `services/beautify.py` (화소), `services/reshape.py`·`body.py` (몸) |
| 내보낼 때 AI 보정 | `services/timeline_beauty.py` |
| 하이라이트 | `core/highlight.py` (점수), `core/highlight_cuts.py` (하이라이트 찾기·자르기·▾ 메뉴 계산), `services/highlight.py` (재료 뽑기·캐시·작업자) |
| 컷편집 화면 | `ui/editor/editor_page.py`, `ui/editor/timeline.py` |
| 프로젝트 정리(캐시 지우기) | `services/cleanup.py` |

## 4. 개발 환경 · 테스트

- 프로젝트 conda 환경(`KS_Recoder`)의 python으로 실행한다. 경로는 PC마다 다르다.
- `PYTHONPATH=src`
- 전체 테스트: `python -m pytest -q`
- 화면이 있는 테스트·확인 스크립트: `QT_QPA_PLATFORM=offscreen`
- **오프스크린은 한글 글꼴이 네모로 나온다.** 배치·색은 확인되지만 글자 모양은 실제 창에서 봐야 한다.
- **소스 파일 대부분이 CRLF다.** 큰 수정은 Write 도구로 만든 파이썬 패치 스크립트로 한다: 바이트로 읽기 → `\r\n`을 `\n`으로 → 바꾸기 → 원래 줄바꿈으로 되돌리기.
  - Bash 히어독 안의 `\\n`은 진짜 줄바꿈으로 바뀌어 문자열이 깨진다. 히어독으로 파이썬 코드를 넣을 때 `\n`이 든 문자열은 피한다.
  - Git Bash의 `sed`로 `\r$`를 다루는 것은 믿을 수 없다.
- 실측·디버그 스크립트는 세션 scratchpad 폴더에 둔다 (저장소에 넣지 않는다).

## 5. 고칠 때(리팩토링) 주의할 점

### 영상·색
- **YUV ↔ RGB는 `services/yuv`의 `convert` / `to_rgb` / `from_rgb`만 쓴다.**
  - PyAV 기본 `reformat`·`to_ndarray(format="rgb24")`는 버림 반올림이라 화면이 1단계 어두워진다.
  - 색 규격 태그도 어긋난다 (+1.4 / −1.1).
  - 규칙: 프레임 태그를 따른다 (1 → bt709, 5/6 → bt601, 9/10 → bt2020). 태그가 없으면 높이 720 이상은 bt709, 미만은 bt601. yuv → yuv는 행렬을 바꾸지 않는다 (`KEEP_MATRIX`).
- **화질을 잴 때는 원시 Y 판으로.** `to_ndarray(format="gray")`는 태그에 따라 달라져서 틀린 숫자가 나온다.
- **색 보정**
  - 보정값을 33³ LUT로 굽고, 영상에는 FFmpeg lut3d로, 미리보기에는 numpy로 입힌다.
  - 노출·화이트밸런스는 선형광(감마 2.2)에서 계산한다.
  - 설정은 `color/settings.json`에 따로 둔다 (컷편집이 project.json을 메모리에 들고 있다가 덮어쓰기 때문).
  - AI 보정본(`ai/<이름>_보정.mp4`)은 원본의 색 보정을 따른다 (`ColorState.grade_for`).
- **`TimelineRenderer._prepare`의 순서:** (내보낼 때) AI 보정 → 원본 색 보정 LUT → 클립 LUT → 자르기·회전·뒤집기 → 클립 색 → 화면 합치기 → (내보내기에서) 글자·자막을 맨 마지막에 굽기 (`OverlayBurner`).
  - 글자를 보정보다 먼저 구우면 몸매 보정이 글자를 휘게 한다.
  - 보정을 합친 화면에 입히면 위 줄 클립·디졸브의 얼굴까지 바뀐다.
- **알려진 성능 문제:** 색 보정 LUT를 **4K 원본 크기 그대로** 입힌 뒤 줄인다 (4K 원본 54.8초 내보내기가 358초). AI 보정 경로만 먼저 줄인다. 고칠 때는 위 순서를 지킨다 (TODO 4).

### 내보낼 때 AI 보정 (`services/timeline_beauty.py`)
- 입히는 조건: AI 보정 탭에서 샘플 컷을 보고 **OK한(`approved`)** 설정 + 효과가 있거나 다른 얼굴 흐리기. 원본(`Clip.source`)마다 하나다 — **클립마다 다른 보정은 TODO**.
- 원본 프레임 번호 = `round(decoded.time * clip.fps)`. 얼굴 점은 원본 좌표라 줄인 크기 비율로 곱한다. 보정 크기는 `work_size` (출력을 채울 만큼, 원본보다 크게는 안 함).
- 얼굴 분석은 타임라인이 쓰는 구간 ± 3프레임만 한다. `cache/<키>.faces_<사람>.spans.npz`에 쌓는다. AI 탭에서 영상 전체를 분석해 둔 `.faces_<사람>.npy`가 있으면 그것을 쓴다.
- `BodyFrames`(몸 분리, 앞 프레임 배경 기억)는 **클립마다** 따로 둔다.
- 컷편집 미리보기(재생)에는 입히지 않는다 (사용자 결정 — 보정한 모습은 AI 보정 탭에서 본다).

### 캐시 · 정리
- 캐시 이름은 `services/sources.cache_name(path)` (원본 이름 + 경로·크기·시각 해시).
- **새 캐시 파일을 만들면 `services/cleanup.py`에 등록한다.** `cache/` 바로 아래 파일은 `CURRENT_SUFFIXES`, 얼굴 분석류(`.faces_` 포함)는 `project_files`의 keep 목록. 등록하지 않으면 프로젝트를 열 때 sweep이 지운다.
- 쓰는 중인 파일은 `.part` 이름으로 쓰고 `os.replace`로 바꾼다.

### 설정 · 저장 파일
- `AppSettings.load`는 모르는 칸을 버린다 (새 칸은 기본값과 함께 더하면 된다).
- `BeautySettings.from_dict`·`CaptionStyle.from_dict`는 없는 칸을 기본값으로 둔다. 옛 파일과 호환되므로 칸을 지우거나 이름을 바꾸지 말 것.
- 컷편집이 설정을 바꾸면 `settings_edited` 신호 → `main_window._apply_settings`가 저장한다.
- AI 보정 효과를 하나 더할 때 고칠 곳:
  - `core/beauty.py`: `BeautySettings` 칸과 `EFFECTS`
  - `services/beautify.py`: `_apply_face` (색 → 변형 순서), `REGION_COLORS`와 `_areas` (적용 범위 보기)
  - `ui/beautify/beautify_page.py`: `EFFECT_TABS`, 필요하면 `EFFECT_TIPS`
  - 테스트

### 컷편집 화면
- 도구모음 버튼은 `_action(text, tip, slot, keys)` 하나로 버튼과 단축키를 같이 만든다. **단축키 이름(버튼 글자에서 기호를 뺀 것)은 겹치면 안 된다.** 긴 도움말은 `_long_tips`에 넣는다 (`_set_keys`가 툴팁을 다시 쓰기 때문).
- 타임라인 맨 아래 점수 줄(`SCORE_H`)은 줄 영역 밖에 고정이다. 줄 영역의 아래 끝은 `_tracks_bottom()` — 높이 계산에 `self.height()`를 바로 쓰지 말 것.

### 하이라이트
- 점수는 **원본 시각**에 붙는다 (`HighlightBook`, 열쇠는 `Clip.source`). 컷을 옮기거나 다듬어도 그 장면을 따라간다.
- 표시 점수는 영상마다 상대값이다 (평소 20백분위 = 0, 99백분위 = 100. 폭이 좁은 영상은 덜 키움).
- 분할 기준은 그 영상 전체 점수에서 정한다. 같은 영상의 조용한 클립에서 하이라이트를 지어내지 않게 하려는 것이다.
- 분석은 영상 디코딩이 대부분이다 (20분 4K30 ≈ 6.5분, 개발 PC). GPU 디코딩(NVDEC)은 오히려 느렸다.

## 6. 실측으로 알게 된 것 (다시 재지 않아도 되는 것)

| 항목 | 값 |
|---|---|
| 내보낼 때 AI 보정 vs 보정본 파일 (4K60 81초 → 54.8초 쇼츠) | 약 14분 30초 vs 약 41분 40초. 결과 PSNR 41.9dB (거의 같음) |
| AI 보정 1080p 프레임당 | 몸 변형·배경 메꾸기 161ms + 몸 분리 57ms + 얼굴 14ms → **몸매 보정이 88%** (TODO 3) |
| 콧볼 축소 100% | 콧볼 폭 −10.6% (목표 쇼츠 −10.7%) |
| 팔자주름 슬라이더 25/50/75/100% | 그늘 21/46/68/93% 지움 (가짜 주름) |
| CapCut 값 → KS 값 | 그대로 옮기면 안 맞는다. 예: 작은 얼굴 0.13 → 목표는 얼굴 크기 그대로, 색온도 −0.154 → KS −0.3, 베이비 핑크 메이크업 ≈ 블러셔 0.6~0.8 |
| 영상 편집 화질 (기본 / 최고) | PSNR 46.3~46.6 / 48.7~49.6 dB, 밝기 차 ±0.1 이내 |

## 7. 샘플 쇼츠 테스트 (저장소 밖 `sample_editing_example/`)

CapCut으로 만든 54.8초 세로 쇼츠(`target_video.mp4`)를 원본(`original_video.MOV`, 4K60)에서 KS Studio 기능만으로 다시 만드는 테스트다. **개인 자료라 저장소에 없다** (.gitignore). 이 폴더가 있는 PC에서만 돌릴 수 있다.

- `ks_edit_steps.py`: 앱이 쓰는 함수를 단계별로 부르는 재현 스크립트. 단계마다 "앱에서는:"에 화면에서 하는 법이 적혀 있다.
  - 실행: `python sample_editing_example/ks_edit_steps.py all`
  - 일부만: `texts preview` (글자만 고치고 목표와 나란히) · `beauty export compare` · `face_compare` · 예전 방식 `beauty_file old beauty_compare`
- `편집_단계_스크립트.md`: 단계별 설명서 (목표 / 앱에서는 / 결과 / ❗수정점), 실행 기록, 수정점 우선순위.
- `calibrate_beauty.py` · `calibrate_grade.py`: 목표 영상과 같은 장면 5곳에서 얼굴 점·피부색·벽색·팔자 그늘·허리를 재서 보정값을 맞추는 도구.
  - `base`: 목표 vs 보정 없음
  - `sweep <칸…>`: 칸마다 값을 바꿔 보기
  - `check`: 지금 값으로 목표와 견주기
- `nose_experiment.py` · `fold_experiment.py`: 콧볼·팔자 방식 비교 그림.
- **지금 맞춘 값** (`ks_edit_steps.py`의 `BEAUTY` · `GRADE`):
  - 얼굴: 콧볼 1.0 · 팔자 1.0 · 갸름 0.65 · 턱 −0.7 · 눈 0.5 · 블러셔 0.8 (조금 진함 — 0.6~0.7 검토) · 매끈 0.5 · 눈 또렷 0.25 · 허리 0.2
  - 색: 하이라이트 +0.11 · 색온도 −0.3 · 노출 +0.02
- **남은 차이:** 피부 노란 기 (KS에 칸이 없음 — TODO 10-1) · 엔딩 그림(투명 PNG를 못 얹음 — TODO 7) · 글자 모양(상자 크기·줄 간격·글꼴 굵기 — TODO 5·8·9).

## 8. 정해 둔 결정 (다시 묻지 않을 것)

- 생성형 배경·날씨 바꾸기는 하지 않는다 (무겁다). 합성 정도만, 요청이 있으면.
- AI 보정은 **내보낼 때** 입힌다. 컷편집 미리보기에는 보이지 않아도 된다.
- 콧볼 축소 100% = 목표 수준 (예전 100% ≈ 지금 60%).
- 팔자주름 100% = 거의 다 지움. 자연스러운 정도는 슬라이더로 (툴팁에 60~80% 안내).
- 하이라이트 분할: V1만 대상, 점수 줄은 맨 아래, 민감도는 사용자가 조절. 기준(가중치) 맞추기는 사용자가 실제 촬영본을 보고 판단한다.
- 클립마다 다른 AI 보정은 나중에.

## 9. `.claude/` (저장소 밖, 이 PC)

- `.claude/CLAUDE.md`: 작업 규칙 — 큰 일 쪼개기 · 버그는 재현 테스트부터 · 요청 안 한 변경 금지 · 확인한 것/못 한 것 나눠 보고 · 오래 걸리는 일은 백그라운드로 · 커밋과 푸시 따로 · 맥락 관리
- 스킬: `/handoff` (`.claude/HANDOFF.md`에 인수인계) · `/verify` (결과를 다시 확인해서 표로) · `/review-instructions` (지시 파일 점검 제안)
- `.claude/output-styles/`: 사용자가 둔 출력 스타일 (건드리지 않는다)
