"""AI 보정 화소 처리: 블러셔 색, 콧볼 변형 방향, 효과 없음/얼굴 없음, 보정본 렌더 (얼굴 모델 없이 가짜 랜드마크로)."""

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("cv2")  # AI 패키지([ai])를 깔지 않았으면 건너뛴다

import av  # noqa: E402

import cv2  # noqa: E402

from ks_studio.core.beauty import (  # noqa: E402
    BeautySettings,
    cheek_ellipses,
    contour_shapes,
    lip_polygons,
    nose_pulls,
    slim_pulls,
    under_eye_ellipses,
)
from ks_studio.services.beautify import apply, apply_frame, regions, render_beautified  # noqa: E402
from ks_studio.services.sources import AUDIO_RATE, decode_audio  # noqa: E402
from tests.core.test_beauty import fake_face  # noqa: E402

GRAY = 128


def canvas() -> np.ndarray:
    return np.full((1000, 1000, 3), GRAY, np.uint8)


def test_blush_tints_cheeks_only():
    lm = fake_face()
    out = apply(canvas(), lm, BeautySettings(blush=1.0))
    left = cheek_ellipses(lm)[0]
    center = out[round(left.cy), round(left.cx)].astype(int)
    assert center[0] > GRAY + 20 and center[1] < GRAY  # 핑크 쪽으로
    assert (out[50, 50] == GRAY).all()  # 얼굴 바깥은 그대로


def test_nose_slim_moves_wing_edge_toward_center():
    lm = fake_face()
    image = canvas()
    image[:, 400:425] = 0  # 왼쪽 콧볼 바깥쪽에 세로 검은 띠 (x 400~424)
    out = apply(image, lm, BeautySettings(nose_slim=1.0))
    pull = nose_pulls(lm, 1.0)[0]
    row = round(pull.cy)
    dark = np.flatnonzero(out[row, :, 0] < 64)
    assert dark.max() > 424 + 5  # 띠의 안쪽 끝이 코 가운데(오른쪽)로 밀려 들어왔다
    assert (out[0, 400:425] == 0).all()  # 반지름 밖(맨 위 줄)은 그대로


def test_turned_or_tiny_face_is_left_alone():
    turned = fake_face()
    turned[86, 0] += 0.5 * 400
    image = canvas()
    assert apply(image, turned, BeautySettings(blush=1.0)) is image
    assert apply(image, fake_face() * 0.02, BeautySettings(blush=1.0)) is image


def test_nothing_to_do_returns_same_image():
    image = canvas()
    assert apply(image, fake_face(), BeautySettings()) is image
    assert apply(image, None, BeautySettings(blush=1.0)) is image
    assert apply(image, np.full((106, 2), np.nan), BeautySettings(blush=1.0)) is image


def make_video(path, frames: int = 30, fps: int = 30) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height = 320, 320
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        for k in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((320, 320, 3), GRAY, np.uint8), format="rgb24")
            frame.pts, frame.time_base = k, Fraction(1, fps)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        samples = (0.5 * np.sin(np.arange(frames * AUDIO_RATE // fps) * 2 * np.pi * 440 / AUDIO_RATE)).astype(np.float32)
        for i in range(0, len(samples), 1024):
            chunk = av.AudioFrame.from_ndarray(samples[None, i : i + 1024], format="flt", layout="mono")
            chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
            container.mux(audio.encode(chunk))
        container.mux(audio.encode())


def test_render_keeps_frames_and_sound_and_applies_effect(tmp_path):
    src, out = tmp_path / "src.mp4", tmp_path / "out.mp4"
    make_video(src)
    lm = fake_face() * 0.32  # 320px 화면에 맞춘 얼굴
    track = np.stack([lm] * 30)
    track[20:] = np.nan  # 뒤쪽은 얼굴 없음 → 그대로
    render_beautified(src, out, track, BeautySettings(blush=1.0))
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
    assert len(set(np.diff(pts))) == 1  # 프레임 시각이 겹치지 않고 일정 간격 (시간 단위를 잘못 쓰면 겹쳤다)
    with av.open(str(out)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(frames) == 30
    cheek = cheek_ellipses(lm)[0]
    y, x = round(cheek.cy), round(cheek.cx)
    assert frames[5][y, x, 0] > frames[25][y, x, 0] + 15  # 얼굴 있는 프레임만 볼이 붉다
    assert abs(len(decode_audio(out)) - len(decode_audio(src))) < AUDIO_RATE // 50  # 소리 그대로


def test_face_slim_moves_jaw_edge_inward():
    lm = fake_face()
    image = canvas()
    pull = slim_pulls(lm, 1.0)[3]  # 왼쪽 턱선 가운데쯤
    row, x = round(pull.cy), round(pull.cx)
    image[:, x - 30 : x] = 0  # 턱선 바깥쪽에 검은 띠
    out = apply(image, lm, BeautySettings(slim=1.0))
    assert np.flatnonzero(out[row, :, 0] < 64).max() > x + 5  # 띠 안쪽 끝이 가운데로 밀려 들어왔다


def test_regions_marks_where_effects_go():
    lm = fake_face()
    image = canvas()
    out = regions(image, lm, BeautySettings(blush=0.5, nose_slim=0.5, slim=0.5))
    left = cheek_ellipses(lm)[0]
    assert out.shape == image.shape and (out[round(left.cy), round(left.cx)] != GRAY).any()  # 볼 자리에 색
    assert (out[20, 20] == GRAY).all() and (image == GRAY).all()  # 얼굴 밖 그대로, 원본은 안 바뀜


def test_apply_frame_keeps_background_bits_exact():
    """얼굴 주변만 RGB로 바꿔 보정한다 — 배경 화소는 원래 YUV 값 그대로 (화면 전체 RGB 왕복은 PSNR 47dB까지 떨어졌다)."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (2000, 2000, 3), dtype=np.uint8)  # 얼굴(1000×1000 안) 밖에 배경이 넉넉하게
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
    before = frame.to_ndarray().copy()
    out = apply_frame(frame, fake_face(), BeautySettings(blush=1.0, slim=1.0)).to_ndarray()
    assert (out[:100] == before[:100]).all() and (out[1100:2000] == before[1100:2000]).all()  # 얼굴 밖 Y 줄은 비트까지 같다
    assert not (out == before).all()  # 얼굴은 바뀌었다
    eyes = apply_frame(frame, fake_face(), BeautySettings(eyes=1.0)).to_ndarray()
    assert (eyes[1100:1150] == before[1100:1150]).all()  # 얼굴 영역 안이라도 효과가 닿지 않은 곳(눈 키우기면 입 둘레)은 그대로
    assert apply_frame(frame, None, BeautySettings(blush=1.0)) is frame


def test_one_pass_from_original_calls_effect_with_copy_frame_numbers(tmp_path):
    """휴대폰 원본에서 회전·색 변환과 보정을 한 번에: 보정 함수는 사본과 같은 프레임 번호(0, 1, 2…)로 불린다."""
    from ks_studio.services.optimize import inspect, make_optimized
    from tests.services.test_optimize import make_source, vfr_times

    src, out = tmp_path / "vfr.mp4", tmp_path / "out.mp4"
    make_source(src, vfr_times(1.0))
    seen = []

    def process(frame):
        seen.append(frame.pts)
        return frame

    make_optimized(src, out, inspect(src), process=process)
    assert seen == list(range(len(seen))) and abs(len(seen) - 30) <= 2


def noisy(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(150, 12, (1000, 1000, 3)), 0, 255).astype(np.uint8)


def test_smooth_softens_skin_not_eyes_or_background():
    image = noisy()
    out = apply(image, fake_face(), BeautySettings(smooth=1.0))
    cheek, eye = (slice(420, 470), slice(230, 280)), (slice(295, 306), slice(350, 371))
    assert out[cheek].std() < 0.6 * image[cheek].std()
    assert (out[eye] == image[eye]).all() and (out[:50] == image[:50]).all()


def test_brighten_and_dark_circle_lift_only_their_areas():
    image = canvas()
    bright = apply(image, fake_face(), BeautySettings(brighten=1.0))
    assert bright[450, 250].mean() > GRAY + 5 and (bright[300, 360] == GRAY).all()  # 볼은 밝게, 눈은 그대로
    ellipse = under_eye_ellipses(fake_face())[0]
    under = apply(image, fake_face(), BeautySettings(dark_circle=1.0))
    assert under[round(ellipse.cy), round(ellipse.cx)].mean() > GRAY + 5
    assert (under[300, 360] == GRAY).all() and (under[500, 500] == GRAY).all()


def test_eye_zoom_makes_dark_eye_bigger():
    image = canvas()
    cv2.circle(image, (360, 300), 20, (0, 0, 0), -1)
    out = apply(image, fake_face(), BeautySettings(eyes=1.0))
    width = lambda a: int((a[300, 300:420, 0] < 64).sum())  # noqa: E731
    assert width(out) > width(image) + 4


def test_lip_color_tints_lips_keeps_brightness_and_mouth_inside():
    image = canvas()
    out = apply(image, fake_face(), BeautySettings(lip=1.0))
    r, g, b = out[645, 500].astype(int)  # 아랫입술
    assert r > g + 20 and r > b
    assert abs(int(cv2.cvtColor(out[645:646, 500:501], cv2.COLOR_RGB2LAB)[0, 0, 0]) - int(cv2.cvtColor(image[:1, :1], cv2.COLOR_RGB2LAB)[0, 0, 0])) < 12
    assert np.abs(out[623, 500].astype(int) - GRAY).max() <= 3  # 입 안쪽은 립 컬러가 아니다 (경계를 부드럽게 한 만큼만)


def test_teeth_whitening_removes_yellow_inside_mouth_only():
    image = canvas()
    _, inner = lip_polygons(fake_face())
    cv2.fillPoly(image, [inner.astype(np.int32)], (225, 210, 150))  # 누런 치아
    image[700, 500] = (225, 210, 150)  # 입 밖 같은 색
    out = apply(image, fake_face(), BeautySettings(teeth=1.0))
    before, after = image[623, 500].astype(int), out[623, 500].astype(int)
    assert after[0] - after[2] < 0.5 * (before[0] - before[2])  # 노란기(빨강-파랑 차이)가 줄었다
    assert (out[700, 500] == image[700, 500]).all()


def test_eye_clear_sharpens_inside_eyes_only():
    image = cv2.GaussianBlur(noisy(1), (0, 0), 2)
    out = apply(image, fake_face(), BeautySettings(eye_clear=1.0))
    eye, cheek = (slice(292, 309), slice(340, 381)), (slice(420, 470), slice(230, 280))
    assert out[eye].astype(float).std() > 1.3 * image[eye].astype(float).std()
    assert (out[cheek] == image[cheek]).all()


def test_regions_shows_only_effects_that_are_on():
    lm = fake_face()
    off = regions(canvas(), lm, BeautySettings())
    assert (off == GRAY).all()
    lips = regions(canvas(), lm, BeautySettings(lip=1.0))
    assert (lips[645, 500] != GRAY).any() and (lips[450, 250] == GRAY).all()


def test_blemish_fills_small_dark_spots_but_not_lines():
    image = canvas()
    for center in ((250, 450), (720, 520), (500, 180)):  # 볼·볼·이마의 작은 점
        cv2.circle(image, center, 7, (90, 70, 60), -1)
    cv2.line(image, (640, 430), (800, 430), (90, 70, 60), 4)  # 머리카락처럼 가늘고 긴 선
    out = apply(image, fake_face(), BeautySettings(blemish=1.0))
    for x, y in ((250, 450), (720, 520), (500, 180)):
        assert np.abs(out[y, x].astype(int) - GRAY).max() < 12  # 둘레 피부색으로 메꿨다
    assert out[430, 720].astype(int).max() < 100  # 선은 그대로
    assert (out[300, 360] == image[300, 360]).all()


def test_smile_moves_mouth_corner_up():
    image = canvas()
    cv2.circle(image, (420, 620), 6, (0, 0, 0), -1)  # 왼쪽 입꼬리
    out = apply(image, fake_face(), BeautySettings(smile=1.0))
    rows = lambda a: np.flatnonzero((a[560:680, 400:440, 0] < 64).any(axis=1)).mean()  # noqa: E731
    assert rows(out) < rows(image) - 3


def test_contour_darkens_shade_and_lightens_highlight():
    image = canvas()
    out = apply(image, fake_face(), BeautySettings(contour=1.0))
    shade, highlight = contour_shapes(fake_face())
    s, h = shade[0], highlight[0]
    assert out[round(s.cy), round(s.cx)].mean() < GRAY - 5
    sr, sg, sb = out[round(s.cy), round(s.cx)].astype(int)
    assert sb < sr  # 따뜻한 그림자 (파랑을 더 뺐다)
    assert out[round(h.cy), round(h.cx)].mean() > GRAY + 5
    assert (out[20, 20] == GRAY).all()


def test_eyeshadow_tints_eyelid_in_chosen_color():
    image = canvas()
    pink = apply(image, fake_face(), BeautySettings(eyeshadow=1.0, eyeshadow_color="pink"))
    r, g, b = pink[272, 360].astype(int)
    assert r > g + 10 and r > b  # 눈두덩이 분홍
    assert np.abs(pink[300, 360].astype(int) - GRAY).max() <= 3  # 눈 안은 (경계를 부드럽게 한 만큼만)
    brown = apply(image, fake_face(), BeautySettings(eyeshadow=1.0, eyeshadow_color="brown"))
    assert not (brown[272, 360] == pink[272, 360]).all()
    odd = apply(image, fake_face(), BeautySettings(eyeshadow=1.0, eyeshadow_color="없는색"))
    assert (odd == brown).all()  # 모르는 색(손으로 고친 파일)은 브라운


def test_side_by_side_clip_has_range_both_halves_and_sound(tmp_path):
    """1분 미리보기 영상: start부터 count장, 왼쪽 원본 | 오른쪽 보정, 그 구간 소리."""
    from ks_studio.services.beautify import CLIP_GAP, render_side_by_side

    src, out = tmp_path / "src.mp4", tmp_path / "clip.mp4"
    make_video(src, frames=30)
    lm = fake_face() * 0.32
    render_side_by_side(src, out, 10, 15, lambda i: lm, BeautySettings(blush=1.0))
    with av.open(str(out)) as c:
        frames = [f.to_ndarray(format="rgb24").astype(int) for f in c.decode(video=0)]
        audio_seconds = c.streams.audio[0].duration * c.streams.audio[0].time_base
    assert len(frames) == 15 and frames[0].shape == (320, 2 * 320 + CLIP_GAP, 3)
    left, right = frames[5][:, :320], frames[5][:, 320 + CLIP_GAP :]
    cheek = cheek_ellipses(lm)[0]
    y, x = round(cheek.cy), round(cheek.cx)
    assert abs(left[y, x] - GRAY).max() < 8 and right[y, x, 0] > right[y, x, 2] + 10  # 오른쪽만 블러셔
    assert abs(float(audio_seconds) - 0.5) < 0.1  # 15장 = 0.5초



def test_lip_colors_differ_and_gloss_brightens_lower_lip_center():
    from ks_studio.core.beauty import LIP_COLORS, gloss_ellipse

    lm = fake_face()
    reds = {key: apply(canvas(), lm, BeautySettings(lip=1.0, lip_color=key))[645, 500].astype(int) for key in LIP_COLORS}
    assert reds["red"][1] < reds["nude"][1] and reds["plum"][0] < reds["coral"][0]  # 색마다 다르다
    painted = apply(canvas(), lm, BeautySettings(lip=1.0))
    glossy = apply(canvas(), lm, BeautySettings(lip=1.0, lip_gloss=1.0))
    sheen = gloss_ellipse(lm)
    x, y = round(sheen.cx), round(sheen.cy)
    assert glossy[y, x].astype(int).sum() > painted[y, x].astype(int).sum() + 30  # 아랫입술 가운데가 반짝
    assert (glossy[50, 50] == painted[50, 50]).all()  # 입술 밖은 그대로


def fold_line(image: np.ndarray, lm: np.ndarray, value: int = 100) -> None:
    """팔자주름처럼: 콧볼 바깥 → 입꼬리 바깥으로 어두운 선 (폭 16px — 눈 사이 400px 얼굴)."""
    for (wing, _), corner in (((77, 78), 52), ((83, 84), 61)):
        a, b = lm[wing], lm[corner]
        side = np.sign(a[0] - 500)
        start = (int(a[0] + side * 15), int(a[1] + 10))
        end = (int(b[0] + side * 40), int(b[1] + 20))
        cv2.line(image, start, end, (value, value, value), 16, cv2.LINE_AA)  # 실제 팔자처럼 잔결보다 넓게


def fold_contrast(image: np.ndarray, lm: np.ndarray) -> float:
    """선 가운데(왼쪽)와 바로 옆 피부의 밝기 차."""
    a, b = lm[77], lm[52]
    mid = ((a[0] - 15 + b[0] - 40) / 2, (a[1] + 10 + b[1] + 20) / 2)
    on = image[int(mid[1]), int(mid[0])].astype(float).mean()
    off = image[int(mid[1]), int(mid[0]) - 25].astype(float).mean()
    return off - on


def test_nasolabial_lifts_fold_shadow_and_keeps_rest():
    lm = fake_face()
    image = canvas()
    fold_line(image, lm)
    before = fold_contrast(image, lm)
    full = apply(image, lm, BeautySettings(nasolabial=1.0))
    half = apply(image, lm, BeautySettings(nasolabial=0.5))
    assert before > 20
    assert fold_contrast(full, lm) < before * 0.2  # 100% = 거의 다
    assert before * 0.35 < fold_contrast(half, lm) < before * 0.75  # 슬라이더에 비례
    assert (full[50, 50] == image[50, 50]).all()  # 띠 바깥은 그대로
    assert apply(image, lm, BeautySettings()) is image


def test_nasolabial_keeps_fine_texture():
    """잔결(작은 무늬)은 남는다 — 그늘만 걷는다."""
    lm = fake_face()
    rng = np.random.default_rng(0)
    image = np.clip(canvas().astype(int) + rng.integers(-6, 7, (1000, 1000, 1)), 0, 255).astype(np.uint8)
    fold_line(image, lm)
    out = apply(image, lm, BeautySettings(nasolabial=1.0))
    a, b = lm[77], lm[52]
    y, x = int((a[1] + b[1]) / 2 + 15), int((a[0] + b[0]) / 2 - 27)
    patch_before = image[y - 8 : y + 8, x - 30 : x - 14, 0].astype(float)
    patch_after = out[y - 8 : y + 8, x - 30 : x - 14, 0].astype(float)
    assert patch_after.std() > 0.6 * patch_before.std()


def test_nose_slim_full_strength_reaches_half_way():
    from ks_studio.core.beauty import NOSE_CENTER, NOSE_WINGS, nose_pulls

    lm = fake_face()
    pulls = nose_pulls(lm, 1.0)
    center = lm[list(NOSE_CENTER)].mean(axis=0)
    wing = (lm[NOSE_WINGS[0][0]] + lm[NOSE_WINGS[0][1]]) / 2
    moved = np.hypot(pulls[0].tx - pulls[0].cx, pulls[0].ty - pulls[0].cy)
    assert moved == pytest.approx(0.5 * np.linalg.norm(center - wing), rel=1e-6)  # 100% = 콧볼이 가운데까지 거리의 절반
