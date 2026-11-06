"""AI 보정 위치 계산: 볼 타원, 콧볼 당기기, 랜드마크 떨림 줄이기. 실제 얼굴 대신 필요한 점만 둔 가짜 얼굴."""

import math

import numpy as np

from ks_studio.core.beauty import (
    AiState,
    BeautySettings,
    cheek_ellipses,
    cheekbone_pulls,
    chin_pulls,
    contour_shapes,
    eyelid_polygons,
    eye_zooms,
    nose_length_pulls,
    skin_polygon,
    under_eye_ellipses,
    frontal_weight,
    load_ai_state,
    nose_pulls,
    save_ai_state,
    size_weight,
    slim_pulls,
    smile_pulls,
    smooth_track,
)


def fake_face(dx: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """정면 얼굴 비슷한 점 배치 (눈 사이 400px, 코 가운데 x=500). angle만큼 (500, 500) 기준으로 돌린다."""
    lm = np.full((106, 2), 500.0)  # 쓰지 않는 점은 얼굴 가운데에 (0,0)에 두면 얼굴 영역이 화면 끝까지 늘어난다
    points = {
        35: (300, 300), 93: (700, 300),  # 눈 바깥 끝
        12: (160, 450), 28: (840, 450),  # 볼 윤곽 (왼쪽 12 ↔ 오른쪽 28)
        77: (420, 480), 78: (430, 510), 83: (580, 480), 84: (570, 510),  # 콧볼
        80: (500, 520), 86: (500, 470),  # 콧구멍 아래 가운데, 코끝
        72: (500, 300), 75: (460, 380), 76: (445, 440), 81: (540, 380), 82: (555, 440),  # 콧대 위, 코 옆
        79: (465, 515), 85: (535, 515),  # 콧구멍 아래
        0: (500, 760),  # 턱 끝
        # 볼 아래 → 턱선 (왼쪽 13,14,15,16,2,3,4 ↔ 오른쪽 29,30,31,32,18,19,20, x = 1000 - 왼쪽)
        13: (165, 490), 14: (178, 535), 15: (195, 580), 16: (215, 625), 2: (240, 665), 3: (270, 700), 4: (305, 728),
        29: (835, 490), 30: (822, 535), 31: (805, 580), 32: (785, 625), 18: (760, 665), 19: (730, 700), 20: (695, 728),
        # 윤곽 나머지: 위쪽(1,9,10,11 ↔ 17,25,26,27)과 턱 끝 옆(5~8 ↔ 21~24)
        1: (150, 300), 9: (152, 340), 10: (154, 380), 11: (157, 420),
        17: (850, 300), 25: (848, 340), 26: (846, 380), 27: (843, 420),
        5: (340, 745), 6: (375, 752), 7: (410, 757), 8: (450, 760),
        21: (660, 745), 22: (625, 752), 23: (590, 757), 24: (550, 760),
        # 눈: 안쪽 끝 39 / 89, 가운데 38 / 88, 위아래 테두리 (y 280~320)
        39: (420, 300), 38: (360, 300), 41: (330, 285), 40: (360, 280), 42: (390, 285), 37: (390, 315), 33: (360, 320), 36: (330, 315),
        89: (580, 300), 88: (640, 300), 95: (610, 285), 94: (640, 280), 96: (670, 285), 91: (670, 315), 87: (640, 320), 90: (610, 315),
        # 눈썹 (y 224~250)
        43: (290, 250), 44: (320, 245), 45: (350, 242), 47: (380, 244), 46: (420, 250),
        48: (295, 232), 49: (330, 225), 51: (370, 224), 50: (415, 232),
        97: (710, 250), 98: (680, 245), 99: (650, 242), 100: (620, 244), 101: (580, 250),
        102: (705, 232), 103: (670, 225), 104: (630, 224), 105: (585, 232),
        # 입술 바깥 (입꼬리 52 / 61) · 입 안쪽 (y 612~634)
        52: (420, 620), 64: (445, 605), 63: (470, 598), 71: (500, 602), 67: (530, 598), 68: (555, 605),
        61: (580, 620), 58: (555, 640), 59: (530, 650), 53: (500, 654), 56: (470, 650), 55: (445, 640),
        65: (435, 620), 66: (465, 612), 62: (500, 614), 70: (535, 612), 69: (565, 620), 57: (535, 630), 60: (500, 634), 54: (465, 630),
    }
    c, s = math.cos(angle), math.sin(angle)
    for i, (x, y) in points.items():
        x, y = x - 500, y - 500
        lm[i] = (500 + dx + x * c - y * s, 500 + x * s + y * c)
    return lm


def test_cheeks_sit_between_eye_corner_nose_and_contour():
    left, right = cheek_ellipses(fake_face())
    assert 160 < left.cx < 420 and 300 < left.cy < 480
    assert 580 < right.cx < 840
    assert math.isclose(left.cx - 160, 840 - right.cx, abs_tol=1)  # 좌우 대칭
    assert math.isclose(left.rx, 0.22 * 400) and left.angle == 0


def test_cheeks_follow_face_tilt():
    left, _ = cheek_ellipses(fake_face(angle=0.2))
    assert math.isclose(left.angle, 0.2, abs_tol=1e-9)


def test_nose_pulls_move_wings_toward_center():
    left, right = nose_pulls(fake_face(), 1.0)
    assert left.tx > left.cx and right.tx < right.cx  # 둘 다 가운데로
    assert math.isclose(left.tx - left.cx, (500 - 425) * 0.3, rel_tol=0.05)
    assert nose_pulls(fake_face(), 0.0) == []


def test_smoothing_reduces_jitter_and_fills_short_gaps():
    rng = np.random.default_rng(0)
    track = np.stack([fake_face(dx=rng.normal(0, 4)) for _ in range(30)])
    track[10] = np.nan  # 한 프레임 놓침
    smooth = smooth_track(track)
    assert not np.isnan(smooth[10]).any()
    jitter = lambda t: np.nanstd(t[:, 35, 0])  # noqa: E731
    assert jitter(smooth) < jitter(track) * 0.6


def test_no_face_far_away_stays_missing():
    track = np.full((20, 106, 2), np.nan)
    track[:3] = fake_face()
    smooth = smooth_track(track)
    assert not np.isnan(smooth[4]).any() and np.isnan(smooth[10]).all()


def test_settings_roundtrip_ignores_unknown_keys():
    settings = BeautySettings(blush=0.4, nose_slim=0.2, approved=True)
    assert BeautySettings.from_dict({**settings.to_dict(), "future": 1}) == settings
    assert BeautySettings().is_identity
    assert not BeautySettings(chin=-0.3).is_identity  # 턱 길이는 음수(짧게)도 효과
    assert BeautySettings.from_dict({"blush": 0.5}) == BeautySettings(blush=0.5)  # 효과가 적던 예전 저장 파일


def test_ai_state_roundtrip(tmp_path):
    path = tmp_path / "ai" / "settings.json"
    assert load_ai_state(path).for_source("x.mp4") == BeautySettings()
    state = AiState({"recordings/a.mp4": BeautySettings(blush=0.5, approved=True)}, ("D:/phone.mov",), preview=False)
    path.parent.mkdir()
    save_ai_state(path, state)
    assert load_ai_state(path) == state


def test_frontal_weight_drops_for_turned_face():
    assert frontal_weight(fake_face()) == 1.0
    turned = fake_face()
    turned[86, 0] += 0.25 * 400  # 코끝이 눈 사이 거리의 25%만큼 옆으로 → 반쯤 돌아간 얼굴
    assert 0.3 < frontal_weight(turned) < 0.7
    turned[86, 0] += 0.2 * 400  # 옆얼굴
    assert frontal_weight(turned) == 0.0


def test_size_weight_ignores_tiny_background_faces():
    assert size_weight(fake_face(), 1920) == 1.0  # 눈 사이 400px
    assert size_weight(fake_face() * 0.05, 1920) == 0.0  # 눈 사이 20px (화면 폭 1%)


def test_slim_pulls_jaw_toward_face_midline():
    lm = fake_face()
    pulls = slim_pulls(lm, 1.0)
    assert len(pulls) == 14
    left, right = pulls[:7], pulls[7:]
    assert all(p.tx > p.cx for p in left) and all(p.tx < p.cx for p in right)  # 둘 다 가운데로
    assert math.isclose(math.hypot(left[0].tx - left[0].cx, left[0].ty - left[0].cy), 0.06 * 400, rel_tol=1e-6)
    assert slim_pulls(lm, 0.0) == []


def test_cheekbone_pulls_go_inward():
    pulls = cheekbone_pulls(fake_face(), 1.0)
    assert len(pulls) == 6 and all((p.tx - p.cx) * (500 - p.cx) > 0 for p in pulls)  # 모두 가운데 쪽
    assert cheekbone_pulls(fake_face(), 0.0) == []


def test_chin_goes_down_for_longer_and_up_for_shorter():
    longer, shorter = chin_pulls(fake_face(), 1.0), chin_pulls(fake_face(), -1.0)
    assert all(p.ty > p.cy for p in longer) and all(p.ty < p.cy for p in shorter)
    assert math.isclose(longer[0].ty - longer[0].cy, 0.08 * 400, rel_tol=0.05)
    assert chin_pulls(fake_face(), 0.0) == []


def test_nose_length_pulls_nose_bottom_up():
    (pull,) = nose_length_pulls(fake_face(), 1.0)
    assert pull.ty < pull.cy and math.isclose(pull.cx, 500)


def test_eye_zooms_sit_on_eye_centers():
    left, right = eye_zooms(fake_face(), 1.0)
    assert (left.cx, left.cy, right.cx) == (360, 300, 640)
    assert math.isclose(left.radius, 120) and 0 < left.amount < 0.2


def test_skin_reaches_above_brows_and_under_eyes_sit_below_eyes():
    polygon = skin_polygon(fake_face())
    assert polygon[:, 1].min() < 224 and polygon[:, 1].max() == 760  # 눈썹 위 이마 아래쪽 ~ 턱 끝
    left, right = under_eye_ellipses(fake_face())
    assert 320 < left.cy < 380 and left.cx == 360 and right.cx == 640


def test_smile_lifts_mouth_corners_up_and_out():
    left, right = smile_pulls(fake_face(), 1.0)
    assert left.ty < left.cy and right.ty < right.cy  # 위로
    assert left.tx < left.cx and right.tx > right.cx  # 바깥으로
    assert smile_pulls(fake_face(), 0.0) == []


def test_contour_shades_cheeks_and_nose_sides_highlights_center():
    shade, highlight = contour_shapes(fake_face())
    assert len(shade) == 4 and len(highlight) == 3
    bridge, forehead, chin = highlight
    assert bridge.cx == 500 and 300 < bridge.cy < 470  # 콧대: 눈 사이 ~ 코끝
    assert forehead.cy < 224 and chin.cy < 760  # 이마는 눈썹 위, 턱 끝은 안쪽
    assert shade[0].cx < 420 and shade[1].cx > 580  # 광대 아래 쉐딩은 양쪽 바깥


def test_eyelids_sit_between_eye_and_brow():
    left, right = eyelid_polygons(fake_face())
    assert left[:, 1].max() == 300 and left[:, 1].min() > 242  # 눈 윗선 ~ 눈썹 아래 사이
    assert right[:, 0].min() >= 580


def test_ai_state_keeps_memos(tmp_path):
    path = tmp_path / "settings.json"
    save_ai_state(path, AiState({}, memos={"recordings/a.mp4": "인트로 쓰기", "b.mp4": ""}))
    assert load_ai_state(path).memos == {"recordings/a.mp4": "인트로 쓰기"}  # 빈 메모는 저장하지 않는다
