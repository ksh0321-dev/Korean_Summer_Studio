"""사람 층 변형 (몸매·얼굴 작게): 모델 없이 가짜 사람 분리·몸의 점으로. 핵심은 '사람 밖 배경은 화소 하나 안 바뀐다'."""

import numpy as np
import pytest

pytest.importorskip("cv2")

import av  # noqa: E402
import cv2  # noqa: E402

from ks_studio.core.beauty import BeautySettings, face_small_scale, skin_polygon  # noqa: E402
from ks_studio.services.beautify import apply, apply_frame  # noqa: E402
from ks_studio.services.body import BACKGROUND, CLOTHES, FACE_SKIN, PARTS_SIZE  # noqa: E402
from ks_studio.services.reshape import BodyInfo, reshape  # noqa: E402
from tests.core.test_beauty import fake_face  # noqa: E402

SIZE = 1000


def parts_of(*layers: tuple[np.ndarray, int]) -> np.ndarray:
    """화면 크기 마스크들(부위 번호와 함께) → (256, 256, 6) 부위 확률."""
    parts = np.zeros((PARTS_SIZE, PARTS_SIZE, 6), np.float32)
    parts[..., BACKGROUND] = 1
    for mask, part in layers:
        small = cv2.resize(mask.astype(np.float32), (PARTS_SIZE, PARTS_SIZE), interpolation=cv2.INTER_AREA)
        parts[..., part] += small
        parts[..., BACKGROUND] -= small
    return parts.astype(np.float16)


def striped() -> np.ndarray:
    image = np.full((SIZE, SIZE, 3), 128, np.uint8)
    image[:, ::20] = 30  # 세로 줄무늬 배경 — 휘면 바로 드러난다
    return image


def keypoints(**moves) -> np.ndarray:
    """서 있는 사람: 어깨 y=300, 엉덩이 y=600, 무릎 y=800 (x 400~600). 팔은 몸에서 멀리 (허리 옆에 없게)."""
    points = np.zeros((17, 3))
    points[:, 2] = 1.0
    points[5, :2], points[6, :2] = (600, 300), (400, 300)  # 어깨 (사람 기준 왼/오른)
    points[11, :2], points[12, :2] = (580, 600), (420, 600)  # 엉덩이
    points[13, :2], points[14, :2] = (580, 800), (420, 800)
    points[15, :2], points[16, :2] = (580, 950), (420, 950)
    points[7, :2], points[8, :2] = (850, 200), (150, 200)  # 팔꿈치·손목은 위로 멀리
    points[9, :2], points[10, :2] = (900, 100), (100, 100)
    for index, xy in moves.items():
        points[int(index[1:]), :2] = xy
    return points


def torso_scene():
    image = striped()
    person = np.zeros((SIZE, SIZE), np.uint8)
    person[280:1000, 380:620] = 1  # 몸통·다리 (폭 240)
    image[person > 0] = (200, 150, 120)
    image[698:703, 380:620] = (0, 0, 0)  # 다리 위 가로선 (다리 길게 확인용)
    return image, person


def outside(mask: np.ndarray, margin: int = 31) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((margin, margin), np.uint8)) == 0


def test_face_small_shrinks_head_and_never_touches_background():
    lm = fake_face()
    image = striped()
    head = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(head, [skin_polygon(lm).astype(np.int32)], 1)
    image[head > 0] = (200, 160, 140)
    cx, cy, _ = face_small_scale(lm, 1.0)
    cv2.circle(image, (round(cx), round(cy)), 150, (0, 0, 0), 3)  # 얼굴 안의 원
    body = BodyInfo(parts_of((head, FACE_SKIN)), None)
    out = apply(image, lm, BeautySettings(face_small=1.0), body=body)
    row = round(cy)
    width = lambda a: np.ptp(np.flatnonzero(a[row, 300:700, 0] < 64))  # noqa: E731
    assert width(out) < width(image) * 0.95  # 얼굴(안의 원)이 작아졌다
    # 얼굴이 비켜난 자리 밖(사람 분리 경계 오차만큼 넉넉히)은 그대로
    assert (out[outside(head, 61)] == image[outside(head, 61)]).all()


def test_face_small_needs_body_info():
    """사람 분리가 없으면 (몸 모델을 안 받았으면) 얼굴 작게만 빠진다 — 예전 얼굴 점 윤곽 방식은 이중 윤곽이 보였다."""
    image = striped()
    assert (apply(image, fake_face(), BeautySettings(face_small=1.0)) == image).all()


def test_waist_narrows_torso_only():
    image, person = torso_scene()
    out = image.copy()
    assert reshape(out, None, {"waist": 1.0}, BodyInfo(parts_of((person, CLOTHES)), keypoints()))
    skin = lambda a, y: np.count_nonzero(a[y, :, 0] > 180)  # noqa: E731  (몸통 색 R=200)
    assert skin(out, 525) < skin(image, 525) - 15  # 허리(어깨~엉덩이 75%)가 좁아졌다
    assert abs(skin(out, 290) - skin(image, 290)) <= 2  # 어깨 위쪽은 거의 그대로
    assert (out[outside(person)] == image[outside(person)]).all()


def test_waist_skips_side_with_arm_next_to_it():
    """팔이 허리 옆에 붙어 있으면 찾은 가장자리가 팔 바깥이라 그쪽은 건너뛴다."""
    image, person = torso_scene()
    arms = keypoints(k9=(630, 520), k10=(370, 520))  # 두 손목이 허리 옆
    out = image.copy()
    assert not reshape(out, None, {"waist": 1.0}, BodyInfo(parts_of((person, CLOTHES)), arms))
    assert (out == image).all()


def test_legs_get_longer_below_hips():
    image, person = torso_scene()
    out = image.copy()
    assert reshape(out, None, {"legs": 1.0}, BodyInfo(parts_of((person, CLOTHES)), keypoints()))
    line = lambda a: np.flatnonzero(a[560:, 510, 0] < 60)  # noqa: E731  가로선의 세로 자리 (엉덩이 아래만)
    assert line(out).mean() > line(image).mean() + 5  # 엉덩이 아래 선이 내려갔다 = 다리가 길어졌다
    above = slice(0, 560)
    assert (out[above] == image[above]).all()  # 엉덩이 위는 그대로
    assert (out[outside(person)] == image[outside(person)]).all()


def test_no_keypoints_means_no_body_change():
    image, person = torso_scene()
    out = image.copy()
    assert not reshape(out, None, {"waist": 1.0, "legs": 1.0}, BodyInfo(parts_of((person, CLOTHES)), None))


def test_apply_frame_reshapes_whole_frame_without_face():
    """몸매는 얼굴이 안 보여도 — 화면 전체를 바꾸되 사람 밖 화소(YUV)는 그대로."""
    image, person = torso_scene()
    frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
    before = frame.to_ndarray()
    after = apply_frame(frame, None, BeautySettings(legs=1.0), BodyInfo(parts_of((person, CLOTHES)), keypoints())).to_ndarray()
    assert (after != before).any()
    y_out = outside(person)
    assert (after[:SIZE][y_out] == before[:SIZE][y_out]).all()  # Y 판
    assert apply_frame(frame, None, BeautySettings(legs=1.0)) is frame  # 몸 정보가 없으면 그대로


def test_head_resegmentation_overrides_coarse_layer_in_its_box():
    """머리 둘레를 다시 분리한 결과가 그 상자 안에서 이긴다 (가장자리에서는 서서히) — 귀걸이·잔머리가 사람 층에 든다."""
    from ks_studio.services.reshape import GRID, _person_layer

    coarse = parts_of()  # 전체 분리: 사람이 없다고 봤다
    head = parts_of((np.ones((SIZE, SIZE), np.uint8), CLOTHES))  # 머리 둘레 다시 분리: 상자 전체가 사람
    body = BodyInfo(coarse, None, (head, (200, 200, 600, 600)))
    alpha = _person_layer(body, SIZE // GRID + 1, SIZE // GRID + 1)
    assert alpha[400 // GRID, 400 // GRID] > 0.99  # 상자 가운데
    assert alpha[100 // GRID, 100 // GRID] == 0  # 상자 밖은 전체 분리 그대로
    assert 0 < alpha[400 // GRID, 205 // GRID] < 0.5  # 상자 가장자리는 서서히


def test_only_target_person_is_reshaped():
    """화면에 두 사람: 얼굴 자리를 준 사람만 사람 층으로 — 옆사람은 다리 길게에도 그대로."""
    from ks_studio.services.reshape import only_person

    image, person = torso_scene()
    other = np.zeros((SIZE, SIZE), np.uint8)
    other[300:1000, 40:200] = 1  # 왼쪽에 떨어져 선 다른 사람
    image[other > 0] = (90, 160, 90)
    image[698:703, 40:200] = (0, 0, 0)
    parts = parts_of((person, CLOTHES), (other, CLOTHES))
    alone = only_person(parts, SIZE, SIZE, near=(500, 300))
    out = image.copy()
    assert reshape(out, None, {"legs": 1.0}, BodyInfo(alone, keypoints()))
    assert (out[:, :260] == image[:, :260]).all()  # 다른 사람은 그대로
    assert (out[600:, 380:620] != image[600:, 380:620]).any()  # 보정할 사람은 바뀐다
