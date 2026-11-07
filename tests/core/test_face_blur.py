"""다른 사람 얼굴 흐리기: 얼굴 줄기 잇기 · 흐리지 않을 사람 · 주인공 보호 · 놓친 프레임 메우기 · 등록한 사람."""

import numpy as np

from ks_studio.core.beauty import BeautySettings
from ks_studio.core.face_blur import HOLD, SeenFace, blur_plan, landmark_boxes, link
from ks_studio.core.people import load_registry, registered_name, save_registry

RNG = np.random.default_rng(0)


def person() -> np.ndarray:
    v = RNG.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def looks_like(v: np.ndarray, noise: float = 0.5) -> np.ndarray:
    """같은 사람의 다른 장면 (인식값이 조금 다르다)."""
    w = v + RNG.normal(size=512).astype(np.float32) * noise / np.sqrt(512)
    return w / np.linalg.norm(w)


def walk(who: np.ndarray, frames, x0: float, step: float = 2.0) -> list[SeenFace]:
    return [SeenFace(f, (x0 + f * step, 100.0, x0 + f * step + 80, 180.0), looks_like(who)) for f in frames]


def test_link_follows_moving_faces_separately():
    a, b = person(), person()
    tracks = link(walk(a, range(10), 0) + walk(b, range(10), 500))
    assert sorted(len(t) for t in tracks) == [10, 10]


def test_plan_blurs_strangers_only_and_bridges_missed_frames():
    hero, friend, stranger = person(), person(), person()
    frames = list(range(30))
    missed = [f for f in frames if f not in (12, 13, 14)]  # 낯선 사람 얼굴을 세 프레임 놓쳤다
    faces = walk(hero, frames, 0) + walk(friend, frames, 300) + walk(stranger, missed, 600)
    plan = blur_plan(faces, keep=[hero, friend])
    assert all(len(plan.get(f, [])) == 1 for f in frames)  # 낯선 사람만, 놓친 프레임도 흐린다
    assert plan[13][0][0] < 600 + 13 * 2  # 흐릴 상자는 얼굴보다 크다 (사방 30%)
    assert max(plan) == 29 + HOLD and min(plan) == 0  # 끝난 뒤로도 잠깐 더 (앞쪽은 0에서 멈춘다)


def test_protagonist_track_is_protected_even_when_recognition_fails():
    hero = person()
    side_faces = walk(person(), range(10), 0)  # 옆얼굴이라 인식값이 주인공과 안 닮았다
    protected = {f.frame: f.box for f in side_faces}  # 얼굴 점으로는 주인공 자리를 쫓고 있다
    assert blur_plan(side_faces, keep=[hero], protected=protected) == {}
    assert len(blur_plan(side_faces, keep=[hero])) > 0


def test_landmark_boxes_skip_missing_frames():
    track = np.full((3, 106, 2), np.nan)
    track[1] = np.column_stack([np.linspace(10, 50, 106), np.linspace(20, 90, 106)])
    assert landmark_boxes(track, start=100) == {101: (10.0, 20.0, 50.0, 90.0)}


def test_registry_roundtrip(tmp_path):
    path = tmp_path / "people.json"
    friend = person()
    assert load_registry(path) == []
    save_registry(path, [("엄마", friend)])
    registry = load_registry(path)
    assert registry[0][0] == "엄마" and registered_name(registry, looks_like(friend)) == "엄마"
    assert registered_name(registry, person()) is None


def test_settings_keep_roundtrip():
    v = tuple(float(x) for x in person())
    settings = BeautySettings.from_dict(BeautySettings(blur_others=True, keep=(v,)).to_dict())
    assert settings.blur_others and len(settings.keep_vectors) == 1 and settings.needs_render
    assert BeautySettings.from_dict({"keep": [[1, 2, 3]]}).keep == ()  # 잘못된 값은 버린다
