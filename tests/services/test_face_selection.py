"""보정할 사람 고르기: 후보 장면의 사람 묶기, 주인공만 고르기, 샘플 컷 고르기 (얼굴 모델 없이 가짜 얼굴·인식값)."""

import numpy as np

from ks_studio.services.face import FoundFace, Probe, choose, people, pick_samples
from tests.core.test_beauty import fake_face
from tests.core.test_people import face_of

WIDTH = 1920


def found(person: int, scale: float = 1.0, turned: bool = False) -> FoundFace:
    lm = fake_face() * scale
    if turned:  # 옆얼굴
        lm[86, 0] += 0.5 * 400 * scale
    return FoundFace(lm, face_of(person))


def probe(frame: int, *faces: FoundFace) -> Probe:
    return Probe(frame, np.zeros((1080, WIDTH, 3), np.uint8), faces)


def scenes() -> list[Probe]:
    # 주인공(0)이 대부분, 1번 장면엔 주인공 없이 더 크게 나온 행인(1), 3번 장면엔 주인공 옆얼굴뿐
    return [
        probe(0, found(0), found(1, 0.5)),
        probe(100, found(1, 1.3)),
        probe(200, found(0, 0.9)),
        probe(300, found(0, turned=True)),
        probe(400, found(0, 1.1), found(2, 0.6)),
    ]


def test_people_sorted_by_appearances_from_frontal_faces_only():
    found_people = people(scenes())
    assert [p.appearances for p in found_people][:2] == [3, 2]  # 주인공 3번 (옆얼굴 장면 제외), 행인 2번
    assert found_people[0].face.embedding is not None


def test_choose_target_ignores_bigger_bystander():
    target = people(scenes())[0].embedding
    faces = list(scenes()[1].faces)  # 행인뿐인 장면
    assert choose(faces, target) is None
    assert choose(faces, None) is faces[0]  # 주인공을 안 정했으면 가장 큰 얼굴


def test_samples_show_only_target():
    target = people(scenes())[0].embedding
    frames = [s.frame for s in pick_samples(scenes(), target)]
    assert frames == [0, 200, 400]  # 행인 장면(100)과 옆얼굴 장면(300)은 빠진다


def test_sample_probes_cache_roundtrip(tmp_path):
    """샘플 후보 장면 캐시: 같은 영상을 다시 고르면 얼굴 찾기 없이 같은 장면·얼굴을 돌려준다."""
    import numpy as np

    from ks_studio.services.face import FoundFace, Probe, load_probes, save_probes

    rng = np.random.default_rng(0)
    rgb = np.full((90, 160, 3), 120, np.uint8)
    face = FoundFace(rng.normal(50, 10, (106, 2)), rng.normal(0, 1, 512).astype(np.float32))
    probes = [Probe(10, rgb, (face, face)), Probe(40, rgb, ())]
    path = tmp_path / "cache" / "ai_samples" / "x.v1.npz"
    save_probes(path, probes)
    back = load_probes(path)
    assert [p.frame for p in back] == [10, 40] and [len(p.faces) for p in back] == [2, 0]
    assert np.allclose(back[0].faces[1].landmarks, face.landmarks) and np.allclose(back[0].faces[0].embedding, face.embedding)
    assert np.abs(back[1].rgb.astype(int) - 120).max() <= 3  # JPEG
    assert load_probes(tmp_path / "missing.npz") is None


def test_face_model_download_detects_and_unpacks(tmp_path):
    """모델이 없으면 '없음', 내려받으면 쓰는 파일 셋이 제자리에 (묶음 안 폴더가 한 겹 더 있어도)."""
    import zipfile

    from ks_studio.services.face import FACE_MODEL_FILES, download_face_models, face_model_dir, face_models_ready

    models = tmp_path / "models"
    assert not face_models_ready(models)
    face_model_dir(models).mkdir(parents=True)  # 받다 끊겨 폴더만 남은 경우
    assert not face_models_ready(models)
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as zf:
        for name in FACE_MODEL_FILES:
            zf.writestr(f"buffalo_l/{name}", b"onnx")
    seen = []
    download_face_models(seen.append, models_dir=models, url=pack.as_uri())
    assert face_models_ready(models) and seen and seen[-1] == 1.0
    assert not any(p.name.endswith(".part") for p in (models / "models").iterdir())  # 임시 파일 정리
