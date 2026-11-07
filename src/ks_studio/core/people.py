"""얼굴 인식값(512차원, 길이 1로 맞춘 벡터)으로 '같은 사람'을 묶고 찾는다. 순수 numpy.

같은 사람이면 코사인 유사도가 높고(보통 0.4 이상), 다른 사람이면 낮다(보통 0.2 아래). 경계는 SAME_PERSON.
"""

import numpy as np

SAME_PERSON = 0.35


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def match(embeddings: list[np.ndarray], target: np.ndarray, threshold: float = SAME_PERSON) -> int | None:
    """target과 가장 닮은 얼굴 번호. 아무도 threshold를 넘지 못하면 None (그 장면에 주인공이 없다)."""
    if not embeddings:
        return None
    scores = [similarity(e, target) for e in embeddings]
    best = int(np.argmax(scores))
    return best if scores[best] >= threshold else None


def cluster(embeddings: list[np.ndarray], threshold: float = SAME_PERSON) -> list[int]:
    """얼굴마다 사람 번호. 앞에서부터, 지금까지 묶인 사람의 평균과 닮았으면 그 사람, 아니면 새 사람."""
    centers: list[np.ndarray] = []
    labels = []
    for embedding in embeddings:
        scores = [similarity(embedding, c) for c in centers]
        if scores and max(scores) >= threshold:
            label = int(np.argmax(scores))
            members = labels.count(label) + 1
            centers[label] = centers[label] + (embedding - centers[label]) / members  # 평균을 조금씩 갱신
        else:
            label = len(centers)
            centers.append(embedding.astype(np.float64))
        labels.append(label)
    return labels


# ── 등록한 사람 (앱 전체 공통, user/people.json) ──────────────────────
# 주인공 말고도 얼굴을 흐리지 않을 사람 (가족·출연자 등). 영상마다 고르지 않아도 알아본다.


def load_registry(path) -> list[tuple[str, np.ndarray]]:
    """[(이름, 인식값)]. 파일이 없거나 깨졌으면 빈 목록."""
    import json

    try:
        items = json.loads(path.read_text(encoding="utf-8")).get("people", [])
    except (OSError, ValueError):
        return []
    found = []
    for item in items:
        vector = np.array(item.get("embedding", []), np.float32)
        if isinstance(item, dict) and str(item.get("name", "")).strip() and vector.shape == (512,):
            found.append((str(item["name"]).strip(), vector))
    return found


def save_registry(path, people: list[tuple[str, np.ndarray]]) -> None:
    from ks_studio.core.jsonio import write_json_atomic

    data = [{"name": name, "embedding": [round(float(x), 5) for x in vector]} for name, vector in people]
    write_json_atomic(path, {"version": 1, "people": data})


def registered_name(registry: list[tuple[str, np.ndarray]], embedding: np.ndarray) -> str | None:
    """이 얼굴이 등록한 사람이면 그 이름."""
    index = match([vector for _, vector in registry], embedding)
    return registry[index][0] if index is not None else None
