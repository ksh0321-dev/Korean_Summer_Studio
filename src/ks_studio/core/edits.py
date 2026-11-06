"""컷편집 연산. 전부 Project를 받아 새 Project를 돌려주는 순수 함수다 (원본은 바뀌지 않는다).

할 수 없는 편집(위 줄에서 겹치게 놓기 등)은 받은 project를 그대로 돌려준다 → 호출하는 쪽은 `is`로 거부를 알 수 있다.
"""

from collections.abc import Callable
from dataclasses import replace

from ks_studio.core.project import MAIN_TRACK, MIN_VIDEO_TRACKS, Clip, Placed, Placement, Project, new_id


def insert_clip(project: Project, index: int, clip: Clip) -> Project:
    clips = list(project.clips)
    clips.insert(max(0, min(index, len(clips))), clip)
    return replace(project, clips=tuple(clips))


def split(project: Project, frame: int, clip_id: str | None = None) -> Project:
    """frame 위치에서 클립을 둘로 나눈다. clip_id가 없으면 frame에 걸린 모든 줄의 클립을 나눈다.

    클립 시작·끝이나 클립 밖이면 그 클립은 그대로.
    """
    targets = [p for p in project.placed() if p.start < frame < p.end and clip_id in (None, p.clip.id)]
    for target in targets:
        project = _split_one(project, target.clip.id, frame - target.start)
    return project


def _split_one(project: Project, clip_id: str, offset: int) -> Project:
    placed = project.find(clip_id)
    clip = placed.clip
    cut = project.source_frame(clip, offset)
    if not clip.in_frame < cut < clip.out_frame:
        return project
    # 페이드 인은 앞 조각에, 페이드 아웃은 뒤 조각에만 남는다 (자른 자리에서 소리가 줄었다 커지지 않게)
    left = replace(clip, out_frame=cut, fade_out_ms=0)
    right = replace(clip, in_frame=cut, id=new_id(), fade_in_ms=0, transition_ms=0)  # 디졸브도 앞 조각에만
    if placed.track == MAIN_TRACK:
        index = project.index_of(clip_id)
        return replace(project, clips=project.clips[:index] + (left, right) + project.clips[index + 1 :])
    # 원본 fps가 다르면 반올림 때문에 두 조각 길이 합이 1프레임 늘 수 있다 → 옆 클립과 겹치면 나누지 않는다
    right_start = placed.start + project.length(left)
    rest = _without(project, clip_id)
    if not rest.fits(placed.track, right_start, project.length(right)):
        return project
    pieces = (Placement(left, placed.track, placed.start), Placement(right, placed.track, right_start))
    return replace(rest, overlays=rest.overlays + pieces)


def trim_to(project: Project, frame: int, side: str, clip_id: str) -> Project:
    """클립을 frame에서 나눠 앞(side="left") 또는 뒤(side="right")를 버린다 (Q/W). V1이면 빈자리가 닫힌다."""
    placed = project.find(clip_id)
    if placed is None or not placed.start < frame < placed.end:
        return project
    cut = _split_one(project, clip_id, frame - placed.start)
    if cut is project:
        return project
    right_id = next(p.clip.id for p in cut.placed() if project.find(p.clip.id) is None)  # 뒤 조각만 새 id
    return delete_clip(cut, clip_id if side == "left" else right_id)


def delete_range(project: Project, start: int, end: int) -> Project:
    """모든 줄에서 [start, end) 구간을 잘라내고 뒤를 당긴다 (I/O 구간 지우기).

    V1은 자석식이라 저절로 당겨지고, 위 줄은 구간 안 조각을 지운 뒤 구간 뒤 클립을 구간 길이만큼 당긴다.
    """
    if not 0 <= start < end:
        return project
    cut = split(split(project, start), end)
    inside = {p.clip.id for p in cut.placed() if start <= p.start and p.end <= end}
    clips = tuple(c for c in cut.clips if c.id not in inside)
    overlays = tuple(
        replace(o, start=o.start - (end - start)) if o.start >= end else o for o in cut.overlays if o.clip.id not in inside
    )
    markers = tuple(m - (end - start) if m >= end else m for m in cut.markers if not start <= m < end)
    return replace(cut, clips=clips, overlays=overlays, markers=markers)


def delete_ranges(project: Project, ranges: list[tuple[int, int]]) -> Project:
    """여러 구간을 한 번에 잘라낸다 (무음 자르기). 뒤에서부터 지워야 앞 구간 위치가 바뀌지 않는다."""
    for start, end in sorted(ranges, reverse=True):
        project = delete_range(project, start, end)
    return project


CLIP_PROPS = {"volume_db", "fade_in_ms", "fade_out_ms", "scale", "pos_x", "pos_y", "opacity", "transition_ms"}


def set_clip_props(project: Project, clip_ids: set[str], **changes) -> Project:
    """선택한 클립들의 소리·화면 속성(볼륨·페이드·크기·위치·투명도·디졸브)을 바꾼다."""
    assert set(changes) <= CLIP_PROPS, changes
    for clip_id in clip_ids:
        project = _replace_clip(project, clip_id, lambda c: replace(c, **changes))
    return project


def edit_points(project: Project) -> list[int]:
    """재생 헤드를 옮길 편집점: 모든 줄의 클립 시작·끝 (↑/↓)."""
    return sorted({0, project.duration, *project.markers, *(e for p in project.placed() for e in (p.start, p.end))})


def delete_clip(project: Project, clip_id: str) -> Project:
    """V1이면 자석식이라 뒤 클립이 자동으로 당겨진다. 위 줄이면 빈자리가 남는다."""
    return _without(project, clip_id)


def _without(project: Project, clip_id: str) -> Project:
    return replace(
        project,
        clips=tuple(c for c in project.clips if c.id != clip_id),
        overlays=tuple(o for o in project.overlays if o.clip.id != clip_id),
    )


def place_overlay(project: Project, clip: Clip, track: int, start: int) -> Project:
    """위 줄 track의 start에 놓는다. 다른 클립과 겹치거나 0보다 앞이면 거부."""
    if not project.fits(track, start, project.length(clip), ignore=clip.id):
        return project
    return replace(project, overlays=project.overlays + (Placement(clip, track, start),))


def move_to(project: Project, clip_id: str, track: int, position: int) -> Project:
    """클립을 다른 자리로 옮긴다. track이 V1이면 position = 끼워 넣을 순서(빼낸 뒤 기준), 위 줄이면 시작 프레임."""
    clip = project.find(clip_id).clip
    rest = _without(project, clip_id)
    if track == MAIN_TRACK:
        return insert_clip(rest, position, clip)
    moved = place_overlay(rest, clip, track, position)
    return project if moved is rest else moved


def move_clip(project: Project, clip_id: str, index: int) -> Project:
    """index = 옮긴 뒤 이 클립이 놓일 위치."""
    clip = project.clips[project.index_of(clip_id)]
    return insert_clip(delete_clip(project, clip_id), index, clip)


def trim_clip(project: Project, clip_id: str, in_frame: int, out_frame: int) -> Project:
    """원본 길이 안으로 자르는 건 호출하는 쪽 책임. 최소 1프레임은 남긴다.

    위 줄 클립은 앞쪽(in)을 바꾸면 끝 위치를, 뒤쪽(out)을 바꾸면 시작 위치를 고정한다. 겹치게 되면 거부.
    """
    in_frame = max(0, in_frame)
    out_frame = max(in_frame + 1, out_frame)
    placed = project.find(clip_id)
    trimmed = replace(placed.clip, in_frame=in_frame, out_frame=out_frame)
    if placed.track == MAIN_TRACK:
        return _replace_clip(project, clip_id, lambda _c: trimmed)
    start = placed.end - project.length(trimmed) if in_frame != placed.clip.in_frame else placed.start
    rest = _without(project, clip_id)
    if not rest.fits(placed.track, start, project.length(trimmed)):
        return project
    return replace(project, overlays=tuple(Placement(trimmed, o.track, start) if o.clip.id == clip_id else o for o in project.overlays))


def set_audio_offset(project: Project, clip_id: str, offset_ms: int) -> Project:
    return _replace_clip(project, clip_id, lambda c: replace(c, audio_offset_ms=offset_ms))


def _replace_clip(project: Project, clip_id: str, change: Callable[[Clip], Clip]) -> Project:
    """줄과 위치는 그대로 두고 클립 내용만 바꾼다."""
    return replace(
        project,
        clips=tuple(change(c) if c.id == clip_id else c for c in project.clips),
        overlays=tuple(replace(o, clip=change(o.clip)) if o.clip.id == clip_id else o for o in project.overlays),
    )


def toggle_track(project: Project, track: int, kind: str) -> Project:
    """줄의 소리 끄기(kind="mute") 또는 화면 숨기기(kind="hide")를 켜고 끈다. 둘은 서로 영향을 주지 않는다."""
    field = {"mute": "muted", "hide": "hidden"}[kind]
    tracks = set(getattr(project, field)) ^ {track}
    return replace(project, **{field: tuple(sorted(tracks))})


def add_track(project: Project, at: int) -> Project:
    """영상 줄을 at번으로 끼워 넣는다 (2 ~ 맨 위+1). 원래 at번 이상이던 줄은 클립·👁·🔊와 함께 하나씩 위로 밀린다."""
    if not MAIN_TRACK < at <= project.video_tracks + 1:
        return project

    def up(track: int) -> int:
        return track + 1 if track >= at else track

    return replace(
        project,
        video_tracks=project.video_tracks + 1,
        overlays=tuple(replace(o, track=up(o.track)) for o in project.overlays),
        muted=tuple(up(t) for t in project.muted),
        hidden=tuple(up(t) for t in project.hidden),
    )


def delete_track(project: Project, track: int) -> Project:
    """영상 줄과 그 줄의 클립을 지운다. 위 줄은 하나씩 내려온다. V1이거나 최소 줄 수면 거부."""
    if not MAIN_TRACK < track <= project.video_tracks or project.video_tracks <= MIN_VIDEO_TRACKS:
        return project

    def down(t: int) -> int:
        return t - 1 if t > track else t

    return replace(
        project,
        video_tracks=project.video_tracks - 1,
        overlays=tuple(replace(o, track=down(o.track)) for o in project.overlays if o.track != track),
        muted=tuple(down(t) for t in project.muted if t != track),
        hidden=tuple(down(t) for t in project.hidden if t != track),
    )


def set_fps(project: Project, fps: int) -> Project:
    """타임라인 fps를 바꾼다. 클립은 원본 기준이라 그대로 두고, 렌더할 때 변환된다.

    위 줄 클립의 시작 위치와 마커는 타임라인 프레임이라 같은 시각이 되도록 바꾼다.
    """
    overlays = tuple(replace(o, start=round(o.start * fps / project.fps)) for o in project.overlays)
    markers = tuple(round(m * fps / project.fps) for m in project.markers)
    return replace(project, fps=fps, overlays=overlays, markers=markers)


def toggle_marker(project: Project, frame: int) -> Project:
    """재생 헤드에 마커가 있으면 지우고, 없으면 놓는다."""
    markers = set(project.markers) ^ {frame}
    return replace(project, markers=tuple(sorted(markers)))


def toggle_transition(project: Project, frame: int, ms: int) -> tuple[Project, str | None]:
    """frame에서 가장 가까운 V1 클립 경계의 디졸브를 넣거나 뺀다. (새 프로젝트, 바뀐 클립 id — 경계가 없으면 None)."""
    main = project.main_placed()
    if len(main) < 2:
        return project, None
    target = min(main[1:], key=lambda p: abs(p.start - frame))
    clip = target.clip
    return set_clip_props(project, {clip.id}, transition_ms=0 if clip.transition_ms else ms), clip.id


def set_linked(project: Project, linked: bool) -> Project:
    return replace(project, linked=linked)


def follow_main(before: Project, after: Project) -> Project:
    """연동: V1이 바뀐 편집(before → after) 뒤에 위 줄 클립이 붙어 있던 V1 장면을 따라가게 한다.

    - 위 줄 클립은 '시작 지점 아래의 V1 클립과 그 원본 시각'에 붙어 있다 (편집할 때마다 새로 계산, 따로 저장하지 않는다).
    - 그 V1 클립이 지워졌으면 같이 지운다. V1 끝 뒤에 놓인 클립은 V1 끝을 따라간다.
    - 붙어 있던 V1 클립이 위 줄로 옮겨졌거나, 편집이 직접 옮긴 위 줄 클립은 그대로 둔다.
    - 따라가다 같은 줄의 다른 클립과 겹치면 바로 위 빈 줄로 올린다. 빈 줄이 없으면 맨 위에 줄을 만든다.
    """
    if not after.linked or before.fps != after.fps or before.clips == after.clips:
        return after
    old_main = [p for p in before.placed() if p.track == MAIN_TRACK]
    old_ids = {p.clip.id for p in old_main}
    new_main = [p for p in after.placed() if p.track == MAIN_TRACK]
    lifted = {o.clip.id for o in after.overlays}
    untouched = set(before.overlays)
    fixed, followers = [], []
    for o in after.overlays:
        if o not in untouched:
            fixed.append(o)
            continue
        anchor = next((p for p in old_main if p.start <= o.start < p.end), None)
        if anchor is None:  # V1 끝 뒤
            followers.append(replace(o, start=o.start + after.main_duration - before.main_duration))
            continue
        if anchor.clip.id in lifted:
            fixed.append(o)
            continue
        seconds = before.source_time(anchor.clip, o.start - anchor.start)
        target = _main_showing(new_main, anchor.clip, seconds, old_ids)
        if target is None:  # 붙어 있던 V1 클립이 지워짐
            continue
        # 앞쪽을 잘라 그 장면이 없어졌으면 클립 맨 앞에 붙인다
        local = max(0, round((seconds - target.clip.in_frame / target.clip.fps) * after.fps))
        followers.append(replace(o, start=target.start + local))
    # 움직이지 않은 클립이 자리를 먼저 차지하고, 움직인 클립이 겹치면 위로 올라간다
    followers.sort(key=lambda o: (o not in untouched, o.start, o.track))
    project = replace(after, overlays=tuple(fixed))
    for o in followers:
        project = _place_free(project, o)
    return project


def _place_free(project: Project, placement: Placement) -> Project:
    """위 줄에 놓되, 겹치면 바로 위 빈 줄로 올린다. 빈 줄이 없으면 맨 위에 줄을 만든다."""
    length, track = project.length(placement.clip), max(placement.track, MAIN_TRACK + 1)
    while not project.fits(track, placement.start, length):
        track += 1
        if track > project.video_tracks:
            project = replace(project, video_tracks=track)
    return replace(project, overlays=project.overlays + (replace(placement, track=track),))


def _main_index_at(project: Project, frame: int) -> int:
    """frame에 끼워 넣을 V1 순서: 클립 경계면 그 자리, 클립 한가운데면 그 클립 뒤."""
    found = project.locate(frame)
    if found is None:
        return len(project.clips)
    return found[0] if found[1] == 0 else found[0] + 1


def delete_clips(project: Project, clip_ids: set[str]) -> Project:
    """여러 클립을 한 번에 지운다. V1은 빈자리가 닫힌다."""
    return replace(
        project,
        clips=tuple(c for c in project.clips if c.id not in clip_ids),
        overlays=tuple(o for o in project.overlays if o.clip.id not in clip_ids),
    )


def paste(project: Project, items: list[Placed], frame: int) -> Project:
    """복사해 둔 클립들을 frame에 붙여 넣는다 (새 id).

    V1 클립은 frame의 V1 편집점에 순서대로 끼워 넣고, 위 줄 클립은 서로 간격을 유지한 채
    frame부터 놓는다 (겹치면 위 빈 줄로).
    """
    if not items:
        return project
    base = min(p.start for p in items)
    index = _main_index_at(project, frame)
    for item in sorted((p for p in items if p.track == MAIN_TRACK), key=lambda p: p.start):
        project = insert_clip(project, index, replace(item.clip, id=new_id()))
        index += 1
    for item in sorted((p for p in items if p.track != MAIN_TRACK), key=lambda p: (p.start, p.track)):
        copy = Placement(replace(item.clip, id=new_id()), item.track, frame + item.start - base)
        project = _place_free(project, copy)
    return project


def duplicate(project: Project, clip_ids: set[str]) -> Project:
    """선택한 클립마다 바로 뒤에 복제본을 놓는다 (V1은 끼워 넣기, 위 줄은 끝에 이어서. 겹치면 위 빈 줄로)."""
    for placed in [p for p in project.placed() if p.clip.id in clip_ids]:
        copy = replace(placed.clip, id=new_id())
        if placed.track == MAIN_TRACK:
            project = insert_clip(project, project.index_of(placed.clip.id) + 1, copy)
        else:
            project = _place_free(project, Placement(copy, placed.track, placed.end))
    return project


def _main_showing(new_main: list[Placed], old_clip: Clip, seconds: float, old_ids: set[str]) -> Placed | None:
    """편집 뒤 V1에서 old_clip의 원본 시각 seconds를 보여 주는 클립.

    분할하면 뒤 조각은 새 id라서, 같은 원본을 쓰는 새 클립도 후보로 본다. 그 시각이 잘려 나갔으면 같은 id의 클립.
    """
    candidates = [p for p in new_main if p.clip.id == old_clip.id]
    candidates += [p for p in new_main if p.clip.id not in old_ids and p.clip.source == old_clip.source]
    for p in candidates:
        if p.clip.in_frame / p.clip.fps <= seconds + 1e-9 and seconds < p.clip.out_frame / p.clip.fps - 1e-9:
            return p
    return next((p for p in candidates if p.clip.id == old_clip.id), None)


def add_media(project: Project, source: str) -> Project:
    if source in project.media:
        return project
    return replace(project, media=project.media + (source,))


def remove_media(project: Project, source: str) -> Project:
    """보관함에서만 뺀다. 타임라인에 이미 놓인 클립은 그대로 둔다."""
    return replace(project, media=tuple(m for m in project.media if m != source))


def clips_using(project: Project, source: str) -> set[str]:
    """source를 쓰는 타임라인 클립 id (모든 줄)."""
    return {c.id for c in project.clips if c.source == source} | {o.clip.id for o in project.overlays if o.clip.source == source}


def remove_from_workspace(project: Project, source: str, recording: bool) -> Project:
    """워크스페이스 목록에서 빼고 그 영상을 쓰는 타임라인 클립도 지운다.

    녹화본은 recordings/에서 저절로 보이므로 '뺀 녹화본'으로 적어 둔다 (원본 파일은 그대로). 가져온 미디어는 목록에서 지운다.
    """
    project = delete_clips(project, clips_using(project, source))
    if recording:
        return replace(project, removed=tuple(dict.fromkeys((*project.removed, source))))
    return remove_media(project, source)
