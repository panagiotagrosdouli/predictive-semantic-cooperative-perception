from src.core import TrackedObject
from src.opv2v import OPV2VFrame
from src.tracks import build_tracks, latest_state


def frame(timestamp: str, objects: tuple[TrackedObject, ...]) -> OPV2VFrame:
    return OPV2VFrame("scenario", "agent", timestamp, 0.0, 0.0, objects)


def test_build_tracks_groups_ids_and_sorts_time() -> None:
    frames = [
        frame("00002", (TrackedObject("a", 3, 1, 0, 0),)),
        frame("00001", (TrackedObject("a", 1, 1, 0, 0), TrackedObject("b", 5, 0, 0, 0))),
    ]
    tracks = build_tracks(frames, min_length=2)
    assert len(tracks) == 1
    assert tracks[0].object_id == "a"
    assert [point.timestamp for point in tracks[0].points] == ["00001", "00002"]


def test_latest_state_estimates_velocity() -> None:
    track = build_tracks(
        [
            frame("00001", (TrackedObject("a", 1, 2, 0, 0),)),
            frame("00002", (TrackedObject("a", 3, 5, 0, 0),)),
        ]
    )[0]
    state = latest_state(track, dt_seconds=0.5)
    assert state.x == 3
    assert state.y == 5
    assert state.vx == 4
    assert state.vy == 6
