"""Load eye-gaze CSVs and map gaze samples to video frame indices for UI overlay.

CSV format matches ``gazed_object.py`` / ``process_gaze_mask_alignment.py``:

* gaze CSV  -- columns ``timestamp [ns]``, ``gaze x [px]``, ``gaze y [px]``
  (one row per gaze sample, typically ~200 Hz)
* world CSV -- column ``timestamp [ns]``, one row per world-camera video frame,
  in temporal order (row order == frame index)

Each gaze sample is aligned to the most recent world-camera frame at or before
its timestamp (``pandas.merge_asof`` with ``direction="backward"``), exactly like
the alignment step in ``process_gaze_mask_alignment.load_gaze_data``.

Gaze pixel coordinates are in world-camera resolution, so they map directly onto
the video frame as displayed by the UI (before any canvas scaling).
"""

from __future__ import annotations

import numpy as np

# Gold -- deliberately distinct in hue *and* shape from the annotation / refinement
# points (which are red/green circles). Arrays in both UIs are RGB.
GAZE_COLOR = (255, 200, 0)
_OUTLINE = (0, 0, 0)


def load_gaze_csv(gaze_csv_path: str, world_csv_path: str) -> dict[int, list[tuple[float, float]]]:
    """Return ``{frame_idx: [(x, y), ...]}`` mapping video frames to gaze samples.

    Raises ``ValueError`` if either CSV is missing required columns.
    """
    import pandas as pd

    gaze = pd.read_csv(gaze_csv_path)
    world = pd.read_csv(world_csv_path)

    required = {"timestamp [ns]", "gaze x [px]", "gaze y [px]"}
    missing = sorted(required.difference(gaze.columns))
    if missing:
        raise ValueError(f"Gaze CSV is missing required column(s): {missing}")
    if "timestamp [ns]" not in world.columns:
        raise ValueError("World-timestamps CSV is missing required column 'timestamp [ns]'")

    # Frame index == original row order of the world CSV (matches the aligner).
    world = world.loc[:, ["timestamp [ns]"]].copy()
    world["_frame_idx"] = np.arange(len(world))
    world = world.sort_values("timestamp [ns]")

    gaze = gaze.loc[:, ["timestamp [ns]", "gaze x [px]", "gaze y [px]"]].copy()
    gaze = gaze.dropna(subset=["timestamp [ns]", "gaze x [px]", "gaze y [px]"])
    gaze = gaze.sort_values("timestamp [ns]")

    aligned = pd.merge_asof(
        gaze,
        world,
        on="timestamp [ns]",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = aligned.dropna(subset=["_frame_idx"])

    out: dict[int, list[tuple[float, float]]] = {}
    for frame_idx, sub in aligned.groupby("_frame_idx", sort=True):
        out[int(frame_idx)] = list(
            zip(sub["gaze x [px]"].astype(float), sub["gaze y [px]"].astype(float))
        )
    return out


def draw_gaze_marker(frame_rgb, points, scale_x: float = 1.0, scale_y: float = 1.0,
                     size: int = 13) -> None:
    """Draw gaze samples for one frame onto ``frame_rgb`` (modified in place).

    Older samples in the frame are small dots joined by a thin trail; the most
    recent sample gets a diamond reticle with gapped outward ticks.
    """
    import cv2

    if not points:
        return
    pts = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points]

    # Trail: connecting line + small dots for the non-latest samples.
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(frame_rgb, [arr], False, _OUTLINE, 3, cv2.LINE_AA)
        cv2.polylines(frame_rgb, [arr], False, GAZE_COLOR, 1, cv2.LINE_AA)
    for p in pts[:-1]:
        cv2.circle(frame_rgb, p, 3, _OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(frame_rgb, p, 2, GAZE_COLOR, -1, cv2.LINE_AA)

    cx, cy = pts[-1]
    diamond = np.array(
        [[cx, cy - size], [cx + size, cy], [cx, cy + size], [cx - size, cy]],
        dtype=np.int32,
    )
    cv2.polylines(frame_rgb, [diamond], True, _OUTLINE, 4, cv2.LINE_AA)
    cv2.polylines(frame_rgb, [diamond], True, GAZE_COLOR, 2, cv2.LINE_AA)
    cv2.circle(frame_rgb, (cx, cy), 2, _OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(frame_rgb, (cx, cy), 1, GAZE_COLOR, -1, cv2.LINE_AA)

    inner, outer = size + 3, size + 10
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p1 = (cx + dx * inner, cy + dy * inner)
        p2 = (cx + dx * outer, cy + dy * outer)
        cv2.line(frame_rgb, p1, p2, _OUTLINE, 4, cv2.LINE_AA)
        cv2.line(frame_rgb, p1, p2, GAZE_COLOR, 2, cv2.LINE_AA)


def summarize(gaze_by_frame: dict[int, list[tuple[float, float]]]) -> str:
    """Short human-readable summary for a message box."""
    if not gaze_by_frame:
        return "No gaze samples mapped to any frame."
    frames = sorted(gaze_by_frame)
    n_samples = sum(len(v) for v in gaze_by_frame.values())
    return (
        f"Mapped {n_samples} gaze sample(s) across {len(frames)} frame(s) "
        f"(frame {frames[0]}-{frames[-1]}).\n\n"
        "Gaze x/y are assumed to be in this video's pixel resolution, and frame "
        "indices assume the loaded video matches the world-timestamps CSV."
    )
