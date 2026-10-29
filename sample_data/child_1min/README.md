# Sample data: `child_1min`

One minute (1,801 frames, 1600x1200 @ 30 fps) of egocentric video from a Pupil Labs
Neon eye tracker worn by a child during free play with a parent, with the matching
gaze data. Use it to try the SAM3 pipeline and the gaze-to-object alignment end to end.

## Files

| File | Purpose |
|---|---|
| `child_1min.mp4` | Scene video (no audio), **not stored in git**: download it from the release (see below). Frame *i* = row *i* of `world_timestamps.csv`. |
| `world_timestamps.csv`, `gaze.csv`, `fixations.csv`, `blinks.csv`, `scene_camera.json` | Pupil Labs Neon "Timeseries Data" format. Timestamps are re-based (not Unix time); ids are placeholders (de-identified). |
| `concepts.json` | 13 text-prompt concepts for `sam3_process.py --concepts`. |
| `concepts_vocabulary.json` | Suggested instance names per concept, for `sam3_ui.py` (Vocabulary -> Load Vocabulary File). |

`hand` is one concept (not separate left/right concepts), so SAM3 keeps the hand masks
mutually exclusive. Name the detected instances afterwards (e.g. `parent_left_hand`).

## Download the video

The 27 MB video is a release asset, not part of the repository. From the repository root:

```bash
curl -L -o sample_data/child_1min/child_1min.mp4 \
    https://github.com/CaiLab-neuro/SegmentVideo4Gaze/releases/download/v0.2.0/child_1min.mp4
```

## Suggested commands

Run from the repository root. Detection of all 13 concepts takes a while; to try
things quickly, process a few with `--filter`:

```bash
# 1. Detect + propagate a few concepts (drop --filter to run all of them)
python sam3_process.py \
    --video   sample_data/child_1min/child_1min.mp4 \
    --project sample_data/child_1min_project \
    --concepts sample_data/child_1min/concepts.json \
    --filter head,hand,table \
    --mask-format npz

# 2. Review, rename and refine instances in the UI
python sam3_ui.py sample_data/child_1min_project
#    Vocabulary -> Load Vocabulary File -> sample_data/child_1min/concepts_vocabulary.json

# 3. Replay refinement clicks made in the UI
python sam3_process.py --project sample_data/child_1min_project --refine

# 4. Which object is gazed at in each frame
python process_gaze_mask_alignment.py \
    --subject-id 1 --camera-id child \
    --gaze-csv  sample_data/child_1min/gaze.csv \
    --world-csv sample_data/child_1min/world_timestamps.csv \
    --blink-csv sample_data/child_1min/blinks.csv \
    sample_data/child_1min_project \
    sample_data/child_1min_gaze
```

Frames where no selected concept is visible have no mask, so they are reported as
"no object". Run more concepts (or all 13) for fuller coverage.

Check `python process_gaze_mask_alignment.py --help` for the gaze options, and
`docs/` for the full guides.
