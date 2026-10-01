# Gaze-Target Alignment (`process_gaze_mask_alignment.py`)

**Purpose**: Use segmentation masks together with gaze and world-camera timestamps to assign each gaze sample to the most likely gazed object, producing a **gaze-target time course**.

This step comes *after* segmentation: complete segmentation first with either the [SAM3 pipeline](sam3_guide.md) or the [SAM2 pipeline](sam2_guide.md), then match the resulting masks against gaze coordinates frame by frame. For each gaze point, the script compares the gaze location to the available object masks in the corresponding frame and outputs the most likely gazed object together with a confidence score. Multiple subjects and cameras can be processed in a single run, or auto-discovered from the filenames in the gaze directory.

For gaze coordinate data, we follow the format of Pupil Labs output.

## Required Inputs

- **Gaze/world-camera directory** containing files named like `{subject_id}_{camera}_gaze.csv` and `{subject_id}_{camera}_world_timestamps.csv` (or pass single files directly with `--gaze-csv` / `--world-csv`)
- **Mask directory** — either:
  - a SAM2-style mask directory containing folders named like `{subject_id}/{camera}/masks/` (or `{subject_id}_{camera}/masks/`), or
  - a **SAM3 project directory** (the script auto-detects it and reads masks from the concept/instance hierarchy)
- **Output directory** for gaze-target annotation results

## Optional Inputs

- **Blink directory** containing `{subject_id}_{camera}_blinks.csv` files if you want to label or remove gaze points during blinks
- **Ignore list** — a text file of object labels to exclude from gaze assignment (`--ignore-object-list`)

## Expected CSV Structure

- **`{subject_id}_{camera}_gaze.csv`**: one row per gaze sample, ordered by time. Required columns are `timestamp [ns]`, `gaze x [px]`, and `gaze y [px]`. Additional columns are allowed and are preserved in the merged/output tables.
- **`{subject_id}_{camera}_world_timestamps.csv`**: one row per world-camera frame, ordered by time. Required column is `timestamp [ns]`. Additional columns are allowed, but the script rebuilds `frame_idx` and `frame_timestamp` from this file during alignment.
- **`{subject_id}_{camera}_blinks.csv`**: used only with `--blink-dir`. The code expects `start timestamp [ns]`, `end timestamp [ns]`, and `blink id`, because those columns are used to mark whether a gaze sample falls inside a blink interval.

**Minimum column examples**:
```csv
# {subject_id}_{camera}_gaze.csv
timestamp [ns],gaze x [px],gaze y [px]
1000000000,640.5,360.2
1000033333,642.1,361.0
```

```csv
# {subject_id}_{camera}_world_timestamps.csv
timestamp [ns]
999999000
1000030000
1000063000
```

```csv
# {subject_id}_{camera}_blinks.csv
blink id,start timestamp [ns],end timestamp [ns]
0,1000200000,1000400000
1,1001000000,1001200000
```

## Usage

```bash
# Process one subject/camera pair
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27 \
  --camera-id child

# Process multiple subjects and cameras in one run
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27,28 \
  --camera-id child,parent

# Auto-discover all subjects and cameras from CSV filenames
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir

# Use a SAM3 project directory as the mask source
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/sam3_project_dir \
  /path/to/output_dir \
  --subject-id 27 --camera-id child

# Point at single gaze/world CSV files directly (no naming convention needed)
python process_gaze_mask_alignment.py \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --gaze-csv /path/to/my_gaze.csv \
  --world-csv /path/to/my_world_timestamps.csv \
  --subject-id 27 --camera-id child

# Remove gaze points during blinks
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27 \
  --camera-id child \
  --blink-dir /path/to/blink_data
```

## Options

| Option | Description |
|--------|-------------|
| `gaze_world_dir` | Directory with `{subject}_{camera}_gaze.csv` and `{subject}_{camera}_world_timestamps.csv`. Optional when `--gaze-csv` and `--world-csv` are both given. |
| `mask_dir` | Directory with `{subject}/{camera}/masks/` subfolders, or a SAM3 project directory |
| `output_dir` | Directory to save output files |
| `--subject-id` | Subject ID(s), e.g. `27` or `27,28`. Auto-discovered if omitted. |
| `--camera-id` | Camera ID(s), e.g. `child` or `child,parent`. Auto-discovered if omitted. |
| `--gaze-csv` / `--world-csv` | Direct paths to a single gaze CSV / world-timestamps CSV, bypassing the naming convention (use together, with exactly one subject/camera pair) |
| `--blink-dir` | Directory with `{subject}_{camera}_blinks.csv` files for blink labeling/removal |
| `--ignore-object-list` | Text file with object labels to ignore, one per line (`#` comments allowed). SAM2: names, IDs, or mask filenames; SAM3: `concept_name`, `user_name`, or `concept_user_name` |
| `--gaze-radius` | Radius (pixels) of the disk around each gaze point used for mask-overlap confidence (default 20). Larger = more tolerant of tracking noise, but blurs nearby objects |
| `--gaze-confidence-threshold` | Minimum overlap confidence to assign a gazed object (default 0.5). The raw confidence is always recorded unthresholded |
| `--pipeline` | Which masks to score for a SAM3 project that was handed off to SAM2: `auto` (default; SAM3 if the mask dir has `project.json`, else SAM2), `sam3` (SAM3 instances only), `sam2` (`sam2_results/masks/` only), `both` (SAM3 instances plus objects added natively in SAM2; SAM3-derived, covered and retired SAM2 ids are skipped to avoid double counting). In `both`, use `gazed_object` (the name) to identify objects: SAM3 and SAM2 ids are separate number ranges, so a `gazed_object_id` such as `id5` alone is ambiguous |
| `--num-workers` | Number of subject-camera pairs processed in parallel (default 1) |
| `--within-job-workers` | Worker processes scoring frames within one pair (default 4; 1 = sequential). Automatically reduced so that `--num-workers` x this value does not exceed the available CPUs. Results are identical for any value |
| `--recompute` | Force recomputation from masks even if a cached probabilities `.pkl` exists. Without it, the cache is reused only if radius, pipeline, ignore list, mask source, gaze and world CSVs and blink setting match those recorded next to it; otherwise it is recomputed automatically |
| `--skip-figures` | Skip trajectory and confidence heatmap figures |
| `--vector-figures` | Keep the per-sample trajectory and heatmap as vector graphics in the PDFs. By default they are rasterized at 300 dpi inside the PDF (axes and text stay vector), which keeps PDFs small and fast for long recordings |
| `--category-sort` | Row ordering in figures: `first_seen` (default) or `frequency` |
| `--start-plot-time` / `--end-plot-time` | Time window (seconds) for method-figure plots |
| `--log-path` | Path for the log file (default: `{output_dir}/gaze_object.log`) |

## Output Files

- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gazed_object.csv`** — gaze samples with assigned object labels and confidence
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_object_probabilities.pkl`** — per-gaze probabilities for all available masks (also serves as a cache for re-runs)
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_object_probabilities.settings.json`** — settings the `.pkl` was computed with, used to decide whether the cache can be reused
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_blink_labeled.csv`** — blink-labeled gaze data when `--blink-dir` is used
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_blink_removed.csv`** — blink-removed gaze data when `--blink-dir` is used
- **`output_dir/{subject_id}/{camera}/figures/`** — trajectory plots and confidence heatmaps (PNG and PDF)

With `--ignore-object-list`, the CSV, `.pkl` and `.settings.json` names get an `_excluding_ignored_objects` suffix.
- **`output_dir/gaze_object.log`** — processing log (or path set by `--log-path`)

The main output CSV keeps the original gaze columns and any extra gaze metadata, then adds:

- `frame_idx`: world-camera frame index matched to the gaze sample
- `frame_timestamp`: timestamp of the matched world-camera frame
- `in_blink`, `blink id`: added only when `--blink-dir` is used
- `gazed_object_id`: object/instance ID of the winning mask
- `gazed_object`: object label of the winning mask
- `gazed_object_confidence`: fraction of pixels inside the gaze-radius disk that overlap the winning object mask

## Mask Formats

For SAM2 mask folders the script reads the per-frame **NPZ** bundle (`masks_f000000.npz`) when present, plus any per-object **PNG** or **NPZ** masks (`mask_f000000_<name>_id<N>.png/.npz`) of objects not already in the bundle. A SAM3 project handed off to SAM2 mixes these in `sam2_results/masks/`: SAM3 masks are hard-linked with their original extension and covered-object unions are per-object files. Both pipelines can produce either format (`--mask-format npz`); see the pipeline guides for details.

## Citation

If you use this tool in your research, please cite our paper ([arXiv:2605.22962](https://arxiv.org/abs/2605.22962)) — full reference and BibTeX in the [main README](../README.md#citation).