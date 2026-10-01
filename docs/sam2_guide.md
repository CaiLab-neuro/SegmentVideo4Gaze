# SAM2 Pipeline Guide — Point-Based Segmentation

The SAM2 pipeline segments objects you mark yourself with **click points**: you click on each object in one or a few frames, and SAM2 propagates the masks through the whole video. Compared with the [SAM3 pipeline](sam3_guide.md), it requires more manual annotation but gives you exact control over *which* objects are tracked — useful when a concept is hard to describe by text, or when a scene contains many instances of a concept and you only need a few.

Two entry points:

- **`sam2_ui.py`** — the graphical interface for annotating, segmenting, and refining
- **`sam2_process.py`** — the command-line tool for batch segmentation (recommended for long videos)

## Using the SAM2 Video UI (`sam2_ui.py`)

### Launch the app

```bash
python sam2_ui.py
```

<!-- TODO: screenshot of the main window -->

### Initial setup

1. Load a video file
2. Create an object list — add each object you want to track and give it a name

### Annotation

1. Navigate to a frame where the object is clearly visible
2. Add annotation points on the object (positive clicks on the object, negative clicks on regions to exclude)
3. Repeat for additional objects or frames as needed
4. Export annotations as a JSON file

<!-- TODO: screenshot of annotation points on a frame -->

### Segmentation

Select a model from the dropdown (larger models such as Base+ or Large generally produce better masks but are slower; only downloaded checkpoints appear), then either segment within the UI or use the processing script (more details in the next section):

```bash
python sam2_process.py annotations.json video.mp4
```

For videos longer than a few hundred frames, using the processing script is recommended as the UI can be slow on long videos.

### Refinement

1. Import segmentation results (this also imports the original annotations)
2. Use quality metrics to identify segments in the video where segmentation is poor
3. Add or adjust annotation points and re-segment the frame to verify
4. Re-export the updated annotations and re-run the processing script, or run segmentation in refinement mode for a range of frames within the UI

### Mouse Controls

| Action | Effect |
|--------|--------|
| Left click on canvas | Add a **positive** point for the current object (or remove a point, in removal mode) |
| Right click on canvas | Add a **negative** point for the current object (or remove a point, in removal mode) |
| Mouse wheel | Previous/next frame |
| Shift + mouse wheel | Jump to previous/next **annotated** frame |
| Ctrl/Cmd + mouse wheel | Zoom in/out, centered on the cursor |

### Keyboard Shortcuts

| Key | Effect |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `↑` / `↓` | Previous / next object in the object list |
| `Page Up` / `Page Down` | Jump to previous / next annotated frame |
| `Home` | Jump to the first object in the list |
| `End` | Jump to the last named (non-generic) object in the list |
| `Ctrl/Cmd + Z` | Undo last point |
| `Ctrl+Y` / `Cmd+Shift+Z` / `Cmd+Y` | Redo last point |
| `Ctrl/Cmd + +` / `Ctrl/Cmd + -` | Zoom in/out, centered on the cursor |
| `r` | Toggle point removal mode (click a point to delete it, instead of adding one) |
| `space` | Play / pause |

Arrow-key and letter shortcuts are ignored while a text field (e.g. an object name entry) has focus, so typing names doesn't accidentally move frames or objects.

## Processing Script (`sam2_process.py`)

**Purpose**: Process an annotation JSON from the SAM2 Video UI to generate the segmented video and masks.

> This script was previously named `process_annotations.py`; the old name still works and forwards to `sam2_process.py`.

**Usage**:

```bash
# Basic usage (uses SAM2.1 Base+ by default)
python sam2_process.py annotations.json video.mp4

# Use SAM2.1 Large model
python sam2_process.py annotations.json video.mp4 --model sam2.1-large

# With custom output directory
python sam2_process.py annotations.json video.mp4 --output-dir results/

# With custom settings
python sam2_process.py annotations.json video.mp4 \
  --output-dir results/ \
  --fps 30 --opacity 0.4

# Re-render output video from existing masks (no re-segmentation)
python sam2_process.py annotations.json video.mp4 \
  --output-dir results/ --video-only --opacity 0.6

# Re-segment only objects that were updated in the annotation file
python sam2_process.py annotations.json video.mp4 \
  --output-dir results/ --only-updated

# Persistent frame directory (avoids re-extracting frames on repeated runs)
python sam2_process.py annotations.json video.mp4 \
  --frame-dir /path/to/persistent/frames
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | `sam2_output` |
| `--model` | SAM2 model to use | auto (best for GPU) |
| `--fps` | Output video FPS | 30.0 |
| `--opacity` | Mask overlay opacity (0.0–1.0) | 0.4 |
| `--video-only` | Re-render video from existing masks, skip segmentation | — |
| `--only-updated` | Re-segment only objects marked as updated; reuse other masks | — |
| `--prev-results` | Directory with previous masks to reuse (with `--only-updated`) | output dir |
| `--offload-to-cpu` | Offload video frames to CPU to reduce GPU memory usage | — |
| `--frame-dir` | Persistent directory for extracted frames (avoids re-extraction) | temp dir |
| `--mask-format` | Mask storage format: `png` or `npz` (compressed, faster I/O) | `png` |
| `--no-bfloat16` | Disable bfloat16 autocast (relevant on pre-Ampere GPUs) | — |

### Available models

- **SAM2.1** (recommended): tiny (156 MB), small (184 MB), base+ (324 MB), large (898 MB)
- **SAM2** (legacy): same sizes, older version

Only models whose checkpoints were downloaded during installation are available; run `install.py` again to add more.

## Output Files

After processing, you'll get:

- **`output_dir/masks/`** — individual mask images (PNG files, or per-frame NPZ bundles with `--mask-format npz`)
- **`output_dir/segmented_video.mp4`** — video with colored mask overlays
- **`output_dir/processing_metadata.json`** — processing statistics

These outputs feed directly into the [gaze-target alignment step](gaze_alignment.md).

## Working with SAM3 Results (Handoff)

If you segmented some objects with the [SAM3 pipeline](sam3_guide.md) first, export them with `sam3_process.py --export-sam2` and load the resulting `sam2_handoff.json` in `sam2_ui.py`. The SAM3-covered objects appear in the object list as **covered**: they need no click points and are skipped during segmentation — their masks are taken from the SAM3 project (as the union of the referenced SAM3 instance masks) at export time. You only annotate the objects SAM3 could not detect. To change or delete a SAM3-derived object, do it in the SAM3 project and re-export (see [Changing SAM3 objects after the handoff](sam3_guide.md#exporting-to-the-sam2-pipeline)); ids of deleted SAM3 instances stay reserved and cannot be annotated here.

## Point-Based Prompts with SAM3 Models

When a SAM3 installation is detected, the SAM2 UI also offers SAM3 checkpoints for point-based (not text) prompting. Empirically we found this may perform *worse* than SAM2 for point prompts — for text-based prompting, use the [SAM3 pipeline](sam3_guide.md) instead.

## Troubleshooting

- **Model not found**: run `install.py` to download the checkpoint
- **GPU out of memory**: use a smaller model (tiny or small), add `--offload-to-cpu`, or reduce video resolution
- **Path errors**: ensure you're running from the repository directory
- **Slow UI on long videos**: use `sam2_process.py` for segmentation instead of the in-UI button
- **dtype-mismatch errors with `--no-bfloat16`**: SAM2 stores some internal tensors in bfloat16 regardless; the default (autocast enabled) is the recommended path

## Citation

If you use this tool in your research, please cite our paper ([arXiv:2605.22962](https://arxiv.org/abs/2605.22962)) — full reference and BibTeX in the [main README](../README.md#citation).