# SAM3 Pipeline Guide — Text-Prompt Segmentation

The SAM3 pipeline detects and tracks objects across a whole video from **text prompts**. You describe a *concept* in plain language (e.g., "person", "toy", "table"); SAM3 finds every *instance* of that concept in the video and tracks each one with its own mask. You then review the results, name the instances you care about, merge instances corresponding to the same identity, and correct mistakes with clicks or boxes.

Two entry points share the same project format:

- **`sam3_ui.py`** — the graphical interface for reviewing, naming, and refining on any computer, as well as testing SAM3 on short videos (if your computer has GPU. It likely works on modern Mac too but please report to us if you encounter failure).
- **`sam3_process.py`** — the command-line tool for heavy batch processing (e.g., on a remote GPU server)

A typical division of labor: run detection in batch with `sam3_process.py`, then open the project in `sam3_ui.py` to inspect and correct.

## Installation

The SAM3 pipeline has stricter requirements than SAM2:

- Python 3.12+
- PyTorch 2.7+
- **CUDA 12.6+**
- A HuggingFace account with approved access to the SAM3 checkpoints

### 1. Install the SAM3 package

When running `install.py` (see the [main README](../README.md#installation)), answer `y` when prompted for SAM3 installation. The installer checks the version requirements, clones the SAM3 repository into `sam_models/sam3/`, and installs it.

### 2. Request checkpoint access

1. Visit https://huggingface.co/facebook/sam3
2. Follow in the instruction on the top to log in/sign up and request access to SAM3 model
3. Wait for approval (usually within 24–48 hours)

### 3. Authenticate with HuggingFace

```bash
pip install huggingface-hub
# Generate an access token at: https://huggingface.co/settings/tokens
huggingface-cli login   # paste your token when prompted
```

### 4. Download the checkpoint

```bash
mkdir -p sam_models/sam3/checkpoints
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3.pt',
    local_dir='sam_models/sam3/checkpoints'
)
"
```

### 5. Verify (optional)
```bash
python -c "from sam3.model_builder import build_sam3_video_predictor; print('SAM3 OK')"
```

## Core Concepts

- **Project**: a directory holding everything about one video — `project.json` (metadata), per-concept masks, refinement annotations. You can close and reopen a project at any stage, or move the entire folder between machines.
- **Concept**: one text prompt (e.g., "person"). Detection runs the prompt over the **entire video** — the "initial preview frame" in the UI only chooses which frame's results you see first; it does not bias detection.
- **Instance**: one detected object of a concept (e.g., person #0, person #1). You can rename instances (e.g., "child", "parent"), delete false detections (e.g., an animal head from a prompt of "head" while you only care about human head), or absorb duplicates into one another.
- Concepts are processed **one at a time** and masks are streamed to disk, which keeps memory usage bounded even for long videos.

We encourage you to test SAM3 behavior with various prompts on a short video from your data to find the best prompt (selective and comprehensive) for your project.

## UI Workflow (`sam3_ui.py`)

```bash
python sam3_ui.py
```

The window has three panels: the **concepts tree** (left), the **video display** (center), and **instance controls** (right).

<!-- TODO: screenshot of the main window with the three panels labeled -->

### 1. Create or load a project

- **File → Load Video** (optional for first usage) — choose a video and a project directory. You can build your own concept list and save it for later usage. If the video is not too long, you can test run it (it works for long video as well but then you need to keep the computer on while it runs)
- **File → Load Project** — reopen an existing project (if the video file moved, the UI prompts you to locate it; **File → Manage Video Paths...** edits stored paths)

A single image can also be loaded — it becomes a one-frame project.

### 2. Add concepts

Click **Add Concept**, type a concept name and text prompt, and optionally cap **Max Instances**. Either **Process Now** (detection runs in a background thread with a progress dialog) or **Save for Later** (queue it and process in batch with `sam3_process.py`, which is recommended for long videos).

You can also prepare many concepts at once: **File → Import Concept List...** loads a JSON file (see [Concepts JSON](#concepts-json) below), and **File → Edit Vocabulary...** manages a reusable vocabulary of prompts.

<!-- TODO: screenshot of the Add Concept dialog -->

### 3. Review instances

- Select a concept or instance in the tree to highlight it; double-click toggles visibility
- **Rename** instances to meaningful names (e.g., "child", "parent")
- **Delete Instance** removes false detections
- **Absorb Selected Into Target...** merges duplicate instances that are actually the same object — see [Absorbing Instances](#absorbing-instances) below
- Navigation: play/pause, frame slider, zoom (in time), **Flash Mask (f)** to blink the selected mask, **Flash Overlap (o)** to reveal where masks overlap, "Focus: selected concept only" to declutter the display

### 4. Refine

When a mask is wrong on some frames (after it has processed your annotation feedback or in its initial processing):

1. Navigate to a problem frame.
2. Click left button to add **positive points** on the object and **negative points** on wrongly included regions, or drag a **box** around the object (toggle with Box Mode button, or 'b' key)
3. **Confirm points**, then either **Propagate** right away or **Save Changes for Refinement** (recommended) and re-run later (in the UI or with `sam3_process.py --refine`)

Corrections from *all* refinement rounds are stored in the project and replayed on every re-run, so earlier fixes are not lost. **New Instance (Manual Points)** lets you add an object the text prompt missed entirely — this is the escape hatch before falling back to the SAM2 pipeline.

<!-- TODO: screenshot of refinement mode with positive/negative points -->

### Mouse Controls

| Action | Effect |
|--------|--------|
| Left click on canvas | Add a **positive** refinement point (or start a box drag, in Box Mode; or remove a point, in removal mode) |
| Right click on canvas | Add a **negative** refinement point (or remove a point, in removal mode) |
| Left click + drag on canvas (Box Mode) | Draw a box prompt around the object |
| Mouse wheel | Previous/next frame |
| Double-click in concepts tree | Toggle visibility of a concept/instance |

### Keyboard Shortcuts

| Key | Effect |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `↑` / `↓` | Previous / next instance |
| `space` | Play / pause |
| `f` | Flash the selected mask |
| `o` | Flash mask overlap |
| `r` | Toggle point removal mode |
| `b` | Toggle Box Mode (drag a box instead of clicking points) |
| `Home` / `End` | Jump to the first / last (renamed) instance within the current concept |
| `Page Up` / `Page Down` | Jump to previous / next frame with saved annotations (points, boxes, or mask anchors) for the selected instance |
| `Ctrl/Cmd + Z` | Undo last point |
| `Ctrl+Y` / `Cmd+Shift+Z` / `Cmd+Y` | Redo last point |
| `Ctrl+S` | Save points for batch (queues current refinement points without propagating) |
| `Delete` (with an instance selected in the tree) | Delete the selected instance |

Note: unlike the SAM2 UI, there is currently no pixel-level zoom on the video canvas in the SAM3 UI (only the time-axis slider zoom used for scrubbing).

Arrow-key and letter shortcuts are ignored while a text field (e.g. a rename entry) has focus.

### 5. Export

- **Export Video** — render the video with colored mask overlays (just for display - we don't use it often). For a single-image project, this button becomes **Export Image** and saves the overlaid image instead
- **Export SAM2 Format...** — hand the project off to the SAM2 pipeline if necessary(see [below](#exporting-to-the-sam2-pipeline))

## Absorbing Instances

Text-prompt detection sometimes splits one real-world object into two instances (e.g. it loses and re-acquires the object, or a brief occlusion starts a new track). **Absorb** merges such a pair using a guided wizard:

- **Direction**: you first select the instance you want to get rid of — the **source** (it will be absorbed and deleted) — then choose a **target** from the remaining instances (kept). In the target list, instances with a **higher** ID than the source (i.e. detected *later*) are grayed out: absorbing into a later-detected instance is less reliable, since the earlier instance usually has the cleaner, less-drifted track record. Prefer merging a later-appearing instance into an earlier one.
- **Mask anchor**: the UI will ask you to *confirm a clean mask* for the source (and, the first time anything is absorbed into it, for the target too). That confirmed frame's mask will serve as an anchor of how this object looks like next next time you re-process the video with a `--refine` argument. It works in similar way as point or bounding box annotations.
- **Suggested frame — first period**: the wizard jumps you to the instance's **first continuously-detected period** and, within it, the first frame with a non-empty mask. You can accept that mask if it covers the object well (no bleed, not blurry) but you may also consider scrolling to a later frame if the object is more intact or better segmented in that frame or if that frame is less blurry — or click "Add point instead" to fall back to manual point/box placement if no frame in that period looks clean.
- **Mask union**: if, at the frame you just confirmed for the source, the target (or an instance previously absorbed into it) already has its own mask, the wizard offers to take the **pixel-wise union (OR)** of the two masks as the conditioning anchor instead of just the source's mask — useful when the "two instances" are genuinely two parts of one object that were tracked separately. You can accept the union, skip it (keep only the source's mask), or switch to point/box mode instead.

After confirming, the target is re-propagated and you're offered the option to delete the source.

## Batch Command Line Processing (`sam3_process.py`)

All heavy operations can run without the graphic user interface (UI) — useful when you have a dedicated GPU server.

```bash
# Detect concepts (from a JSON file, or just name them directly)
python sam3_process.py --video video.mp4 --project my_project --concepts concepts.json
python sam3_process.py --video video.mp4 --project my_project --concepts person,table

# Check project status and suggested next commands (no GPU needed)
python sam3_process.py --project my_project --status

# Re-run refinement after adding correction points (in the UI or synced from elsewhere)
python sam3_process.py --project my_project --refine              # all concepts
python sam3_process.py --project my_project --refine person,car   # only these

# Redo detection for a concept, keeping (but not using) saved corrections (--redo) or discarding them (--reset)
python sam3_process.py --project my_project --redo person
python sam3_process.py --project my_project --reset person

# Delete a concept or specific instances
python sam3_process.py --project my_project --delete person
python sam3_process.py --project my_project --delete person:inst1,inst2

# Export overlay video
python sam3_process.py --project my_project --export output.mp4
```

### Concepts JSON

```json
{
  "concepts": [
    {"name": "child",  "text_prompt": "person", "detection_frame": 0},
    {"name": "toy",    "text_prompt": "toy",    "detection_frame": 0},
    {"name": "table",  "text_prompt": "table",  "detection_frame": 0}
  ]
}
```

Each entry may also include a new field `max_instances` to cap how many instances SAM3 may create for that concept, e.g., {"name": "hand",  "text_prompt": "hand",  "detection_frame": 0, "max_instances": 10}

### Commonly Used Options

| Option | Description |
|--------|-------------|
| `--project DIR` | Project directory (required) |
| `--video PATH` | Input video (needed when creating a new project) |
| `--concepts FILE\|name,name` | Concepts to detect: JSON file or comma-separated names |
| `--filter name,name` | Process only a subset of the given concepts |
| `--max-instances N` | Cap instances per concept (for name-list concepts) |
| `--refine [name,...]` | Apply the saved correction points and re-segment |
| `--redo [name,...]` | Re-run detection, keeping saved corrections |
| `--reset [name,...]` | Re-run detection from scratch, discarding corrections |
| `--delete concept[:inst,...]` | Delete a concept or specific instances |
| `--export PATH` | Export overlay video |
| `--status` | Print project summary and suggested next commands |
| `--device cuda:0` | GPU selection; a device string or comma-separated list (`cuda:0,cuda:1`) or `all` for multi-GPU |
| `--parallel N` | Concurrent worker processes (workers are spread over `--device` GPUs; each loads its own model, ~8 GB GPU memory) |
| `--frame-dir DIR` | Persistent directory for extracted frames (reused on later runs on the same project) |
| `--mask-format png\|npz` | Mask storage format for new projects (`npz` appears slightly faster). NPZ masks also store the mask's bounding box under the key `bbox` (`[x_min, y_min, x_max, y_max]`, pixels, inclusive; `-1`s if empty) |
| `--sam-version 3\|3.1` | `3.1` uses the Object Multiplex checkpoint — faster with many instances across multiple GPUs(requires Hopper grade GPU, untested) |
| `--cache-size N` | Decoded frames kept in RAM during propagation (default 50) |
| `--max-cond-frames N` | Bound GPU memory/compute when a concept has many correction frames (e.g., 4–8) |
| `-f` / `--force` | Skip confirmation prompts (for `--delete`, `--reset`, `--redo`, ...) |

Run `python sam3_process.py -h` for the full list, including advanced refinement-behavior flags (`--new-instance-policy`, `--restore-cond`, `--save-cond-states`, ...).

## Remote Annotation Workflow (`sam3_sync.py`)

Annotation (light, interactive) and processing (heavy, GPU) can happen on different machines. After adding correction points in the UI on your local machine, push just the small annotation files to the server — masks and video are never transferred:

```bash
python sam3_sync.py ./my_project user@server:/data/projects/my_project
# then, on the server:
python sam3_process.py --project /data/projects/my_project --refine
```

`sam3_sync.py --dry-run ...` previews what would be copied. (On Windows it uses rsync from Git for Windows, Cygwin, or WSL — see the notes in the script header.)

## Exporting to the SAM2 Pipeline

SAM3 results can be handed off to the [SAM2 pipeline](sam2_guide.md), so SAM2 point annotation only needs to cover the objects SAM3 could not:

```bash
python sam3_process.py --project my_project --export-sam2

# Optionally match instances to an existing SAM2 object list by name
python sam3_process.py --project my_project --export-sam2 \
  --sam2-object-list objects.csv
```

This writes `sam2_handoff.json` in the project directory. Opening it from `sam2_ui.py` marks the SAM3-covered objects as **covered**: their masks come from the SAM3 project (no re-segmentation, no click points needed), and you annotate only the remaining objects with points. Masks are referenced in place — nothing is copied. Running `--export-sam2` again after further SAM3 work adds new instances without changing existing ID assignments.

**Changing SAM3 objects after the handoff.** Objects that came from SAM3 stay owned by the SAM3 project: refine or delete them in `sam3_ui.py` / `sam3_process.py`, not in `sam2_ui.py` (which shows them read-only), then re-run `--export-sam2` and `sam2_process.py`. On re-export, the SAM2 id of a deleted instance moves to `retired_ids` in `sam2_handoff.json`: it disappears from `sam2_ui.py`, `sam2_process.py` removes its old masks from `sam2_results/masks/` (the SAM3 originals are untouched), and the id stays reserved so it is never given to another object. If you undo the deletion in SAM3, the next export restores the same id. A union of several instances loses only its deleted parts. Use `sam2_ui.py` only for new objects SAM3 did not cover.

With `--sam2-object-list` (a CSV exported from `sam2_ui.py` via "Export Object List"), SAM3 instances are matched to existing SAM2 object IDs by name; unmatched instances are resolved interactively. The same export is available in the UI as **Export SAM2 Format...**.

The reverse direction (porting SAM2 results into a SAM3 project) is not yet supported.

## Troubleshooting

- **0 detections for a prompt**: try a broader prompt ("person" instead of "head of a child"). Detection always scans the whole video, so this is not a frame-choice issue. 
Or, you can manually create a new instance under the concept, and add a few annotation points or bounding boxes for it.
- **CUDA version too old**: SAM3 requires CUDA 12.6+. Check with `nvidia-smi`.
- **PyTorch too old**: `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- **Checkpoint access not granted**: request access at https://huggingface.co/facebook/sam3 and wait for approval.
- **Checkpoint not found**: ensure the `.pt` file is in `sam_models/sam3/checkpoints/`.
- **Import error**: verify installation with `pip list | grep -i sam3`.
- **GPU out of memory during refinement**: try `--max-cond-frames 4` (or 8), a smaller `--cache-size`, or process fewer concepts at a time, or down-sample your video.
- **Slow playback in the UI**: the frame cache handles most videos; for videos with sparse keyframes, randomly selecting a frame turns to be slow. `sam3_process.py --reencode` (or **File → Re-encode Video (MJPEG)...**) creates a fast-seeking copy.

## Citation

If you use this tool in your research, please cite our paper ([arXiv:2605.22962](https://arxiv.org/abs/2605.22962)) — full reference and BibTeX in the [main README](../README.md#citation).