# Gaze Target Annotator — Video Segmentation with SAM3 / SAM2

Gaze Target Annotator is part of the [GazeBehavior Annotation Toolkit (GBAT)](https://github.com/CaiLab-neuro/GazeBehaviorAnnotationToolkit).

It is a video object segmentation tool built on Meta's [SAM3](https://github.com/facebookresearch/sam3) and [SAM2](https://github.com/facebookresearch/sam2). It was originally designed for segmenting object instances in scene videos acquired from head-mounted eye-trackers in eye-tracking studies, but it is equally suitable as a general-purpose tool for video and image instance segmentation.

For eye-tracking research, it additionally maps gaze coordinates from the eye-tracker onto the segmented objects, producing a **time course of gaze targets** (which object the participant looks at, moment by moment).

## Key Advantages

- **Long-video support** — we have optimized GPU and CPU memory footprint to allow you to process long videos (we tested on 30,000+ frames, but it should work on even longer videos) on a machine with modest GPU (> 16 GB VRAM).
- **Fully local computing** — all models run on your own machine/server. No video ever goes to a third-party cloud unless you send by yourself, which matters for sensitive research recordings that contain faces of participants and others.
- **Friendly graphical interface** — annotate with text prompts, clicks, or boxes in a point-and-click UI; no coding required for the core workflow.
- **Gaze-target time course** — combine segmentation masks with eye-tracker gaze coordinates (Pupil Labs format) to label the gazed object for every gaze sample, with confidence scores and diagnostic figures.
- **Text-prompt detection (SAM3)** — describe objects in plain language ("person", "toy", "table") and detect every instance across the whole video automatically.
- **Interactive refinement** — correct any mistake with positive/negative clicks or boxes and run SAM3/2 again to refine segmentation; corrections from all rounds are preserved as feedback to the SAM model.
- **Headless batch processing** — as the processing time is long for long videos, the processing jobs can run on a remote GPU server, while you provide feedback or add annotation for the next video to process.
- **Resumable projects** — annotations, refinements, and masks are stored on disk in a project directory; you can stop and resume at any stage.

## Two Pipelines: SAM3 and SAM2

The toolkit contains two parallel pipelines that share the same installation and output conventions:

| | SAM3 pipeline (recommended) | SAM2 pipeline |
|---|---|---|
| Prompting | Text prompt per concept (+ click/box refinement) | Click points per object |
| Detection | Finds **all** instances of a concept automatically | You annotate each object yourself in a few frames |
| UI | `sam3_ui.py` | `sam2_ui.py` |
| Batch CLI | `sam3_process.py` | `sam2_process.py` |
| Guide | **[SAM3 Guide](docs/sam3_guide.md)** | **[SAM2 Guide](docs/sam2_guide.md)** |

**SAM3 is likely the pipeline you want to start with.** The SAM2 pipeline remains useful in two situations:

1. **Concepts that are hard to prompt by text** — if SAM3 cannot reliably detect an object from a text description, a few manual clicks in the SAM2 UI can segment it directly.
2. **Many instances, few of interest** — SAM3 tracks *every* instance of a concept, which may increase GPU memory usage and processing time when a scene contains many (e.g., hundreds of toys) but you only need a few. With SAM2 you can click only the instances you care about.

The two pipelines interoperate: SAM3 results can be **exported into the SAM2 workflow** (`sam3_process.py --export-sam2`), so you can let SAM3 handle the easy concepts and pick up the remainder with SAM2 point annotation — without re-segmenting what SAM3 already covered. See the [SAM3 guide](docs/sam3_guide.md#exporting-to-the-sam2-pipeline) for details. SAM2-to-SAM3 porting is not yet supported.

After segmentation (from either pipeline), the **[Gaze-Target Alignment guide](docs/gaze_alignment.md)** describes how to generate the gaze-target time course from the masks and eye-tracker data.

## Installation

**IMPORTANT**: Before running setup, create and activate a dedicated conda environment or virtualenv.

```bash
# Recommended: conda environment with Python 3.12 (SAM3 requires 3.12+; SAM2 alone needs 3.10+)
conda create -n sam python=3.12 -y
conda activate sam

# Run the installer
python install.py
```

The installer:
- Checks the Python version (3.10+ for SAM2, 3.12+ for SAM3)
- Installs Python dependencies (torch, opencv, numpy, ...)
- Clones and installs **SAM2** into `sam_models/sam2/` with interactive checkpoint selection
- Optionally clones and installs **SAM3** into `sam_models/sam3/` (requires PyTorch 2.7+, CUDA 12.6+, and HuggingFace access approval — see the [SAM3 guide](docs/sam3_guide.md#installation))
- Creates launcher scripts (`run.sh` / `run.bat`) and verifies the installation
- SAM3 checkpoint needs to be downloaded by yourself as it requires approval from Huggingface.

Alternative with venv:

```bash
python3 -m venv sam_env
source sam_env/bin/activate   # Linux/Mac
# sam_env\Scripts\activate    # Windows
python install.py
```

### Troubleshooting Setup

- **Python version**: requires 3.10+ for SAM2, 3.12+ for SAM3 (`python --version`)
- **Git not installed**: download from https://git-scm.com/downloads
- **Failed to install some packages**: try downgrading Python from the newest version, then rerun `install.py`. If it still fails, install the failed package with conda or pip, then rerun `install.py`. You can report to us such cases in github issue.
- **Failed to load pytorch_python dll**: remove torch and torchvision, then let `install.py` reinstall them
- **Model not found at runtime**: run `install.py` again and download the missing checkpoint

## Documentation

- **[SAM3 Guide](docs/sam3_guide.md)** — text-prompt segmentation: concepts, instances, refinement, batch CLI, SAM3 installation, export to SAM2
- **[SAM2 Guide](docs/sam2_guide.md)** — point-based segmentation: annotation UI, batch processing, refinement
- **[Gaze-Target Alignment](docs/gaze_alignment.md)** — from segmentation masks + gaze data to a gaze-target time course

## Citation

If you use this toolkit in your research, please cite our paper:

> Iba Baig, Kevin Li, Yanbin Xu, Seiji Cattelain, Marie Hallo, Hayato Ono, Sho Tsuji, and Ming Bo Cai (2026). *GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for automatic annotation of egocentric eye-tracking and video data of child-caregiver interaction.* arXiv:2605.22962. https://arxiv.org/abs/2605.22962

```bibtex
@article{baig2026gazebehavior,
  title   = {GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for
             automatic annotation of egocentric eye-tracking and video data
             of child-caregiver interaction},
  author  = {Baig, Iba and Li, Kevin and Xu, Yanbin and Cattelain, Seiji and
             Hallo, Marie and Ono, Hayato and Tsuji, Sho and Cai, Ming Bo},
  journal = {arXiv preprint arXiv:2605.22962},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.22962}
}
```

## Staying Up to Date

We keep improving the toolkit — check this repository for new versions from time to time. If you installed by cloning the repository (the normal route), updating takes a single command from the `SegmentVideo4Gaze` directory:

```bash
git pull
```


## Update Timeline

- **2025-09** — Initial development: SAM2 point-based annotation UI (`sam2_ui.py`) and batch processing script
- **2025-10 – 2026-01** — Memory optimizations that make long videos (30,000+ frames) feasible on ordinary hardware; segmentation quality metrics to help spot frames that need correction
- **2026-02** — Gaze-target alignment: turn segmentation masks + eye-tracker data into a gaze-target time course; improved in-UI refinement
- **2026-05** — GBAT preprint posted on arXiv ([arXiv:2605.22962](https://arxiv.org/abs/2605.22962)) and this repository made public.
- **2026-07** — SAM3 pipline becomes available. Doumentation restructure; scripts renamed for consistency (`process_annotations.py` → `sam2_process.py`, `sync_annotations_sam3.py` → `sam3_sync.py`; the old names still work but please consider calling new script names)

## Feedback and Contributing

We would love to hear from you! If you run into a problem, find the documentation unclear, or have an idea for a feature, please open an issue on the [Issues tab](../../issues) — reports from real research workflows are the most valuable guide for where to improve the toolkit.

Code and documentation contributions are also very welcome, from a one-line fix to a new feature. See the [Contribution Guide](CONTRIBUTING.md) for how to set up a development environment, our coding conventions, and the pull-request workflow.

## License

This toolkit is released under the [PolyForm Noncommercial License 1.0.0](LICENSE). You are free to use, modify, and share it for any noncommercial purpose, including academic research, teaching, and personal projects, and it may be used by noncommercial organizations such as universities, charities, and government institutions. Commercial use is not permitted, including commercial use of modified versions or of copies redistributed by others.

For commercial licensing, please contact Ming Bo Cai (mingbo.cai@miami.edu).

The license covers this repository's code only. SAM2 and SAM3, which `install.py` downloads separately, remain under their own licenses from Meta.

By submitting a contribution, you agree that it is licensed under the same terms.
