#!/usr/bin/env python3
"""
SAM2 Annotation Processor
========================

Takes annotation JSON from SAM2 Video UI and video file,
then exports segmented video and masks.

Usage:
    python sam2_process.py <annotation_file> <video_file> [options]

Examples:
    # Use default model (sam2 base+)
    python sam2_process.py annotations.json video.mp4
    
    # Use SAM2.1 large model
    python sam2_process.py annotations.json video.mp4 --model sam2.1-large
    
    # Use custom config and checkpoint
    python sam2_process.py annotations.json video.mp4 --config configs/sam2/sam2_hiera_l.yaml --checkpoint checkpoints/sam2_hiera_large.pt
    
"""

import os
import sys
import contextlib

# Reduce GPU memory fragmentation — set before any torch import
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import argparse
import re
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Import lazy loader BEFORE importing SAM2
from sam_lazy_loader import enable_lazy_loading

# Import quality metrics utilities and CUDA helpers
from utils import (
    IncrementalQualityMetricsCalculator,
    save_quality_metrics,
    DisableCUDADuringInit,
    should_disable_cuda_for_device,
    patch_sam3_modules_for_device,
    export_video_from_dict,
)

# Import shared segmentation module
from segment import (
    PointAnnotation,
    SegmentationConfig,
    SegmentationResult,
    VideoSegmenter,
    ProgressCallback,
)

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please run setup.py first to install dependencies.")
    sys.exit(1)

# Check SAM3 availability
def _check_sam3_available():
    """Check if SAM3 is installed and usable.

    NOTE: We intentionally avoid importing sam3.model_builder here because
    it triggers the full import chain including modules with hardcoded "cuda"
    allocations. Instead, we use a lightweight check with importlib.util.find_spec.
    """
    try:
        script_dir = Path(__file__).parent
        sam3_path = script_dir / "sam_models" / "sam3"
        if not sam3_path.exists():
            return False
        import importlib.util
        spec = importlib.util.find_spec("sam3")
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False

SAM3_AVAILABLE = _check_sam3_available()


def _read_mask(mask_path, cv2_mod):
    """Read a binary mask from a PNG or NPZ file. Returns a uint8 numpy array or None."""
    p = Path(mask_path)
    if p.suffix == ".npz":
        try:
            import numpy as _np
            return _np.load(str(p))["mask"]
        except Exception:
            return None
    else:
        return cv2_mod.imread(str(p), cv2_mod.IMREAD_GRAYSCALE)


def _write_mask(dst_path, mask, cv2_mod):
    """Write a uint8 mask array to dst_path (PNG or NPZ determined by suffix).

    Writes to a temporary file and renames it over dst_path. dst_path may be a hard link
    into a SAM3 project left by an earlier run (_link_sam3_masks_to_output); writing in
    place would overwrite the SAM3 original through the shared inode, while the rename
    only replaces this directory entry.
    """
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(f".{dst_path.stem}.tmp{os.getpid()}{dst_path.suffix}")
    try:
        if dst_path.suffix == ".npz":
            import numpy as _np
            _np.savez_compressed(str(tmp_path), mask=mask)
            ok = True
        else:
            ok = cv2_mod.imwrite(str(tmp_path), mask)
        if ok:
            os.replace(str(tmp_path), str(dst_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _make_merged_mask_loader(masks_dir, masks_by_frame):
    """Return a load_mask_func(frame_idx, obj_id) that reads from ``masks_dir`` using
    the filename recorded in ``masks_by_frame[frame_idx][obj_id]``.

    Handles both per-frame stacked NPZ (``npz_key``) and per-object NPZ / covered-union
    NPZ (which store the array under the plain ``mask`` key), plus PNG. Used to
    recompute quality metrics over the full merged mask set (SAM2 + SAM3 links +
    covered unions) after all merging has happened.
    """
    masks_dir = Path(masks_dir)

    def _loader(frame_idx, obj_id):
        obj_data = masks_by_frame.get(frame_idx, {}).get(obj_id)
        if obj_data is None:
            return None
        mask_path = masks_dir / obj_data["filename"]
        if obj_data["filename"].endswith(".npz"):
            try:
                d = np.load(str(mask_path))
                k = obj_data.get("npz_key", "")
                if k not in d.files and "mask" in d.files:
                    # Per-object NPZ (SAM3 link / covered-union) uses key 'mask'
                    k = "mask"
                return (d[k] > 0).astype(np.uint8) * 255 if k in d.files else None
            except Exception:
                return None
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (m > 0).astype(np.uint8) * 255 if m is not None else None

    return _loader


_PER_OBJECT_MASK_RE = re.compile(r"^mask_f\d{6}_.+_id(\d+)\.(png|npz)$")


def _read_retired_sam3_ids(sam3_project) -> set:
    """SAM2 ids of deleted SAM3 instances, from <sam3_project>/sam2_handoff.json "retired_ids"."""
    if sam3_project is None:
        return set()
    handoff_path = Path(sam3_project) / "sam2_handoff.json"
    if not handoff_path.exists():
        return set()
    try:
        with open(handoff_path) as f:
            return {int(k) for k in json.load(f).get("retired_ids", {})}
    except Exception as e:
        print(f"WARNING: could not read retired_ids from {handoff_path}: {e}")
        return set()


def _remove_retired_mask_files(masks_dir, retired_ids) -> int:
    """Delete per-object mask files (mask_f*_<name>_id<N>.png/.npz) of retired ids.

    These are hard links (or copies) of SAM3 masks made by an earlier run, or covered/
    multi-instance unions. Unlinking a hard link leaves the SAM3 original intact.
    Per-frame NPZ bundles are never touched (SAM3 masks are not stored in them).
    """
    masks_dir = Path(masks_dir)
    if not retired_ids or not masks_dir.is_dir():
        return 0
    removed = 0
    for f in masks_dir.iterdir():
        m = _PER_OBJECT_MASK_RE.match(f.name)
        if m and int(m.group(1)) in retired_ids:
            f.unlink()
            removed += 1
    return removed


def _link_sam3_masks_to_output(sam3_project, sam3_mask_dirs_rel, obj_ids,
                                output_dir, num_frames, mask_format="png"):
    """
    Link (or copy) SAM3 native masks into output_dir/masks/ using SAM2 filename conventions.

    Single-instance entries use hard links (zero storage overhead on the same filesystem;
    falls back to shutil.copy2 on cross-filesystem or Windows).  The destination filename
    extension matches the source file's actual format (PNG or NPZ).

    Multi-instance entries (``mask_sub_instances`` key) compute a pixel-wise OR union of
    their constituent masks and write the result in ``mask_format`` (png or npz).

    Returns masks_meta: {frame_idx: {obj_id: {"filename": str}}} for the linked objects.
    """
    import cv2 as _cv2

    out_masks_dir = Path(output_dir) / "masks"
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    masks_meta: dict = {}
    for str_obj_id, entry in sam3_mask_dirs_rel.items():
        obj_id = int(str_obj_id)
        if obj_id not in obj_ids:
            continue
        name = entry.get("name", f"Object_{obj_id}")

        sub_instances = entry.get("mask_sub_instances")
        if sub_instances:
            # Multi-instance union: pixel-wise OR across all sub-instance masks.
            # Output format follows mask_format (png or npz).
            ext = "npz" if mask_format == "npz" else "png"
            frames_written = 0
            for frame_idx in range(num_frames):
                union_mask = None
                for sub in sub_instances:
                    mask_dir = sam3_project / sub["mask_dir_rel"]
                    mask_path = mask_dir / sub["mask_filename_pattern"].format(frame=frame_idx)
                    if not mask_path.exists():
                        continue
                    m = _read_mask(mask_path, _cv2)
                    if m is None:
                        continue
                    union_mask = m if union_mask is None else (union_mask | m)
                if union_mask is not None:
                    dst_name = f"mask_f{frame_idx:06d}_{name}_id{obj_id}.{ext}"
                    dst = out_masks_dir / dst_name
                    _write_mask(dst, union_mask, _cv2)
                    masks_meta.setdefault(frame_idx, {})[obj_id] = {
                        "filename": dst_name, "name": name}
                    frames_written += 1
            n_subs = len(sub_instances)
            sub_names = ", ".join(f"'{s['name']}'" for s in sub_instances)
            if frames_written:
                print(f"  Union mask for '{name}' (id {obj_id}) from {n_subs} sub-instances "
                      f"[{sub_names}]: {frames_written} frames written.")
            else:
                print(f"  WARNING: No sub-instance masks found for '{name}' (id {obj_id}). "
                      f"Sub-instances: [{sub_names}]")
        else:
            # Single instance — hard-link into output dir (zero storage overhead).
            # Preserve the source file's extension (PNG or NPZ) in the destination name.
            mask_dir = sam3_project / entry["mask_dir_rel"]
            pattern = entry["mask_filename_pattern"]
            for frame_idx in range(num_frames):
                src = mask_dir / pattern.format(frame=frame_idx)
                if not src.exists():
                    continue
                dst_name = f"mask_f{frame_idx:06d}_{name}_id{obj_id}{src.suffix}"
                dst = out_masks_dir / dst_name
                if not dst.exists():
                    try:
                        os.link(str(src), str(dst))
                    except OSError:
                        shutil.copy2(str(src), str(dst))
                masks_meta.setdefault(frame_idx, {})[obj_id] = {"filename": dst_name}

    linked = sum(len(v) for v in masks_meta.values())
    print(f"SAM3 masks linked to output: {linked} files for objects {sorted(obj_ids)}")
    return masks_meta


def _write_png_batch(write_tasks, sem):
    """Write a list of (path, uint8_array) pairs to PNG files, then release semaphore."""
    try:
        for path, arr in write_tasks:
            cv2.imwrite(str(path), arr)
    finally:
        sem.release()


def _write_npz_frame(path, arrays, sem):
    """Write all masks for one frame to a compressed NPZ file, then release semaphore."""
    try:
        np.savez_compressed(str(path), **arrays)
    finally:
        sem.release()


def _write_npz_frame_merging_unchanged(path, new_arrays, unchanged_ids, key_pattern, sem):
    """
    Write an NPZ frame bundle, preserving any pre-existing keys that belong to
    unchanged objects (identified by obj-id parsed from the key name).

    This is the streaming alternative to pre-loading: we read the old file
    immediately before overwriting it, so only one frame's data is ever in RAM
    at a time.  Safe to call from a background thread as long as no other thread
    touches the same path concurrently (guaranteed by shutting down the write
    pool between forward and backward passes).
    """
    try:
        merged = {}
        if path.exists():
            try:
                old = np.load(str(path))
                for k in old.files:
                    m = key_pattern.match(k)
                    if m and int(m.group(3)) in unchanged_ids:
                        merged[k] = old[k]
            except Exception:
                pass
        merged.update(new_arrays)          # updated-object keys always win
        np.savez_compressed(str(path), **merged)
    finally:
        sem.release()


# Model configuration mappings
# NOTE: Config paths are absolute filesystem paths to sam_models/sam2/sam2/configs/
# (matching sam2_ui.py approach; Hydra requires a leading '//' workaround on Linux — see load_model())
# Checkpoint paths are relative to the script directory
_SCRIPT_DIR = Path(__file__).parent
_SAM2_CONFIG_DIR = _SCRIPT_DIR / "sam_models" / "sam2" / "sam2" / "configs"
MODEL_CONFIGS = {
    # SAM2.1 models (recommended)
    "sam2.1-tiny": (str(_SAM2_CONFIG_DIR / "sam2.1" / "sam2.1_hiera_t.yaml"), "sam_models/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
    "sam2.1-small": (str(_SAM2_CONFIG_DIR / "sam2.1" / "sam2.1_hiera_s.yaml"), "sam_models/sam2/checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1-base+": (str(_SAM2_CONFIG_DIR / "sam2.1" / "sam2.1_hiera_b+.yaml"), "sam_models/sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
    "sam2.1-large": (str(_SAM2_CONFIG_DIR / "sam2.1" / "sam2.1_hiera_l.yaml"), "sam_models/sam2/checkpoints/sam2.1_hiera_large.pt"),

    # SAM2 models (legacy)
    "sam2-tiny": (str(_SAM2_CONFIG_DIR / "sam2" / "sam2_hiera_t.yaml"), "sam_models/sam2/checkpoints/sam2_hiera_tiny.pt"),
    "sam2-small": (str(_SAM2_CONFIG_DIR / "sam2" / "sam2_hiera_s.yaml"), "sam_models/sam2/checkpoints/sam2_hiera_small.pt"),
    "sam2-base+": (str(_SAM2_CONFIG_DIR / "sam2" / "sam2_hiera_b+.yaml"), "sam_models/sam2/checkpoints/sam2_hiera_base_plus.pt"),
    "sam2-large": (str(_SAM2_CONFIG_DIR / "sam2" / "sam2_hiera_l.yaml"), "sam_models/sam2/checkpoints/sam2_hiera_large.pt"),

    # SAM3 model
    "sam3": (None, "sam_models/sam3/checkpoints/sam3.pt"),  # Config loaded automatically
}

def _get_cuda_device_index(device: str) -> int:
    """
    Extract CUDA device index from device string.

    Args:
        device: Device string (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')

    Returns:
        Device index (0 for 'cuda'), or None if device is 'cpu'
    """
    if not device or device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            return 0
    return 0


def _auto_select_default_model():
    """Auto-select best SAM2.1 model based on GPU memory (mimics UI behavior)"""
    preference_order = [
        "sam2.1-large",       # 898MB - Best quality
        "sam2.1-base+",       # 324MB - High quality
        "sam2.1-small",       # 184MB - Good balance
        "sam2.1-tiny",        # 156MB - Fastest
    ]

    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Adjust preference based on GPU memory
            if gpu_mem_gb >= 8:
                pass  # Keep original order (prefer large)
            elif gpu_mem_gb >= 4:
                preference_order = [p for p in preference_order if 'large' not in p]
            else:
                preference_order = [p for p in preference_order if 'large' not in p and 'base' not in p]

            print(f"Auto-selecting model based on GPU memory ({gpu_mem_gb:.1f}GB)...")
        else:
            preference_order = ["sam2.1-small", "sam2.1-tiny", "sam2.1-base+"]
            print("Auto-selecting model for CPU mode...")
    except:
        preference_order = ["sam2.1-base+", "sam2.1-small"]

    # Find first available model
    for model_name in preference_order:
        config_path, checkpoint_path = MODEL_CONFIGS.get(model_name, (None, None))
        if config_path and os.path.exists(checkpoint_path):
            print(f"Model to use if sam3 is not chosen: {model_name}")
            return model_name

    # Fallback to any available SAM2.1 model
    for model_name in MODEL_CONFIGS:
        if model_name.startswith("sam2.1"):
            config_path, checkpoint_path = MODEL_CONFIGS[model_name]
            if os.path.exists(checkpoint_path):
                print(f"  Selected: {model_name} (fallback)")
                return model_name

    print("  Selected: sam2.1-base+ (default, checkpoint may be missing)")
    return "sam2.1-base+"


class ConsoleProgressCallback:
    """
    Console-based progress callback for VideoSegmenter.

    Prints progress updates with memory monitoring for CLI usage.
    """

    def __init__(self, verbose: bool = True, device: str = None):
        """
        Initialize the console progress callback.

        Args:
            verbose: Whether to print detailed progress updates
            device: Device being used (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')
        """
        self.verbose = verbose
        self._phase_total = 0
        self._current_phase = ""
        self.device = device or "cuda"
        self._cuda_device_index = _get_cuda_device_index(self.device)

    def on_phase_start(self, phase: str, total_steps: int) -> None:
        """Called when a new phase begins."""
        self._current_phase = phase
        self._phase_total = total_steps

        phase_labels = {
            "extracting": "Extracting frames...",
            "initializing": "Initializing SAM inference...",
            "adding_points": "Adding annotation prompts...",
            "forward": f"Propagating masks forward (0->{total_steps-1})...",
            "backward": f"Propagating masks backward ({total_steps-1}->0)...",
        }
        print(f"\n{phase_labels.get(phase, f'Processing {phase}...')}")

    def on_progress(self, phase: str, current: int, total: int, message: str) -> None:
        """Report progress within a phase."""
        if not self.verbose:
            return

        # Only report every 50 frames for propagation phases
        if phase in ("forward", "backward") and current % 50 != 0:
            return

        # Monitor memory usage
        if torch.cuda.is_available() and self._cuda_device_index is not None:
            gpu_allocated = torch.cuda.memory_allocated(device=self._cuda_device_index) / (1024**3)
            gpu_peak = torch.cuda.max_memory_allocated(device=self._cuda_device_index) / (1024**3)
            process = psutil.Process()
            ram_used = process.memory_info().rss / (1024**3)

            print(f"  {phase.capitalize()}: Frame {current}/{total} | "
                  f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                  f"RAM: {ram_used:.2f}GB")
        else:
            print(f"  {phase.capitalize()}: Frame {current}/{total}")

    def on_phase_complete(self, phase: str) -> None:
        """Called when a phase completes."""
        print(f"  {phase.capitalize()} complete.")


class SAM2Processor:
    def __init__(self, config_file=None, checkpoint_file=None, model_name="sam2.1-base+", offload_to_cpu=False, async_loading=False, smooth_masks=False, device=None, frame_format="jpg", exclusive_masks=False, mask_format="png", vos_optimized=False, no_bfloat16=False, max_cond_frames_in_attn=-1):
        """
        Initialize SAM2 Processor

        Args:
            config_file: Path to model config YAML (overrides model_name)
            checkpoint_file: Path to checkpoint file (overrides model_name)
            model_name: Preset model name (e.g., 'sam2-base+', 'sam2.1-large', 'sam3')
            offload_to_cpu: Use SAM2's CPU offloading for memory optimization
            async_loading: Use async frame loading (experimental, may reduce memory)
            smooth_masks: Apply morphological smoothing to reduce pixelation in masks
            device: Device to use (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1'). If None, auto-detect.
            frame_format: Format for extracted frames ("jpg" or "png")
            vos_optimized: Use torch.compile on SAM2's core components for faster propagation.
                The first run will be very slow (5-30 min) while kernels are compiled and cached.
                Subsequent videos in the same process reuse the cache and run faster.
                Only useful for SAM2 batch jobs processing many videos; ignored for SAM3.
                Requires PyTorch 2.5.1+. (default: False)
            max_cond_frames_in_attn: SAM3 only — how many temporally-closest conditioning
                (correction) frames the tracker cross-attends to per propagated frame.
                -1 (default) matches SAM2's own unlimited default; SAM3's model_builder
                otherwise defaults this to 4, which can let older point corrections fall
                out of attention as propagation moves away from them. Ignored for SAM2.
        """
        # Store model name for detection
        self.model_name = model_name
        self.vos_optimized = vos_optimized
        self.no_bfloat16 = no_bfloat16
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Check if SAM3 was requested but is not available
        if model_name == "sam3" and not SAM3_AVAILABLE:
            print("=" * 60)
            print("WARNING: SAM3 requested but not available")
            print("=" * 60)
            print("Possible causes:")
            print("  1. SAM3 not installed (run setup.py and choose SAM3)")
            print("  2. Missing dependencies (huggingface-hub, decord, einops)")
            print("  3. Python < 3.12 or PyTorch < 2.7")
            print()
            print("Falling back to SAM2.1 base+...")
            print("=" * 60)
            print()
            # Fallback to SAM2.1 base+
            self.model_name = "sam2.1-base+"

        # Determine config and checkpoint paths
        if config_file and checkpoint_file:
            # Use custom paths
            self.config_file = config_file
            self.checkpoint_file = checkpoint_file
        elif model_name in MODEL_CONFIGS:
            # Use preset model
            self.config_file, self.checkpoint_file = MODEL_CONFIGS[model_name]
        else:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

        # Validate paths exist (skip config validation for SAM3 — config is bundled in the package)
        if self.model_name != "sam3" and self.config_file and not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # Checkpoint is optional for some use cases, but warn if missing
        if self.checkpoint_file and not os.path.exists(self.checkpoint_file):
            print(f"WARNING: Checkpoint file not found: {self.checkpoint_file}")
            print("Model will be initialized without pre-trained weights.")
            self.checkpoint_file = None

        self.sam2_model = None
        self.video_predictor = None

        # Memory optimization configuration
        self.offload_to_cpu = offload_to_cpu
        self.async_loading = async_loading
        self.smooth_masks = smooth_masks
        self.no_backward_propagation = False  # Will be set from command line args
        self.exclusive_masks = exclusive_masks  # Winner-takes-all per pixel
        self.requested_device = device  # User-requested device (None = auto-detect)
        self.device = None  # Will be set after loading model
        self.frame_format = frame_format  # Format for extracted frames
        self.mask_format = mask_format  # Output format for masks: "png" or "npz"

    @property
    def use_sam3(self):
        """Returns True if using SAM3 model"""
        return self.model_name == "sam3"

    def load_model(self):
        """Load SAM2 or SAM3 model with correct API usage"""
        model_type = "SAM3" if self.use_sam3 else "SAM2"
        print(f"Loading {model_type} model...")

        if not self.use_sam3:
            print(f"  Config: {self.config_file}")
        print(f"  Checkpoint: {self.checkpoint_file or 'None (random init)'}")

        try:
            # Determine device: use requested device or auto-detect
            if self.requested_device:
                device = self.requested_device
                # Validate device selection
                if device.startswith("cuda"):
                    if not torch.cuda.is_available():
                        print(f"  WARNING: CUDA requested but not available, falling back to CPU")
                        device = "cpu"
                    elif device.startswith("cuda:"):
                        gpu_id = int(device.split(":")[1])
                        if gpu_id >= torch.cuda.device_count():
                            print(f"  WARNING: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPU(s) found), falling back to CPU")
                            device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {device}")

            # Store device for later use (e.g., in export functions)
            self.device = device

            # Common optimizations for both SAM2 and SAM3
            if device != "cpu" and torch.cuda.is_available():
                # Determine GPU ID for TF32 check
                gpu_id = 0 if device == "cuda" else int(device.split(":")[1])

                # Set default CUDA device so internal SAM2 ops that don't specify
                # a device (e.g. position encoding warmup, freqs_cis) land on the
                # correct GPU instead of defaulting to cuda:0, which would cause
                # CUDA illegal memory access errors when mixing tensors across devices.
                torch.cuda.set_device(gpu_id)

                # Enable TF32 for Ampere GPUs (RTX 30xx+, A100) for better performance
                if torch.cuda.get_device_properties(gpu_id).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("  TensorFloat32 (TF32) enabled for Ampere GPU")

                # Both SAM2 and SAM3/SAM3.1 unconditionally store maskmem_features in
                # bfloat16 internally. Global autocast is required so that all other
                # operations also run in bfloat16, avoiding "BFloat16 vs Float" matmul
                # errors during propagation.
                major = torch.cuda.get_device_properties(gpu_id).major
                _enable_bfloat16 = not self.no_bfloat16
                if self.no_bfloat16:
                    print("  BFloat16 mode: DISABLED via --no-bfloat16")
                elif major < 8:
                    # Pre-Ampere GPU: no native bfloat16 hardware — warn and ask.
                    # SAM2/SAM3 store maskmem_features in bfloat16 regardless of
                    # autocast, so disabling autocast may cause dtype mismatch errors.
                    # Enabling it runs in software emulation, which may be slow or wrong.
                    # Neither path is guaranteed to work; let the user decide.
                    print(f"\n  WARNING: Your GPU has compute capability {major}.x "
                          f"(pre-Ampere, < 8.0) with no native bfloat16 hardware.")
                    print("  SAM2/SAM3 store internal tensors (maskmem_features) in")
                    print("  bfloat16 regardless of this setting. Your options:")
                    print("    [Y] Enable bfloat16 autocast (recommended — keeps compute")
                    print("        consistent with internal tensors; may run slowly or")
                    print("        error depending on PyTorch version and driver)")
                    print("    [N] Disable bfloat16 autocast (may cause dtype mismatch")
                    print("        errors because internal tensors remain bfloat16)")
                    print("  Pass --no-bfloat16 to always choose N without this prompt.")
                    print("  If neither option works, please report your GPU model,")
                    print("  PyTorch version, CUDA version, and the error message at:")
                    print("  https://github.com/MingboCai/Sam2UI/issues")
                    try:
                        answer = input("  Enable bfloat16 autocast? [Y/n]: ").strip().lower()
                    except EOFError:
                        answer = ""  # non-interactive: fall back to recommended default
                    _enable_bfloat16 = answer not in ("n", "no")
                    if not _enable_bfloat16:
                        print("  BFloat16 mode: DISABLED (user choice)")
                if _enable_bfloat16:
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                    print("  BFloat16 mode: GLOBAL autocast enabled")
                    print("  Model weights remain in float32 (checkpoint dtype)")

            # Load model based on type
            # Use context manager to prevent hardcoded CUDA allocations when using
            # CPU or a specific GPU other than cuda:0
            if self.use_sam3:
                # SAM3 / SAM3.1 loading
                if not SAM3_AVAILABLE:
                    raise ImportError("SAM3 not available. Run setup.py to install.")

                # IMPORTANT: Must patch SAM3 modules BEFORE importing model builders
                # because the import triggers the full module chain with hardcoded "cuda"
                patch_sam3_modules_for_device(device)

                from sam3.model_builder import build_sam3_video_model
                print("  Building SAM3 model ...")
                ckpt_path = self.checkpoint_file
                # Build SAM3 model and extract tracker with SAM2-compatible API
                if ckpt_path and os.path.exists(ckpt_path):
                    sam3_model = build_sam3_video_model(checkpoint_path=ckpt_path, device=device)
                else:
                    sam3_model = build_sam3_video_model(device=device)
                # Extract the tracker component (has init_state, add_new_points, etc.)
                self.video_predictor = sam3_model.tracker
                # Attach backbone for feature extraction
                self.video_predictor.backbone = sam3_model.detector.backbone
                # build_sam3_video_model() defaults max_cond_frames_in_attn to 4 (only the
                # 4 temporally-closest correction points get cross-attended per frame), unlike
                # SAM2's own default of -1 (unlimited). Override to self.max_cond_frames_in_attn
                # (default -1) so point corrections don't fall out of attention as propagation
                # moves away from them, matching SAM2's default behavior.
                if hasattr(self.video_predictor, "max_cond_frames_in_attn"):
                    self.video_predictor.max_cond_frames_in_attn = self.max_cond_frames_in_attn
                    print(f"  SAM3 tracker: max_cond_frames_in_attn set to {self.max_cond_frames_in_attn}")

            else:
                # SAM2 loading
                # Hydra on Linux strips the leading '/' from absolute paths, so prepend
                # an extra '/' (matching sam2_ui.py behavior at line ~6472)
                config_for_hydra = self.config_file
                if config_for_hydra and config_for_hydra.startswith('/'):
                    config_for_hydra = '/' + config_for_hydra

                if should_disable_cuda_for_device(device):
                    with DisableCUDADuringInit():
                        self.video_predictor = build_sam2_video_predictor(
                            config_file=config_for_hydra,
                            ckpt_path=self.checkpoint_file,
                            device=device,
                            vos_optimized=self.vos_optimized,
                        )
                else:
                    self.video_predictor = build_sam2_video_predictor(
                        config_file=config_for_hydra,
                        ckpt_path=self.checkpoint_file,
                        device=device,
                        vos_optimized=self.vos_optimized,
                    )

            print(f"OK: {model_type} model loaded successfully")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_annotations(self, annotation_file):
        """Load annotation data from JSON file"""
        print(f"Loading annotations from {annotation_file}...")
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            if "annotations" not in data:
                raise ValueError("Invalid annotation file: missing 'annotations' field")
            
            print(f"OK: Loaded {len(data['annotations'])} annotations")
            print(f"   Video: {data.get('video_path', 'Unknown')}")
            print(f"   Total frames: {data.get('total_frames', 'Unknown')}")
            print(f"   Objects: {len(data.get('object_names', {}))}")
            
            return data
        except Exception as e:
            print(f"ERROR: Failed to load annotations: {e}")
            return None

    def get_video_info(self, video_path):
        """Get video frame count and properties without loading all frames"""
        print(f"Reading video info from {video_path}...")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            print(f"OK: Video info - {frame_count} frames, {fps:.2f} fps, {width}x{height}")
            return frame_count, fps, width, height
        except Exception as e:
            print(f"ERROR: Failed to read video info: {e}")
            return None, None, None, None
    
    def group_annotations_by_frame(self, annotations):
        """Group annotations by frame index"""
        frame_annotations = {}
        for annotation in annotations:
            frame_idx = annotation["frame_index"]
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append(annotation)
        return frame_annotations

    def _cleanup_inference_state(self, inference_state, current_frame_idx, frames_to_keep=20, reverse=False, verbose=False):
        """
        Clean up old frames from inference state to prevent memory growth.

        Handles both SAM2 and SAM3 inference state structures:
        - SAM3: Cleans multiple caches (non_cond_frame_outputs, cond_frame_outputs, cached_frame_outputs, tracker states)
        - SAM2: Cleans per-object output dicts

        Args:
            inference_state: SAM2/SAM3 inference state object
            current_frame_idx: Current frame being processed
            frames_to_keep: Number of recent frames to keep (default 20)
            reverse: If True, propagating backward (delete frames ahead), else forward (delete frames behind)
            verbose: If True, print debug info every frame
        """
        total_deleted = 0

        # Direction-aware cleanup logic:
        # - Forward (reverse=False): Delete frames BEHIND current (f < current - keep)
        # - Backward (reverse=True): Delete frames AHEAD of current (f > current + keep)
        if reverse:
            should_delete = lambda f: f > current_frame_idx + frames_to_keep
        else:
            should_delete = lambda f: f < current_frame_idx - frames_to_keep

        # Try SAM3 structure first (direct attribute access)
        if hasattr(inference_state, 'non_cond_frame_outputs'):
            # Clean non-conditioning frame outputs (tracker state)
            non_cond = inference_state.non_cond_frame_outputs
            old_frames = [f for f in non_cond.keys() if should_delete(f)]
            for old_frame in old_frames:
                del non_cond[old_frame]
            total_deleted += len(old_frames)

            # Clean conditioning frame outputs (tracker state)
            if hasattr(inference_state, 'cond_frame_outputs'):
                cond = inference_state.cond_frame_outputs
                old_frames = [f for f in cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del cond[old_frame]
                total_deleted += len(old_frames)

            # Clean cached_frame_outputs (Sam3VideoInference level)
            # This is a dict attribute, not a method
            if hasattr(inference_state, '__dict__') and 'cached_frame_outputs' in inference_state.__dict__:
                cached = inference_state.__dict__['cached_frame_outputs']
                if isinstance(cached, dict):
                    old_frames = [f for f in cached.keys() if isinstance(f, int) and should_delete(f)]
                    for old_frame in old_frames:
                        del cached[old_frame]
                    total_deleted += len(old_frames)

            # Clean tracker inference states (per-object tracker states)
            if hasattr(inference_state, '__dict__') and 'tracker_inference_states' in inference_state.__dict__:
                tracker_states = inference_state.__dict__['tracker_inference_states']
                if isinstance(tracker_states, list):
                    for tracker_state in tracker_states:
                        if hasattr(tracker_state, 'output_dict'):
                            output_dict = tracker_state.output_dict
                            for cache_name in ['non_cond_frame_outputs', 'cond_frame_outputs']:
                                if cache_name in output_dict:
                                    cache = output_dict[cache_name]
                                    old_frames = [f for f in cache.keys() if should_delete(f)]
                                    for old_frame in old_frames:
                                        del cache[old_frame]
                                    total_deleted += len(old_frames)

            

        # Fall back to SAM2/SAM3 dict structure (SAM3 uses this for tracker inference_state)
        elif isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
            # CRITICAL: Clean the MAIN output_dict first (SAM3 stores frame outputs here)
            # This is where sam3_tracking_predictor.py:859 stores outputs: output_dict[storage_key][frame_idx] = current_out
            # But ONLY clean non_cond_frame_outputs, NOT cond_frame_outputs!
            if "output_dict" in inference_state:
                output_dict = inference_state["output_dict"]

                # Only clean non-conditional frames (intermediate propagation frames - safe to delete)
                if "non_cond_frame_outputs" in output_dict:
                    cache = output_dict["non_cond_frame_outputs"]
                    old_frames = [f for f in cache.keys() if should_delete(f)]
                    for old_frame in old_frames:
                        del cache[old_frame]
                    total_deleted += len(old_frames)

                # DO NOT clean cond_frame_outputs - SAM3 needs these for entire propagation!
                # These are frames with user prompts that SAM3 references during tracking.
                # Deleting these causes: AssertionError at sam3_tracker_base.py:591

            # Then clean per-object dicts (these are slices/views of the main dict)
            for obj_idx in range(len(inference_state.get("obj_ids", []))):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                old_frames = [f for f in non_cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del non_cond[old_frame]
                total_deleted += len(old_frames)

            

        # Periodic GPU memory cleanup
        if current_frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # def process_segmentation(self, video_path, annotations_data, output_dir, frame_dir=None):
    #     """Process segmentation using SAM2"""
    #     return self.process_segmentation_full(video_path, annotations_data, output_dir, frame_dir=frame_dir)

    def process_segmentation(self, video_path, annotations_data, output_dir, frame_dir=None, skip_quality_save=False, mask_format=None, unchanged_ids=None):
        """Process segmentation with streaming mask export to reduce memory usage"""
        print("Starting segmentation process (streaming export mode)...")

        # Group annotations by frame and object
        frame_annotations = self.group_annotations_by_frame(annotations_data["annotations"])

        # Create or reuse frame directory
        if frame_dir:
            # Use persistent frame directory
            temp_dir = Path(frame_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            is_persistent = True

            # Check if frames already exist (support both jpg and png formats)
            existing_frames = sorted(temp_dir.glob("*.jpg"))
            if not existing_frames:
                existing_frames = sorted(temp_dir.glob("*.png"))
            if existing_frames:
                print(f"  Reusing existing frames from: {temp_dir}")
                print(f"  Found {len(existing_frames)} frames")
                skip_extraction = True
            else:
                print(f"  Extracting frames to persistent directory: {temp_dir} (format: {self.frame_format})")
                skip_extraction = False
        else:
            # Use temporary directory (will be deleted after processing)
            temp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
            is_persistent = False
            skip_extraction = False
            print(f"  Extracting frames to temporary directory: {temp_dir} (format: {self.frame_format})")

        try:
            # Extract frames if needed
            if not skip_extraction:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video file: {video_path}")

                save_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = temp_dir / f"{save_idx:05d}.{self.frame_format}"
                    if self.frame_format == "png":
                        # Use fast PNG compression (level 1 is fast, level 9 is slow but smaller)
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    else:
                        # Default JPEG quality
                        cv2.imwrite(str(frame_path), frame)
                    save_idx += 1

                    if save_idx % 500 == 0:
                        print(f"  Extracted {save_idx} frames...")

                cap.release()
                print(f"  Extracted {save_idx} frames")
            else:
                print(f"  Skipping frame extraction (using existing frames)")

            # Get frame dimensions (needed for SAM3 coordinate conversion and quality metrics)
            # Support both jpg and png formats
            frame_files = sorted(temp_dir.glob("*.jpg"))
            if not frame_files:
                frame_files = sorted(temp_dir.glob("*.png"))
            if not frame_files:
                raise ValueError(f"No frame files found in: {temp_dir}")

            first_frame_path = frame_files[0]
            first_frame = cv2.imread(str(first_frame_path))
            if first_frame is None:
                raise ValueError(f"Cannot read first frame: {first_frame_path}")
            frame_height, frame_width = first_frame.shape[:2]
            print(f"  Frame dimensions: {frame_width}x{frame_height}")
            del first_frame  # Free memory

            # Initialize SAM2 inference state with JPEG directory
            print("Initializing SAM inference state...")

            # Build init_state parameters
            init_params = {'video_path': str(temp_dir)}

            offload = bool(getattr(self, 'offload_to_cpu', False))
            if offload:
                init_params['offload_video_to_cpu'] = offload
                init_params['offload_state_to_cpu'] = offload
                print("  Using CPU offloading for memory optimization")

            # Add async loading if configured
            if hasattr(self, 'async_loading') and self.async_loading:
                init_params['async_loading_frames'] = True
                print("  Using async frame loading (experimental)")

            # Get device from stored attribute (self.video_predictor.device works for SAM2 but
            # SAM3's Sam3VideoTrackingMultiplexDemo doesn't expose .device on the module)
            device = self.device or "cpu"

            # No local autocast needed - global autocast is always enabled on CUDA in load_model()
            # (see benchmark.py line 20 for this pattern)
            if device == "cuda":
                autocast_mode = "BFloat16 (global autocast)"
            else:
                autocast_mode = "CPU"

            inference_state = self.video_predictor.init_state(**init_params)
            num_frames = inference_state["num_frames"]

            print(f"  Initialized state for {num_frames} frames (autocast: {autocast_mode})")

            # Initialize quality metrics calculator for incremental calculation during propagation
            quality_calculator = IncrementalQualityMetricsCalculator(
                frame_dimensions=(frame_height, frame_width),
                num_frames=num_frames
            )
            print(f"  Quality metrics calculator initialized for {num_frames} frames")

            # Process each annotated frame
            object_names = annotations_data.get("object_names", {})
            object_colors = annotations_data.get("object_colors", {})

            # Group annotations by object ID
            objects_with_annotations = {}
            for frame_idx, annotations in frame_annotations.items():
                for annotation in annotations:
                    obj_id = annotation["object_id"]
                    if obj_id not in objects_with_annotations:
                        objects_with_annotations[obj_id] = {}
                    if frame_idx not in objects_with_annotations[obj_id]:
                        objects_with_annotations[obj_id][frame_idx] = []
                    objects_with_annotations[obj_id][frame_idx].append(annotation)

            print(f"Processing {len(objects_with_annotations)} objects...")

            # Build flat (frame_idx, obj_id, annotations) list for iteration.
            _all_ann_tuples = [
                (frame_idx, obj_id, annotations)
                for obj_id, obj_frames in objects_with_annotations.items()
                for frame_idx, annotations in sorted(obj_frames.items())
            ]

            _last_obj_id = None
            for frame_idx, obj_id, annotations in _all_ann_tuples:
                if obj_id != _last_obj_id:
                    obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                    print(f"\nProcessing {obj_name} (ID: {obj_id})...")
                    _last_obj_id = obj_id

                if frame_idx >= num_frames:
                    print(f"  WARNING: Skipping frame {frame_idx} (beyond video length)")
                    continue

                points = []
                labels = []

                for annotation in annotations:
                    x, y = annotation["x"], annotation["y"]
                    is_positive = annotation["is_positive"]
                    points.append([x, y])
                    labels.append(1 if is_positive else 0)

                if not points:
                    continue

                print(f"  Frame {frame_idx}: {len(points)} points")

                try:
                    if self.use_sam3:
                        # SAM3 (non-multiplex): direct add_new_points on tracker
                        rel_points = [[x / frame_width, y / frame_height] for x, y in points]
                        points_t = torch.tensor(rel_points, dtype=torch.float32)
                        labels_t = torch.tensor(labels, dtype=torch.int32)
                        print(f"    SAM3 coord: pixel {points[0]} → rel [{rel_points[0][0]:.4f}, {rel_points[0][1]:.4f}]")
                        _ = self.video_predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=points_t,
                            labels=labels_t,
                        )
                    else:
                        # SAM2: pixel coordinates
                        points_np = np.array(points, dtype=np.float32)
                        labels_np = np.array(labels, dtype=np.int32)
                        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            points=points_np,
                            labels=labels_np,
                        )
                except Exception as e:
                    import traceback as _tb
                    print(f"    WARNING: Error adding points: {e}")
                    _tb.print_exc()
                    continue

            # Create output directories for streaming export
            masks_dir = Path(output_dir) / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Propagate annotations - FORWARD direction (frame 0 → end)
            # IMPORTANT: Always propagate from frame 0 to ensure full video coverage
            print(f"\nPropagating annotations FORWARD (frames 0 to {num_frames-1})...")

            masks_metadata = {}  # Only metadata, not actual mask arrays

            # Async mask writer: GPU loop submits writes to a background thread so it never
            # blocks on disk I/O.  A semaphore caps the queue depth to 8 submitted-but-not-yet-
            # written frames (~80 MB at 1080p × 5 objects) so memory stays bounded.
            _mask_format = mask_format if mask_format is not None else self.mask_format
            _unchanged_ids = set(unchanged_ids) if unchanged_ids else set()
            # Pre-compile the key pattern once; only needed for NPZ merge-write.
            _npz_key_pat = re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)$") if (_mask_format == "npz" and _unchanged_ids) else None
            _write_pool = ThreadPoolExecutor(max_workers=4)
            _io_sem = threading.Semaphore(8)
            _write_futures = []

            # Free any cached GPU memory before the propagation loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Nested autocast mirrors the global autocast in load_model() for CPU-offloaded
            # tensors that arrive back from RAM during propagation.  Skipped on CPU where
            # CUDA autocast is unsupported.
            _fwd_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if "cuda" in self.device and not self.no_bfloat16
                else contextlib.nullcontext()
            )
            with _fwd_autocast:
                # Build forward propagation iterator
                if self.use_sam3:
                    propagate_iterator = self.video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=num_frames,
                        reverse=False,
                        propagate_preflight=True,
                    )
                else:
                    propagate_iterator = self.video_predictor.propagate_in_video(
                        inference_state, reverse=False
                    )

                for result in propagate_iterator:
                    # Unpack result
                    if self.use_sam3:
                        out_frame_idx, out_obj_ids, out_low_res_masks, out_mask_logits, out_obj_scores = result
                    else:
                        out_frame_idx, out_obj_ids, out_mask_logits = result
                        out_low_res_masks = None
                        out_obj_scores = None

                    frame_masks = {}
                    frame_masks_for_quality = {}

                    # Exclusive masks: winner-takes-all per pixel
                    if self.exclusive_masks and len(out_obj_ids) > 0:
                        _logits_2d = out_mask_logits.squeeze(1)
                        _bg = torch.zeros(1, _logits_2d.shape[1], _logits_2d.shape[2],
                                          device=_logits_2d.device, dtype=_logits_2d.dtype)
                        _all_logits = torch.cat([_bg, _logits_2d], dim=0)
                        _winners = torch.argmax(_all_logits, dim=0)
                        del _logits_2d, _all_logits
                    else:
                        _winners = None

                    _frame_png_writes = []
                    _frame_npz_arrays = {}

                    _n_masks = out_mask_logits.shape[0] if out_mask_logits is not None else 0
                    for i, obj_id in enumerate(out_obj_ids):
                        if i >= _n_masks:
                            continue
                        # Extract mask
                        if _winners is not None:
                            mask = (_winners == i + 1).cpu().numpy()
                        else:
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                        obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                        obj_color = object_colors.get(str(obj_id), [255, 0, 0])
                        score = (float(out_obj_scores[i]) if out_obj_scores is not None else 1.0)
                        mask_u8 = (mask * 255).astype(np.uint8)
                        del mask

                        if _mask_format == "npz":
                            _npz_key = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}"
                            _frame_npz_arrays[_npz_key] = mask_u8
                            frame_masks[obj_id] = {
                                'filename': f"masks_f{out_frame_idx:06d}.npz",
                                'npz_key': _npz_key,
                                'score': score,
                                'name': obj_name,
                                'color': obj_color,
                            }
                        else:
                            mask_filename = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}.png"
                            _frame_png_writes.append((masks_dir / mask_filename, mask_u8))
                            frame_masks[obj_id] = {
                                'filename': mask_filename,
                                'score': score,
                                'name': obj_name,
                                'color': obj_color,
                            }

                        frame_masks_for_quality[obj_id] = mask_u8

                    _io_sem.acquire()
                    if _mask_format == "npz":
                        _npz_out = masks_dir / f"masks_f{out_frame_idx:06d}.npz"
                        if _unchanged_ids:
                            _write_futures.append(_write_pool.submit(
                                _write_npz_frame_merging_unchanged,
                                _npz_out, _frame_npz_arrays, _unchanged_ids, _npz_key_pat, _io_sem,
                            ))
                        else:
                            _write_futures.append(_write_pool.submit(
                                _write_npz_frame, _npz_out, _frame_npz_arrays, _io_sem,
                            ))
                    else:
                        _write_futures.append(_write_pool.submit(
                            _write_png_batch, _frame_png_writes, _io_sem
                        ))

                    masks_metadata[out_frame_idx] = frame_masks
                    quality_calculator.update_forward(out_frame_idx, frame_masks_for_quality)
                    del frame_masks_for_quality

                    # Delete output tensors/arrays
                    if _winners is not None:
                        del _winners
                    if out_mask_logits is not None:
                        del out_mask_logits
                    if out_low_res_masks is not None:
                        del out_low_res_masks
                    if out_obj_scores is not None:
                        del out_obj_scores
                    del result

                    # SAM2/SAM3 only: clean up cached frames to bound memory
                    if inference_state is not None:
                        self._cleanup_inference_state(inference_state, out_frame_idx, frames_to_keep=20, reverse=False, verbose=True)

                    if (out_frame_idx + 1) % 50 == 0:
                        cuda_device_index = _get_cuda_device_index(device)
                        if cuda_device_index is not None:
                            gpu_allocated = torch.cuda.memory_allocated(device=cuda_device_index) / (1024**3)
                            gpu_peak = torch.cuda.max_memory_allocated(device=cuda_device_index) / (1024**3)
                        else:
                            gpu_allocated = 0.0
                            gpu_peak = 0.0
                        process = psutil.Process()
                        ram_used = process.memory_info().rss / (1024**3)
                        print(f"  Forward: Frame {out_frame_idx + 1}/{num_frames} | "
                              f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                              f"RAM: {ram_used:.2f}GB")
                        torch.cuda.reset_peak_memory_stats()

            # Always flush forward writes before starting the backward pass.
            # Both passes write to the same per-frame files (forward and backward cover
            # overlapping frame indices), so in-flight forward writes must complete before
            # backward writes begin to avoid concurrent writes to the same path.
            _write_pool.shutdown(wait=True)
            for _wf in _write_futures:
                _wf.result()
            _write_futures.clear()
            _write_pool = ThreadPoolExecutor(max_workers=2)
            print("  Forward mask writes flushed.")

            # Propagate annotations - BACKWARD direction (last frame → 0)
            # IMPORTANT: Always propagate backward to ensure full video coverage from both directions
            # This provides better quality by having bidirectional temporal context

            if self.no_backward_propagation:
                print("\n  Skipping backward propagation (disabled by --no-backward flag)")
                print("  WARNING: This may result in lower quality segmentation as only forward propagation is used")
            else:
                print(f"\nPropagating annotations BACKWARD (frames {num_frames-1} to 0)...")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _bwd_autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if "cuda" in self.device
                    else contextlib.nullcontext()
                )
                with _bwd_autocast:
                    # Build backward propagation iterator
                    if self.use_sam3:
                        propagate_iterator = self.video_predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=num_frames - 1,
                            max_frame_num_to_track=num_frames,
                            reverse=True,
                            propagate_preflight=True,
                        )
                    else:
                        propagate_iterator = self.video_predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=num_frames - 1,
                            max_frame_num_to_track=num_frames,
                            reverse=True,
                        )

                    for result in propagate_iterator:
                        # Unpack result
                        if self.use_sam3:
                            out_frame_idx, out_obj_ids, out_low_res_masks, out_mask_logits, out_obj_scores = result
                        else:
                            out_frame_idx, out_obj_ids, out_mask_logits = result
                            out_low_res_masks = None
                            out_obj_scores = None

                        frame_masks = {}
                        frame_masks_for_quality = {}

                        if self.exclusive_masks and len(out_obj_ids) > 0:
                            _logits_2d = out_mask_logits.squeeze(1)
                            _bg = torch.zeros(1, _logits_2d.shape[1], _logits_2d.shape[2],
                                              device=_logits_2d.device, dtype=_logits_2d.dtype)
                            _all_logits = torch.cat([_bg, _logits_2d], dim=0)
                            _winners = torch.argmax(_all_logits, dim=0)
                            del _logits_2d, _all_logits
                        else:
                            _winners = None

                        _frame_png_writes = []
                        _frame_npz_arrays = {}

                        _n_masks = out_mask_logits.shape[0] if out_mask_logits is not None else 0
                        for i, obj_id in enumerate(out_obj_ids):
                            if i >= _n_masks:
                                continue
                            if _winners is not None:
                                mask = (_winners == i + 1).cpu().numpy()
                            else:
                                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                            obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                            obj_color = object_colors.get(str(obj_id), [255, 0, 0])
                            score = (float(out_obj_scores[i]) if out_obj_scores is not None else 1.0)
                            mask_u8 = (mask * 255).astype(np.uint8)
                            del mask

                            if _mask_format == "npz":
                                _npz_key = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}"
                                _frame_npz_arrays[_npz_key] = mask_u8
                                frame_masks[obj_id] = {
                                    'filename': f"masks_f{out_frame_idx:06d}.npz",
                                    'npz_key': _npz_key,
                                    'score': score,
                                    'name': obj_name,
                                    'color': obj_color,
                                }
                            else:
                                mask_filename = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}.png"
                                _frame_png_writes.append((masks_dir / mask_filename, mask_u8))
                                frame_masks[obj_id] = {
                                    'filename': mask_filename,
                                    'score': score,
                                    'name': obj_name,
                                    'color': obj_color,
                                }

                            frame_masks_for_quality[obj_id] = mask_u8

                        _io_sem.acquire()
                        if _mask_format == "npz":
                            _npz_out = masks_dir / f"masks_f{out_frame_idx:06d}.npz"
                            if _unchanged_ids:
                                _write_futures.append(_write_pool.submit(
                                    _write_npz_frame_merging_unchanged,
                                    _npz_out, _frame_npz_arrays, _unchanged_ids, _npz_key_pat, _io_sem,
                                ))
                            else:
                                _write_futures.append(_write_pool.submit(
                                    _write_npz_frame, _npz_out, _frame_npz_arrays, _io_sem,
                                ))
                        else:
                            _write_futures.append(_write_pool.submit(
                                _write_png_batch, _frame_png_writes, _io_sem
                            ))

                        masks_metadata[out_frame_idx] = frame_masks
                        quality_calculator.update_backward(out_frame_idx, frame_masks_for_quality)
                        del frame_masks_for_quality

                        if _winners is not None:
                            del _winners
                        if out_mask_logits is not None:
                            del out_mask_logits
                        if out_low_res_masks is not None:
                            del out_low_res_masks
                        if out_obj_scores is not None:
                            del out_obj_scores
                        del result

                        if inference_state is not None:
                            self._cleanup_inference_state(inference_state, out_frame_idx, frames_to_keep=20, reverse=True, verbose=True)

                        if (out_frame_idx + 1) % 50 == 0:
                            cuda_device_index = _get_cuda_device_index(device)
                            if cuda_device_index is not None:
                                gpu_allocated = torch.cuda.memory_allocated(device=cuda_device_index) / (1024**3)
                                gpu_peak = torch.cuda.max_memory_allocated(device=cuda_device_index) / (1024**3)
                            else:
                                gpu_allocated = 0.0
                                gpu_peak = 0.0
                            process = psutil.Process()
                            ram_used = process.memory_info().rss / (1024**3)
                            print(f"  Backward: Frame {out_frame_idx + 1}/{num_frames} | "
                                  f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                                  f"RAM: {ram_used:.2f}GB")
                            torch.cuda.reset_peak_memory_stats()

            # Flush all pending mask writes
            _write_pool.shutdown(wait=True)
            for _wf in _write_futures:
                _wf.result()
            print(f"  All mask writes complete ({len(_write_futures)} frames flushed).")

            # Clean up inference state memory
            model_type = "SAM3" if self.use_sam3 else "SAM2"
            print(f"\nCleaning up {model_type} inference state...")
            if hasattr(inference_state, 'non_cond_frame_outputs'):
                inference_state.non_cond_frame_outputs.clear()
            elif isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
                for obj_idx in range(len(inference_state.get("obj_ids", []))):
                    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                    obj_output_dict.get("non_cond_frame_outputs", {}).clear()
            print(f"OK: Cleaned {model_type} memory")

            torch.cuda.empty_cache()

            # Save quality metrics calculated during propagation
            # Skip saving when --only-updated is used with unchanged objects: the metrics here
            # only cover updated objects (high background ratio due to missing unchanged masks).
            # The caller will recalculate and save with all objects merged.
            if not skip_quality_save:
                inter_frame_changes, background_ratios, overlap_ratios = quality_calculator.get_results()
                quality_calculator.print_summary()
                save_quality_metrics(str(output_dir), inter_frame_changes, background_ratios, overlap_ratios)
            else:
                print("  Skipping partial quality metrics save (will be recalculated with all objects after merging unchanged masks)")

            print(f"\nOK: Generated masks for {len(masks_metadata)} frames")
            # Return frame_dir to allow reuse during export, defer cleanup to caller
            return masks_metadata, object_names, object_colors, num_frames, str(temp_dir), is_persistent

        except Exception:
            # Clean up on error only
            if not is_persistent:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary frames after error: {temp_dir}")
                except Exception as cleanup_error:
                    print(f"WARNING: Could not clean up temp directory: {cleanup_error}")
            raise
    
    def export_masks(self, masks_by_frame, video_path, object_names, output_dir):
        """Verify mask files (already written during propagation)."""
        print("Verifying mask files...")

        masks_dir = Path(output_dir) / "masks"
        verified_count = 0
        missing_count = 0
        checked_files = set()  # NPZ files are shared across objects; only check each path once

        for frame_idx in sorted(masks_by_frame.keys()):
            frame_masks = masks_by_frame[frame_idx]
            for obj_id, mask_data in frame_masks.items():
                mask_path = masks_dir / mask_data['filename']
                if mask_path in checked_files:
                    verified_count += 1  # already confirmed present
                    continue
                checked_files.add(mask_path)
                if not mask_path.exists():
                    print(f"WARNING: Missing mask file: {mask_path}")
                    missing_count += 1
                else:
                    verified_count += 1

        if missing_count > 0:
            print(f"WARNING: {missing_count} mask entries are missing!")
        print(f"OK: Verified {verified_count} mask entries in {masks_dir}")
        return verified_count

    def export_metadata(self, annotations_data, masks_by_frame, output_dir, num_frames=None, video_path=None, overlay_opacity=0.4):
        """Export processing metadata with file paths"""
        print("Exporting metadata...")

        metadata = {
            "processing_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_file": self.config_file,
                "checkpoint_file": self.checkpoint_file,
                "offload_to_cpu": self.offload_to_cpu,
                "total_frames_processed": len(masks_by_frame),
                "total_masks_generated": sum(len(masks) for masks in masks_by_frame.values()),
                "objects_detected": list(set(
                    obj_id for masks in masks_by_frame.values()
                    for obj_id in masks.keys()
                )),
                "overlay_opacity": overlay_opacity
            },
            "file_paths": {
                "original_video_path": str(Path(video_path).resolve()) if video_path else None,
                "segmented_video_filename": "segmented_video.avi",
                "metadata_filename": "processing_metadata.json"
            },
            "original_annotations": annotations_data
        }

        # Save metadata
        metadata_path = Path(output_dir) / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"OK: Exported metadata to {metadata_path}")
        return True

    @staticmethod
    def cleanup_frame_dir(frame_dir: str, is_persistent: bool) -> None:
        """
        Clean up the frame directory after video export.

        Args:
            frame_dir: Path to the frame directory
            is_persistent: If True, keep the directory; if False, delete it
        """
        if is_persistent:
            print(f"Keeping frames in persistent directory: {frame_dir}")
        else:
            try:
                shutil.rmtree(frame_dir)
                print(f"Cleaned up temporary frames: {frame_dir}")
            except Exception as e:
                print(f"WARNING: Could not clean up temp directory: {e}")


def _load_masks_metadata(masks_dir: Path, obj_ids=None):
    """
    Scan a masks directory and return metadata dicts keyed by frame_idx.
    Supports both PNG (mask_f{frame:06d}_{name}_id{obj_id}.png) and
    NPZ bundles (masks_f{frame:06d}.npz with matching keys inside).

    Returns: (masks_by_frame, object_names_by_id)
    """
    png_pattern = re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)\.png$")
    npz_key_pattern = re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)$")
    masks_by_frame = {}
    object_names_found = {}

    if not masks_dir.exists():
        return masks_by_frame, object_names_found

    # Scan individual PNG files
    for filepath in sorted(masks_dir.glob("mask_f*.png")):
        m = png_pattern.match(filepath.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        obj_name = m.group(2)
        obj_id = int(m.group(3))
        if obj_ids is not None and obj_id not in obj_ids:
            continue
        masks_by_frame.setdefault(frame_idx, {})[obj_id] = {"filename": filepath.name}
        object_names_found[obj_id] = obj_name

    # Scan NPZ bundles (masks_f{frame:06d}.npz)
    for filepath in sorted(masks_dir.glob("masks_f*.npz")):
        try:
            data = np.load(str(filepath))
            for key in data.files:
                km = npz_key_pattern.match(key)
                if not km:
                    continue
                frame_idx = int(km.group(1))
                obj_name = km.group(2)
                obj_id = int(km.group(3))
                if obj_ids is not None and obj_id not in obj_ids:
                    continue
                masks_by_frame.setdefault(frame_idx, {})[obj_id] = {
                    "filename": filepath.name,
                    "npz_key": key,
                }
                object_names_found[obj_id] = obj_name
        except Exception:
            continue

    return masks_by_frame, object_names_found


def main():
    """Main processing function"""
    os.umask(0o002)  # ensure group-writable output for shared results dirs
    parser = argparse.ArgumentParser(
        description="Process SAM2 annotations and generate segmented output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available preset models:
  {', '.join(MODEL_CONFIGS.keys())}

Examples:
  # Use default model (sam2-base+)
  python sam2_process.py annotations.json video.mp4

  # Use SAM2.1 large model
  python sam2_process.py annotations.json video.mp4 --model sam2.1-large

  # Use CPU offloading for memory optimization
  python sam2_process.py annotations.json video.mp4 --offload-to-cpu

  # Use a specific GPU (e.g., cuda:1)
  python sam2_process.py annotations.json video.mp4 --device cuda:1

  # Use CPU explicitly
  python sam2_process.py annotations.json video.mp4 --device cpu

  # Use custom config and checkpoint
  python sam2_process.py annotations.json video.mp4 \\
    --config configs/sam2/sam2_hiera_l.yaml \\
    --checkpoint checkpoints/sam2_hiera_large.pt
        """
    )

    parser.add_argument("annotation_file", help="Path to annotation JSON file from SAM2 Video UI")
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument("--output-dir", default="sam2_output", help="Output directory (default: sam2_output)")

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=None,
                           choices=list(MODEL_CONFIGS.keys()),
                           help="Preset model name (default: auto-select based on GPU memory)")
    parser.add_argument("--max-cond-frames-in-attn", type=int, default=-1,
                       help="SAM3 only: number of temporally-closest correction frames the "
                            "tracker attends to per propagated frame. -1 (default) = unlimited, "
                            "matching SAM2's own default. Ignored when using SAM2.")
    model_group.add_argument("--config", help="Custom config YAML path (requires --checkpoint)")

    parser.add_argument("--checkpoint", help="Custom checkpoint path (requires --config)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Mask overlay opacity (default: 0.4)")
    parser.add_argument("--no-bfloat16", action="store_true", dest="no_bfloat16",
                       help="Disable bfloat16 autocast on CUDA. SAM2/SAM3 store internal "
                            "tensors in bfloat16 regardless, so this may cause dtype mismatch "
                            "errors. Primarily useful on pre-Ampere GPUs if autocast causes "
                            "problems. On pre-Ampere GPUs without this flag you will be asked "
                            "interactively.")
    parser.add_argument("--offload-to-cpu", action="store_true",
                       help="Offload video frames and model state to CPU to reduce GPU memory usage (slightly increases CPU memory usage and at a cost of a slightly slower speed)")
    parser.add_argument("--async-loading", action="store_true",
                       help="Use async frame loading (not much benefit. Just kept this feature as it is available)")
    parser.add_argument("--smooth-masks", action="store_true",
                       help="Apply morphological smoothing to reduce pixelation in exported masks (preserves binary masks)")
    parser.add_argument("--frame-dir", type=str, default=None,
                       help="Persistent directory for video frames (default: auto-generated in the system temp directory). If specified, frames will be reused from previous runs and not deleted after processing.")
    parser.add_argument("--frame-cache-size", type=int, default=20,
                       help="Number of frames to keep in memory cache (default: 20, ~2GB). Minimum: 10, Recommended: 20-50.")
    parser.add_argument("--frame-format", type=str, default="jpg", choices=["jpg", "png"],
                       help="Format for extracted frames (default: jpg). Use 'png' for lossless quality at the cost of larger files.")
    parser.add_argument("--no-backward", action="store_true", dest="no_backward",
                       help="Disable backward propagation (not recommended, may result in lower quality segmentation for frames before first annotation)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1'). Default: auto-detect (CUDA if available, else CPU)")
    parser.add_argument("--only-updated", action="store_true",
                       help="Re-segment only objects marked as updated in the annotation file (updated_objects field). Unchanged objects reuse existing masks from --prev-results or the output directory.")
    parser.add_argument("--prev-results", type=str, default=None,
                       help="Directory with previous segmentation results to reuse masks from (used with --only-updated). Defaults to the output directory.")
    parser.add_argument("--sam3-project", type=str, default=None,
                       help="Path to SAM3 project directory. Required when the annotation file references SAM3 objects and processing runs on a different machine from where annotations were created.")
    parser.add_argument("--no-final-quality-recompute", action="store_true",
                       dest="no_final_quality_recompute",
                       help="Skip the end-of-run quality-metrics recompute over the full merged mask set (--only-updated / SAM3-linked / covered-union). quality_metrics.npz then covers only SAM2-propagated objects.")
    parser.add_argument("--video-only", action="store_true",
                       help="Skip segmentation entirely; create/recreate the output video from existing masks in the output directory.")
    parser.add_argument("--exclusive-masks", action="store_true", dest="exclusive_masks",
                       help="Winner-takes-all per pixel: for each pixel, only the object with the highest logit is assigned (background wins if all logits < 0). Incompatible with --only-updated.")
    parser.add_argument("--mask-format", type=str, default="png", choices=["png", "npz"],
                       help="Output format for segmentation masks (default: png). Use 'npz' to write one compressed archive per frame (all objects bundled); ~3x faster to load in downstream tools, ~50%% smaller files on disk.")
    parser.add_argument("--vos-optimized", action="store_true", dest="vos_optimized",
                       help="Enable torch.compile on SAM2's core components for faster propagation. "
                            "WARNING: the first run compiles and profiles GPU kernels, which takes 5-30 min. "
                            "Subsequent videos in the same process reuse the compiled cache and run faster. "
                            "Only useful for SAM2 batch jobs; ignored for SAM3. Requires PyTorch 2.5.1+.")
    parser.add_argument("--overwrite", action="store_true",
                       help="If the output directory has existing results, proceed without deleting them and overwrite on name conflicts (non-interactive).")

    args = parser.parse_args()

    # Validate custom config/checkpoint usage
    if args.config and not args.checkpoint:
        parser.error("--config requires --checkpoint")
    if args.checkpoint and not args.config:
        parser.error("--checkpoint requires --config")

    # Exclusive masks requires all objects to be segmented in the same run
    if getattr(args, 'exclusive_masks', False) and args.only_updated:
        parser.error("--exclusive-masks is incompatible with --only-updated: winner-takes-all requires all objects' logits to be present in the same segmentation run")

    # Auto-select model if not specified
    if args.model is None and not args.config:
        args.model = _auto_select_default_model()

    # Validate input files
    if not os.path.exists(args.annotation_file):
        print(f"ERROR: Annotation file not found: {args.annotation_file}")
        return 1

    if not os.path.exists(args.video_file):
        print(f"ERROR: Video file not found: {args.video_file}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check if output already exists
    masks_dir = output_dir / "masks"
    video_file = output_dir / "segmented_video.avi"
    metadata_file = output_dir / "processing_metadata.json"

    existing_items = []
    if masks_dir.exists():
        _mask_glob = "*.npz" if args.mask_format == "npz" else "*.png"
        _existing_masks = list(masks_dir.glob(_mask_glob))
        if _existing_masks:
            existing_items.append(f"masks/ ({len(_existing_masks)} files)")
    if video_file.exists():
        existing_items.append("segmented_video.avi")
    if metadata_file.exists():
        existing_items.append("processing_metadata.json")

    if existing_items:
        print("\nWARNING: Output directory contains existing files:")
        for item in existing_items:
            print(f"  - {item}")

        if args.only_updated or args.video_only:
            # Deleting would destroy masks that --only-updated or --video-only needs to reuse
            if args.overwrite:
                print("Proceeding without deletion (--overwrite)...")
            else:
                print("\nOptions:")
                print("  1. Proceed without deleting (existing masks will be reused)")
                print("  2. Abort")

                while True:
                    choice = input("\nEnter choice (1/2): ").strip()
                    if choice == '1':
                        print("Proceeding without deletion...")
                        break
                    elif choice == '2':
                        print("Aborted by user")
                        return 1
                    else:
                        print("Invalid choice. Please enter 1 or 2")
        else:
            if args.overwrite:
                print("Proceeding without deletion (--overwrite)...")
            else:
                print("\nOptions:")
                print("  1. Delete all and proceed")
                print("  2. Proceed without deleting (may overwrite)")
                print("  3. Abort")

                while True:
                    choice = input("\nEnter choice (1/2/3): ").strip()
                    if choice == '1':
                        # Delete existing
                        if masks_dir.exists():
                            shutil.rmtree(masks_dir)
                        if video_file.exists():
                            video_file.unlink()
                        if metadata_file.exists():
                            metadata_file.unlink()
                        print("Deleted existing files. Proceeding...")
                        break
                    elif choice == '2':
                        print("Proceeding without deletion...")
                        break
                    elif choice == '3':
                        print("Aborted by user")
                        return 1
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3")

    print("=" * 60)
    print("SAM2 Annotation Processor")
    print("=" * 60)
    print(f"Annotation file: {args.annotation_file}")
    print(f"Video file: {args.video_file}")
    print(f"Output directory: {output_dir}")
    if args.offload_to_cpu:
        print(f"Memory optimization: CPU offloading enabled")
    print()

    # --video-only: recreate output video from existing masks, no segmentation
    if args.video_only:
        masks_dir = output_dir / "masks"
        masks_by_frame, object_names_found = _load_masks_metadata(masks_dir)
        if not masks_by_frame:
            print(f"ERROR: No masks found in {masks_dir}")
            return 1
        # Load object colors from metadata if available
        object_colors = {}
        meta_file = output_dir / "processing_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            orig = meta.get("original_annotations", {})
            object_names_meta = orig.get("object_names", {})
            object_colors = orig.get("object_colors", {})
            object_names_found = {int(k): v for k, v in object_names_meta.items()} if object_names_meta else object_names_found
        object_names = {str(k): v for k, v in object_names_found.items()}
        fps_val = args.fps
        cap = cv2.VideoCapture(args.video_file)
        if cap.isOpened() and fps_val == 30.0:
            fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        import torch
        use_gpu_overlay = torch.cuda.is_available()
        export_video_from_dict(
            video_path=args.video_file,
            masks_by_frame=masks_by_frame,
            object_names=object_names,
            object_colors=object_colors,
            output_dir=str(output_dir),
            fps=fps_val,
            overlay_opacity=args.opacity,
            compress=True,
            crf=23,
            use_gpu=use_gpu_overlay,
        )
        print(f"\nVideo recreated from {len(masks_by_frame)} frames of existing masks.")
        return 0

    # Enable lazy loading BEFORE creating SAM2/SAM3 model
    # This prevents loading all frames into memory at once (huge memory savings for long videos)
    enable_lazy_loading(cache_size=args.frame_cache_size, enable_sam3=True)

    # Initialize processor
    try:
        if args.config:
            processor = SAM2Processor(config_file=args.config, checkpoint_file=args.checkpoint,
                                     offload_to_cpu=args.offload_to_cpu, async_loading=args.async_loading,
                                     smooth_masks=args.smooth_masks,
                                     device=args.device, frame_format=args.frame_format,
                                     exclusive_masks=args.exclusive_masks,
                                     mask_format=args.mask_format,
                                     vos_optimized=args.vos_optimized,
                                     no_bfloat16=args.no_bfloat16,
                                     max_cond_frames_in_attn=args.max_cond_frames_in_attn)
        else:
            processor = SAM2Processor(model_name=args.model, offload_to_cpu=args.offload_to_cpu,
                                     async_loading=args.async_loading, smooth_masks=args.smooth_masks,
                                     device=args.device,
                                     frame_format=args.frame_format,
                                     exclusive_masks=args.exclusive_masks,
                                     mask_format=args.mask_format,
                                     vos_optimized=args.vos_optimized,
                                     no_bfloat16=args.no_bfloat16,
                                     max_cond_frames_in_attn=args.max_cond_frames_in_attn)

        # Set no_backward flag
        processor.no_backward_propagation = args.no_backward

    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    
    # Load model
    if not processor.load_model():
        return 1
    
    # Load annotations
    annotations_data = processor.load_annotations(args.annotation_file)
    if not annotations_data:
        return 1

    # --only-updated: filter to just changed objects, remember which ones to reuse
    unchanged_ids = set()
    prev_results_dir = None
    full_annotations_data = annotations_data  # preserved for export_metadata (Fix: bug where metadata loses unchanged objects)
    if args.only_updated:
        updated_ids = set(annotations_data.get("updated_objects", []))
        if not updated_ids:
            print("WARNING: No updated_objects found in annotation file. Processing all objects.")
        else:
            all_ids = {a["object_id"] for a in annotations_data["annotations"]}
            unchanged_ids = all_ids - updated_ids
            prev_results_dir = Path(args.prev_results) if args.prev_results else output_dir
            print(f"--only-updated: re-segmenting {sorted(updated_ids)}, reusing masks for {sorted(unchanged_ids)}")
            # Filter annotations to updated objects only (for segmentation; full data kept in full_annotations_data)
            annotations_data = dict(annotations_data)
            annotations_data["annotations"] = [
                a for a in annotations_data["annotations"]
                if a["object_id"] in updated_ids
            ]

    # Extract SAM3 objects from annotation data (always, not just when --only-updated)
    sam3_obj_ids = {int(x) for x in annotations_data.get("sam3_objects", [])}
    sam3_mask_dirs_rel = annotations_data.get("sam3_mask_dirs", {})
    sam3_project = Path(args.sam3_project) if getattr(args, 'sam3_project', None) else None

    # SAM2 objects declared "covered by SAM3" — excluded from segmentation;
    # their masks are computed as the union of sam3_sub_ids at output time.
    sam2_covered_raw = annotations_data.get("sam2_covered_ids", {})
    sam2_covered_ids = {int(k): v for k, v in sam2_covered_raw.items()}
    if sam2_covered_ids:
        covered_set = set(sam2_covered_ids.keys())
        sam3_obj_ids.update(covered_set)  # treat as SAM3-managed (skip segmentation)
        print(f"SAM2 covered IDs (declared done by SAM3, excluded from re-segmentation): "
              f"{sorted(covered_set)}")

    # SAM3 instances deleted since this annotation file was saved: sam3_process.py
    # --export-sam2 moved their ids to "retired_ids" in the handoff. The handoff is the
    # source of truth, so drop them here even if this (older) annotation file still lists
    # them, and remove their stale masks from the output below. An id that has its own
    # click annotations is a SAM2-native object and is left alone.
    annotated_ids = {a["object_id"] for a in full_annotations_data.get("annotations", [])}
    retired_ids = _read_retired_sam3_ids(sam3_project) - annotated_ids
    if retired_ids:
        dropped = sorted(retired_ids & (sam3_obj_ids | {int(k) for k in sam3_mask_dirs_rel}))
        sam3_obj_ids -= retired_ids
        sam3_mask_dirs_rel = {k: v for k, v in sam3_mask_dirs_rel.items() if int(k) not in retired_ids}
        unchanged_ids -= retired_ids
        if dropped:
            print(f"Skipping retired SAM3 ids {dropped} (instances deleted in the SAM3 project)")

    if sam3_obj_ids and sam3_project:
        # Remove SAM3 objects from unchanged_ids — they'll be linked from native paths
        unchanged_ids -= sam3_obj_ids
        print(f"SAM3 objects detected: {sorted(sam3_obj_ids - set(sam2_covered_ids))} "
              f"(will link from {sam3_project})")
    elif sam3_obj_ids and not sam3_project:
        print(f"SAM3 objects detected: {sorted(sam3_obj_ids)} (no --sam3-project; treating as unchanged)")

    # Get video info
    frame_count, fps, width, height = processor.get_video_info(args.video_file)
    if frame_count is None:
        return 1

    # Use video FPS if not specified
    if args.fps == 30.0 and fps:
        args.fps = fps
    
    frame_dir_used = None
    is_persistent = False

    # Quality metrics recompute: whenever masks that process_segmentation does NOT score
    # during propagation end up in the final set - reused --only-updated masks, SAM3-linked
    # instances, or SAM2 covered-union masks - recompute the metrics over the full merged
    # set at the end. Plain SAM2 runs skip this and keep the cheaper incremental metrics.
    # SAM3-linked / covered-union masks only get merged into masks_by_frame when a
    # --sam3-project is available to read the native masks from.
    _sam3_will_merge = bool(
        (sam3_obj_ids or sam2_covered_ids) and sam3_project and sam3_mask_dirs_rel
    )
    _will_recompute_quality = bool(
        unchanged_ids or _sam3_will_merge
    ) and not args.no_final_quality_recompute

    try:
        # Process segmentation
        # Skip the intermediate quality metrics save when we will recompute over the full
        # merged set below: the partial metrics from propagation would otherwise be written
        # and then immediately overwritten (and are misleading in the meantime).
        result = processor.process_segmentation(
            args.video_file, annotations_data, output_dir, frame_dir=args.frame_dir,
            skip_quality_save=_will_recompute_quality,
            unchanged_ids=unchanged_ids if unchanged_ids else None,
        )
        masks_by_frame, object_names, object_colors, num_frames, frame_dir_used, is_persistent = result

        if masks_by_frame is None:
            print("ERROR: No masks generated")
            return 1

        print(f"\nOK: Generated masks for {len(masks_by_frame)} frames")

        # --only-updated: merge unchanged masks from prev_results into masks_by_frame
        if unchanged_ids and prev_results_dir is not None:
            prev_masks_dir = prev_results_dir / "masks"
            prev_masks, prev_names = _load_masks_metadata(prev_masks_dir, unchanged_ids)
            if prev_masks:
                out_masks_dir = output_dir / "masks"
                out_masks_dir.mkdir(exist_ok=True)

                if prev_results_dir.resolve() != output_dir.resolve() and args.mask_format != "npz":
                    # PNG: copy individual per-object files only when coming from a different dir
                    for frame_masks in prev_masks.values():
                        for data in frame_masks.values():
                            src = prev_masks_dir / data["filename"]
                            dst = out_masks_dir / data["filename"]
                            if src.exists() and not dst.exists():
                                shutil.copy2(str(src), str(dst))

                elif prev_results_dir.resolve() != output_dir.resolve() and args.mask_format == "npz":
                    # NPZ + different output dir: process_segmentation wrote only updated objects
                    # into the new dir (its merge-at-write reads the OUTPUT dir which was empty).
                    # Merge unchanged objects from prev dir into each output NPZ bundle now.
                    print(f"Merging unchanged NPZ masks from {prev_masks_dir} into {out_masks_dir}...")
                    for frame_idx, frame_obj_map in sorted(prev_masks.items()):
                        npz_out = out_masks_dir / f"masks_f{frame_idx:06d}.npz"
                        bundle: dict = {}
                        if npz_out.exists():
                            try:
                                existing = np.load(str(npz_out))
                                bundle = {k: existing[k] for k in existing.files}
                            except Exception:
                                pass
                        for oid, data in frame_obj_map.items():
                            src_path = prev_masks_dir / data["filename"]
                            if not src_path.exists():
                                continue
                            npz_key = data.get("npz_key", "")
                            if data["filename"].endswith(".npz") and npz_key:
                                try:
                                    src_data = np.load(str(src_path))
                                    if npz_key in src_data.files:
                                        bundle[npz_key] = src_data[npz_key]
                                except Exception:
                                    pass
                            elif data["filename"].endswith(".png"):
                                arr = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
                                if arr is not None:
                                    obj_name = prev_names.get(oid, f"Object_{oid}")
                                    bundle[f"mask_f{frame_idx:06d}_{obj_name}_id{oid}"] = arr
                        if bundle:
                            np.savez_compressed(str(npz_out), **bundle)

                # Merge metadata into masks_by_frame, enriching each dict with 'name' and 'color'
                # so export_video_from_dict can use them (it falls back to object_names[int_key]
                # but the dict has string keys from JSON, causing "Object_N" names)
                for frame_idx, frame_masks in prev_masks.items():
                    for oid, data in frame_masks.items():
                        data['name'] = prev_names.get(oid, f"Object_{oid}")
                        raw_color = object_colors.get(str(oid), [255, 0, 0])
                        data['color'] = list(raw_color) if not isinstance(raw_color, list) else raw_color
                    masks_by_frame.setdefault(frame_idx, {}).update(frame_masks)
                for oid, name in prev_names.items():
                    object_names[str(oid)] = name
                print(f"Reused masks for unchanged objects {sorted(unchanged_ids)} from {prev_masks_dir}")
                # Quality metrics for the merged set are recomputed once below, after SAM3
                # links and covered-union masks have also been added to masks_by_frame.
            else:
                print(f"WARNING: No masks found in {prev_masks_dir} for unchanged objects {sorted(unchanged_ids)}")

        if retired_ids:
            n_removed = _remove_retired_mask_files(output_dir / "masks", retired_ids)
            if n_removed:
                print(f"Removed {n_removed} stale mask file(s) of retired SAM3 ids "
                      f"{sorted(retired_ids)} from {output_dir / 'masks'}")

        # Link SAM3 native masks into output_dir/masks/ (zero storage via hard links)
        if sam3_obj_ids and sam3_project and sam3_mask_dirs_rel:
            sam3_meta = _link_sam3_masks_to_output(
                sam3_project, sam3_mask_dirs_rel, sam3_obj_ids,
                output_dir, num_frames, mask_format=args.mask_format)
            # Merge SAM3 entries into masks_by_frame so video overlay + quality metrics include them
            for frame_idx, frame_objs in sam3_meta.items():
                masks_by_frame.setdefault(frame_idx, {}).update(frame_objs)
            # Add SAM3 object names (from sam3_mask_dirs_rel) to object_names
            for str_obj_id, entry in sam3_mask_dirs_rel.items():
                obj_id = int(str_obj_id)
                if obj_id in sam3_obj_ids:
                    object_names[str(obj_id)] = entry.get("name", f"Object_{obj_id}")

        # Compute union masks for SAM2-covered IDs (covered_id = OR of sam3_sub_ids masks).
        # These IDs were declared done by SAM3 but have no direct mask entry; their mask is
        # the pixel-wise union of the sub-instance masks already linked above.
        if sam2_covered_ids and sam3_project and sam3_mask_dirs_rel:
            import cv2 as _cv2
            out_masks_dir = output_dir / "masks"
            out_masks_dir.mkdir(exist_ok=True)
            _ext = "npz" if args.mask_format == "npz" else "png"
            for covered_id, info in sam2_covered_ids.items():
                sub_ids = [int(x) for x in info.get("sam3_sub_ids", [])]
                if not sub_ids:
                    continue
                covered_name = info.get("name", f"Object_{covered_id}")
                object_names[str(covered_id)] = covered_name
                frames_written = 0
                for frame_idx in range(num_frames):
                    union_mask = None
                    for sub_id in sub_ids:
                        sub_entry = sam3_mask_dirs_rel.get(str(sub_id))
                        if sub_entry is None:
                            continue
                        mask_dir = sam3_project / sub_entry["mask_dir_rel"]
                        pattern = sub_entry["mask_filename_pattern"]
                        mask_path = mask_dir / pattern.format(frame=frame_idx)
                        if not mask_path.exists():
                            continue
                        m = _read_mask(mask_path, _cv2)
                        if m is None:
                            continue
                        union_mask = m if union_mask is None else (union_mask | m)
                    if union_mask is not None:
                        out_path = out_masks_dir / (
                            f"mask_f{frame_idx:06d}_{covered_name}_id{covered_id}.{_ext}")
                        _write_mask(out_path, union_mask, _cv2)
                        masks_by_frame.setdefault(frame_idx, {})[covered_id] = {
                            "filename": out_path.name, "name": covered_name}
                        frames_written += 1
                if frames_written:
                    print(f"  Union mask for '{covered_name}' (id {covered_id}): "
                          f"{frames_written} frames written.")
                else:
                    print(f"  WARNING: No sub-id masks found for covered '{covered_name}' "
                          f"(id {covered_id}). Sub-ids: {sub_ids}")

        # Recompute quality metrics over the full merged mask set (SAM2-propagated +
        # --only-updated reused + SAM3-linked instances + SAM2 covered-union masks).
        # Gated by _will_recompute_quality so plain SAM2 runs keep the incremental metrics.
        # Always recompute when the intermediate save was skipped, even if no merge happened
        # (e.g. --only-updated found no previous masks): otherwise no metrics file is written.
        if _will_recompute_quality:
            print("Recalculating quality metrics over all objects "
                  "(SAM2 + SAM3 + covered unions)...")
            from utils import calculate_quality_metrics
            _quality_loader = _make_merged_mask_loader(output_dir / "masks", masks_by_frame)
            _inter, _bg, _overlap = calculate_quality_metrics(
                masks_by_frame, _quality_loader, (height, width), num_frames
            )
            save_quality_metrics(str(output_dir), _inter, _bg, _overlap)
            print("OK: Saved quality metrics for all objects")
        elif _sam3_will_merge and args.no_final_quality_recompute:
            print("NOTE: --no-final-quality-recompute set; quality_metrics.npz covers only "
                  "SAM2-propagated objects, not SAM3/covered-union masks.")

        # Export results
        processor.export_masks(masks_by_frame, args.video_file, object_names, output_dir)

        # Use shared export function from utils - reuses extracted frames for efficiency
        # Auto-enable GPU overlay if inference device is CUDA (unless export would hit memory limits)
        use_gpu_overlay = processor.device.startswith("cuda") if processor.device else False
        export_video_from_dict(
            video_path=args.video_file,
            masks_by_frame=masks_by_frame,
            object_names=object_names,
            object_colors=object_colors,
            output_dir=str(output_dir),
            fps=args.fps,
            overlay_opacity=args.opacity,
            compress=True,
            crf=23,
            frame_dir=frame_dir_used,  # Reuse extracted frames
            use_gpu=use_gpu_overlay,
            gpu_device=processor.device if use_gpu_overlay else None,
        )

        processor.export_metadata(full_annotations_data, masks_by_frame, output_dir, num_frames,
                                video_path=args.video_file, overlay_opacity=args.opacity)

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Frames processed: {num_frames}")
        print()
        print("Generated files:")
        print(f"  - Masks: {output_dir}/masks/")
        print(f"  - Video: {output_dir}/segmented_video.avi")
        print(f"  - Metadata: {output_dir}/processing_metadata.json")
        print()

        return 0

    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up frame directory after export
        if frame_dir_used:
            SAM2Processor.cleanup_frame_dir(frame_dir_used, is_persistent)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
