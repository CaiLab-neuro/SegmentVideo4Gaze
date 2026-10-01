"""
SAM3 Utilities - Helper functions for SAM3 pipeline.

Includes frame extraction, video re-encoding, color generation, instance operations,
and dynamic frame compositing.
"""

import os
import json
import shutil
import time
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import colorsys

from sam3_project import SAM3Concept, SAM3Instance, SAM3Project


def _existing_frame_indices(output_dir: str) -> set:
    """Return the set of frame indices already extracted as NNNNNN.jpg in output_dir."""
    indices = set()
    if not os.path.isdir(output_dir):
        return indices
    for name in os.listdir(output_dir):
        stem, ext = os.path.splitext(name)
        if ext == ".jpg" and len(stem) == 6 and stem.isdigit():
            indices.add(int(stem))
    return indices


def frames_dir_is_complete(frames_dir: str, expected_frames: int) -> bool:
    """
    True if frames_dir holds (approximately) expected_frames extracted JPEGs.

    A killed extraction leaves a shorter (usually contiguous) prefix, so a plain
    os.path.isdir() check would wrongly reuse a partial directory.

    expected_frames comes from cv2.CAP_PROP_FRAME_COUNT, a container-metadata
    estimate that can over-report the frames cap.read() actually decodes (VFR,
    some codecs). A strict >= would then re-extract forever, so allow the same
    max(2, 1%) tolerance used by _verify_frames_dir_count in sam3_ui.py.
    """
    if not expected_frames or expected_frames <= 0:
        # Without a trusted expected count we can't judge completeness; treat
        # any non-empty directory as complete (legacy behavior).
        return bool(_existing_frame_indices(frames_dir))
    tolerance = max(2, int(expected_frames * 0.01))
    return len(_existing_frame_indices(frames_dir)) >= expected_frames - tolerance


def extract_frames_from_video(video_path: str, output_dir: str,
                              max_frames: Optional[int] = None,
                              overwrite: bool = False):
    """
    Extract frames from video to JPG images.
    Reuses SAM2 frame naming convention: 000000.jpg, 000001.jpg, etc.

    Resume-safe: if a previous extraction was killed midway, already-extracted
    frames are kept and only the missing ones are written. The highest existing
    index is always rewritten, since a kill mid-cv2.imwrite can leave that one
    file truncated. When the existing frames form a contiguous prefix 0..N-1
    (the normal interrupted case), the capture seeks directly to N-1 instead of
    decoding the whole prefix again; the seek is validated by comparing the
    decoded frame against the JPEG on disk, falling back to a full decode from
    frame 0 if they disagree (imprecise-seek codecs).

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        max_frames: Optional limit on number of frames to extract
        overwrite: If True, delete existing frame files and re-extract everything
    """
    os.makedirs(output_dir, exist_ok=True)

    if overwrite:
        # Remove ALL existing frame files, not just the indices we are about to
        # rewrite: a previously longer video would otherwise leave stale
        # higher-index frames that mix into (and pass the completeness count of)
        # the new extraction.
        stale = _existing_frame_indices(output_dir)
        for idx in stale:
            try:
                os.remove(os.path.join(output_dir, f"{idx:06d}.jpg"))
            except OSError:
                pass
        if stale:
            print(f"Overwrite: removed {len(stale)} existing frames from {output_dir}")
        have = set()
    else:
        have = _existing_frame_indices(output_dir)
    last_have = max(have) if have else -1
    # Holes in 0..last_have. have is a subset of {0..last_have} by construction,
    # so the count alone determines contiguity.
    n_holes = (last_have + 1) - len(have)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # Fast path: contiguous prefix with no holes — seek straight to the last
    # existing frame instead of decoding the prefix. CAP_PROP_POS_FRAMES after
    # a set() just echoes the requested value, so it cannot detect an imprecise
    # seek; instead decode the frame at the seek target and compare it against
    # the JPEG already on disk. On mismatch (or unreadable file) fall back to
    # decoding from frame 0.
    if last_have > 0 and n_holes == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_have)
        ret, probe = cap.read()
        existing = cv2.imread(os.path.join(output_dir, f"{last_have:06d}.jpg"))
        seek_ok = False
        if ret and existing is not None and probe.shape == existing.shape:
            # JPEG quality-95 round-trip stays within a few gray levels (MAD
            # ~1-3 on real footage); a different frame of a moving video
            # differs far more. A false reject here only costs a full decode
            # from frame 0 (the safe path), so err toward rejecting.
            mad = float(np.mean(np.abs(probe.astype(np.int16) - existing.astype(np.int16))))
            seek_ok = mad < 6.0
        if seek_ok:
            # probe was frame last_have; the read advanced the capture, so the
            # loop below continues at last_have + 1. Rewrite last_have from the
            # probe first (truncation guard for a kill mid-imwrite).
            cv2.imwrite(os.path.join(output_dir, f"{last_have:06d}.jpg"),
                        probe, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx = last_have + 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Some backends need a reopen to reliably rewind after a failed seek
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != 0:
                cap.release()
                cap = cv2.VideoCapture(video_path)

    if have:
        print(f"Resuming extraction in {output_dir}: {len(have)} frames present, "
              f"continuing from frame {frame_idx}"
              + (f" ({n_holes} hole(s) to repair)" if n_holes else ""))

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        if frame_idx not in have or frame_idx == last_have:
            frame_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        frame_idx += 1

    cap.release()
    total = len(_existing_frame_indices(output_dir))
    print(f"Extracted {written} frames to {output_dir} ({total} total)")


def ensure_frames_extracted(video_path: str, frames_dir: str,
                            expected_frames: Optional[int] = None) -> str:
    """
    Make sure frames_dir holds a complete extraction of video_path, resuming a
    previously interrupted extraction if needed. Returns frames_dir.

    Drop-in replacement for the `if not os.path.isdir(frames_dir): extract`
    pattern at call sites, which silently reused partial directories.
    """
    if frames_dir_is_complete(frames_dir, expected_frames or 0):
        print(f"Reusing existing frames in {frames_dir}")
    else:
        n_have = len(_existing_frame_indices(frames_dir))
        if n_have:
            print(f"Frame cache {frames_dir} is incomplete "
                  f"({n_have}/{expected_frames or '?'} frames) — resuming extraction...")
        else:
            print(f"Extracting frames to {frames_dir}...")
        extract_frames_from_video(video_path, frames_dir)
    return frames_dir


def sam3_mask_path(output_dir: str, frame_idx: int, mask_format: str = "png") -> str:
    """Return the on-disk path for a mask without writing it."""
    ext = "npz" if mask_format == "npz" else "png"
    return os.path.join(output_dir, f"{frame_idx:06d}.{ext}")


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    """Bounding box of the nonzero pixels of a 2D mask.

    Returns int32 ``[x_min, y_min, x_max, y_max]`` in pixels, max inclusive (the same
    convention as SAM3's own masks_to_boxes, before its normalization to xywh), or
    ``[-1, -1, -1, -1]`` for an empty mask.
    """
    rows = np.flatnonzero(np.any(mask, axis=1))
    if rows.size == 0:
        return np.full(4, -1, dtype=np.int32)
    cols = np.flatnonzero(np.any(mask, axis=0))
    return np.array([cols[0], rows[0], cols[-1], rows[-1]], dtype=np.int32)


def save_sam3_mask(mask: np.ndarray, output_dir: str, frame_idx: int, mask_format: str = "png") -> str:
    """
    Save SAM3 mask with simple frame-number filename.

    Directory path encodes concept + instance hierarchy, so filename only needs frame number.
    Example: concepts/person/instances/0/masks/000000.png (or .npz)

    NPZ files also store the mask's bounding box under the key ``bbox`` (see mask_bbox).
    It is computed from the mask actually saved rather than taken from SAM3's
    out_boxes_xywh, which SAM3 computes before its non-overlap step and which is absent
    for refined or merged masks. ``np.load(path)["bbox"]`` reads it without decompressing
    the mask. PNG files have no room for it.

    Args:
        mask: Binary mask (H, W) numpy array (0/255 uint8 or bool)
        output_dir: Directory to save mask (should be .../instances/{id}/masks/)
        frame_idx: Frame index
        mask_format: "png" (default) or "npz" (compressed, faster I/O)

    Returns:
        Path to saved mask file
    """
    os.makedirs(output_dir, exist_ok=True)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    mask_path = sam3_mask_path(output_dir, frame_idx, mask_format)
    if mask_format == "npz":
        np.savez_compressed(mask_path, mask=mask, bbox=mask_bbox(mask))
    else:
        cv2.imwrite(mask_path, mask)
    return mask_path


class AsyncMaskWriter:
    """
    Background thread that encodes and writes masks to disk so the GPU
    propagation loop is not stalled on PNG/NPZ encoding and disk I/O.

    The bounded queue caps RAM held by in-flight masks (a 1600x1200 uint8
    mask is ~1.9 MB, so the default 128 entries is at most ~250 MB).
    If the worker hits an error it keeps draining the queue (so submit()
    never deadlocks) and the first error is re-raised by close().

    Usage:
        writer = AsyncMaskWriter(mask_format="png")
        try:
            for ...:
                path = writer.submit(mask_np, output_dir, frame_idx)
        finally:
            writer.close()  # drains queue, joins thread, re-raises worker error
    """

    def __init__(self, mask_format: str = "png", max_queue: int = 128):
        import queue
        import threading
        self.mask_format = mask_format
        self._queue = queue.Queue(maxsize=max_queue)
        self._error: Optional[Exception] = None
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="AsyncMaskWriter")
        self._thread.start()

    def submit(self, mask: np.ndarray, output_dir: str, frame_idx: int) -> str:
        """Queue a mask for writing; returns its final path immediately."""
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        self._queue.put((mask, output_dir, frame_idx))
        return sam3_mask_path(output_dir, frame_idx, self.mask_format)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            mask, output_dir, frame_idx = item
            if self._error is not None:
                continue  # keep draining so producers never block
            try:
                save_sam3_mask(mask, output_dir, frame_idx, self.mask_format)
            except Exception as e:
                self._error = e

    def close(self):
        """Wait for all queued masks to be written; re-raise any worker error."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        if self._error is not None:
            err, self._error = self._error, None
            raise err


def load_sam3_mask(mask_dir: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load a SAM3 mask, checking NPZ then PNG format.

    Args:
        mask_dir: Directory containing mask files (e.g., .../instances/{id}/masks/)
        frame_idx: Frame index to load

    Returns:
        Grayscale uint8 mask array, or None if not found
    """
    npz_path = os.path.join(mask_dir, f"{frame_idx:06d}.npz")
    if os.path.exists(npz_path):
        try:
            return np.load(npz_path)["mask"]
        except Exception:
            pass
    png_path = os.path.join(mask_dir, f"{frame_idx:06d}.png")
    if os.path.exists(png_path):
        return cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    return None


def _find_ffmpeg() -> Optional[str]:
    """Return path to ffmpeg executable, or None if not found."""
    return shutil.which("ffmpeg")


def _find_ffprobe() -> Optional[str]:
    """Return path to ffprobe executable, or None if not found."""
    return shutil.which("ffprobe")


def check_and_reencode_video(video_path: str, output_dir: str) -> str:
    """
    Check if video has sparse keyframes. If so, re-encode with MJPEG for fast random access.
    Returns path to video to use (original or re-encoded).
    Falls back to original video silently if ffmpeg/ffprobe are not installed.

    Args:
        video_path: Path to original video
        output_dir: Directory to save re-encoded video

    Returns:
        Path to video to use (original if keyframes dense, MJPEG if re-encoded)
    """
    ffprobe = _find_ffprobe()
    ffmpeg = _find_ffmpeg()
    if not ffprobe or not ffmpeg:
        print("Warning: ffmpeg/ffprobe not found — skipping keyframe check and re-encoding.")
        return video_path

    probe_cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        frames = result.stdout.strip().split('\n')

        # Count frames between I-frames
        i_frame_indices = [i for i, f in enumerate(frames) if f == 'I']
        if len(i_frame_indices) > 1:
            avg_gop_size = sum(i_frame_indices[i+1] - i_frame_indices[i]
                              for i in range(len(i_frame_indices)-1)) / (len(i_frame_indices)-1)
        else:
            avg_gop_size = len(frames)

        print(f"Average GOP size: {avg_gop_size:.1f}")

        # If sparse (GOP > 30), re-encode with MJPEG
        if avg_gop_size > 30:
            reencoded_path = os.path.join(output_dir, "video_mjpeg.avi")
            reencode_cmd = [
                ffmpeg, "-i", video_path,
                "-c:v", "mjpeg",
                "-q:v", "2",
                "-y",
                reencoded_path
            ]
            print(f"Re-encoding video with MJPEG for fast random access...")
            subprocess.run(reencode_cmd, check=True)
            print(f"Re-encoded video saved to: {reencoded_path}")
            return reencoded_path

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Warning: Failed to probe/re-encode video: {e}")
        print("Using original video without re-encoding.")

    return video_path


def force_reencode_video_mjpeg(video_path: str, output_dir: str) -> str:
    """
    Re-encode video to MJPEG (all-keyframe) format unconditionally.
    MJPEG gives O(1) random frame access with no codec seeking overhead.

    Args:
        video_path: Path to source video
        output_dir: Directory to write video_mjpeg.avi

    Returns:
        Path to re-encoded MJPEG file

    Raises:
        RuntimeError: if ffmpeg is not installed
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install it and ensure it is on your PATH.\n"
            "  Linux/Mac: sudo apt install ffmpeg  or  brew install ffmpeg\n"
            "  Windows:   https://ffmpeg.org/download.html"
        )
    os.makedirs(output_dir, exist_ok=True)
    reencoded_path = os.path.join(output_dir, "video_mjpeg.avi")
    cmd = [ffmpeg, "-i", video_path, "-c:v", "mjpeg", "-q:v", "2", "-y", reencoded_path]
    print(f"Re-encoding to MJPEG: {reencoded_path}")
    subprocess.run(cmd, check=True)
    print(f"Done: {reencoded_path}")
    return reencoded_path


def _van_der_corput(n: int) -> float:
    """Van der Corput sequence in base 2 (bit-reversal fraction).

    Produces a low-discrepancy sequence where any prefix is spread
    maximally across [0, 1]: 1→0.5, 2→0.25, 3→0.75, 4→0.125, …
    Using this for hue ensures adjacent instance indices always have
    large hue differences, regardless of total instance count.
    """
    q, d = 0, 1
    while n:
        d <<= 1
        q = q * 2 + (n & 1)
        n >>= 1
    return q / d


def generate_concept_color(index: int) -> Tuple[int, int, int]:
    """
    Generate visually distinct color for concept using HSV color space.

    Args:
        index: Concept index (0-based)

    Returns:
        RGB color tuple (0-255)
    """
    # Use golden ratio for hue spacing to maximize distinctness
    golden_ratio = 0.618033988749895
    hue = (index * golden_ratio) % 1.0
    saturation = 0.7 + (index % 3) * 0.1  # Vary saturation slightly
    value = 0.9

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def generate_instance_color(index: int, total_instances: int = None) -> Tuple[int, int, int]:
    """
    Generate highly distinct colors for instances within a concept.

    Uses the Van der Corput (bit-reversal) sequence so that any two
    consecutive indices always have a large hue difference.  This is
    much better than even spacing when the total count is unknown or
    changes dynamically, and better than golden-ratio when the count
    is small (it guarantees the maximum minimum pairwise distance for
    any prefix of the sequence).

    Args:
        index: Instance index (0-based)
        total_instances: Unused; kept for backwards compatibility.

    Returns:
        RGB color tuple (0-255)
    """
    # Van der Corput: index 0→0.5 (cyan), 1→0.25 (chartreuse), 2→0.75 (violet),
    # 3→0.125 (orange-yellow), 4→0.625 (teal), 5→0.375 (green), 6→0.875 (rose), …
    hue = _van_der_corput(index + 1)

    # Alternate saturation and value for an extra distinguishing dimension
    saturation = 0.88 + 0.08 * (index % 2)        # 0.88 or 0.96
    value      = 0.95 - 0.08 * ((index // 2) % 3) # 0.95 / 0.87 / 0.79

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def validate_text_prompt(text: str) -> bool:
    """
    Validate text prompt format.

    Args:
        text: Text prompt to validate

    Returns:
        True if valid, False otherwise
    """
    if not text or len(text.strip()) == 0:
        return False

    if len(text) > 200:
        return False

    return True


def estimate_memory_usage(num_frames: int, frame_dimensions: Tuple[int, int],
                         num_instances: int, use_lazy_loading: bool = True) -> float:
    """
    Estimate memory usage for processing.

    Args:
        num_frames: Number of frames in video
        frame_dimensions: (width, height) of frames
        num_instances: Expected number of instances
        use_lazy_loading: Whether lazy loading is enabled

    Returns:
        Estimated memory usage in GB
    """
    width, height = frame_dimensions

    # Frame memory
    if use_lazy_loading:
        frame_memory = 20 * width * height * 3  # 20 frames in cache
    else:
        frame_memory = num_frames * width * height * 3

    # Mask memory (binary masks)
    mask_memory = num_instances * num_frames * width * height

    # Model memory (approximate)
    model_memory = 2 * 1024 * 1024 * 1024  # ~2GB for model

    total_bytes = frame_memory + mask_memory + model_memory
    return total_bytes / (1024 ** 3)  # Convert to GB


def delete_instance(
    concept: SAM3Concept,
    sam3_obj_id: int,
    inference_state: dict,
    sam3_model
):
    """
    Delete instance using SAM3's remove_object().

    Args:
        concept: Concept containing the instance
        sam3_obj_id: SAM3 object ID to delete
        inference_state: SAM3 inference state
        sam3_model: SAM3 model instance
    """
    instance = concept.get_instance_by_sam3_id(sam3_obj_id)
    if instance is None:
        raise ValueError(f"Instance {sam3_obj_id} not found in concept '{concept.name}'")

    # Remove from SAM3 inference state
    sam3_model.remove_object(inference_state, obj_id=sam3_obj_id, is_user_action=True)

    # Mark as deleted (keep metadata for undo)
    instance.deleted = True
    instance.visible = False


def merge_instances(
    concept: SAM3Concept,
    instance_ids: List[int],
    new_name: str,
    project_dir: str
) -> SAM3Instance:
    """
    Merge instances by union of masks frame-by-frame.

    Args:
        concept: Concept containing the instances
        instance_ids: List of SAM3 object IDs to merge
        new_name: Name for merged instance
        project_dir: Project directory

    Returns:
        New merged instance
    """
    # Collect all frames that have masks in any instance
    all_frames = set()
    for inst_id in instance_ids:
        instance = concept.get_instance_by_sam3_id(inst_id)
        if instance is None:
            continue

        mask_dir = os.path.join(project_dir, "concepts", concept.name,
                                "instances", str(inst_id), "masks")
        if not os.path.exists(mask_dir):
            continue

        mask_files = list(Path(mask_dir).glob("*.png")) + list(Path(mask_dir).glob("*.npz"))
        all_frames.update([int(f.stem) for f in mask_files])

    # Create merged instance
    merged_id = max(inst.sam3_obj_id for inst in concept.instances) + 1
    merged_instance = SAM3Instance(
        sam3_obj_id=merged_id,
        user_name=new_name,
        concept_name=concept.name,
        first_detection_frame=min(all_frames) if all_frames else 0,
        last_detection_frame=max(all_frames) if all_frames else 0,
        num_frames_with_mask=len(all_frames)
    )

    # Union masks frame-by-frame
    for frame_idx in sorted(all_frames):
        masks = []
        for inst_id in instance_ids:
            inst_mask_dir = os.path.join(project_dir, "concepts", concept.name,
                                         "instances", str(inst_id), "masks")
            mask = load_sam3_mask(inst_mask_dir, frame_idx)
            if mask is not None:
                masks.append(mask)

        if masks:
            # Union: logical OR of all masks
            merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            output_dir = os.path.join(project_dir, "concepts", concept.name,
                                     "instances", str(merged_id), "masks")

            # Save with simple frame-number filename
            mask_path = save_sam3_mask(merged_mask, output_dir, frame_idx)

    # Mark originals as deleted
    for inst_id in instance_ids:
        instance = concept.get_instance_by_sam3_id(inst_id)
        if instance:
            instance.deleted = True

    # Add merged instance to concept
    concept.instances.append(merged_instance)
    return merged_instance


class DynamicFrameCompositor:
    """
    Composite frames on-demand from original video + per-concept masks.

    CRITICAL: Composite all masks into a single overlay FIRST,
    then blend with original frame ONCE to avoid repeated downweighting.
    """

    def __init__(self, project: SAM3Project, use_mjpeg: bool = True):
        """
        Initialize compositor.

        Frame loading priority: persistent frames_dir (JPEGs) → MJPEG video → original video.

        Args:
            project: SAM3 project
            use_mjpeg: Use MJPEG re-encoded video if available (faster random access)
        """
        self.project = project

        # Extracted-frame directory (fastest: true random access, no seeking)
        candidate = project.get_frames_dir()
        self.frames_dir = candidate if os.path.isdir(candidate) else None
        if self.frames_dir:
            kind = "persistent" if project.frames_dir else "temporary"
            print(f"Using {kind} frame cache for playback: {self.frames_dir}")

        # Open video fallback (MJPEG preferred over original for seeking performance)
        if use_mjpeg and project.mjpeg_video_path and os.path.exists(project.mjpeg_video_path):
            video_path = project.mjpeg_video_path
            if not self.frames_dir:
                print(f"Using MJPEG video for fast playback: {video_path}")
        else:
            video_path = project.video_path

        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.video_cap.release()
            self.video_cap = None
            if not self.frames_dir:
                raise ValueError(f"Failed to open video: {video_path}")
            print(f"Warning: cannot open video ({video_path}); using frame cache only.")
        # Track the next expected frame so sequential reads skip the seek call.
        self._video_cap_next_frame: int = -1

        # Per-frame mask cache: populated by get_composited_frame, read by callers
        # Maps (concept_name, obj_id) -> mask ndarray for the most-recently composited frame.
        self._last_masks: Dict[tuple, np.ndarray] = {}

        # Rate-limit [COMP] diagnostics: print at most once every 2 seconds.
        self._last_comp_print: float = 0.0

        # Pre-allocate ALL per-frame working buffers at init time and force-touch them
        # so OS maps physical pages now.  This eliminates cold-page spikes on the first
        # few render calls (which otherwise took 50–160 seconds on NFS-backed storage).
        probe = self._load_video_frame(0)
        ph, pw = probe.shape[:2]
        self.native_h, self.native_w = ph, pw  # original video resolution
        self._alloc_buffers(ph, pw)

    def _alloc_buffers(self, h: int, w: int):
        """Allocate (or reallocate) all per-frame working buffers."""
        self._frame_rgb_u8  = np.empty((h, w, 3), dtype=np.uint8)
        self._frame_rgb_f32 = np.empty((h, w, 3), dtype=np.float32)
        self._result_u8     = np.empty((h, w, 3), dtype=np.uint8)
        self._overlay_rgb   = np.empty((h, w, 3), dtype=np.float32)
        self._overlay_alpha = np.empty((h, w),    dtype=np.float32)
        self._s1            = np.empty((h, w),    dtype=np.float32)
        self._s2            = np.empty((h, w),    dtype=np.float32)
        self._s3            = np.empty((h, w),    dtype=np.float32)

    def _load_video_frame(self, frame_idx: int) -> np.ndarray:
        """Load a raw BGR frame using the best available source."""
        # Priority 1: persistent JPEG frame directory (O(1) random access, no codec seeking)
        if self.frames_dir:
            jpg_path = os.path.join(self.frames_dir, f"{frame_idx:06d}.jpg")
            if os.path.exists(jpg_path):
                frame = cv2.imread(jpg_path)
                if frame is not None:
                    return frame
        # Priority 2/3: video file (MJPEG or original)
        if self.video_cap is None:
            raise ValueError(f"Failed to read frame {frame_idx} (no video source)")
        # Skip the seek on sequential access — a bare read() is much faster.
        if self._video_cap_next_frame != frame_idx:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if ret:
            self._video_cap_next_frame = frame_idx + 1
            return frame
        self._video_cap_next_frame = -1
        raise ValueError(f"Failed to read frame {frame_idx}")

    def get_last_masks(self) -> Dict[tuple, np.ndarray]:
        """Return masks loaded during the most recent get_composited_frame call.

        Keys are (concept_name, sam3_obj_id). Callers (e.g. _draw_instance_labels,
        _highlight_selected_instance) use this to avoid re-reading the same files.
        """
        return self._last_masks

    def get_composited_frame(self, frame_idx: int, alpha_multiplier: float = 1.0,
                             focus_concept_name: Optional[str] = None,
                             target_hw: Optional[tuple] = None,
                             render_absorbed_for: Optional[str] = None) -> np.ndarray:
        """
        Composite a single frame with all visible concepts.

        CRITICAL: Composite all masks into a single overlay FIRST,
        then blend with original frame ONCE to avoid repeated downweighting.
        Loaded masks are cached in self._last_masks for the caller.

        Args:
            frame_idx: Frame index to composite.
            alpha_multiplier: Global scale for all mask opacities (0.0 = raw frame, no masks).
            focus_concept_name: When set, only composite this concept's instances.
            target_hw: (height, width) to downscale to before compositing.  All numpy
                work (cvtcolor, blend, final_blend) runs at this smaller resolution,
                giving a ~6× speedup for 1600×1200 → 640×480.  Masks are also resized
                (NEAREST) before blending.  The returned frame is at target_hw size so
                the caller can skip a second PIL resize step.
            render_absorbed_for: Concept name whose absorbed-source instances should also
                be composited, using their absorbing target's color at reduced opacity.
                Useful when showing the selected concept so the user can see the original
                masks of absorbed instances (e.g. during the union-check dialog).

        Returns:
            Composited frame (RGB uint8) at target_hw size if given, else native size.
        """
        _t0 = time.perf_counter()

        # Load original frame via priority-ordered sources
        frame = self._load_video_frame(frame_idx)
        _t_video = time.perf_counter()

        # Downscale to target resolution before any numpy work — reduces all subsequent
        # ops (cvtcolor, fill, per-mask blend, final_blend) proportionally.
        if target_hw is not None:
            th, tw = target_hw
            if frame.shape[0] != th or frame.shape[1] != tw:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        # Reallocate buffers if size changed (target_hw change or rare native resolution change)
        if self._overlay_rgb.shape[0] != h or self._overlay_rgb.shape[1] != w:
            self._alloc_buffers(h, w)

        # Short-circuit: no mask overlay needed
        if alpha_multiplier <= 0.0:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_u8)
            self._last_masks.clear()
            return self._frame_rgb_u8

        # Convert BGR→RGB in-place (no new array); copy to float32 in-place (no new array)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_u8)
        np.copyto(self._frame_rgb_f32, self._frame_rgb_u8, casting='unsafe')
        frame_rgb = self._frame_rgb_f32
        _t_cvtcolor = time.perf_counter()

        _t_bufinit = _t_cvtcolor  # no lazy init needed; buffers already allocated

        overlay_rgb = self._overlay_rgb
        overlay_alpha = self._overlay_alpha
        overlay_rgb.fill(0.0)
        overlay_alpha.fill(0.0)
        _t_fill = time.perf_counter()

        # Clear per-frame mask cache before building composite
        self._last_masks.clear()

        _t_mask_io = 0.0
        _t_color = 0.0
        _t_blend = 0.0
        _n_masks = 0

        # Composite all masks into overlay (or only the focused concept's masks)
        for concept_name in self.project.concept_order:
            if focus_concept_name is not None and concept_name != focus_concept_name:
                continue
            concept = self.project.get_concept_by_name(concept_name)
            if not concept or not concept.visible:
                continue

            for instance in concept.get_visible_instances():
                inst_mask_dir = os.path.join(
                    self.project.get_concept_dir(concept.name),
                    "instances", str(instance.sam3_obj_id), "masks",
                )
                _tm = time.perf_counter()
                mask = load_sam3_mask(inst_mask_dir, frame_idx)
                _t_mask_io += time.perf_counter() - _tm
                if mask is None:
                    continue
                # Downscale mask to match the (already-resized) frame dimensions
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                _n_masks += 1
                self._last_masks[(concept.name, instance.sam3_obj_id)] = mask

                _tc = time.perf_counter()
                color = np.array(instance.get_effective_color(concept.color_rgb), dtype=np.float32)
                _t_color += time.perf_counter() - _tc

                _tb = time.perf_counter()
                # s1 = mask_alpha: uint8 mask → float, scaled by opacity + global multiplier
                np.multiply(mask, alpha_multiplier / 255.0, out=self._s1, casting='unsafe')
                # s2 = 1 - mask_alpha
                np.subtract(1.0, self._s1, out=self._s2)

                # overlay_rgb = overlay_rgb*(1-alpha) + color*alpha, channel-by-channel in-place
                # Zero (H,W,3) temporaries: only two reused (H,W) scratch arrays touched
                for c in range(3):
                    overlay_rgb[:, :, c] *= self._s2                    # scale down existing
                    np.multiply(self._s1, color[c], out=self._s3)       # s3 = color[c]*alpha
                    overlay_rgb[:, :, c] += self._s3                    # add new color contrib

                # overlay_alpha = max(overlay_alpha, mask_alpha), in-place
                np.maximum(overlay_alpha, self._s1, out=overlay_alpha)
                _t_blend += time.perf_counter() - _tb

        _t_loop_end = time.perf_counter()

        # Absorbed-source ghost rendering: for the selected concept only, draw masks of
        # instances that were absorbed into a target, using the target's color at 60% of
        # normal opacity so they appear as a faint "ghost" overlay.  This lets the user
        # see original mask boundaries during the union-check dialog and while annotating.
        if render_absorbed_for is not None and alpha_multiplier > 0.0:
            abs_concept = self.project.get_concept_by_name(render_absorbed_for)
            if abs_concept and abs_concept.visible:
                # Build reverse mapping: absorbed source obj_id -> absorbing target instance
                src_to_target = {}
                for inst in abs_concept.instances:
                    if not inst.deleted and inst.absorbed_source_ids:
                        for sid in inst.absorbed_source_ids:
                            src_to_target[sid] = inst
                for inst in abs_concept.instances:
                    if not (inst.deleted and inst.absorbed_source):
                        continue
                    target_inst = src_to_target.get(inst.sam3_obj_id)
                    if target_inst is None:
                        continue
                    abs_mask_dir = os.path.join(
                        self.project.get_concept_dir(abs_concept.name),
                        "instances", str(inst.sam3_obj_id), "masks",
                    )
                    mask = load_sam3_mask(abs_mask_dir, frame_idx)
                    if mask is None:
                        continue
                    if mask.shape[0] != h or mask.shape[1] != w:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    self._last_masks[(abs_concept.name, inst.sam3_obj_id)] = mask
                    color = np.array(
                        target_inst.get_effective_color(abs_concept.color_rgb),
                        dtype=np.float32,
                    )
                    # 60% of normal alpha so ghosts are visually distinct from active masks
                    np.multiply(mask, 0.6 * alpha_multiplier / 255.0,
                                out=self._s1, casting='unsafe')
                    np.subtract(1.0, self._s1, out=self._s2)
                    for c in range(3):
                        overlay_rgb[:, :, c] *= self._s2
                        np.multiply(self._s1, color[c], out=self._s3)
                        overlay_rgb[:, :, c] += self._s3
                    np.maximum(overlay_alpha, self._s1, out=overlay_alpha)

        # Final blend: frame_rgb = frame_rgb*(1-alpha) + overlay_rgb*alpha, zero (H,W,3) temps
        _tf = time.perf_counter()
        np.subtract(1.0, overlay_alpha, out=self._s2)  # s2 = 1 - overlay_alpha
        for c in range(3):
            overlay_rgb[:, :, c] *= overlay_alpha           # overlay * alpha, in-place
            frame_rgb[:, :, c] *= self._s2                  # frame * (1-alpha), in-place
            frame_rgb[:, :, c] += overlay_rgb[:, :, c]     # combine, in-place
        _t_final_blend = time.perf_counter() - _tf

        # Clip in-place (float32), then copy to pre-allocated uint8 result — zero new allocations
        np.clip(frame_rgb, 0.0, 255.0, out=frame_rgb)
        np.copyto(self._result_u8, frame_rgb, casting='unsafe')
        result = self._result_u8
        _t_clip = time.perf_counter()

        # Rate-limited diagnostics: print at most once every 2 seconds
        if _t_clip - self._last_comp_print >= 2.0:
            _total = _t_clip - _t0
            _tracked = ((_t_video-_t0) + (_t_cvtcolor-_t_video) + (_t_bufinit-_t_cvtcolor) +
                        (_t_fill-_t_bufinit) + _t_mask_io + _t_color + _t_blend +
                        _t_final_blend + (_t_clip - _tf - _t_final_blend))
            focus_tag = f" focus={focus_concept_name}" if focus_concept_name else ""
            print(f"[COMP] frame={frame_idx}{focus_tag} total={_total*1000:.1f}ms | "
                  f"video={(_t_video-_t0)*1000:.1f}ms | "
                  f"cvtcolor={(_t_cvtcolor-_t_video)*1000:.1f}ms | "
                  f"fill={(_t_fill-_t_bufinit)*1000:.1f}ms | "
                  f"mask_io({_n_masks})={_t_mask_io*1000:.1f}ms | "
                  f"blend={_t_blend*1000:.1f}ms | "
                  f"final_blend={_t_final_blend*1000:.1f}ms | "
                  f"clip={(_t_clip-_tf-_t_final_blend)*1000:.1f}ms | "
                  f"untracked={(_total-_tracked)*1000:.1f}ms")
            self._last_comp_print = _t_clip

        return result

    def export_video(self, output_path: str, progress_callback=None):
        """
        Export composited video.

        Args:
            output_path: Output video path
            progress_callback: Optional callback(frame_idx, num_frames)
        """
        from utils import compress_video_with_ffmpeg

        # Create temporary uncompressed video
        temp_path = output_path.replace(".mp4", "_temp.avi")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(
            temp_path, fourcc, self.project.fps,
            self.project.frame_dimensions
        )

        try:
            for frame_idx in range(self.project.num_frames):
                frame_rgb = self.get_composited_frame(frame_idx)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

                if progress_callback:
                    progress_callback(frame_idx, self.project.num_frames)

        finally:
            out.release()

        # Compress with ffmpeg
        print(f"Compressing video with ffmpeg...")
        compress_video_with_ffmpeg(temp_path, output_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def release(self):
        """Release video capture"""
        if self.video_cap:
            self.video_cap.release()

    def __del__(self):
        """Cleanup on destruction"""
        self.release()


def retire_deleted_handoff_ids(
    project: SAM3Project,
    mapping: dict,
    retired: dict,
    covered: Optional[dict] = None,
) -> Tuple[dict, dict, List[str]]:
    """Reconcile a sam2_handoff.json object_mapping with the project's live instances.

    Deleting an instance in SAM3 only sets a flag (its mask files stay on disk), so a
    mapping entry carried over from an earlier export would keep the instance alive in
    sam2_ui.py and sam2_process.py. Here such entries move from ``mapping`` to ``retired``
    (handoff "retired_ids"), which keeps the SAM2 id reserved so it is never reused.
    Multi-instance union entries (``mask_sub_instances``) lose their deleted parts and are
    retired only when no part is left. Any id present in ``mapping`` is removed from
    ``retired``, so an instance restored in SAM3 (and re-assigned its old id by the caller)
    is reinstated. Entries without SAM3 instance fields are left untouched.

    Args:
        project: The SAM3 project.
        mapping: object_mapping after the caller's (re)assignments, {sam2_id_str: entry}.
        retired: retired_ids from the existing handoff, {sam2_id_str: entry}.
        covered: sam2_covered_ids (only used to warn about retired sub-ids).

    Returns:
        (new_mapping, new_retired, messages) - messages describe each change.
    """
    live = {
        (concept.name, inst.sam3_obj_id)
        for concept in project.concepts
        for inst in concept.instances
        if not inst.deleted
    }
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    new_mapping = dict(mapping)
    new_retired = {k: v for k, v in retired.items() if k not in new_mapping}
    messages: List[str] = []

    for sam2_id, entry in list(new_mapping.items()):
        subs = entry.get("mask_sub_instances")
        if subs:
            kept = [s for s in subs if (s.get("concept"), s.get("instance_id")) in live]
            if len(kept) == len(subs):
                continue
            dropped = ", ".join(f"'{s.get('name')}' ({s.get('concept')})"
                                for s in subs if s not in kept)
            if kept:
                new_mapping[sam2_id] = dict(entry, mask_sub_instances=kept)
                messages.append(f"SAM2 id {sam2_id} ('{entry.get('name', '')}'): deleted "
                                f"part(s) {dropped} removed from its union.")
            else:
                new_retired[sam2_id] = {"name": entry.get("name", ""),
                                        "mask_sub_instances": subs, "retired": now}
                del new_mapping[sam2_id]
                messages.append(f"Retired SAM2 id {sam2_id} ('{entry.get('name', '')}'): all "
                                f"parts of its union were deleted. The id stays reserved.")
            continue
        concept_name, instance_id = entry.get("concept"), entry.get("instance_id")
        if concept_name is None or instance_id is None:
            continue
        if (concept_name, instance_id) not in live:
            new_retired[sam2_id] = {"concept": concept_name, "instance_id": instance_id,
                                    "name": entry.get("name", ""), "retired": now}
            del new_mapping[sam2_id]
            messages.append(f"Retired SAM2 id {sam2_id} ('{entry.get('name', '')}', "
                            f"concept={concept_name}): SAM3 instance deleted. "
                            f"The id stays reserved.")

    for cid, info in (covered or {}).items():
        stale = [s for s in info.get("sam3_sub_ids", []) if str(s) in new_retired]
        if stale:
            messages.append(f"WARNING: covered object '{info.get('name', cid)}' (id {cid}) "
                            f"lists retired sub-id(s) {stale}; their masks are no longer "
                            f"part of its union.")
    return new_mapping, new_retired, messages


def export_to_sam2_format(
    project: SAM3Project,
    output_dir: str,
    assignments: List[Dict],
) -> Dict:
    """
    Export SAM3 masks to SAM2-compatible flat structure.

    Each assignment must have keys:
      - "instance": SAM3Instance
      - "concept": SAM3Concept
      - "sam2_obj_id": int  (target SAM2 object ID)

    To merge multiple instances into one SAM2 object, pass multiple
    assignments with the same sam2_obj_id; their masks are OR-merged.

    Writes:
      output_dir/masks/mask_f{frame:06d}_{user_name}_id{sam2_obj_id}.png
      output_dir/segmentation_metadata.json

    Returns dict with object_names, object_colors, and mask counts.
    """
    from utils import load_mask

    masks_dir = Path(output_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Group assignments by sam2_obj_id for merged export
    by_sam2_id: Dict[int, List[Dict]] = defaultdict(list)
    for assignment in assignments:
        by_sam2_id[assignment["sam2_obj_id"]].append(assignment)

    object_names = {}
    object_colors = {}
    total_masks = 0

    for sam2_obj_id, group in by_sam2_id.items():
        # Use the first assignment's instance name as the SAM2 object name
        first = group[0]
        obj_name = first["instance"].user_name
        concept = first["concept"]
        color = first["instance"].get_effective_color(concept.color_rgb or (255, 0, 0))

        object_names[str(sam2_obj_id)] = obj_name
        object_colors[str(sam2_obj_id)] = list(color)

        # Collect all frame→(inst_mask_dir, frame_idx) pairs across all instances in this group.
        # Detect source format: NPZ takes priority if any instance uses it.
        frame_to_dirs: Dict[int, List[Path]] = defaultdict(list)
        source_is_npz = False
        for assignment in group:
            inst = assignment["instance"]
            inst_mask_dir = (
                Path(project.project_dir)
                / "concepts" / concept.name
                / "instances" / str(inst.sam3_obj_id)
                / "masks"
            )
            if not inst_mask_dir.exists():
                continue
            for mask_file in sorted(inst_mask_dir.glob("*.npz")):
                frame_idx = int(mask_file.stem)
                frame_to_dirs[frame_idx].append(inst_mask_dir)
                source_is_npz = True
            if not source_is_npz:
                for mask_file in sorted(inst_mask_dir.glob("*.png")):
                    frame_idx = int(mask_file.stem)
                    frame_to_dirs[frame_idx].append(inst_mask_dir)

        for frame_idx, inst_dirs in sorted(frame_to_dirs.items()):
            masks = []
            for inst_mask_dir in inst_dirs:
                m = load_sam3_mask(str(inst_mask_dir), frame_idx)
                if m is not None:
                    masks.append(m > 0)

            if not masks:
                continue

            merged = np.any(masks, axis=0).astype(np.uint8) * 255

            npz_key = f"mask_f{frame_idx:06d}_{obj_name}_id{sam2_obj_id}"
            if source_is_npz:
                # Write into a per-frame NPZ bundle (compatible with sam2_process.py --mask-format npz)
                npz_path = masks_dir / f"masks_f{frame_idx:06d}.npz"
                existing: dict = {}
                if npz_path.exists():
                    try:
                        old = np.load(str(npz_path))
                        existing = {k: old[k] for k in old.files}
                    except Exception:
                        pass
                existing[npz_key] = merged
                np.savez_compressed(str(npz_path), **existing)
            else:
                dst_path = masks_dir / f"mask_f{frame_idx:06d}_{obj_name}_id{sam2_obj_id}.png"
                cv2.imwrite(str(dst_path), merged)
            total_masks += 1

        # Persist sam2_object_id back to project instances
        for assignment in group:
            assignment["instance"].sam2_object_id = sam2_obj_id

    # Write metadata compatible with sam2_process.py / sam2_ui.py import
    metadata = {
        "processing_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "sam3_export",
            "total_masks_generated": total_masks,
            "objects_detected": [int(k) for k in object_names.keys()],
        },
        "file_paths": {
            "original_video_path": str(project.video_path),
            "segmented_video_filename": "segmented_video.avi",
            "metadata_filename": "segmentation_metadata.json",
        },
        "original_annotations": {
            "object_names": object_names,
            "object_colors": object_colors,
        },
    }
    metadata_path = Path(output_dir) / "segmentation_metadata.json"
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"SAM2-compatible export: {total_masks} masks → {output_dir}")
    return {"object_names": object_names, "object_colors": object_colors, "total_masks": total_masks}


def build_sam3_masks_metadata(project: SAM3Project):
    """
    Build mask enumeration structures compatible with calculate_quality_metrics() from utils.py.

    Assigns flat integer obj_ids (1-based) across all non-deleted instances in the project.

    Returns:
        masks_by_frame: {frame_idx: {obj_id: {"filename": str, "name": str}}}
        load_fn:        callable(frame_idx, obj_id) -> np.ndarray | None
        obj_id_to_info: {obj_id: {"name": str, "color": list, "concept": str, "mask_dir": str}}
    """
    obj_id_counter = 1
    obj_id_to_info: Dict[int, dict] = {}

    for concept in project.concepts:
        for inst in concept.instances:
            if inst.deleted:
                continue
            mask_dir = os.path.join(
                project.project_dir, "concepts", concept.name,
                "instances", str(inst.sam3_obj_id), "masks",
            )
            if not os.path.isdir(mask_dir):
                continue
            color = inst.get_effective_color(concept.color_rgb or (200, 200, 200))
            obj_id_to_info[obj_id_counter] = {
                "name": inst.user_name,
                "color": list(color),
                "concept": concept.name,
                "mask_dir": mask_dir,
            }
            obj_id_counter += 1

    # Enumerate per-frame presence by scanning mask directories
    masks_by_frame: Dict[int, Dict[int, dict]] = {}
    for flat_id, info in obj_id_to_info.items():
        mask_dir = info["mask_dir"]
        for fname in sorted(os.listdir(mask_dir)):
            stem, ext = os.path.splitext(fname)
            if ext not in (".png", ".npz"):
                continue
            try:
                frame_idx = int(stem)
            except ValueError:
                continue
            masks_by_frame.setdefault(frame_idx, {})[flat_id] = {
                "filename": fname,
                "name": info["name"],
            }

    def load_fn(frame_idx: int, obj_id: int):
        info = obj_id_to_info.get(obj_id)
        if info is None:
            return None
        return load_sam3_mask(info["mask_dir"], frame_idx)

    return masks_by_frame, load_fn, obj_id_to_info
