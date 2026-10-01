#!/usr/bin/env python3
import cv2
import json
import numpy as np
import pandas as pd
import os
import glob
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})
import logging, sys
import argparse
import re
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from pathlib import Path
from typing import Optional


@functools.lru_cache(maxsize=8)
def _disk_template(r: int) -> np.ndarray:
    """Pre-compute a boolean disk of radius r, shape (2r+1, 2r+1). Cached across calls."""
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return xx ** 2 + yy ** 2 <= r ** 2


def _score_gaze_all_masks(
    stacked: np.ndarray, xi: int, yi: int, r: int, disk: np.ndarray
) -> np.ndarray:
    """Score one gaze point against N stacked binary masks in a single vectorized pass.

    Args:
        stacked: (N, H, W) uint8 array of binary masks.
        xi, yi: Gaze pixel coordinates (col, row).
        r: Radius in pixels.
        disk: Pre-computed boolean disk of shape (2r+1, 2r+1) from _disk_template(r).

    Returns:
        (N,) float array of per-mask confidence scores.
    """
    N, H, W = stacked.shape
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return np.zeros(N, dtype=float)
    x0 = max(0, xi - r);  x1 = min(W, xi + r + 1)
    y0 = max(0, yi - r);  y1 = min(H, yi + r + 1)
    dy0 = y0 - (yi - r);  dy1 = dy0 + (y1 - y0)
    dx0 = x0 - (xi - r);  dx1 = dx0 + (x1 - x0)
    circle_local = disk[dy0:dy1, dx0:dx1]
    circle_area = int(circle_local.sum())
    if circle_area == 0:
        return np.zeros(N, dtype=float)
    patch = stacked[:, y0:y1, x0:x1]          # (N, h, w)
    hits = (patch[:, circle_local] > 0).sum(axis=1)  # (N,)
    return hits.astype(float) / circle_area


# ---------------------------------------------------------------------------
# SAM3 helpers (module-level so they work in ProcessPoolExecutor workers)
# ---------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _sam3_virtual_key(concept_name: str, user_name: str, sam3_obj_id: int, frame_idx: int) -> str:
    """Return the virtual mask key used for a SAM3 instance.

    Format mirrors SAM2 convention so that _mask_name_to_category and the
    gazed_object/gazed_object_id extraction in process_subject work unchanged.

    Key: mask_f{frame:06d}_{concept_name}_{user_name}_id{obj_id}.png
    → gazed_object    = "{concept_name}_{user_name}"
    → gazed_object_id = "id{obj_id}"
    """
    return f"mask_f{frame_idx:06d}_{concept_name}_{user_name}_id{sam3_obj_id}.png"


def _sam3_mask_labels(mask_key: str) -> set:
    """Labels that can match a SAM3 virtual key in an ignore_object_list."""
    stem = os.path.splitext(mask_key)[0]
    parts = stem.split("_")
    labels = {_normalize_label(mask_key), _normalize_label(stem)}
    if len(parts) >= 4 and parts[0] == "mask" and parts[1].startswith("f") and parts[-1].startswith("id"):
        object_name = "_".join(parts[2:-1])
        object_id = parts[-1]
        labels.add(_normalize_label(object_name))
        labels.add(_normalize_label(object_id))
        if object_id.startswith("id"):
            labels.add(object_id[2:])
    return labels


def _load_sam3_masks_for_frame(
    project_dir: str,
    frame_idx: int,
    sam3_instances: list,
    ignore_labels: set = None,
) -> dict:
    """Load SAM3 masks for one frame from the hierarchical project directory.

    Args:
        project_dir: SAM3 project directory (the one containing project.json).
        frame_idx: Frame index to load.
        sam3_instances: List of (concept_name, user_name, sam3_obj_id) for non-deleted instances.
        ignore_labels: Normalized label set from --ignore-object-list; entries that match
            any label of an instance are skipped.

    Returns:
        Dict mapping virtual mask keys to (H, W) uint8 binary arrays.
    """
    frame_id_str = f"{frame_idx:06d}"
    masks = {}
    for concept_name, user_name, sam3_obj_id in sam3_instances:
        mask_key = _sam3_virtual_key(concept_name, user_name, sam3_obj_id, frame_idx)
        if ignore_labels and (_sam3_mask_labels(mask_key) & ignore_labels):
            continue
        inst_mask_dir = os.path.join(
            project_dir, "concepts", concept_name, "instances", str(sam3_obj_id), "masks"
        )
        # Try NPZ first (compressed), then PNG
        npz_path = os.path.join(inst_mask_dir, f"{frame_id_str}.npz")
        png_path = os.path.join(inst_mask_dir, f"{frame_id_str}.png")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            masks[mask_key] = (data["mask"] > 0).astype(np.uint8)
        elif os.path.exists(png_path):
            mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[mask_key] = (mask > 0).astype(np.uint8)
    return masks


def _sam2_mask_labels(name: str) -> set:
    """Normalized labels that can match a SAM2-style mask filename / npz key.

    Mirrors GazeObjectAligner.mask_labels so module-level workers can filter without
    an aligner instance.
    """
    filename = os.path.basename(name)
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    labels = {_normalize_label(filename), _normalize_label(stem)}
    if len(parts) >= 4 and parts[0] == "mask" and parts[1].startswith("f"):
        object_name = '_'.join(parts[2:-1])
        object_id = parts[-1]
        labels.add(_normalize_label(object_name))
        labels.add(_normalize_label(object_id))
        if object_id.startswith("id"):
            labels.add(object_id[2:])
    return labels


def _load_sam2_masks_for_frame(
    mask_dir: str,
    frame_idx: int,
    ignore_labels: set = None,
    exclude_id_labels: set = None,
) -> dict:
    """Load SAM2-style masks for one frame (per-frame NPZ, else per-object PNGs).

    Args:
        mask_dir: Directory holding ``masks_f{frame:06d}.npz`` or ``*mask_f{frame:06d}*.png``.
        ignore_labels: Normalized labels (from --ignore-object-list) - any mask whose
            labels intersect this set is skipped.
        exclude_id_labels: Normalized id tokens (e.g. ``{"id5", "5"}``) - masks with a
            matching object id are skipped. Used in ``pipeline=both`` to drop SAM2 masks
            that are really SAM3 instances linked/unioned by sam2_process.py.

    Returns:
        Dict mapping mask filename (npz keys get a ``.png`` suffix, matching legacy
        behaviour) to (H, W) uint8 binary arrays.
    """
    frame_id_str = f"{frame_idx:06d}"

    def _skip(name: str) -> bool:
        labels = _sam2_mask_labels(name)
        if ignore_labels and (labels & ignore_labels):
            return True
        if exclude_id_labels and (labels & exclude_id_labels):
            return True
        return False

    npz_path = Path(mask_dir) / f"masks_f{frame_id_str}.npz"
    masks = {}
    if npz_path.exists():
        data = np.load(str(npz_path))
        for k in data.files:
            key = k + ".png"
            if _skip(key):
                continue
            masks[key] = (data[k] > 0).astype(np.uint8)
    else:
        pattern = os.path.join(mask_dir, f"*mask_f{frame_id_str}*.png")
        for mask_path in sorted(glob.glob(pattern)):
            if _skip(os.path.basename(mask_path)):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[os.path.basename(mask_path)] = np.where(mask > 0, 1, 0).astype(np.uint8)
    return masks


def _sam2_native_masks_dir(project_dir) -> Optional[Path]:
    """Return the SAM2 results masks/ dir inside a converted SAM3 project, or None.

    Reads ``sam2_results_subdir`` from sam2_handoff.json (default ``sam2_results``).
    """
    project_dir = Path(project_dir)
    subdir = "sam2_results"
    handoff = project_dir / "sam2_handoff.json"
    if handoff.exists():
        try:
            with open(handoff) as f:
                subdir = json.load(f).get("sam2_results_subdir", subdir)
        except Exception:
            pass
    candidate = project_dir / subdir / "masks"
    return candidate if candidate.is_dir() else None


def _read_handoff_sam3_derived_id_labels(project_dir) -> set:
    """Normalized id tokens for every SAM2 object in sam2_handoff.json that is actually a
    SAM3 instance (object_mapping) or a covered-union (sam2_covered_ids).

    Used by ``pipeline=both`` to avoid double-counting: those SAM2 masks in
    sam2_results/masks/ are dropped in favour of the canonical SAM3 hierarchy masks.
    """
    project_dir = Path(project_dir)
    handoff = project_dir / "sam2_handoff.json"
    labels: set = set()
    if not handoff.exists():
        return labels
    try:
        with open(handoff) as f:
            data = json.load(f)
    except Exception:
        return labels
    for key_group in (data.get("object_mapping", {}), data.get("sam2_covered_ids", {})):
        for sam2_id in key_group:
            labels.add(_normalize_label(f"id{sam2_id}"))
            labels.add(_normalize_label(str(sam2_id)))
    return labels


def _frame_group_worker(args):
    """Worker: load masks for one frame, score all gaze points, return partial probabilities.

    args is a 4- to 6-tuple:
        (frame_idx, gaze_points, mask_dir, r)                              # SAM2 mode
        (frame_idx, gaze_points, project_dir, r, sam3_instances)           # SAM3 mode
        (frame_idx, gaze_points, project_dir, r, sam3_instances, sam2_extra)  # both mode
    where sam2_extra = (sam2_masks_dir, exclude_id_labels, ignore_labels) or None.
    """
    frame_idx, gaze_points, mask_dir, r = args[:4]
    sam3_instances = args[4] if len(args) > 4 else None
    sam2_extra = args[5] if len(args) > 5 else None
    disk = _disk_template(r)

    if sam3_instances is not None:
        # SAM3 mode: masks live under concepts/<concept>/instances/<id>/masks/
        masks = _load_sam3_masks_for_frame(mask_dir, frame_idx, sam3_instances)
        if sam2_extra is not None:
            # both mode: add SAM2-native objects, minus SAM3-derived / covered ids.
            s2_dir, exclude_id_labels, ignore_labels = sam2_extra
            masks.update(_load_sam2_masks_for_frame(
                s2_dir, frame_idx, set(ignore_labels), set(exclude_id_labels)))
    else:
        # SAM2 mode
        masks = _load_sam2_masks_for_frame(mask_dir, frame_idx)

    if not masks:
        return {}, False

    mask_names = list(masks.keys())
    stacked = np.stack(list(masks.values()))
    result = {}
    for gaze_idx, xi, yi in gaze_points:
        confs = _score_gaze_all_masks(stacked, xi, yi, r, disk)
        result[gaze_idx] = dict(zip(mask_names, confs.tolist()))
    return result, True


"""
Published-version aligner to assign an object label to each gaze sample using
segmentation masks and gaze/world-camera timestamps.

Supports two mask source formats:
    SAM2 (default): flat mask directory with files named
        mask_f{frame_id}_{name}_{id}.png  or  masks_f{frame_id}.npz
    SAM3: hierarchical project directory (contains project.json) with masks under
        concepts/{concept}/instances/{obj_id}/masks/{frame_id:06d}.png  or .npz
    SAM3 is auto-detected by the presence of project.json in the resolved directory.

This script:
    1) Resolves the mask source using three fallback strategies (see resolve_mask_dir).
    2) Loads *{subject_id}_{camera}_gaze.csv* and
       *{subject_id}_{camera}_world_timestamps.csv* from the gaze/world directory.
    3) Aligns each gaze timestamp to the most recent world-camera frame at or
       before that timestamp.
    4) Optionally labels blinks from *{subject_id}_{camera}_blinks.csv* and
       removes blink-period gaze samples from the object-assignment stage.
    5) Scores each aligned gaze sample against all readable masks in the matched
       frame and assigns the highest-confidence object label.
    6) Saves the per-gaze object assignments, a pickle of per-mask confidence
       values, and method-figure trajectory/heatmap plots.

Authors: Yanbin Xu
Date: Jan.7 2026

Inputs:
    1) Subject ID(s) and Camera ID(s), provided with the optional CLI arguments
       *--subject-id* and *--camera-id*.
       Subject ID examples: *27* or *27,28*
       Camera ID examples: *child* or *child,parent*
       If omitted, both are auto-discovered from the CSV filenames in
       *gaze_world_dir*.

    2) Gaze and World Camera data directory: /path/to/gaze_world_data
    Within this directory, the code expects
        a) Gaze data CSV files named as *{subject_id}_{camera}_gaze.csv*
        Rows: Each gaze recording ordered by timestamp.
        Columns(required/Column Name Case Specific):
            timestamp [ns]: Timestamp of the gaze point.
            gaze x [px]: Gaze x coordinate in pixels.
            gaze y [px]: Gaze y coordinate in pixels.
        b) World Camera frame timestamps CSV files named as *{subject_id}_{camera}_world_timestamps.csv*
        Rows: Each frame in the egocentric video ordered by timestamp.
        Columns(required/Column Name Case Specific):
            timestamp [ns]: Timestamp of the frame.
        Note: these files can include additional columns such as
              *source_frame_idx*.

        Alternatively, pass *--gaze-csv* and *--world-csv* to point directly at a
        single pair of CSV files, bypassing the *{subject_id}_{camera}_*.csv* naming
        convention. Requires exactly one *--subject-id* and one *--camera-id*
        (no auto-discovery). gaze_world_dir is still required positionally but is
        unused for path resolution in this mode.

    3) Segmentation mask directory: /path/to/segmentation_masks
    The script resolves the mask source for each subject/camera in this order:
        a) Direct SAM3: mask_dir itself contains project.json
           (useful when processing a single subject/camera)
        b) Direct SAM2: mask_dir itself contains mask_f*.png or masks_f*.npz files
        c) Structured layout (subject/camera subdirectories):
           - SAM3 project at {mask_dir}/{subject_id}/{camera}/
           - SAM3 project at {mask_dir}/{subject_id}_{camera}/
           - SAM2 masks at  {mask_dir}/{subject_id}/{camera}/masks/
           - SAM2 masks at  {mask_dir}/{subject_id}_{camera}/masks/
           - SAM2 masks directly at {mask_dir}/{subject_id}/{camera}/ or {subject_id}_{camera}/

        SAM2 mask name format: mask_f{frame_id}_{name}_{id}.png
        SAM3 mask location:    concepts/{concept}/instances/{id}/masks/{frame_id:06d}.png

    4) (Optional) Blink data directory: /path/to/blink_data
    If provided, blink CSVs named *{subject_id}_{camera}_blinks.csv* must contain:
        start timestamp [ns], end timestamp [ns], blink id

Outputs:
    1) Output directory: /path/to/output_directory
    Within this directory, the code will create folders named as *{subject_id}/{camera}/*
    Each folder contains:
        {subject_id}_{camera}_gazed_object[_excluding_ignored_objects].csv
        {subject_id}_{camera}_gaze_object_probabilities[_excluding_ignored_objects].pkl
        If blink removal enabled:
          {subject_id}_{camera}_gaze_blink_labeled.csv
          {subject_id}_{camera}_gaze_blink_removed.csv
        figures/
          {subject_id}_{camera}_trajectory_plot.{png,pdf}
          {subject_id}_{camera}_confidence_heatmap.{png,pdf}

    2) A log file is written to *--log-path* if provided, otherwise to
       *{output_dir}/gaze_object.log*.

Notes:
    SAM2 NPZ mask format: If masks_f{frame_id}.npz files exist alongside PNGs, they are used
    automatically (no flag needed). NPZ reads are ~3x faster than per-object PNG reads.

    SAM3 mode: auto-detected when project.json is present. No extra flag is required.
    Output column gazed_object = "{concept_name}_{user_name}" for SAM3 instances.

    Fast reuse from cached pkl: If {output_dir}/{subject}/{camera}/{subject}_{camera}_gaze_object_probabilities.pkl
    already exists from a previous run, this script will skip all mask I/O and reassign gaze
    objects directly from the cached per-mask confidence scores. A WARNING is logged when this
    shortcut is taken. To force recomputation from masks regardless, pass --recompute.
"""


class RawDescriptionDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass

def make_log_handlers(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    return handlers


def setup_logging(log_path: Path, use_queue: bool = False):
    handlers = make_log_handlers(log_path)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not use_queue:
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers,
            force=True,  # important if running in notebooks / reused envs
        )
        return None, None, None

    log_manager = Manager()
    log_queue = log_manager.Queue()
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    root_logger.handlers = [QueueHandler(log_queue)]
    return log_queue, listener, log_manager


def setup_worker_logging(log_queue):
    if log_queue is None:
        return

    root_logger = logging.getLogger()
    root_logger.handlers = [QueueHandler(log_queue)]
    root_logger.setLevel(logging.INFO)


class SubjectCameraLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"subj={self.extra['subject_id']} camera={self.extra['camera']} | {msg}", kwargs


def pair_logger(logger, subject_id, camera):
    return SubjectCameraLoggerAdapter(logger, {"subject_id": subject_id, "camera": camera})


def format_log_datetime(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now().astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")


class GazeObjectAligner:
    def __init__(
        self,
        gaze_world_dir: str,
        mask_dir: str,
        output_dir: str,
        blink_dir: Optional[str] = None,
        start_plot_time: Optional[float] = None,
        end_plot_time: Optional[float] = None,
        ignore_object_list: Optional[str] = None,
        plot_figures: bool = True,
        logger: Optional[logging.Logger] = None,
        recompute: bool = False,
        within_job_workers: int = 1,
        gaze_csv_path: Optional[str] = None,
        world_csv_path: Optional[str] = None,
        category_sort: str = "first_seen",
        gaze_confidence_threshold: float = 0.5,
        gaze_radius: int = 20,
        pipeline: str = "auto",
    ):
        # pipeline: "auto" (SAM3 if project.json else SAM2), "sam3" (SAM3 instances only),
        # "sam2" (SAM2 masks only; for a converted project uses sam2_results/masks/),
        # "both" (SAM3 instances + SAM2-native objects, de-duplicated via sam2_handoff.json)
        self.pipeline = pipeline
        self.gaze_world_dir = gaze_world_dir
        self.gaze_csv_path = gaze_csv_path
        self.world_csv_path = world_csv_path
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.blink_dir = blink_dir
        self.start_plot_time = start_plot_time
        self.end_plot_time = end_plot_time
        self.logger = logger or logging.getLogger(__name__)
        self.ignore_object_list = ignore_object_list
        self.plot_figures = plot_figures
        self.recompute = recompute
        self.within_job_workers = within_job_workers
        self.category_sort = category_sort
        self.gaze_confidence_threshold = gaze_confidence_threshold
        self.gaze_radius = gaze_radius
        self.ignore_objects = self.load_ignore_objects(ignore_object_list)
        if self.ignore_objects:
            self.logger.info(
                "Ignoring %d object mask labels from %s: %s",
                len(self.ignore_objects),
                ignore_object_list,
                sorted(self.ignore_objects),
            )

    def normalize_object_label(self, label):
        """Normalize object labels so txt entries can use spaces or underscores."""
        return _normalize_label(label)

    def mask_labels(self, mask_path):
        """Return labels that can be used to match one mask filename (SAM2 or SAM3 virtual key)."""
        filename = os.path.basename(mask_path)
        stem = os.path.splitext(filename)[0]
        parts = stem.split('_')
        labels = {self.normalize_object_label(filename), self.normalize_object_label(stem)}

        if len(parts) >= 4 and parts[0] == "mask" and parts[1].startswith("f"):
            object_name = '_'.join(parts[2:-1])
            object_id = parts[-1]
            labels.add(self.normalize_object_label(object_name))
            labels.add(self.normalize_object_label(object_id))
            if object_id.startswith("id"):
                labels.add(object_id[2:])

        return labels

    def load_ignore_objects(self, ignore_object_list):
        """Load optional object labels to ignore, one per line."""
        if not ignore_object_list:
            return set()

        ignore_path = Path(ignore_object_list)
        with ignore_path.open("r") as f:
            labels = {
                self.normalize_object_label(line.split("#", 1)[0])
                for line in f
                if line.split("#", 1)[0].strip()
            }

        return labels

    def excluded_objects_output_suffix(self):
        """Label outputs that were generated after excluding ignored objects."""
        return "_excluding_ignored_objects" if self.ignore_objects else ""

    # ------------------------------------------------------------------
    # SAM3 project helpers
    # ------------------------------------------------------------------

    def _is_sam3_project(self, path: Path) -> bool:
        """Return True if path is a SAM3 project directory (contains project.json)."""
        return (path / "project.json").exists()

    @staticmethod
    def _looks_like_sam2_masks_dir(path: Path) -> bool:
        """Quick check: does this directory contain SAM2-style mask files?"""
        try:
            for f in path.iterdir():
                if f.name.startswith("mask_f") and f.suffix == ".png":
                    return True
                if f.name.startswith("masks_f") and f.suffix == ".npz":
                    return True
        except OSError:
            pass
        return False

    def load_sam3_project_info(self, project_dir: Path) -> list:
        """Load non-deleted instance info from a SAM3 project.

        Reads each concept's concept_metadata.json (which holds the full instance
        list including user_name and deleted flag).

        Returns:
            List of (concept_name, user_name, sam3_obj_id) for all non-deleted instances.
        """
        instances = []
        concepts_dir = project_dir / "concepts"
        if not concepts_dir.is_dir():
            self.logger.warning("SAM3 project has no concepts/ directory: %s", project_dir)
            return instances

        for concept_dir in sorted(concepts_dir.iterdir()):
            if not concept_dir.is_dir():
                continue
            metadata_path = concept_dir / "concept_metadata.json"
            if not metadata_path.exists():
                self.logger.debug("No concept_metadata.json in %s, skipping", concept_dir)
                continue
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
            except Exception as e:
                self.logger.warning("Could not read %s: %s", metadata_path, e)
                continue

            concept_name = meta.get("name", concept_dir.name)
            for inst in meta.get("instances", []):
                if inst.get("deleted", False):
                    continue
                sam3_obj_id = inst["sam3_obj_id"]
                user_name = inst.get("user_name", f"{concept_name}_{sam3_obj_id}")
                instances.append((concept_name, user_name, sam3_obj_id))

        self.logger.info(
            "SAM3 project at %s: %d non-deleted instance(s)", project_dir, len(instances)
        )
        return instances

    # ------------------------------------------------------------------
    # Mask directory resolution
    # ------------------------------------------------------------------

    def resolve_mask_dir(self, subj: str, camera: str) -> Path:
        """Resolve the mask source for a given subject/camera pair.

        Strategy (first match wins):
        1. Direct SAM3: mask_dir itself contains project.json.
        2. Direct SAM2: mask_dir itself contains SAM2-style mask files.
        3. Structured layout with subject/camera subdirectories:
             SAM3 at  {mask_dir}/{subj}/{camera}/         (project.json present)
             SAM3 at  {mask_dir}/{subj}_{camera}/         (project.json present)
             SAM2 at  {mask_dir}/{subj}/{camera}/masks/   (masks/ subdir)
             SAM2 at  {mask_dir}/{subj}_{camera}/masks/   (masks/ subdir)
             SAM2 at  {mask_dir}/{subj}/{camera}/          (mask_f*.png present)
             SAM2 at  {mask_dir}/{subj}_{camera}/          (mask_f*.png present)

        Returns:
            Path to the SAM3 project directory or SAM2 masks directory.
        """
        root = Path(self.mask_dir)

        # Strategy 1 & 2: mask_dir is the mask source directly
        if self._is_sam3_project(root):
            return root
        if self._looks_like_sam2_masks_dir(root):
            return root

        # Strategy 3: structured subject/camera subdirectories
        candidate_roots = [
            root / subj / camera,
            root / f"{subj}_{camera}",
        ]
        for candidate in candidate_roots:
            if not candidate.exists():
                continue
            # SAM3: project.json present
            if (candidate / "project.json").exists():
                return candidate
            # SAM2: explicit masks/ subdirectory
            masks_sub = candidate / "masks"
            if masks_sub.is_dir():
                return masks_sub
            # SAM2: mask files directly in candidate
            if self._looks_like_sam2_masks_dir(candidate):
                return candidate

        checked = [str(root)] + [str(c) for c in candidate_roots]
        raise FileNotFoundError(
            f"Could not find a mask source for subject={subj}, camera={camera}. "
            f"Checked: {', '.join(checked)}"
        )

    def summarize_mask_frames(self, mask_dir: Path) -> dict[str, object]:
        """Summarize which frame ids are present in a mask directory (SAM2 or SAM3)."""
        if self._is_sam3_project(mask_dir):
            return self._summarize_sam3_mask_frames(mask_dir)
        return self._summarize_sam2_mask_frames(mask_dir)

    def _summarize_sam2_mask_frames(self, mask_dir: Path) -> dict:
        frame_pattern = re.compile(r"(?:masks?_f|mask_f)(\d{6})")
        mask_dir = Path(mask_dir)
        frame_ids = sorted(
            {
                int(match.group(1))
                for mask_path in list(mask_dir.glob("masks_f*.npz")) + list(mask_dir.glob("*.png"))
                for match in [frame_pattern.search(mask_path.name)]
                if match is not None
            }
        )
        if not frame_ids:
            return {"count": 0, "frame_ids": set(), "contiguous": False,
                    "min_frame": None, "max_frame": None}
        contiguous = frame_ids == list(range(frame_ids[0], frame_ids[-1] + 1))
        return {
            "count": len(frame_ids),
            "frame_ids": set(frame_ids),
            "contiguous": contiguous,
            "min_frame": frame_ids[0],
            "max_frame": frame_ids[-1],
        }

    def _summarize_sam3_mask_frames(self, project_dir: Path) -> dict:
        """Collect frame IDs present across all SAM3 instance mask directories."""
        frame_ids_set = set()
        concepts_dir = project_dir / "concepts"
        if not concepts_dir.is_dir():
            return {"count": 0, "frame_ids": set(), "contiguous": False,
                    "min_frame": None, "max_frame": None}
        for concept_dir in concepts_dir.iterdir():
            if not concept_dir.is_dir():
                continue
            instances_dir = concept_dir / "instances"
            if not instances_dir.is_dir():
                continue
            for inst_dir in instances_dir.iterdir():
                if not inst_dir.is_dir():
                    continue
                masks_dir = inst_dir / "masks"
                if not masks_dir.is_dir():
                    continue
                for f in masks_dir.iterdir():
                    if f.suffix in (".png", ".npz"):
                        try:
                            frame_ids_set.add(int(f.stem))
                        except ValueError:
                            pass
        frame_ids = sorted(frame_ids_set)
        if not frame_ids:
            return {"count": 0, "frame_ids": set(), "contiguous": False,
                    "min_frame": None, "max_frame": None}
        contiguous = frame_ids == list(range(frame_ids[0], frame_ids[-1] + 1))
        return {
            "count": len(frame_ids),
            "frame_ids": set(frame_ids),
            "contiguous": contiguous,
            "min_frame": frame_ids[0],
            "max_frame": frame_ids[-1],
        }

    def load_mask(self, frame_id: int, mask_dir: str) -> dict[str, np.ndarray]:
        """Load segmentation masks for one frame (SAM2 or SAM3).

        This is the serial path used when within_job_workers == 1.
        For SAM3, sam3_instances must have been set via self._sam3_instances_cache
        before calling this method (done by process_subject).
        """
        mask_dir_path = Path(mask_dir)
        if self._is_sam3_project(mask_dir_path):
            instances = getattr(self, "_sam3_instances_cache", [])
            return _load_sam3_masks_for_frame(
                mask_dir, frame_id, instances, self.ignore_objects
            )

        # SAM2 path
        frame_id_str = f"{frame_id:06d}"
        npz_path = mask_dir_path / f"masks_f{frame_id_str}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path))
            return {k + ".png": (data[k] > 0).astype(np.uint8) for k in data.files}

        pattern = os.path.join(mask_dir, f"*mask_f{frame_id_str}*.png")
        mask_paths = sorted(glob.glob(pattern))
        if not mask_paths:
            self.logger.debug("No masks found for frame %s in %s", frame_id, mask_dir)
            return {}

        masks = {}
        for mask_path in mask_paths:
            if self.ignore_objects and self.mask_labels(mask_path) & self.ignore_objects:
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            masks[os.path.basename(mask_path)] = binary_mask

        return masks

    @staticmethod
    def _mask_name_to_category(mask_name: str) -> str:
        """Convert a mask filename (SAM2 or SAM3 virtual key) into an object category name."""
        stem = Path(mask_name).stem
        parts = stem.split('_')
        if len(parts) >= 4 and parts[0] == "mask" and parts[1].startswith("f") and parts[-1].startswith("id"):
            category_parts = parts[2:-1]
        elif len(parts) > 1:
            category_parts = parts[:-1]
        else:
            category_parts = parts
        category = "_".join(category_parts).strip("_")
        return category if category else stem

    @staticmethod
    def _build_time_edges(time_s: np.ndarray, default_step: float = 1.0 / 30.0) -> np.ndarray:
        """Build bin edges from sample timestamps for bar and heatmap plotting."""
        if len(time_s) == 0:
            return np.array([0.0, default_step], dtype=float)
        if len(time_s) == 1:
            t0 = max(0.0, float(time_s[0]))
            return np.array([t0, t0 + default_step], dtype=float)

        midpoints = (time_s[:-1] + time_s[1:]) / 2.0
        left_edge = max(0.0, float(time_s[0] - (midpoints[0] - time_s[0])))
        right_edge = float(time_s[-1] + (time_s[-1] - midpoints[-1]))
        if right_edge <= midpoints[-1]:
            right_edge = float(midpoints[-1] + default_step)

        edges = np.empty(len(time_s) + 1, dtype=float)
        edges[0] = left_edge
        edges[1:-1] = midpoints
        edges[-1] = right_edge
        return edges

    def plot_method_figures(
        self,
        subject_id_temp: str,
        camera_temp: str,
        gaze_df: pd.DataFrame,
        subject_gaze_probabilities: dict[int, dict[str, float]],
        out_dir: Path,
        category_display_map: Optional[dict] = None,
    ) -> None:
        """Create method figures: trajectory trace and confidence heatmap."""
        if gaze_df.empty:
            self.logger.warning(
                "No gaze rows available for plotting for subject=%s camera=%s",
                subject_id_temp,
                camera_temp,
            )
            return

        if "timestamp [ns]" not in gaze_df.columns:
            self.logger.warning(
                "Cannot create method figures for subject=%s camera=%s because column 'timestamp [ns]' is missing",
                subject_id_temp,
                camera_temp,
            )
            return

        plot_df = gaze_df.sort_values("timestamp [ns]").copy()
        timestamp_ns = pd.to_numeric(plot_df["timestamp [ns]"], errors="coerce").to_numpy(dtype=float)
        valid_time = np.isfinite(timestamp_ns)
        if not valid_time.any():
            self.logger.warning(
                "Cannot create method figures for subject=%s camera=%s because all timestamps are invalid",
                subject_id_temp,
                camera_temp,
            )
            return
        plot_df = plot_df.loc[valid_time].copy()
        timestamp_ns = timestamp_ns[valid_time]
        full_time_s = (timestamp_ns - timestamp_ns[0]) / 1e9

        requested_start_s = 0.0 if self.start_plot_time is None else float(self.start_plot_time)
        requested_end_s = float(full_time_s[-1]) if self.end_plot_time is None else float(self.end_plot_time)

        if requested_start_s < 0.0:
            self.logger.warning(
                "start_plot_time=%.3f is negative for subject=%s camera=%s. Clamping to 0.0.",
                requested_start_s,
                subject_id_temp,
                camera_temp,
            )
            requested_start_s = 0.0
        if requested_end_s < 0.0:
            self.logger.warning(
                "end_plot_time=%.3f is negative for subject=%s camera=%s. Clamping to 0.0.",
                requested_end_s,
                subject_id_temp,
                camera_temp,
            )
            requested_end_s = 0.0
        if requested_end_s < requested_start_s:
            self.logger.warning(
                (
                    "Cannot create method figures for subject=%s camera=%s because "
                    "start_plot_time (%.3f) is greater than end_plot_time (%.3f)."
                ),
                subject_id_temp,
                camera_temp,
                requested_start_s,
                requested_end_s,
            )
            return

        segment_mask = (full_time_s >= requested_start_s) & (full_time_s <= requested_end_s)
        if not segment_mask.any():
            self.logger.warning(
                (
                    "No gaze samples in requested plot window [%.3f, %.3f] s for subject=%s camera=%s "
                    "(available range [0.000, %.3f] s)."
                ),
                requested_start_s,
                requested_end_s,
                subject_id_temp,
                camera_temp,
                float(full_time_s[-1]),
            )
            return

        plot_df = plot_df.loc[segment_mask].copy()
        time_s = full_time_s[segment_mask]
        self.logger.info(
            (
                "Plotting method figures for subject=%s camera=%s using time window [%.3f, %.3f] s "
                "(available [0.000, %.3f] s, points=%d)."
            ),
            subject_id_temp,
            camera_temp,
            requested_start_s,
            requested_end_s,
            float(full_time_s[-1]),
            len(plot_df),
        )
        time_edges = self._build_time_edges(time_s)

        if "gazed_object" in plot_df.columns:
            raw_categories = plot_df["gazed_object"].fillna("no_object").astype(str)
        else:
            raw_categories = pd.Series(["no_object"] * len(plot_df), index=plot_df.index)

        gaze_categories = []
        for category in raw_categories:
            cleaned = category.strip()
            if cleaned == "" or cleaned.lower() in {"none", "nan"}:
                cleaned = "no_object"
            gaze_categories.append(cleaned)

        seen_order = list(dict.fromkeys(gaze_categories))
        has_no_object = "no_object" in seen_order
        real_categories = [cat for cat in seen_order if cat != "no_object"]

        # Discover every category scored in subject_gaze_probabilities, including ones that
        # never won the per-point argmax label (e.g. consistently second-best behind a
        # larger overlapping mask), so both the heatmap and frequency-based sorting reflect
        # the full population of scores, not just who "won" each point.
        heatmap_categories = list(real_categories)
        for probs_one_gaze in subject_gaze_probabilities.values():
            for mask_name in probs_one_gaze.keys():
                category = self._mask_name_to_category(mask_name)
                if category not in heatmap_categories:
                    heatmap_categories.append(category)

        heatmap_row = {cat: i for i, cat in enumerate(heatmap_categories)}
        confidence_matrix = np.zeros((len(heatmap_categories), len(plot_df)), dtype=float)
        for col_idx, gaze_index in enumerate(plot_df.index):
            probs_one_gaze = subject_gaze_probabilities.get(gaze_index, {})
            if probs_one_gaze:
                per_category_conf = {}
                for mask_name, conf in probs_one_gaze.items():
                    category = self._mask_name_to_category(mask_name)
                    conf_val = float(conf)
                    if category not in per_category_conf or conf_val > per_category_conf[category]:
                        per_category_conf[category] = conf_val
                for category, conf_val in per_category_conf.items():
                    confidence_matrix[heatmap_row[category], col_idx] = conf_val
            elif "gazed_object_confidence" in plot_df.columns:
                category = gaze_categories[col_idx]
                if category in heatmap_row:
                    conf_val = float(plot_df.iloc[col_idx]["gazed_object_confidence"])
                    if np.isfinite(conf_val):
                        confidence_matrix[heatmap_row[category], col_idx] = max(0.0, conf_val)

        if self.category_sort == "frequency":
            # "Frequency" is approximated by mean confidence across ALL gaze points (not a
            # count of argmax wins): a category that consistently scores moderately but
            # rarely wins outright (e.g. a small mask overlapping a much larger one) still
            # ranks appropriately, instead of being under-counted by a hard win-count.
            avg_conf = confidence_matrix.mean(axis=1)
            order_indices = sorted(range(len(heatmap_categories)), key=lambda i: avg_conf[i], reverse=True)
            heatmap_categories = [heatmap_categories[i] for i in order_indices]
            confidence_matrix = confidence_matrix[order_indices, :]
            heatmap_row = {cat: i for i, cat in enumerate(heatmap_categories)}

        real_categories = [cat for cat in heatmap_categories if cat in real_categories]

        # "no_object" is not a real gazed object; pin it to the last row with a fixed
        # neutral color instead of letting first-seen/frequency order place it arbitrarily
        # and assign it a near-invisible tab20 color.
        if has_no_object:
            heatmap_categories.append("no_object")
            confidence_matrix = np.vstack([confidence_matrix, np.zeros((1, confidence_matrix.shape[1]))])
        heatmap_row = {cat: i for i, cat in enumerate(heatmap_categories)}

        category_order = real_categories + (["no_object"] if has_no_object else [])
        category_to_y = {cat: i for i, cat in enumerate(category_order)}
        display_map = category_display_map or {}

        figures_dir = Path(out_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        trajectory_height = float(np.clip(1.3 + 0.35 * len(category_order), 3.5, 18.0))
        trajectory_fig, trajectory_ax = plt.subplots(figsize=(20, trajectory_height))
        cmap = plt.get_cmap("tab20")
        NO_OBJECT_COLOR = (0.55, 0.55, 0.55, 1.0)
        category_colors = {"no_object": NO_OBJECT_COLOR}
        for idx, cat in enumerate(real_categories):
            category_colors[cat] = cmap(idx % cmap.N)
        y_values = np.array([category_to_y[cat] for cat in gaze_categories], dtype=float)
        point_colors = [category_colors[cat] for cat in gaze_categories]

        trajectory_ax.plot(
            time_s,
            y_values,
            color="#4a4a4a",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )
        trajectory_ax.scatter(
            time_s,
            y_values,
            c=point_colors,
            s=16,
            edgecolors="none",
            zorder=3,
        )
        trajectory_ax.set_yticks(np.arange(len(category_order)))
        trajectory_ax.set_yticklabels([display_map.get(cat, cat) for cat in category_order])
        trajectory_ax.set_xlabel("Time (s)")
        trajectory_ax.set_ylabel("Gazed Category")
        trajectory_ax.set_title(f"Gaze trajectory ({subject_id_temp}, {camera_temp})")
        trajectory_ax.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        trajectory_ax.set_ylim(-0.5, len(category_order) - 0.5)
        trajectory_ax.grid(axis="x", linestyle="--", alpha=0.35)
        trajectory_fig.tight_layout()

        trajectory_png = figures_dir / f"{subject_id_temp}_{camera_temp}_trajectory_plot.png"
        trajectory_pdf = figures_dir / f"{subject_id_temp}_{camera_temp}_trajectory_plot.pdf"
        trajectory_fig.savefig(trajectory_png, dpi=300, bbox_inches="tight")
        trajectory_fig.savefig(trajectory_pdf, bbox_inches="tight")
        plt.close(trajectory_fig)
        self.logger.info("Saved trajectory plot to %s and %s", trajectory_png, trajectory_pdf)

        heatmap_height = float(np.clip(1.3 + 0.35 * len(heatmap_categories), 3.5, 20.0))
        heatmap_fig, heatmap_ax = plt.subplots(figsize=(20, heatmap_height))
        y_edges = np.arange(len(heatmap_categories) + 1, dtype=float)
        mesh = heatmap_ax.pcolormesh(
            time_edges,
            y_edges,
            confidence_matrix,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        heatmap_ax.set_yticks(np.arange(len(heatmap_categories)) + 0.5)
        heatmap_ax.set_yticklabels([display_map.get(cat, cat) for cat in heatmap_categories])
        heatmap_ax.set_xlabel("Time (s)")
        heatmap_ax.set_ylabel("Gazed Category")
        heatmap_ax.set_title(f"Gaze confidence heatmap ({subject_id_temp}, {camera_temp})")
        heatmap_ax.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        cbar = heatmap_fig.colorbar(mesh, ax=heatmap_ax)
        cbar.set_label("Confidence")
        heatmap_fig.tight_layout()

        heatmap_png = figures_dir / f"{subject_id_temp}_{camera_temp}_confidence_heatmap.png"
        heatmap_pdf = figures_dir / f"{subject_id_temp}_{camera_temp}_confidence_heatmap.pdf"
        heatmap_fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
        heatmap_fig.savefig(heatmap_pdf, bbox_inches="tight")
        plt.close(heatmap_fig)
        self.logger.info("Saved confidence heatmap to %s and %s", heatmap_png, heatmap_pdf)

    def load_gaze_data(self, subj: str, camera: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
        """Load gaze data from a CSV file.
        Select gaze data within the cut video duration.
        Label each gaze with the corresponding frame id in the cut video.

        Args:
            subj (str): The subject identifier.
            camera (str): The camera identifier.
        Returns:
            pd.DataFrame: A DataFrame containing the gaze data within the cut video duration.
        """
        gaze_path = Path(self.gaze_csv_path) if self.gaze_csv_path else (
            Path(self.gaze_world_dir) / f"{subj}_{camera}_gaze.csv"
        )
        gaze_dic = pd.read_csv(gaze_path)
        world_cam_path = Path(self.world_csv_path) if self.world_csv_path else (
            Path(self.gaze_world_dir) / f"{subj}_{camera}_world_timestamps.csv"
        )
        world_cam_dic = pd.read_csv(world_cam_path)
        self.logger.info(
            "Loaded gaze/world CSV: gaze_rows=%d world_rows=%d",
            len(gaze_dic),
            len(world_cam_dic),
        )

        required_cols = {"timestamp [ns]", "gaze x [px]", "gaze y [px]"}
        if not required_cols.issubset(gaze_dic.columns):
            missing = sorted(required_cols.difference(gaze_dic.columns))
            raise ValueError(f"Missing required gaze columns: {missing}")

        required_cols_world = {"timestamp [ns]"}
        if not required_cols_world.issubset(world_cam_dic.columns):
            missing = sorted(required_cols_world.difference(world_cam_dic.columns))
            raise ValueError(f"Missing required world camera timestamp columns: {missing}")
        self.logger.info("Required columns verified")

        # Keep world-camera mapping columns authoritative from world CSV to avoid
        # merge suffix collisions when gaze CSV already contains these columns.
        mapping_cols = ["frame_idx", "frame_timestamp", "source_frame_idx"]
        gaze_dic = gaze_dic.drop(columns=[c for c in mapping_cols if c in gaze_dic.columns])

        world_cam_dic['frame_idx'] = world_cam_dic.index
        world_cam_dic['frame_timestamp'] = world_cam_dic['timestamp [ns]']
        gaze_dic = gaze_dic.sort_values("timestamp [ns]")
        world_cam_dic = world_cam_dic.sort_values("timestamp [ns]")
        aligned = pd.merge_asof(
            gaze_dic,
            world_cam_dic,
            left_on="timestamp [ns]",
            right_on="timestamp [ns]",
            direction="backward",        # <= gaze time (closest before)
            allow_exact_matches=True
        )
        aligned = aligned.dropna(subset=['frame_idx'])
        return aligned, world_cam_dic, world_cam_path

    def label_blinks(self, aligned_gaze_df: pd.DataFrame, subj: str, camera: str) -> pd.DataFrame:
        """Label each gaze point with blink information, whether they are in blink periods."""
        blink_df = pd.read_csv(os.path.join(self.blink_dir, f"{subj}_{camera}_blinks.csv"))
        blink_df = blink_df.sort_values("start timestamp [ns]")
        blink_starts = blink_df['start timestamp [ns]'].to_numpy()
        blink_ends   = blink_df['end timestamp [ns]'].to_numpy()
        gaze_ts      = aligned_gaze_df['timestamp [ns]'].to_numpy()

        pos = np.searchsorted(blink_starts, gaze_ts, side="right") - 1
        valid = pos >= 0

        in_blink = np.zeros_like(gaze_ts, dtype=bool)
        in_blink[valid] = gaze_ts[valid] <= blink_ends[pos[valid]]
        aligned_gaze_df['in_blink'] = in_blink

        blink_id = -1 * np.ones_like(gaze_ts, dtype=int)
        mask = valid & in_blink
        blink_id[mask] = blink_df.iloc[pos[mask]]['blink id'].values
        aligned_gaze_df['blink id'] = blink_id

        return aligned_gaze_df


    def gaze_to_object_radius(self, mask: np.ndarray, x: float, y: float, r: int = 20) -> float:
        """Compute confidence that gaze hits any object (mask>0) within a radius r circle."""
        if mask.ndim != 2:
            raise ValueError(f"`mask` must be 2D (H,W). Got shape {mask.shape}.")

        H, W = mask.shape
        xi, yi = int(round(x)), int(round(y))

        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            return 0.0

        x0 = max(0, xi - r)
        x1 = min(W, xi + r + 1)
        y0 = max(0, yi - r)
        y1 = min(H, yi + r + 1)

        disk = _disk_template(r)
        dy0 = y0 - (yi - r);  dy1 = dy0 + (y1 - y0)
        dx0 = x0 - (xi - r);  dx1 = dx0 + (x1 - x0)
        circle_local = disk[dy0:dy1, dx0:dx1]

        mask_patch = mask[y0:y1, x0:x1]
        circle_area = int(circle_local.sum())
        if circle_area == 0:
            return 0.0

        hit = (mask_patch[circle_local] > 0).sum()
        return float(hit) / circle_area

    def process_subject(self, subject_id_temp: str, camera_temp: str) -> None:
        """Process gaze data for a specific subject and camera."""
        subject_run_started_at = datetime.now().astimezone()
        self.logger.info(
            "Subject-camera run started at (local time): %s",
            format_log_datetime(subject_run_started_at),
        )
        out_dir = Path(self.output_dir) / str(subject_id_temp) / str(camera_temp)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_status = "completed"
        try:
            gaze, world_cam_dic, world_cam_path = self.load_gaze_data(subject_id_temp, camera_temp)
            if len(gaze) == 0:
                self.logger.warning(f"No gaze data found for subject {subject_id_temp} and camera {camera_temp}. Skipping.")
                return

            unique_aligned_frames = int(gaze["frame_idx"].nunique())
            cut_world_frames = len(world_cam_dic)

            if self.blink_dir is None:
                gaze_blink_removed = gaze
            else:
                gaze_blink_labeled = self.label_blinks(gaze, subject_id_temp, camera_temp)
                gaze_blink_labeled.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv"), index=False)
                self.logger.info(f"Saved blink labeled gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv')}")
                gaze_blink_removed = gaze_blink_labeled.loc[~gaze_blink_labeled['in_blink']].copy()
                gaze_blink_removed.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv"), index=False)
                self.logger.info(f"Saved blink removed gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv')}")

            unique_post_blink_frames = int(gaze_blink_removed["frame_idx"].nunique())
            mask_subject = self.resolve_mask_dir(subject_id_temp, camera_temp)

            # Detect SAM3 vs SAM2 and apply the requested pipeline.
            is_sam3 = self._is_sam3_project(mask_subject)
            sam3_instances = None
            sam2_extra = None  # set for pipeline="both": (sam2_masks_dir, exclude_ids, ignore)

            if self.pipeline == "sam3" and not is_sam3:
                raise FileNotFoundError(
                    f"--pipeline sam3 requires a SAM3 project (project.json) at {mask_subject}")

            if self.pipeline in ("sam2", "both") and is_sam3:
                s2_dir = _sam2_native_masks_dir(mask_subject)
                if self.pipeline == "sam2":
                    if s2_dir is None:
                        raise FileNotFoundError(
                            f"--pipeline sam2: no SAM2 results (sam2_results/masks/) under {mask_subject}. "
                            f"Run sam2_process.py on sam2_handoff.json first.")
                    self.logger.info("pipeline=sam2: using SAM2-native masks at %s", s2_dir)
                    is_sam3 = False
                    mask_subject = s2_dir
                else:  # both
                    if s2_dir is None:
                        self.logger.warning(
                            "pipeline=both: no SAM2 results under %s; scoring SAM3 instances only.",
                            mask_subject)
                    else:
                        exclude_ids = _read_handoff_sam3_derived_id_labels(mask_subject)
                        sam2_extra = (str(s2_dir), sorted(exclude_ids), sorted(self.ignore_objects))
                        self.logger.info(
                            "pipeline=both: SAM3 instances + SAM2-native masks at %s "
                            "(excluding %d SAM3-derived/covered id label(s)).",
                            s2_dir, len(exclude_ids))

            if is_sam3:
                sam3_instances = self.load_sam3_project_info(mask_subject)
                # Apply ignore_objects filtering at instance level
                if self.ignore_objects:
                    orig_count = len(sam3_instances)
                    sam3_instances = [
                        (c, u, oid) for c, u, oid in sam3_instances
                        if not (_sam3_mask_labels(
                            _sam3_virtual_key(c, u, oid, 0)) & self.ignore_objects)
                    ]
                    skipped = orig_count - len(sam3_instances)
                    if skipped:
                        self.logger.info(
                            "Ignored %d SAM3 instance(s) based on ignore_object_list.", skipped
                        )
                # Cache for load_mask() serial path
                self._sam3_instances_cache = sam3_instances
                # Category strings are "{concept_name}_{user_name}" (see _mask_name_to_category);
                # map them back to just the instance (user) name for plot y-tick labels.
                category_display_map = {f"{c}_{u}": u for c, u, _ in sam3_instances}
            else:
                self.logger.info("SAM2 mode: mask directory at %s", mask_subject)
                category_display_map = {}

            mask_summary = self.summarize_mask_frames(mask_subject)
            if sam2_extra is not None:
                # both mode: a frame counts as covered if either source has it.
                s2_summary = self._summarize_sam2_mask_frames(Path(sam2_extra[0]))
                merged_ids = mask_summary["frame_ids"] | s2_summary["frame_ids"]
                mask_summary = {
                    "count": len(merged_ids),
                    "frame_ids": merged_ids,
                    "contiguous": sorted(merged_ids) == list(range(min(merged_ids), max(merged_ids) + 1)) if merged_ids else False,
                    "min_frame": min(merged_ids) if merged_ids else None,
                    "max_frame": max(merged_ids) if merged_ids else None,
                }
            post_blink_frame_ids = set(gaze_blink_removed["frame_idx"].astype(int).unique())
            post_blink_missing_mask_frames = sorted(post_blink_frame_ids.difference(mask_summary["frame_ids"]))

            self.logger.info(
                "Number of frames in camera timestamp: %d from %s",
                cut_world_frames,
                world_cam_path,
            )
            if mask_summary["count"] == 0:
                self.logger.info(
                    "Mask frames present: 0, no parsed frame ids, from %s",
                    mask_subject,
                )
            else:
                self.logger.info(
                    "Number of frames in masks: %d, min frame: %d, max frame: %d, from %s",
                    mask_summary["count"],
                    mask_summary["min_frame"],
                    mask_summary["max_frame"],
                    mask_subject,
                )
            self.logger.info(
                "Number of frames with gaze data before blink removal: %d",
                unique_aligned_frames,
            )
            if self.blink_dir is None:
                self.logger.info(
                    "Number of frames with gaze data after blink removal: %d (blink removal not applied)",
                    unique_post_blink_frames,
                )
            else:
                self.logger.info(
                    "Number of frames with gaze data after blink removal: %d",
                    unique_post_blink_frames,
                )
            self.logger.info(
                "Post-blink frames missing masks: %d",
                len(post_blink_missing_mask_frames),
            )
            if post_blink_missing_mask_frames:
                preview = ", ".join(str(frame_id) for frame_id in post_blink_missing_mask_frames[:20])
                self.logger.warning(
                    "Sample post-blink frame ids without masks for subject=%s camera=%s: %s",
                    subject_id_temp,
                    camera_temp,
                    preview,
                )

            subject_gaze_probabilities = {}
            total_gaze_points = len(gaze_blink_removed)
            total_frames = unique_post_blink_frames
            frames_with_masks = 0
            frames_without_masks = 0
            assigned_gaze_points = 0
            assigned_confidences = []

            gaze_blink_removed['gazed_object_id'] = None
            gaze_blink_removed['gazed_object'] = None
            gaze_blink_removed['gazed_object_confidence'] = 0.0
            _r = self.gaze_radius

            excluded_objects_suffix = self.excluded_objects_output_suffix()
            pkl_path = out_dir / f"{subject_id_temp}_{camera_temp}_gaze_object_probabilities{excluded_objects_suffix}.pkl"
            _use_cached_pkl = not self.recompute and pkl_path.exists()

            if _use_cached_pkl:
                self.logger.warning(
                    "Found existing probabilities at %s — skipping mask I/O and reassigning from "
                    "cached scores. Pass --recompute to force a full run from scratch.",
                    pkl_path,
                )
                with open(pkl_path, 'rb') as f:
                    subject_gaze_probabilities = pickle.load(f)
                for i in gaze_blink_removed.index:
                    probs = subject_gaze_probabilities.get(i, {})
                    if probs:
                        best_name = max(probs, key=lambda k: probs[k])
                        best_conf = float(probs[best_name])
                        if best_conf > 0.0:
                            gaze_blink_removed.loc[i, 'gazed_object_confidence'] = best_conf
                        if best_conf >= self.gaze_confidence_threshold:
                            gaze_blink_removed.loc[i, 'gazed_object_id'] = best_name.split('.')[0].split('_')[-1]
                            gaze_blink_removed.loc[i, 'gazed_object'] = '_'.join(best_name.split('.')[0].split('_')[2:-1])
                            assigned_gaze_points += 1
                            assigned_confidences.append(best_conf)
            else:
                self.logger.info(f"Loading masks from {mask_subject}")
                frame_groups = []
                for frame_idx, gdf in gaze_blink_removed.groupby('frame_idx', sort=False):
                    gaze_points = [
                        (i, int(round(gdf.loc[i, 'gaze x [px]'])), int(round(gdf.loc[i, 'gaze y [px]'])))
                        for i in gdf.index
                    ]
                    if is_sam3 and sam2_extra is not None:
                        frame_groups.append((int(frame_idx), gaze_points, str(mask_subject), _r, sam3_instances, sam2_extra))
                    elif is_sam3:
                        frame_groups.append((int(frame_idx), gaze_points, str(mask_subject), _r, sam3_instances))
                    else:
                        frame_groups.append((int(frame_idx), gaze_points, str(mask_subject), _r))

                if self.within_job_workers > 1:
                    with ProcessPoolExecutor(max_workers=self.within_job_workers) as executor:
                        for partial, has_masks in executor.map(_frame_group_worker, frame_groups):
                            if not has_masks:
                                frames_without_masks += 1
                            else:
                                frames_with_masks += 1
                                subject_gaze_probabilities.update(partial)
                else:
                    for args in frame_groups:
                        partial, has_masks = _frame_group_worker(args)
                        if not has_masks:
                            frames_without_masks += 1
                        else:
                            frames_with_masks += 1
                            subject_gaze_probabilities.update(partial)

                for i in gaze_blink_removed.index:
                    probs = subject_gaze_probabilities.get(i, {})
                    if not probs:
                        continue
                    best_name = max(probs, key=probs.__getitem__)
                    best_conf = float(probs[best_name])
                    if best_conf > 0.0:
                        gaze_blink_removed.loc[i, 'gazed_object_confidence'] = best_conf
                    if best_conf >= self.gaze_confidence_threshold:
                        gaze_blink_removed.loc[i, 'gazed_object_id'] = best_name.split('.')[0].split('_')[-1]
                        gaze_blink_removed.loc[i, 'gazed_object'] = '_'.join(best_name.split('.')[0].split('_')[2:-1])
                        assigned_gaze_points += 1
                        assigned_confidences.append(best_conf)
                        self.logger.debug(
                            "Processed gaze index %s: gazed object=%s confidence=%.4f",
                            i,
                            best_name,
                            best_conf,
                        )

            assignment_rate = (assigned_gaze_points / total_gaze_points * 100.0) if total_gaze_points > 0 else 0.0
            mean_conf = float(np.mean(assigned_confidences)) if assigned_confidences else 0.0
            median_conf = float(np.median(assigned_confidences)) if assigned_confidences else 0.0

            if _use_cached_pkl:
                self.logger.info(
                    "Gaze-object assignment summary (from cached pkl) for subject=%s camera=%s: "
                    "total_gaze_points=%d assigned=%d assignment_rate=%.2f%% "
                    "mean_conf=%.4f median_conf=%.4f",
                    subject_id_temp,
                    camera_temp,
                    total_gaze_points,
                    assigned_gaze_points,
                    assignment_rate,
                    mean_conf,
                    median_conf,
                )
            else:
                self.logger.info(
                    (
                        "Gaze-object assignment summary for subject=%s camera=%s: "
                        "total_gaze_points=%d assigned=%d assignment_rate=%.2f%% "
                        "frames_with_readable_masks=%d/%d post_blink_frames "
                        "mean_conf=%.4f median_conf=%.4f"
                    ),
                    subject_id_temp,
                    camera_temp,
                    total_gaze_points,
                    assigned_gaze_points,
                    assignment_rate,
                    frames_with_masks,
                    total_frames,
                    mean_conf,
                    median_conf,
                )
                if frames_without_masks > 0:
                    self.logger.info(
                        "No readable segmentation masks were found for %d post-blink frame(s) for subject=%s camera=%s.",
                        frames_without_masks,
                        subject_id_temp,
                        camera_temp,
                    )

            output_path = os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gazed_object{excluded_objects_suffix}.csv")
            gaze_blink_removed.to_csv(output_path, index=False)
            self.logger.info(f"Saved gaze object results to {output_path}")

            if not _use_cached_pkl:
                with open(pkl_path, 'wb') as f:
                    pickle.dump(subject_gaze_probabilities, f)
                self.logger.info(f"Saved probabilities of each mask for each eye gaze to {pkl_path}")

            if self.plot_figures:
                self.plot_method_figures(
                    subject_id_temp=subject_id_temp,
                    camera_temp=camera_temp,
                    gaze_df=gaze_blink_removed,
                    subject_gaze_probabilities=subject_gaze_probabilities,
                    out_dir=out_dir,
                    category_display_map=category_display_map,
                )
            else:
                self.logger.info("Skipping method figures because --skip-figures was provided.")

        except Exception:
            run_status = "failed"
            raise
        finally:
            subject_run_finished_at = datetime.now().astimezone()
            self.logger.info(
                "Subject-camera run %s at (local time): %s",
                run_status,
                format_log_datetime(subject_run_finished_at),
            )


def process_subject_camera_pair(args):
    """Worker entry point for one subject-camera pair."""
    (
        subject_id_temp,
        camera_temp,
        gaze_world_dir,
        mask_dir,
        output_dir,
        blink_dir,
        start_plot_time,
        end_plot_time,
        ignore_object_list,
        plot_figures,
        recompute,
        within_job_workers,
        gaze_csv_path,
        world_csv_path,
        category_sort,
        gaze_confidence_threshold,
        gaze_radius,
        log_queue,
        pipeline,
    ) = args

    setup_worker_logging(log_queue)
    logger = pair_logger(logging.getLogger(__name__), subject_id_temp, camera_temp)
    gaze_aligner = GazeObjectAligner(
        gaze_world_dir,
        mask_dir,
        output_dir,
        blink_dir=blink_dir,
        start_plot_time=start_plot_time,
        end_plot_time=end_plot_time,
        ignore_object_list=ignore_object_list,
        plot_figures=plot_figures,
        logger=logger,
        recompute=recompute,
        within_job_workers=within_job_workers,
        gaze_csv_path=gaze_csv_path,
        world_csv_path=world_csv_path,
        category_sort=category_sort,
        gaze_confidence_threshold=gaze_confidence_threshold,
        gaze_radius=gaze_radius,
        pipeline=pipeline,
    )
    logger.info("Processing started.")
    gaze_aligner.process_subject(subject_id_temp, camera_temp)
    return subject_id_temp, camera_temp


def main():
    parser = argparse.ArgumentParser(
    description='Detect gazed objects based on gaze data and SAM2/SAM3 masks.',
    formatter_class=RawDescriptionDefaultsHelpFormatter,
    epilog=(
        'Examples:\n'
        '  # Process one subject and one camera (SAM2 or SAM3, auto-detected)\n'
        '  python process_gaze_mask_alignment.py \\\n'
        '    /path/to/gaze_world_data \\\n'
        '    /path/to/segmentation_masks \\\n'
        '    /path/to/output_directory \\\n'
        '    --subject-id 27 \\\n'
        '    --camera-id child\n\n'

        '  # Direct SAM3 project (mask_dir is the project folder itself)\n'
        '  python process_gaze_mask_alignment.py \\\n'
        '    /path/to/gaze_world_data \\\n'
        '    /path/to/sam3_project_dir \\\n'
        '    /path/to/output_directory \\\n'
        '    --subject-id 27 --camera-id child\n\n'

        '  # Process multiple subjects/cameras and apply blink removal\n'
        '  python process_gaze_mask_alignment.py \\\n'
        '    /path/to/gaze_world_data \\\n'
        '    /path/to/segmentation_masks \\\n'
        '    /path/to/output_directory \\\n'
        '    --subject-id 27,28 \\\n'
        '    --camera-id child,parent \\\n'
        '    --blink-dir /path/to/blink_data \\\n'
        '    --log-path /path/to/gaze_object.log\n\n'

        '  # Auto-discover subjects and cameras from gaze/world CSV filenames\n'
        '  python process_gaze_mask_alignment.py \\\n'
        '    /path/to/gaze_world_data \\\n'
        '    /path/to/segmentation_masks \\\n'
        '    /path/to/output_directory\n\n'

        '  # Direct gaze/world CSV paths (bypasses {subject}_{camera}_*.csv naming)\n'
        '  python process_gaze_mask_alignment.py \\\n'
        '    /path/to/gaze_world_data \\\n'
        '    /path/to/segmentation_masks \\\n'
        '    /path/to/output_directory \\\n'
        '    --subject-id 27 --camera-id child \\\n'
        '    --gaze-csv /path/to/any_gaze.csv \\\n'
        '    --world-csv /path/to/any_world_timestamps.csv\n\n'

        'Notes:\n'
        '  - gaze_world_dir must contain both {subject}_{camera}_gaze.csv and\n'
        '    {subject}_{camera}_world_timestamps.csv, unless --gaze-csv/--world-csv\n'
        '    are used to point directly at a single pair of CSV files (requires\n'
        '    exactly one --subject-id and one --camera-id; gaze_world_dir is still\n'
        '    required positionally but unused for path resolution in that mode).\n'
        '  - mask_dir resolution order (first match used):\n'
        '      1) mask_dir itself is a SAM3 project (contains project.json)\n'
        '      2) mask_dir itself contains SAM2 mask files\n'
        '      3) {mask_dir}/{subject}/{camera}/ or {mask_dir}/{subject}_{camera}/\n'
        '         with SAM3 project.json, SAM2 masks/ subdirectory, or SAM2 mask files\n'
        '  - SAM3 is auto-detected; no extra flag is needed.\n'
        '    Output gazed_object = "{concept}_{user_name}" for SAM3 instances.\n'
        '  - If --blink-dir is provided, blink CSVs must be named\n'
        '    {subject}_{camera}_blinks.csv and include start timestamp [ns],\n'
        '    end timestamp [ns], and blink id.\n'
        '  - Output folders are created as {output_dir}/{subject}/{camera}/.\n'
        '  - If --ignore-object-list is provided, ignored masks are excluded and\n'
        '    final CSV/PKL outputs receive an _excluding_ignored_objects suffix.\n'
        '    For SAM3: entries can match concept_name, user_name, or concept_user_name.\n'
        '  - Method figures are written under each subject-camera output folder unless\n'
        '    --skip-figures is provided.'
    ))

    parser.add_argument(
        'gaze_world_dir', nargs='?', default=None,
        help='Path to the gaze timestamps files and Egocentric Video Frame timestamps files. '
             'Optional when both --gaze-csv and --world-csv are provided.',
    )
    parser.add_argument('mask_dir', help='Directory containing SAM2 mask files or SAM3 project directory')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('--subject-id', dest='subject_id', help='Subject ID e.g. 27 or 27,28')
    parser.add_argument('--camera-id', help='Camera ID e.g. child or child,parent')
    parser.add_argument(
        '--gaze-csv', dest='gaze_csv', default=None,
        help='Optional direct path to a single gaze CSV file, bypassing the '
             '{subject}_{camera}_gaze.csv naming convention inside gaze_world_dir. '
             'Must be used together with --world-csv, and requires exactly one '
             'subject/camera pair (via --subject-id/--camera-id).',
    )
    parser.add_argument(
        '--world-csv', dest='world_csv', default=None,
        help='Optional direct path to a single world-camera timestamps CSV file, '
             'bypassing the {subject}_{camera}_world_timestamps.csv naming convention. '
             'Must be used together with --gaze-csv.',
    )
    parser.add_argument('--blink-dir', help='If remove gaze during blinks', required=False)
    parser.add_argument('--log-path', help='Path to the log file', required=False)
    parser.add_argument(
        '--ignore-object-list', default=None,
        help='Optional txt file with object mask labels to ignore, one per line. '
             'Comments start with #. '
             'For SAM2: entries can be object names, ids, or exact mask filenames. '
             'For SAM3: entries can be concept_name, user_name, or concept_user_name.',
    )
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help='Number of subject-camera pairs to process in parallel. Use 1 for sequential processing.',
    )
    parser.add_argument(
        '--within-job-workers', type=int, default=1, dest='within_job_workers',
        help='Number of parallel workers for frame-group processing within one subject-camera pair. '
             'Default 1 (sequential). Higher values parallelize mask I/O but may saturate disk.',
    )
    parser.add_argument(
        '--pipeline',
        choices=['auto', 'sam3', 'sam2', 'both'],
        default='auto',
        help='Which segmentation pipeline(s) to score gaze against. "auto" (default): SAM3 if '
             'the resolved mask dir is a SAM3 project (project.json), else SAM2. "sam3": SAM3 '
             'instances only. "sam2": SAM2 masks only - for a converted SAM3 project this reads '
             'sam2_results/masks/. "both": SAM3 instances plus SAM2-native objects added in '
             'sam2_ui.py, de-duplicated against SAM3-derived/covered ids via sam2_handoff.json.',
    )
    parser.add_argument('--skip-figures', action='store_true', help='Skip trajectory and confidence heatmap figure generation.')
    parser.add_argument(
        '--start-plot-time',
        type=float,
        default=None,
        help='Optional start time in seconds (from recording start) for method-figure plotting.',
    )
    parser.add_argument(
        '--end-plot-time',
        type=float,
        default=None,
        help='Optional end time in seconds (from recording start) for method-figure plotting.',
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Force recomputation from masks even if a cached probabilities pkl already exists.',
    )
    parser.add_argument(
        '--gaze-confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum mask overlap confidence required to label a gaze point as gazing at that '
             'object (gazed_object/gazed_object_id). The raw gazed_object_confidence value is '
             'still recorded unthresholded. Default 0.5.',
    )
    parser.add_argument(
        '--category-sort',
        choices=['first_seen', 'frequency'],
        default='first_seen',
        help='Ordering of categories (rows) in the trajectory/heatmap figures: "first_seen" '
             '(default, chronological order of first assignment) or "frequency" (most-assigned '
             'category first).',
    )
    parser.add_argument(
        '--gaze-radius',
        type=int,
        default=20,
        help='Radius in pixels of the disk around each gaze point used to compute mask-overlap '
             'confidence. Larger values are more tolerant of gaze-tracking noise but blur '
             'distinctions between nearby/adjacent objects. Default 20.',
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1.")
    if args.within_job_workers < 1:
        parser.error("--within-job-workers must be >= 1.")
    if args.start_plot_time is not None and args.start_plot_time < 0:
        parser.error("--start-plot-time must be >= 0.")
    if not (0.0 <= args.gaze_confidence_threshold <= 1.0):
        parser.error("--gaze-confidence-threshold must be between 0.0 and 1.0.")
    if args.gaze_radius < 1:
        parser.error("--gaze-radius must be >= 1.")
    if args.end_plot_time is not None and args.end_plot_time < 0:
        parser.error("--end-plot-time must be >= 0.")
    if (
        args.start_plot_time is not None
        and args.end_plot_time is not None
        and args.start_plot_time > args.end_plot_time
    ):
        parser.error("--start-plot-time must be <= --end-plot-time.")

    if bool(args.gaze_csv) != bool(args.world_csv):
        parser.error("--gaze-csv and --world-csv must be provided together.")
    if args.gaze_csv and (not args.subject_id or ',' in args.subject_id or not args.camera_id or ',' in args.camera_id):
        parser.error(
            "--gaze-csv/--world-csv require exactly one subject (--subject-id) "
            "and one camera (--camera-id); auto-discovery from gaze_world_dir is "
            "not available when direct CSV paths are given."
        )
    if args.gaze_world_dir is None and not args.gaze_csv:
        parser.error("gaze_world_dir is required unless --gaze-csv and --world-csv are both provided.")

    if args.subject_id:
        subj_ids = [id.strip() for id in args.subject_id.split(',')]
    else:
        gaze_worldcam_dir = Path(args.gaze_world_dir)
        subj_ids = sorted({
            p.stem.split('_')[0]
            for p in gaze_worldcam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv"}
        })

    if args.camera_id:
        camera_list = [i.strip() for i in args.camera_id.split(',')]
    else:
        gaze_worldcam_dir = Path(args.gaze_world_dir)
        camera_list = sorted({
            p.stem.split('_')[1]
            for p in gaze_worldcam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv"} and len(p.stem.split('_')) >= 2
        })

    subject_camera_pairs = [
        (str(subject_id_temp), str(camera_temp))
        for subject_id_temp in subj_ids
        for camera_temp in camera_list
    ]

    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, 'gaze_object.log')
    log_path = Path(args.log_path)
    use_queue_logging = args.num_workers > 1 and len(subject_camera_pairs) > 1
    log_queue, log_listener, log_manager = setup_logging(log_path, use_queue=use_queue_logging)

    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to: {log_path.resolve()}")
        pipeline_run_started_at = datetime.now().astimezone()
        logger.info("Pipeline run started at (local time): %s", format_log_datetime(pipeline_run_started_at))

        logger.info("------- Loaded Files and Directories -------")
        logger.info(f"Subjects: {subj_ids}, Cameras: {camera_list}")
        logger.info(f"Gaze Directory: {args.gaze_world_dir}")
        logger.info(f"Mask Directory: {args.mask_dir}")
        logger.info(f"Pipeline: {args.pipeline}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Blink Directory: {args.blink_dir}")
        logger.info(f"Ignore Object List: {args.ignore_object_list}")
        logger.info(f"Plot Start Time (s): {args.start_plot_time}")
        logger.info(f"Plot End Time (s): {args.end_plot_time}")
        logger.info(f"Plot Figures: {not args.skip_figures}")
        logger.info(f"Num Workers: {args.num_workers}")
        logger.info(f"Within-Job Workers: {args.within_job_workers}")
        logger.info(f"Gaze Confidence Threshold: {args.gaze_confidence_threshold}")
        logger.info(f"Category Sort: {args.category_sort}")
        logger.info(f"Gaze Radius (px): {args.gaze_radius}")
        logger.info(f"Queue Logging: {use_queue_logging}")
        if args.blink_dir is None:
            logger.info("No blink directory provided, skipping blink labeling.")

        if args.num_workers <= 1 or len(subject_camera_pairs) <= 1:
            for subj, cam in subject_camera_pairs:
                logger.info("------- Start Gaze Mask Processing -------")
                logger.info(f"------- Processing subject {subj}, camera {cam} -------")
                gaze_aligner = GazeObjectAligner(
                    args.gaze_world_dir,
                    args.mask_dir,
                    args.output_dir,
                    blink_dir=args.blink_dir,
                    start_plot_time=args.start_plot_time,
                    end_plot_time=args.end_plot_time,
                    ignore_object_list=args.ignore_object_list,
                    plot_figures=not args.skip_figures,
                    logger=pair_logger(logger, subj, cam),
                    recompute=args.recompute,
                    within_job_workers=args.within_job_workers,
                    gaze_csv_path=args.gaze_csv,
                    world_csv_path=args.world_csv,
                    category_sort=args.category_sort,
                    gaze_confidence_threshold=args.gaze_confidence_threshold,
                    gaze_radius=args.gaze_radius,
                    pipeline=args.pipeline,
                )
                gaze_aligner.process_subject(subj, cam)
                logger.info(f"------- Finished processing subject {subj}, camera {cam} -------")
        else:
            max_workers = min(args.num_workers, len(subject_camera_pairs))
            logger.info(f"Processing {len(subject_camera_pairs)} subject-camera pairs with {max_workers} workers.")
            worker_args = [
                (
                    subj,
                    cam,
                    args.gaze_world_dir,
                    args.mask_dir,
                    args.output_dir,
                    args.blink_dir,
                    args.start_plot_time,
                    args.end_plot_time,
                    args.ignore_object_list,
                    not args.skip_figures,
                    args.recompute,
                    args.within_job_workers,
                    args.gaze_csv,
                    args.world_csv,
                    args.category_sort,
                    args.gaze_confidence_threshold,
                    args.gaze_radius,
                    log_queue,
                    args.pipeline,
                )
                for subj, cam in subject_camera_pairs
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(process_subject_camera_pair, worker_arg): (worker_arg[0], worker_arg[1])
                    for worker_arg in worker_args
                }
                for future in as_completed(future_to_pair):
                    subj, cam = future_to_pair[future]
                    try:
                        future.result()
                    except Exception:
                        logger.exception(f"Failed processing subject {subj} and camera {cam}.")
                        raise
                    logger.info(f"------- Finished processing subject {subj}, camera {cam} -------")

        logger.info("Gaze object detection complete!")
        logger.info("Pipeline run finished at (local time): %s", format_log_datetime(datetime.now().astimezone()))
    finally:
        if log_listener is not None:
            log_listener.stop()
        if log_manager is not None:
            log_manager.shutdown()

if __name__ == "__main__":
    main()
