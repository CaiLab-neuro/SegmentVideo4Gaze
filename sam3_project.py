"""
SAM3 Project Management - Data models and state serialization for hierarchical segmentation.

This module provides project structure management, metadata persistence, and
inference state serialization for SAM3's hierarchical object segmentation pipeline.
"""

import os
import json
import pickle
import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path



# Manually-added instances (created in the UI, never text-detected) get obj_ids from this
# reserved band, kept far above anything SAM3's own auto-detector could plausibly reach
# (its ids start at 0 and increment only as new distinct matches for the concept's text
# prompt are found — realistically never within orders of magnitude of this).
MANUAL_ID_BASE = 100_000


class ProjectStaleError(Exception):
    """Raised when on-disk project/concept files changed since they were loaded.

    Means another process (typically a `sam3_process.py --refine` run) wrote
    new results after this project was loaded; saving now would silently
    clobber them. Caller should tell the user to reload the project.
    """
    def __init__(self, changed_files: List[str]):
        self.changed_files = changed_files
        super().__init__(f"{len(changed_files)} file(s) changed on disk since load: {changed_files}")


def _atomic_json_dump(data, path: str, indent: int = 2) -> None:
    """Write JSON to path atomically (temp file + os.replace).

    A plain open(path, 'w') truncates immediately, so a kill mid-json.dump
    (OOM-killer, Ctrl-C, power loss) leaves truncated JSON and the next
    json.load raises — for project.json that makes the whole project
    unopenable. os.replace is atomic on the same filesystem.

    tempfile.mkstemp() always creates its file mode 0600 (owner-only),
    ignoring the process umask by design. os.replace() preserves that mode,
    so without the explicit chmod below every file written through here
    would end up unreadable/unwritable by the rest of the group even when
    the caller has set a group-friendly umask (e.g. 0o002) for everything
    else.
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
        os.chmod(tmp_path, 0o664)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _refine_lock_path(project_dir: str) -> str:
    return os.path.join(project_dir, ".refine.lock")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_refine_lock(project_dir: str) -> None:
    """Mark project_dir as owned by a running refine/redetect job (this process)."""
    with open(_refine_lock_path(project_dir), 'w') as f:
        json.dump({"pid": os.getpid(), "started": time.time()}, f)


def release_refine_lock(project_dir: str) -> None:
    lock_path = _refine_lock_path(project_dir)
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass


def check_refine_lock(project_dir: str) -> Optional[dict]:
    """Return lock info ({"pid", "started"}) if an active job holds the lock.

    A lock file whose PID is no longer running is treated as stale (e.g. the
    job crashed without cleaning up) and is reported as not-locked.
    """
    lock_path = _refine_lock_path(project_dir)
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path) as f:
            info = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    pid = info.get("pid")
    if pid is None or not _pid_alive(pid):
        return None
    return info


class ConceptStatus(Enum):
    """Status of concept processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SAM3Instance:
    """Single detected instance from SAM3 (one object ID within a concept)"""
    sam3_obj_id: int
    user_name: str              # e.g., "child_head", "parent_head"
    concept_name: str
    color_rgb: Optional[Tuple[int, int, int]] = None  # Override concept color
    visible: bool = True
    deleted: bool = False
    score: float = 0.0
    first_detection_frame: int = 0
    last_detection_frame: int = 0
    num_frames_with_mask: int = 0
    sam2_object_id: Optional[int] = None  # SAM2 object ID assigned during export
    # Maximal runs of consecutive frames that have a mask: [(start, end), ...] inclusive
    continuous_periods: List[Tuple[int, int]] = field(default_factory=list)
    # Best (peak-area) frame per period, computed during propagation while masks are in memory.
    # Each entry: {"start": int, "end": int, "best_frame": int, "pixel_count": int,
    #              "avg_pixel_ratio": float, "min_pixel_ratio": float, "max_pixel_ratio": float}
    # pixel_ratio fields absent on entries from older project.json files — use .get(..., 0.0).
    period_peaks: List[dict] = field(default_factory=list)
    # 99th-percentile pixel ratio across all detected frames; used to normalize presence bar
    # height so typical frames read near full height rather than relative to the single maximum.
    # 0.0 means not yet computed (old projects); UI falls back to max avg_pixel_ratio.
    presence_norm: float = 0.0
    # True when instance was created manually in the UI (not returned by text detection).
    # replay_concept_refinements initializes it via point prompts instead of text detection.
    manually_added: bool = False
    # True once this instance has had at least one OTHER instance absorbed into it.
    # Gates the one-time self anchor added in absorb_instance() (see sam3_ui.py) so the
    # target's own first-period identity is pinned only the first time it gains an
    # absorbed instance — later absorbs into the same target don't repeat it.
    received_absorb_anchor: bool = False
    # True when this instance was the SOURCE of an absorb operation (deleted=True AND
    # absorbed_source=True).  The pipeline calls remove_object() for absorbed sources so
    # their pixels are freed for the absorbing target.  Contrast with purely-deleted
    # instances (deleted=True, absorbed_source=False) whose sub-states are kept alive to
    # suppress re-detection of unwanted objects via non-overlapping constraints.
    absorbed_source: bool = False
    # Frame indices in concepts/<name>/instances/<id>/mask_anchors/ that are used as
    # SAM3 mask-conditioning anchors (set via the absorb wizard or mask-removal undo).
    mask_anchor_frames: List[int] = field(default_factory=list)
    # sam3_obj_ids of instances that were absorbed into this instance (via the absorb
    # wizard).  Used to surface each component's original masks when a future instance
    # is absorbed into this one, so the user can decide whether to union each separately.
    absorbed_source_ids: List[int] = field(default_factory=list)

    def get_effective_color(self, concept_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get instance color (override if set, otherwise use concept color)"""
        return self.color_rgb if self.color_rgb is not None else concept_color

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export"""
        return {
            "sam3_obj_id": self.sam3_obj_id,
            "user_name": self.user_name,
            "concept_name": self.concept_name,
            "color_rgb": list(self.color_rgb) if self.color_rgb else None,
            "visible": self.visible,
            "deleted": self.deleted,
            "score": self.score,
            "first_detection_frame": self.first_detection_frame,
            "last_detection_frame": self.last_detection_frame,
            "num_frames_with_mask": self.num_frames_with_mask,
            "sam2_object_id": self.sam2_object_id,
            "continuous_periods": [list(p) for p in self.continuous_periods],
            "period_peaks": self.period_peaks,
            "presence_norm": self.presence_norm,
            "manually_added": self.manually_added,
            "received_absorb_anchor": self.received_absorb_anchor,
            "absorbed_source": self.absorbed_source,
            "mask_anchor_frames": self.mask_anchor_frames,
            "absorbed_source_ids": self.absorbed_source_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SAM3Instance':
        """Deserialize from dictionary"""
        color = tuple(data["color_rgb"]) if data.get("color_rgb") else None
        return cls(
            sam3_obj_id=data["sam3_obj_id"],
            user_name=data["user_name"],
            concept_name=data["concept_name"],
            color_rgb=color,
            visible=data.get("visible", True),
            deleted=data.get("deleted", False),
            score=data.get("score", 0.0),
            first_detection_frame=data.get("first_detection_frame", 0),
            last_detection_frame=data.get("last_detection_frame", 0),
            num_frames_with_mask=data.get("num_frames_with_mask", 0),
            sam2_object_id=data.get("sam2_object_id", None),
            continuous_periods=[tuple(p) for p in data.get("continuous_periods", [])],
            period_peaks=data.get("period_peaks", []),
            presence_norm=data.get("presence_norm", 0.0),
            manually_added=data.get("manually_added", False),
            received_absorb_anchor=data.get("received_absorb_anchor", False),
            absorbed_source=data.get("absorbed_source", False),
            mask_anchor_frames=data.get("mask_anchor_frames", []),
            absorbed_source_ids=data.get("absorbed_source_ids", []),
        )


@dataclass
class SAM3Concept:
    """High-level semantic concept from one text prompt (or one example box)"""
    name: str
    text_prompt: str
    color_rgb: Tuple[int, int, int]
    opacity: float = 0.5
    visible: bool = True
    status: ConceptStatus = ConceptStatus.PENDING
    # ISO timestamp set when status last transitioned to COMPLETED (initial detection or
    # a full re-detect). None for concepts processed before this field existed, or that
    # have never completed. Not updated by refine/replay — see each instance's
    # refinements.json for per-point timestamps instead.
    completed_at: Optional[str] = None
    instances: List[SAM3Instance] = field(default_factory=list)
    detection_frame: int = 0
    max_instances: int = -1  # -1 = no limit; >0 caps how many instances SAM3 may create
    # Highest sam3_obj_id ever assigned to any instance of this concept (detected or
    # manually-added). add_new_instance() (sam3_ui.py) assigns highest_obj_id + 1 to a new
    # manual instance instead of deriving it from max(existing ids) in `instances` — that
    # list can shrink after a delete+purge, and re-deriving from it would let the id space
    # shrink back down too, risking collision with whatever small integer SAM3's own
    # auto-detector assigns to a genuinely new object in a later refine session.
    # Only reset (to -1) on a full concept reset/re-detect; otherwise it only grows,
    # regardless of how many instances are later deleted/consolidated.
    highest_obj_id: int = -1

    def get_visible_instances(self) -> List[SAM3Instance]:
        """Get all non-deleted visible instances"""
        return [inst for inst in self.instances if inst.visible and not inst.deleted]

    def get_instance_by_sam3_id(self, sam3_obj_id: int) -> Optional[SAM3Instance]:
        """Find instance by SAM3 object ID"""
        for inst in self.instances:
            if inst.sam3_obj_id == sam3_obj_id:
                return inst
        return None

    def prompt_label(self) -> str:
        """Human-readable description of how this concept is defined."""
        return f"text: '{self.text_prompt}'"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export"""
        return {
            "name": self.name,
            "text_prompt": self.text_prompt,
            "color_rgb": list(self.color_rgb),
            "opacity": self.opacity,
            "visible": self.visible,
            "status": self.status.value,
            "completed_at": self.completed_at,
            "instances": [inst.to_dict() for inst in self.instances],
            "detection_frame": self.detection_frame,
            "max_instances": self.max_instances,
            "highest_obj_id": self.highest_obj_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SAM3Concept':
        """Deserialize from dictionary"""
        instances = [SAM3Instance.from_dict(inst_data) for inst_data in data.get("instances", [])]
        # Fall back to the current instances' own max for project.json files saved before
        # highest_obj_id existed, so we never silently regress below ids already in use.
        default_highest = max((inst.sam3_obj_id for inst in instances), default=-1)
        return cls(
            name=data["name"],
            text_prompt=data["text_prompt"],
            color_rgb=tuple(data["color_rgb"]),
            opacity=data.get("opacity", 0.5),
            visible=data.get("visible", True),
            status=ConceptStatus(data.get("status", "pending")),
            completed_at=data.get("completed_at"),
            instances=instances,
            detection_frame=data.get("detection_frame", 0),
            max_instances=data.get("max_instances", -1),
            # max(...) guards against a stored value that's stale/missing relative to
            # instances actually on record (e.g. an older project.json) — never regress.
            highest_obj_id=max(data.get("highest_obj_id", -1), default_highest),
        )


@dataclass
class SAM3Project:
    """Top-level project managing multiple concepts"""
    project_dir: str
    video_path: str
    num_frames: int
    frame_dimensions: Tuple[int, int]
    concepts: List[SAM3Concept] = field(default_factory=list)
    concept_order: List[str] = field(default_factory=list)
    device: str = "cuda:0"
    default_opacity: float = 0.5
    fps: float = 30.0
    mjpeg_video_path: Optional[str] = None  # Re-encoded MJPEG video if created
    frames_dir: Optional[str] = None        # Persistent extracted-frame directory (JPEGs)
    mask_format: str = "png"  # "png" or "npz" — applies to all mask saves in this project
    # Path to the concept→instance-name vocabulary JSON last associated with this project.
    # The UI auto-loads it on project open (re-reading the file each time, so external
    # edits are picked up). None if no vocabulary has been linked.
    vocabulary_path: Optional[str] = None
    # Last gaze CSV / world-timestamps CSV pair loaded for this project. The UI auto-loads
    # this pair on project open (re-reading the files each time) if both still exist.
    gaze_csv_path: Optional[str] = None
    gaze_world_path: Optional[str] = None
    # Alternative locations tried when the primary video/frames paths are unreachable.
    # Populated automatically by the UI when the user manually locates a missing file.
    # Stored as plain strings; paths from other OSes are skipped gracefully at load time.
    alt_video_paths: List[str] = field(default_factory=list)
    alt_frames_dirs: List[str] = field(default_factory=list)
    # True when the source is a single image (treated as a 1-frame video internally).
    is_image: bool = False
    # ISO timestamp set every time save() writes project.json. None for projects saved
    # before this field existed. Reflects the last metadata write, not necessarily the
    # last completed detection/refine run (a load-only session never sets it).
    last_saved: Optional[str] = None
    # path -> mtime at last load/save, for detecting external writes (e.g. a concurrent
    # `sam3_process.py --refine` run). Not persisted to project.json.
    _loaded_mtimes: Dict[str, float] = field(default_factory=dict, init=False, repr=False, compare=False)

    def get_frames_dir(self) -> str:
        """Return the frame cache directory to use on this machine.

        If frames_dir is set and the directory (or its parent) is reachable on this machine,
        return it.  This covers both "already extracted" and "about to extract for the first
        time" on the originating host.

        If frames_dir is set but neither the directory nor its parent exists, the project was
        likely copied from another machine.  Fall back to a machine-local temp directory so
        that the original frames_dir value is left untouched in project.json — the originating
        host can still reuse it after the project is copied back.

        Callers that extract frames to the returned path should not persist the path in
        project.frames_dir; only explicit --frame-dir / UI "Extract Frames" actions may do so.
        """
        if self.frames_dir:
            if os.path.isdir(self.frames_dir):
                return self.frames_dir
            parent = os.path.dirname(os.path.abspath(self.frames_dir))
            if os.path.isdir(parent):
                return self.frames_dir
        video_hash = hashlib.md5(self.video_path.encode()).hexdigest()[:8]
        return os.path.join(tempfile.gettempdir(), f"sam3_frames_{video_hash}")

    def get_concept_dir(self, concept_name: str) -> str:
        """Get directory for concept data"""
        return os.path.join(self.project_dir, "concepts", concept_name)

    def get_concept_by_name(self, name: str) -> Optional[SAM3Concept]:
        """Find concept by name"""
        for concept in self.concepts:
            if concept.name == name:
                return concept
        return None

    def add_concept(self, concept: SAM3Concept):
        """Add concept to project and update concept_order"""
        if self.get_concept_by_name(concept.name) is not None:
            raise ValueError(f"Concept '{concept.name}' already exists")
        self.concepts.append(concept)
        if concept.name not in self.concept_order:
            self.concept_order.append(concept.name)

    def remove_concept(self, concept_name: str):
        """Remove concept from project"""
        self.concepts = [c for c in self.concepts if c.name != concept_name]
        if concept_name in self.concept_order:
            self.concept_order.remove(concept_name)

    def save(self, write_concept_metadata: bool = True):
        """Save project metadata to disk.

        Args:
            write_concept_metadata: Also rewrite concepts/<name>/concept_metadata.json
                for every concept. Pass False when worker processes own those files
                (parallel processing) so a stale in-memory copy can't clobber a
                concept another process just finished writing.
        """
        from datetime import datetime
        self.last_saved = datetime.now().isoformat()
        project_data = {
            "version": "1.0",
            "video_path": self.video_path,
            "num_frames": self.num_frames,
            "frame_dimensions": list(self.frame_dimensions),
            "fps": self.fps,
            "mjpeg_video_path": self.mjpeg_video_path,
            "frames_dir": self.frames_dir,
            "alt_video_paths": self.alt_video_paths,
            "alt_frames_dirs": self.alt_frames_dirs,
            "is_image": self.is_image,
            "vocabulary_path": self.vocabulary_path,
            "gaze_csv_path": self.gaze_csv_path,
            "gaze_world_path": self.gaze_world_path,
            "last_saved": self.last_saved,
            "concepts": [
                {
                    "name": c.name,
                    "text_prompt": c.text_prompt,
                    "color": list(c.color_rgb),
                    "status": c.status.value,
                    "num_instances": len(c.instances),
                }
                for c in self.concepts
            ],
            "concept_order": self.concept_order,
            "global_settings": {
                "device": self.device,
                "default_opacity": self.default_opacity,
                "mask_format": self.mask_format,
            }
        }

        # Save main project.json (atomic: a kill mid-write must not corrupt it)
        project_json_path = os.path.join(self.project_dir, "project.json")
        _atomic_json_dump(project_data, project_json_path)
        self._loaded_mtimes[project_json_path] = os.path.getmtime(project_json_path)

        # Save per-concept metadata
        if write_concept_metadata:
            for concept in self.concepts:
                self._save_concept_metadata(concept)

    def _save_concept_metadata(self, concept: SAM3Concept):
        """Save concept metadata to concepts/<name>/concept_metadata.json"""
        concept_dir = self.get_concept_dir(concept.name)
        os.makedirs(concept_dir, exist_ok=True)

        metadata_path = os.path.join(concept_dir, "concept_metadata.json")
        _atomic_json_dump(concept.to_dict(), metadata_path)
        self._loaded_mtimes[metadata_path] = os.path.getmtime(metadata_path)

    def check_staleness(self) -> List[str]:
        """Return paths (project.json / concept_metadata.json) modified on disk since load/save.

        A non-empty result means another process (e.g. a `--refine` run) wrote new
        results after this in-memory copy was last loaded or saved; saving now would
        overwrite them. Caller should reload the project instead of saving.
        """
        changed = []
        for path, known_mtime in self._loaded_mtimes.items():
            if not os.path.exists(path):
                continue
            if os.path.getmtime(path) != known_mtime:
                changed.append(path)
        return changed

    @classmethod
    def load(cls, project_dir: str) -> 'SAM3Project':
        """Load project from disk"""
        project_json_path = os.path.join(project_dir, "project.json")
        with open(project_json_path, 'r') as f:
            project_data = json.load(f)
        project_json_mtime = os.path.getmtime(project_json_path)

        # Create project instance
        project = cls(
            project_dir=project_dir,
            video_path=project_data["video_path"],
            num_frames=project_data["num_frames"],
            frame_dimensions=tuple(project_data["frame_dimensions"]),
            fps=project_data.get("fps", 30.0),
            mjpeg_video_path=project_data.get("mjpeg_video_path"),
            frames_dir=project_data.get("frames_dir"),
            alt_video_paths=project_data.get("alt_video_paths", []),
            alt_frames_dirs=project_data.get("alt_frames_dirs", []),
            is_image=project_data.get("is_image", False),
            vocabulary_path=project_data.get("vocabulary_path"),
            gaze_csv_path=project_data.get("gaze_csv_path"),
            gaze_world_path=project_data.get("gaze_world_path"),
            last_saved=project_data.get("last_saved"),
            concept_order=project_data.get("concept_order", []),
            device=project_data.get("global_settings", {}).get("device", "cuda:0"),
            default_opacity=project_data.get("global_settings", {}).get("default_opacity", 0.5),
            mask_format=project_data.get("global_settings", {}).get("mask_format", "png"),
        )
        project._loaded_mtimes[project_json_path] = project_json_mtime

        # Load per-concept metadata
        for concept_summary in project_data.get("concepts", []):
            concept = project._load_concept_metadata(concept_summary["name"])
            if concept:
                project.concepts.append(concept)

        return project

    def _load_concept_metadata(self, concept_name: str) -> Optional[SAM3Concept]:
        """Load concept metadata from concepts/<name>/concept_metadata.json"""
        metadata_path = os.path.join(self.get_concept_dir(concept_name), "concept_metadata.json")
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            concept_data = json.load(f)
        self._loaded_mtimes[metadata_path] = os.path.getmtime(metadata_path)

        return SAM3Concept.from_dict(concept_data)

    @classmethod
    def create_new(cls, project_dir: str, video_path: str, num_frames: int,
                   frame_dimensions: Tuple[int, int], fps: float = 30.0,
                   device: str = "cuda:0", mask_format: str = "png",
                   is_image: bool = False, frames_dir: Optional[str] = None) -> 'SAM3Project':
        """Create new project with directory structure"""
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "concepts"), exist_ok=True)

        project = cls(
            project_dir=project_dir,
            video_path=video_path,
            num_frames=num_frames,
            frame_dimensions=frame_dimensions,
            fps=fps,
            device=device,
            mask_format=mask_format,
            is_image=is_image,
            frames_dir=frames_dir,
        )

        # Save video info
        video_info = {
            "video_path": video_path,
            "num_frames": num_frames,
            "frame_dimensions": list(frame_dimensions),
            "fps": fps,
        }
        video_info_path = os.path.join(project_dir, "video_info.json")
        _atomic_json_dump(video_info, video_info_path)

        project.save()
        return project


@dataclass
class SerializableInferenceState:
    """Serializable subset of SAM3 inference_state"""
    image_size: int
    num_frames: int
    orig_height: int
    orig_width: int
    tracker_metadata: dict         # obj_ids_per_gpu, obj_id_to_score, etc.
    action_history: list           # User actions for reconstruction
    resource_path: str             # Path to frames directory
    offload_config: dict

    @classmethod
    def from_inference_state(cls, inference_state: dict, frames_dir: str) -> 'SerializableInferenceState':
        """Extract serializable data from SAM3 inference_state"""
        return cls(
            image_size=inference_state["image_size"],
            num_frames=inference_state["num_frames"],
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
            tracker_metadata=inference_state.get("tracker_metadata", {}),
            action_history=inference_state.get("action_history", []),
            resource_path=frames_dir,
            offload_config={
                "offload_video_to_cpu": True,
                "async_loading_frames": True
            }
        )

    def to_inference_state(self, sam3_model, video_path: str) -> str:
        """
        Reconstruct a SAM3 session from saved state.

        Creates a new session via start_session(), then restores
        tracker_metadata and action_history into the session state.

        Args:
            sam3_model: SAM3 model instance (Sam3VideoPredictorMultiGPU)
            video_path: Path to original video (to re-extract frames if /tmp was cleared)

        Returns:
            session_id: UUID string for the new session
        """
        import uuid
        from sam3_utils import ensure_frames_extracted

        # Re-extract frames if tmp dir was cleared (e.g. after system reboot).
        # Completeness-checked: an existence-only test would silently reuse a
        # partial directory left by a killed extraction.
        ensure_frames_extracted(video_path, self.resource_path, self.num_frames)

        session_id = str(uuid.uuid4())
        sam3_model.start_session(resource_path=self.resource_path, session_id=session_id)

        # Restore saved tracker state into the new session
        state = sam3_model._all_inference_states[session_id]["state"]
        if self.tracker_metadata:
            state["tracker_metadata"] = self.tracker_metadata
        if self.action_history:
            state["action_history"] = self.action_history

        return session_id

    def save(self, concept_dir: str):
        """Save to concepts/<name>/inference_state.pkl"""
        state_path = os.path.join(concept_dir, "inference_state.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, concept_dir: str) -> Optional['SerializableInferenceState']:
        """Load from concepts/<name>/inference_state.pkl"""
        state_path = os.path.join(concept_dir, "inference_state.pkl")
        if not os.path.exists(state_path):
            return None

        with open(state_path, 'rb') as f:
            return pickle.load(f)


def save_concept_state(concept: SAM3Concept, inference_state: dict, project_dir: str,
                       frames_dir: str):
    """Save serializable inference state for a concept"""
    concept_dir = os.path.join(project_dir, "concepts", concept.name)
    os.makedirs(concept_dir, exist_ok=True)
    serializable_state = SerializableInferenceState.from_inference_state(inference_state, frames_dir)
    serializable_state.save(concept_dir)


def load_concept_state(concept: SAM3Concept, project_dir: str, sam3_model, video_path: str) -> Optional[str]:
    """
    Load saved inference state for a concept and create a new SAM3 session from it.

    Returns:
        session_id string if state was restored, None if no saved state exists.
    """
    concept_dir = os.path.join(project_dir, "concepts", concept.name)
    serializable_state = SerializableInferenceState.load(concept_dir)

    if serializable_state is None:
        return None

    return serializable_state.to_inference_state(sam3_model, video_path)


def cleanup_frames_dir(project: SAM3Project):
    """Delete the session-scoped frame cache if it was not pinned by the user on this machine.

    Three cases:
    - frames_dir not set → always temp → delete the temp dir.
    - frames_dir set and accessible (get_frames_dir() returns it) → user-managed → keep.
    - frames_dir set but inaccessible (cross-machine copy) → get_frames_dir() fell back to
      a temp dir on this machine → delete that temp dir; leave frames_dir in project.json so
      the originating host can still reuse it.
    """
    effective = project.get_frames_dir()
    if project.frames_dir and effective == project.frames_dir:
        return  # user-managed path on this machine; don't touch
    if os.path.exists(effective):
        shutil.rmtree(effective)
        print(f"Cleaned up temporary frames: {effective}")
