"""
SAM3 Pipeline - Core segmentation logic for hierarchical object detection.

This module implements the main processing pipeline for SAM3-based video segmentation,
including concept detection, instance management, and refinement operations.
"""

import os
# Must be set before any torch import so the CUDA allocator uses expandable segments,
# which avoids fragmentation-driven monotonic growth in nvidia-smi memory readings.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import torch
import hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional, Callable
from pathlib import Path

from sam3_project import (
    SAM3Project, SAM3Concept, SAM3Instance, ConceptStatus,
    save_concept_state, load_concept_state
)
from sam3_utils import ensure_frames_extracted


def ensure_bf16_autocast():
    """Guarantee CUDA bf16 autocast is active on the current thread.

    SAM3's tracker enters a process-wide ``torch.autocast`` in its ``__init__``
    (``bf16_context.__enter__()``, "keep using for the entire model process").
    But autocast state is *thread-local*, so a worker thread that calls into the
    model after the thread which built it (e.g. the SAM3 UI spawns a fresh
    ``threading.Thread`` per operation) runs with autocast disabled and hits
    "mat1 and mat2 must have the same dtype, but got BFloat16 and Float" — the
    model's internal maskmem tensors stay bf16 regardless of autocast.

    Call this at the start of any thread that will invoke the SAM3 model. The
    entered context is intentionally never exited (matching SAM3's own pattern);
    the UI's worker threads are short-lived and single-purpose.
    """
    try:
        already = torch.is_autocast_enabled("cuda")
    except TypeError:  # older torch: no device-string arg
        already = torch.is_autocast_enabled()
    if already:
        return None
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    ctx.__enter__()
    return ctx


def _prune_non_cond_outputs(
    inner_state: dict,
    frame_idx: int,
    keep: int = 20,
    auto_cond_window: int = 48,
) -> None:
    """Prune stale frame outputs from every tracker state.

    SAM3's multi-GPU structure stores maskmem tensors at:
      inner_state["tracker_inference_states"][gpu]["output_dict"]["non_cond_frame_outputs"]
    SAM3.1 multiplex uses inner_state["sam2_inference_states"] with the same keys.

    Non-cond pruning
    ----------------
    Evict non_cond_frame_outputs entries older than `keep` frames.  SAM3 attends to at
    most num_maskmem=7 previous non-cond frames; keep is set to num_maskmem + 2 at each
    call site via _non_cond_keep_window(), so entries beyond that are never read.

    Cond frame pruning — two categories
    ------------------------------------
    Cond frames come from two sources:

    1. User-correction clicks: stored in point_inputs_per_obj.  NEVER evict.
       select_closest_cond_frames picks the temporally nearest cond frames, so a
       correction at frame 500 is still selected when processing frame 38000 — it is
       the most informative anchor for that instance.

    2. Auto-reconditioning: every recondition_every_nth_frame=16 frames,
       _recondition_masklets feeds the VG detector mask back via add_new_mask(),
       which stores a mask-only entry in mask_inputs_per_obj and promotes the frame
       to cond_frame_outputs (because add_all_frames_to_correct_as_cond=True on the
       tracker).  These are NOT user corrections.  For a 38 K-frame video that is
       ~2,400 auto-recondition cond frames that accumulate without bound.
       Safe to evict: select_closest_cond_frames always picks the NEAREST cond frame
       before the current position, so a recondition entry at F-32 is fully superseded
       by the closer entry at F-16.  We keep the most recent `auto_cond_window` frames
       worth (default 48 = 3 × 16-frame intervals), which is always more than enough.

    When evicting an auto-recondition cond frame we must maintain the invariant that
    all four of {output_dict["cond_frame_outputs"], consolidated_frame_inds["cond_frame_outputs"],
    output_dict_per_obj[*]["cond_frame_outputs"], mask_inputs_per_obj[*][frame]} are
    consistent — removal from one must propagate to all.
    """
    cutoff = frame_idx - keep
    auto_cond_cutoff = frame_idx - auto_cond_window

    tracker_states = (inner_state.get("tracker_inference_states", [])
                      + inner_state.get("sam2_inference_states", []))
    for ts in tracker_states:
        # --- Non-cond pruning (unchanged) ---
        non_cond = ts.get("output_dict", {}).get("non_cond_frame_outputs", {})
        stale = [f for f in non_cond if f < cutoff]
        for f in stale:
            del non_cond[f]

        for obj_dict in ts.get("output_dict_per_obj", {}).values():
            non_cond = obj_dict.get("non_cond_frame_outputs", {})
            stale = [f for f in non_cond if f < cutoff]
            for f in stale:
                del non_cond[f]

        # --- Cond frame pruning: auto-recondition entries only ---
        # Collect frames that have user-supplied point inputs — never evict these.
        user_point_frames: set = set()
        for per_frame in ts.get("point_inputs_per_obj", {}).values():
            user_point_frames.update(per_frame.keys())

        cond = ts.get("output_dict", {}).get("cond_frame_outputs", {})
        # Never evict the earliest cond frame — it is the initial detection anchor
        # (added by _tracker_add_new_objects → add_new_mask → propagate_in_video_preflight).
        # Evicting it would empty cond_frame_outputs and trigger "No points" on the
        # next tracker propagation call.  All later frames are auto-recondition entries
        # that are safe to evict once they fall outside the auto_cond_window.
        initial_frame = min(cond) if cond else None
        evict = {
            f for f in cond
            if f < auto_cond_cutoff and f not in user_point_frames
            and f != initial_frame
        }
        if not evict:
            continue

        for f in evict:
            del cond[f]

        # Keep consolidated_frame_inds in sync.
        consolidated_cond = ts.get("consolidated_frame_inds", {}).get("cond_frame_outputs")
        if consolidated_cond is not None:
            consolidated_cond -= evict

        # Per-object slices of cond_frame_outputs.
        for obj_dict in ts.get("output_dict_per_obj", {}).values():
            obj_cond = obj_dict.get("cond_frame_outputs", {})
            for f in evict:
                obj_cond.pop(f, None)

        # Remove mask_inputs_per_obj entries for evicted frames so the
        # point_inputs ∪ mask_inputs == cond_frame_inds invariant holds if
        # propagate_in_video_preflight is ever called again (e.g. refinement restart).
        # point_inputs_per_obj is intentionally left untouched (we only evict
        # frames that had no point inputs to begin with).
        for per_frame in ts.get("mask_inputs_per_obj", {}).values():
            for f in evict:
                per_frame.pop(f, None)


def _non_cond_keep_window(sam3_model, extra: int = 1) -> int:
    """Frames to retain in non_cond_frame_outputs: tracker's num_maskmem + a small buffer.

    SAM3's tracker attends to at most num_maskmem (default 7) non-cond frames per forward
    pass, so entries older than that are never read.  We keep num_maskmem + extra (=8) as
    a buffer in case the model variant differs.  num_maskmem is an architectural constant
    trained into the checkpoint, not a user-tunable knob.
    """
    try:
        return sam3_model.model.tracker.num_maskmem + extra
    except AttributeError:
        return 7 + extra  # SAM3 default num_maskmem=7



def _keep_nearest_cond_frames(frame_idx: int, cond_keys, max_cond: int, keep_first: bool) -> set:
    """Return the frame indices to keep GPU-resident, computed directly rather than via
    select_closest_cond_frames.

    Mirrors that function's own before/after/nearest-fill algorithm, but deliberately
    WITHOUT its "if len(cond_frame_outputs) <= max_cond_frame_num: keep everything"
    shortcut. That shortcut exists because select_closest_cond_frames' real job is
    deciding what to feed into *this step's* attention — correctly "just use all of it"
    when under budget — not deciding long-term GPU residency. Reused verbatim for
    eviction, it meant a bucket that individually never accumulates more than max_cond
    entries (common under SAM3's per-object/bucket tracker_inference_states split, even
    though the *total* committed entries across all buckets keeps growing) never gets
    evicted from at all — confirmed empirically: cond_counts stayed under max_cond=7 in
    every bucket for an entire 176-frame replay while allocated GPU memory grew ~0.3 GiB
    on every single add_prompt call, tracking output_dict_per_obj's ever-growing count
    of committed cond-frame entries almost exactly. Applying the same proximity-based
    selection unconditionally (regardless of total count) evicts the earliest entries
    that this step doesn't need, resident or not.
    """
    if not cond_keys:
        return set()
    keep = set()
    if keep_first:
        idx_first = min((t for t in cond_keys if t < frame_idx), default=None)
        if idx_first is None:
            idx_first = max((t for t in cond_keys if t > frame_idx), default=None)
        if idx_first is not None:
            keep.add(idx_first)
    idx_before = max((t for t in cond_keys if t < frame_idx), default=None)
    if idx_before is not None:
        keep.add(idx_before)
    idx_after = min((t for t in cond_keys if t >= frame_idx), default=None)
    if idx_after is not None:
        keep.add(idx_after)
    remaining = sorted(
        (t for t in cond_keys if t not in keep), key=lambda x: abs(x - frame_idx)
    )
    keep.update(remaining[: max(0, max_cond - len(keep))])
    return keep


def _offload_unselected_cond_frames(inner_state: dict, frame_idx: int, tracker) -> None:
    """Move cond-frame maskmem tensors to CPU for every cond frame not among the
    max_cond_frames_in_attn nearest to frame_idx, freeing GPU memory for concepts with
    many annotation/mask-anchor frames (each becomes a permanent cond frame that
    _prune_non_cond_outputs deliberately never evicts, since any of them could be the
    closest anchor thousands of frames later).

    Uses _keep_nearest_cond_frames (see its docstring for why this can't delegate to
    select_closest_cond_frames directly) so eviction actually happens even when a given
    tracker-state bucket's own cond-frame count never exceeds max_cond.

    No-op whenever tracker.max_cond_frames_in_attn == -1 (the default): in that mode
    every cond frame is always "nearest" (nothing to prune against). Only worth
    enabling together with a finite --max-cond-frames-in-attn.

    Reload is NOT handled here. Sam3TrackerBase._prepare_memory_conditioned_features
    (patched in utils.py's _patch_sam3_tracker_base to use .to(device) instead of a
    hardcoded .cuda()) already reloads a cond frame's maskmem_features/maskmem_pos_enc
    onto the correct GPU the moment select_closest_cond_frames selects it again for real
    attention use — so a frame evicted here by our own (slightly different, always-on)
    proximity policy still reloads correctly the next time SAM3's own selection picks it.
    """
    max_cond = getattr(tracker, "max_cond_frames_in_attn", -1)
    if max_cond is None or max_cond == -1:
        return
    keep_first = getattr(tracker, "keep_first_cond_frame", False)

    # Deliberately excludes inner_state["sam2_inference_states"] (SAM3.1 multiplex).
    # Its own _prepare_memory_conditioned_features (video_tracking_multiplex.py) still
    # hardcodes bare .cuda() for the reload side (never patched in utils.py, unlike the
    # regular Sam3TrackerBase version which uses .to(device)). A bare .cuda() only
    # happens to land on the right GPU because this codebase calls
    # torch.cuda.set_device(device) once per process at model load — true under today's
    # one-GPU-per-process usage, but would silently misroute a CPU-offloaded tensor to
    # the wrong GPU under multi-GPU sharding within one process. Skip multiplex sessions
    # here until that reload path is patched the same way.
    def _to_cpu(entry: dict) -> None:
        feats = entry.get("maskmem_features")
        if torch.is_tensor(feats) and feats.is_cuda:
            entry["maskmem_features"] = feats.cpu()
        pos_enc = entry.get("maskmem_pos_enc")
        if pos_enc is not None:
            entry["maskmem_pos_enc"] = [
                p.cpu() if torch.is_tensor(p) and p.is_cuda else p for p in pos_enc
            ]

    tracker_states = inner_state.get("tracker_inference_states", [])
    for ts in tracker_states:
        cond = ts.get("output_dict", {}).get("cond_frame_outputs", {})
        if not cond:
            continue
        keep = _keep_nearest_cond_frames(frame_idx, cond.keys(), max_cond, keep_first)
        unselected = [f for f in cond if f not in keep]
        if not unselected:
            continue

        for f in unselected:
            entry = cond.get(f)
            if entry is not None:
                _to_cpu(entry)
            # output_dict_per_obj's cond-frame slices are independent tensor objects
            # from _add_output_per_object (not views that follow the parent's new CPU
            # copy), so they must be offloaded separately or the original GPU tensor
            # they still reference never gets freed.
            for obj_dict in ts.get("output_dict_per_obj", {}).values():
                obj_entry = obj_dict.get("cond_frame_outputs", {}).get(f)
                if obj_entry is not None:
                    _to_cpu(obj_entry)


def _consolidate_and_evict_feature_cache(
    inner_state: dict, sam3_model, frame_indices,
) -> None:
    """Consolidate pending temp outputs and evict feature_cache for the given frames.

    Companion to the frame_idx-descending conditioning pass in replay_concept_refinements
    (and any other loop that calls add_prompt/add_new_mask for many out-of-order frames
    before propagation starts). Each such call populates inner_state["feature_cache"]
    with a full per-frame backbone+FPN feature set (sam3_video_base.py's
    run_backbone_and_detection), keyed by frame_idx. SAM3's own eviction there —
    `feature_cache.pop(frame_idx - 1, None)` — only ever drops the *immediately
    adjacent* frame, which assumes dense sequential propagation; visiting arbitrary,
    non-adjacent annotated frames means that eviction essentially never fires, so
    every touched frame's feature set piles up simultaneously (documented, previously
    tolerated pileup — see the comment above the frame_tasks loop in
    replay_concept_refinements). For a concept with hundreds of annotated frames this
    dwarfs the (already-bounded, see _offload_unselected_cond_frames) cond_frame_outputs
    memory and was the actual cause of the OOMs seen even with --max-cond-frames set.

    A raw frame's feature_cache entry can only be safely dropped once its conditioning
    is consolidated out of temp_output_dict_per_obj and into output_dict — otherwise
    propagate_in_video_preflight's own (eventual, automatic) consolidation pass would
    hit "Image features for frame N are not cached" (the tracker submodule has no
    backbone of its own to recompute a miss). propagate_in_video_preflight() only
    processes frames with PENDING temp outputs (it clears temp_output_dict_per_obj as it
    goes), so calling it here early and repeatedly — once per batch of `frame_indices`
    instead of once for the whole replay set — is safe and merely does in several smaller
    passes what would otherwise happen once at the end automatically inside
    sam3_model.propagate_in_video(). Once consolidated, this batch's frames no longer
    depend on feature_cache and can be evicted immediately, bounding feature_cache to
    roughly one batch's worth of frames instead of the entire annotated set.

    Restricted to inner_state["tracker_inference_states"] (not the SAM3.1 multiplex
    "sam2_inference_states") to match _offload_unselected_cond_frames — feature_cache
    eviction itself is a plain dict pop with no device transfer, so it would be safe for
    multiplex too, but propagate_in_video_preflight's consolidation path has not been
    audited there and this fix hasn't been exercised on that code path.
    """
    if not frame_indices:
        return
    for ts in inner_state.get("tracker_inference_states", []):
        sam3_model.model.tracker.propagate_in_video_preflight(ts, run_mem_encoder=True)
    feature_cache = inner_state.get("feature_cache", {})
    for f in frame_indices:
        feature_cache.pop(f, None)
    _clear_multigpu_buffer(inner_state)


def _clear_multigpu_buffer(inner_state: dict) -> None:
    """Drop every entry in feature_cache["multigpu_buffer"] — the VG detector's own
    chunk cache (sam3_image.py's forward_video_grounding_multigpu), nested inside the
    same feature_cache dict under a string key our frame-indexed feature_cache.pop(f)
    calls never reach.

    That detector caches BOTH the current frame's chunk and proactively prefetches the
    NEXT sequential chunk (frame_idx + world_size) for compute/transfer overlap, evicting
    only the chunk immediately behind it (frame_idx - world_size). With world_size=1
    (single-GPU, the only case audited here) that means every call strands at least the
    prefetched next-frame entry — our conditioning loop visits sparse, arbitrary
    annotated frames, never the true next/previous sequential frame, so neither the
    eviction nor a same/adjacent-frame cache hit ever actually happens. Confirmed via
    targeted memory instrumentation as the actual source of a steady ~0.3 GiB-per-call
    growth previously misattributed to cond_frame_outputs (which turned out to already
    be tiny and CPU-resident).

    Clearing it outright is safe for our access pattern specifically: this cache exists
    purely to let a later call for a NEARBY frame_idx skip recomputation, and we never
    revisit a frame or its immediate neighbor during this conditioning pass. It is NOT
    safe to clear during propagate_in_video()'s own real (sequential) propagation loop,
    where consecutive frames genuinely reuse adjacent chunks — this function must only be
    called from the pre-propagation conditioning passes, not the propagation loops.
    """
    feature_cache = inner_state.get("feature_cache", {})
    buf = feature_cache.get("multigpu_buffer")
    if buf:
        buf.clear()


def compute_period_peaks(periods: List[Tuple[int, int]],
                         frame_pixel_counts: "dict[int, int]",
                         total_pixels: int = 0) -> List[dict]:
    """For each (start, end) period return the peak frame and pixel-ratio stats.

    Returns a list of dicts:
        {"start": int, "end": int, "best_frame": int, "pixel_count": int,
         "avg_pixel_ratio": float, "min_pixel_ratio": float, "max_pixel_ratio": float}

    pixel_ratio fields are 0.0 when total_pixels is 0 or no counts are available.
    Intended to be computed once during propagation (while masks are in memory)
    and stored on SAM3Instance so later operations (delete/absorb anchor
    generation) can find the best frame per period without re-reading any PNGs.
    """
    peaks = []
    inv = 1.0 / total_pixels if total_pixels > 0 else 0.0
    for start, end in periods:
        best_frame = None
        best_count = 0
        total_count = 0
        min_count = None
        max_count = 0
        n_present = 0
        for f in range(start, end + 1):
            c = frame_pixel_counts.get(f, 0)
            if c > best_count:
                best_count = c
                best_frame = f
            if c > 0:
                total_count += c
                max_count = max(max_count, c)
                min_count = c if min_count is None else min(min_count, c)
                n_present += 1
        if best_frame is not None and best_count > 0:
            avg_ratio = (total_count * inv / n_present) if n_present > 0 else 0.0
            peaks.append({
                "start": start, "end": end,
                "best_frame": best_frame, "pixel_count": best_count,
                "avg_pixel_ratio": avg_ratio,
                "min_pixel_ratio": (min_count or 0) * inv,
                "max_pixel_ratio": max_count * inv,
            })
    return peaks


def compute_presence_norm(frame_pixel_counts: "dict[int, int]", total_pixels: int) -> float:
    """99th-percentile pixel ratio across all detected frames (for presence bar normalization).

    Returns 0.0 when total_pixels is 0 or no non-zero frame counts are available.
    The UI uses this as the bar-height denominator so typical frames map near 100% height
    while only the top 1% of frames (outlier peaks) clip at the ceiling.
    """
    if total_pixels <= 0 or not frame_pixel_counts:
        return 0.0
    ratios = sorted(c / total_pixels for c in frame_pixel_counts.values() if c > 0)
    if not ratios:
        return 0.0
    n = len(ratios)
    p = 0.99 * (n - 1)
    lo = int(p)
    hi = min(lo + 1, n - 1)
    return ratios[lo] + (p - lo) * (ratios[hi] - ratios[lo])


def _backfill_stale_period_counts(
    periods: List[Tuple[int, int]],
    pcounts: "dict[int, int]",
    mask_dir: str,
) -> "dict[int, int]":
    """Fill in pixel counts (from on-disk mask files) for periods this run's
    propagation produced no live output for.

    `pcounts` only has entries for frames this propagation pass actually emitted
    for the object. If the tracker stops including an object in `out_obj_ids`
    partway through a refine run (e.g. it drops out of the joint batch while a
    prior, more complete run's masks are still on disk for a later stretch),
    `continuous_periods` — read straight from disk — still reports that stretch,
    but `pcounts` has nothing for it. Left alone, `compute_period_peaks` silently
    produces no peak for that period and `compute_presence_norm` is computed only
    from the covered frames, so the presence bar can under-report (or hide
    entirely) a period with a large real mask. Only periods with zero live
    coverage are re-read from disk, so fully-covered runs pay no extra cost.
    """
    from sam3_utils import load_sam3_mask
    merged = dict(pcounts)
    for start, end in periods:
        if any(f in pcounts for f in range(start, end + 1)):
            continue  # at least partially covered by this run's live output
        for f in range(start, end + 1):
            mask_np = load_sam3_mask(mask_dir, f)
            if mask_np is None:
                continue
            count = int((mask_np > 0).sum())
            if count > 0:
                merged[f] = count
    return merged


def _promote_new_instances(concept: "SAM3Concept", new_obj_ids, pixel_counts_by_obj: dict,
                           instances_root: str, total_pixels: int, num_frames: int) -> None:
    """Create new SAM3Instance records for object ids that produced mask output during
    propagation but matched none of live/deleted/absorbed — i.e. objects the detector
    found for the first time this round, anywhere in the video (not just at the initial
    detection frame). Only called when new_instance_policy == "allow"; the "latent"
    policy leaves such ids unpersisted and un-promoted (same as today's default before
    this option existed), letting their sub-state simply keep competing unsaved.
    """
    from sam3_utils import generate_instance_color
    existing_ids = {inst.sam3_obj_id for inst in concept.instances}
    for i, obj_id in enumerate(sorted(new_obj_ids)):
        if obj_id in existing_ids:
            continue
        mask_dir = os.path.join(instances_root, str(obj_id), "masks")
        periods = compute_continuous_periods(mask_dir)
        if not periods:
            continue  # nothing actually written for this id — nothing to promote
        pcounts = pixel_counts_by_obj.get(obj_id, {})
        idx = len(concept.instances) + i
        instance = SAM3Instance(
            sam3_obj_id=obj_id,
            user_name=f"{concept.name}_{obj_id}",
            concept_name=concept.name,
            color_rgb=generate_instance_color(idx, idx + 1),
            score=len(pcounts) / max(num_frames, 1),
            first_detection_frame=periods[0][0],
            last_detection_frame=periods[-1][1],
            num_frames_with_mask=sum(e - s + 1 for s, e in periods),
            continuous_periods=periods,
            period_peaks=compute_period_peaks(periods, pcounts, total_pixels),
            presence_norm=compute_presence_norm(pcounts, total_pixels),
        )
        concept.instances.append(instance)
        concept.highest_obj_id = max(concept.highest_obj_id, obj_id)
        print(f"  New instance detected mid-video: obj_id={obj_id} "
              f"('{instance.user_name}'), {len(pcounts)} frame(s) with mask.")


def _inject_mask_anchor_at_frame(
    sam3_model,
    session_id: str,
    concept: "SAM3Concept",
    project_dir: str,
    orig_width: int,
    orig_height: int,
    frame_idx: int,
    inst: "SAM3Instance",
) -> None:
    """Inject one instance's on-disk mask_anchors/ frame into the SAM3 tracker via
    add_new_mask(), for a single (frame_idx, inst) pair.

    Must be called AFTER text detection (which initialises tracker states and
    assigns obj_ids) but BEFORE propagation, and — like add_prompt(points=...) —
    as part of a single GLOBAL frame_idx-DESCENDING pass shared with any point/box
    conditioning calls for the same concept (see the call site for why: both this
    function and add_prompt(points=...) funnel through _prepare_backbone_feats /
    run_backbone_and_detection, which unconditionally evicts
    feature_cache[frame_idx - 1] — confirmed at sam3_video_inference.py's
    add_tracker_new_points and below. Calling this in a separate pass after all
    point/box prompts have been added, as a prior version of this function did,
    could silently evict an already-cached point/box frame's features whenever a
    mask-anchor frame equals (some point/box frame + 1), causing
    propagate_in_video_preflight() to fail with "Image features for frame N are not
    cached" — exactly the failure the descending-order discipline exists to prevent,
    just not closed across the two prompt types until they share one pass).

    SAM3 stores the mask in mask_inputs_per_obj and promotes the frame to
    cond_frame_outputs, so it is selected by select_closest_cond_frames during
    propagation — exactly like a user-provided mask prompt.

    If an instance has no sub-state yet this round (not redetected at the initial
    frame, no point/box conditioning) its mask anchor is not simply dropped: a
    synthetic centroid point first seeds a brand-new sub-state (add_new_mask alone
    cannot create one — it only registers into an already-existing, not-yet-tracked
    sub-state), then add_new_mask overwrites that frame's conditioning with the real
    mask, discarding the synthetic point. Same technique as _preload_original_masks().
    """
    import cv2
    import numpy as np
    inner_state = sam3_model._all_inference_states[session_id]["state"]
    if not (inner_state.get("tracker_inference_states") or inner_state.get("sam2_inference_states")):
        return

    anchors_dir = os.path.join(
        project_dir, "concepts", concept.name,
        "instances", str(inst.sam3_obj_id), "mask_anchors",
    )
    # Prefer PNG; fall back to NPZ via cv2 (single channel)
    candidate = None
    for ext in (".png", ".npz"):
        p = os.path.join(anchors_dir, f"{frame_idx:06d}{ext}")
        if os.path.exists(p):
            candidate = p
            break
    if candidate is None:
        return
    if candidate.endswith(".npz"):
        data = np.load(candidate)
        key = list(data.keys())[0]
        mask_np = data[key]
    else:
        mask_np = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        return
    mask_tensor = torch.from_numpy((mask_np > 127).astype("float32"))
    # Inject only into the owning sub-state (not every sub-state).
    # Injecting into a foreign sub-state registers a spurious local index
    # that corrupts propagate_in_video_preflight's assertion.
    target_states = sam3_model.model._get_tracker_inference_states_by_obj_ids(
        inner_state, [inst.sam3_obj_id]
    )
    if not target_states:
        binary = (mask_np > 127).astype(np.uint8)
        ys, xs = np.where(binary)
        if ys.size == 0:
            print(f"  [skip] mask anchor obj_id={inst.sam3_obj_id} frame={frame_idx}: "
                  f"empty mask, cannot seed a new sub-state")
            return
        cx_px, cy_px = float(xs.mean()), float(ys.mean())
        # The moments-style centroid (mean pixel position) can land outside the
        # mask for non-convex shapes (crescent, ring, two blobs joined by a thin
        # neck) — fall back to the distance-transform peak (point most distant
        # from the mask boundary, guaranteed inside). Same fallback already used
        # elsewhere in this codebase (sam3_ui.py's per-period anchor point calc).
        if not binary[int(round(cy_px)), int(round(cx_px))]:
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
            cx_px, cy_px = float(ix), float(iy)
        cx = cx_px / orig_width
        cy = cy_px / orig_height
        try:
            sam3_model.add_prompt(
                session_id=session_id, frame_idx=frame_idx, obj_id=inst.sam3_obj_id,
                points=[[cx, cy]], point_labels=[1],
            )
        except Exception as e:
            print(f"  [skip] mask anchor obj_id={inst.sam3_obj_id} frame={frame_idx}: "
                  f"could not seed new sub-state: {e}")
            return
        target_states = sam3_model.model._get_tracker_inference_states_by_obj_ids(
            inner_state, [inst.sam3_obj_id]
        )
        if not target_states:
            print(f"  [skip] mask anchor obj_id={inst.sam3_obj_id} frame={frame_idx}: "
                  f"sub-state seed did not register")
            return
        print(f"  Seeded new sub-state for obj_id={inst.sam3_obj_id} frame={frame_idx} "
              f"from mask anchor centroid.")
    # Pre-populate backbone features for this frame.  SAM3's add_new_mask calls
    # _run_single_frame_inference which needs ts["cached_features"][frame_idx].
    # ts["cached_features"] is a shared reference to inner_state["feature_cache"].
    # Must be called inside torch.inference_mode() — SAM3's backbone stores
    # inference tensors that cannot be saved for backward otherwise.
    # Do NOT pop features after injection: propagate_in_video_preflight calls
    # _run_memory_encoder → _get_image_feature for each conditioning frame and
    # requires those features to still be in feature_cache.
    try:
        with torch.inference_mode():
            sam3_model.model._prepare_backbone_feats(inner_state, frame_idx, reverse=False)
    except Exception as e:
        print(f"  [skip] mask anchor obj_id={inst.sam3_obj_id} frame={frame_idx}: "
              f"could not prepare backbone features: {e}")
        return
    injected = False
    for ts in target_states:
        try:
            sam3_model.model.tracker.add_new_mask(
                ts, frame_idx, inst.sam3_obj_id, mask_tensor
            )
            print(f"  Injected mask anchor: obj_id={inst.sam3_obj_id} frame={frame_idx}")
            injected = True
        except Exception as e:
            print(f"  [warn] mask anchor injection failed "
                  f"obj_id={inst.sam3_obj_id} frame={frame_idx}: {e}")

    if injected:
        # add_new_mask() is called directly on the raw tracker (sam3_model.model.tracker),
        # bypassing the video-level Sam3VideoInferenceWithInstanceInteractivity wrapper
        # entirely — so it never records anything in inference_state["action_history"].
        # parse_action_history_for_propagation() (sam3_video_inference.py) decides which
        # object ids get real Tracker propagation vs. just "merge in whatever VG's own
        # per-frame guess was" purely from that history — an object with a mask anchor but
        # no point/box refinement this round (recorded via add_tracker_new_points's own
        # add_action_history("refine", ...) call) is invisible to it, so it's silently
        # excluded from Tracker propagation and falls back to VG's raw per-frame detection
        # everywhere except the exact anchor frame itself (which add_new_mask wrote directly
        # into that frame's own stored output) — i.e. the anchor appears to "not take effect"
        # beyond its own frame. Registering the same "refine" action that a point/box
        # correction would have logged fixes this with no other behavior change.
        sam3_model.model.add_action_history(
            inner_state, "refine", frame_idx=frame_idx, obj_ids=[inst.sam3_obj_id]
        )


def _inject_mask_anchors(
    sam3_model,
    session_id: str,
    concept: "SAM3Concept",
    project_dir: str,
    orig_width: int,
    orig_height: int,
) -> None:
    """Inject every non-deleted instance's on-disk mask_anchors/ frames, in a single
    GLOBAL frame_idx-DESCENDING pass across ALL instances together (not per-instance).

    Standalone entry point for callers with no point/box prompts of their own to
    interleave with (e.g. online_replay_concept_refinements' fast path). Callers that
    ALSO add point/box prompts in the same session (e.g. replay_concept_refinements)
    must NOT call this afterward as a second pass — see _inject_mask_anchor_at_frame's
    docstring for why that reintroduces the exact eviction race this ordering
    discipline prevents. Those callers should instead merge mask-anchor frames into
    their own descending frame loop and call _inject_mask_anchor_at_frame directly.
    """
    anchor_tasks = []  # (frame_idx, inst)
    for inst in concept.instances:
        if inst.deleted or not inst.mask_anchor_frames:
            continue
        for frame_idx in inst.mask_anchor_frames:
            anchor_tasks.append((frame_idx, inst))
    anchor_tasks.sort(key=lambda t: t[0], reverse=True)

    for frame_idx, inst in anchor_tasks:
        _inject_mask_anchor_at_frame(
            sam3_model, session_id, concept, project_dir, orig_width, orig_height,
            frame_idx, inst,
        )


def compute_continuous_periods(mask_dir: str) -> List[Tuple[int, int]]:
    """
    Return maximal runs of consecutive frame indices that have a mask file.
    Result is a sorted list of (start_frame, end_frame) inclusive tuples.
    """
    if not os.path.exists(mask_dir):
        return []
    frame_nums = []
    for fname in os.listdir(mask_dir):
        try:
            frame_nums.append(int(os.path.splitext(fname)[0]))
        except ValueError:
            continue
    if not frame_nums:
        return []
    frame_nums.sort()
    periods: List[Tuple[int, int]] = []
    start = prev = frame_nums[0]
    for f in frame_nums[1:]:
        if f == prev + 1:
            prev = f
        else:
            periods.append((start, prev))
            start = prev = f
    periods.append((start, prev))
    return periods


def _preload_original_masks(
    sam3_model,
    session_id: str,
    concept: "SAM3Concept",
    project_dir: str,
    orig_width: int,
    orig_height: int,
) -> None:
    """Pre-initialise LIVE instances from their original first-period masks.

    For hotstart-detected instances: injects a cond frame at the first detection
    frame, reinforcing the tracker's initial region.
    For late-appearing instances (no sub-state yet): creates the sub-state via a
    centroid point prompt, then immediately overwrites it with the full mask.

    After this call, max_obj_id = max(live obj_ids).  remove_object() can then be
    called for deleted instances; any ghost replacements during propagation will
    receive IDs above max_obj_id and are blocked by the live_obj_ids output gate.
    """
    import cv2
    MIN_PIXELS = 50
    inner_state = sam3_model._all_inference_states[session_id]["state"]

    for inst in concept.instances:
        if inst.deleted:
            continue
        mask_dir = os.path.join(
            project_dir, "concepts", concept.name,
            "instances", str(inst.sam3_obj_id), "masks",
        )
        periods = compute_continuous_periods(mask_dir)
        if not periods:
            continue

        # Walk the first period to find the earliest frame with enough pixels.
        first_frame, first_mask_np = None, None
        for f in range(periods[0][0], periods[0][1] + 1):
            for ext in (".npz", ".png"):
                p = os.path.join(mask_dir, f"{f:06d}{ext}")
                if not os.path.exists(p):
                    continue
                if ext == ".npz":
                    d = np.load(p)
                    arr = d[list(d.keys())[0]]
                else:
                    arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if arr is not None and int((arr > 0).sum()) >= MIN_PIXELS:
                    first_frame, first_mask_np = f, arr
                    break
            if first_frame is not None:
                break
        if first_frame is None:
            continue

        mask_tensor = torch.from_numpy((first_mask_np > 0).astype("float32"))

        # Ensure a sub-state exists for this obj_id.
        target_states = sam3_model.model._get_tracker_inference_states_by_obj_ids(
            inner_state, [inst.sam3_obj_id]
        )
        if not target_states:
            # Late-appearing instance — create sub-state via centroid point.
            ys, xs = np.where(first_mask_np > 0)
            cx = float(xs.mean()) / orig_width
            cy = float(ys.mean()) / orig_height
            sam3_model.add_prompt(
                session_id=session_id,
                frame_idx=first_frame,
                obj_id=inst.sam3_obj_id,
                points=[[cx, cy]],
                point_labels=[1],
            )
            target_states = sam3_model.model._get_tracker_inference_states_by_obj_ids(
                inner_state, [inst.sam3_obj_id]
            )

        for ts in target_states:
            try:
                sam3_model.model.tracker.add_new_mask(
                    ts, first_frame, inst.sam3_obj_id, mask_tensor
                )
                # Consolidate this cond frame NOW, while its raw backbone features are
                # still hot in the shared feature_cache. add_new_mask only stores a
                # *temporary* per-object output (temp_output_dict_per_obj); without an
                # immediate preflight call it stays uncommitted until whatever add_prompt
                # call happens next. run_backbone_and_detection evicts feature_cache
                # entries adjacent to the frame it just computed (frame_idx - 1) — so if
                # a later refinement point lands on first_frame + 1 (a common case: the
                # correction is right after where the original detection ended), that
                # call evicts first_frame's features out from under this still-pending
                # temp output, and the next preflight (which consolidates ALL objects'
                # pending temp frames together, not just the newly added one) crashes
                # with "Image features for frame N are not cached" since the tracker
                # component has no backbone of its own to recompute them. Calling
                # preflight here — mirroring what add_tracker_new_points already does
                # right after every point add — commits the mask-anchor output into
                # output_dict immediately, so it no longer depends on feature_cache.
                sam3_model.model.tracker.propagate_in_video_preflight(
                    ts, run_mem_encoder=True
                )
                print(f"  Pre-initialised obj_id={inst.sam3_obj_id} "
                      f"from original mask at frame {first_frame}")
            except Exception as e:
                print(f"  [warn] pre-init failed obj_id={inst.sam3_obj_id} "
                      f"frame={first_frame}: {e}")


def save_cond_frame_states(sam3_model, session_id: str, concept_dir: str) -> int:
    """
    Save maskmem cond frame states to disk after propagation.

    Must be called BEFORE close_session() while tracker states are still live.
    Saves the combined [B, ...] tensors per cond frame so that
    restore_cond_frame_states() can inject them back in a fresh session.

    tracker_inference_states is a flat list of tracker states — despite being
    loosely called "ranks" elsewhere, each entry is actually a separate batch
    of objects (SAM3 spins up a new one whenever new_instance_policy="allow"
    registers objects mid-video), not a GPU shard. Per-frame tensors are read
    from each object's own `output_dict_per_obj[obj_idx]` slice, NOT sliced by
    position out of the shared batched `output_dict["cond_frame_outputs"]`
    tensor: that batched tensor is frozen at whatever object count existed at
    THAT frame's own consolidation time, which can be smaller than the
    tracker_state's current/final object count (SAM3 keeps registering new
    objects into the same tracker_state as detection/refinement continues, and
    never retroactively re-consolidates older frames to match), so positional
    slicing by today's obj_ids goes out of bounds or silently mispairs objects
    with the wrong vectors. `output_dict_per_obj` has no such problem: it's
    keyed by each object's own stable obj_idx and always stores that object's
    own [1, ...] slice, populated fresh every time that frame is consolidated
    for that object, regardless of how many other objects exist. Entries are
    concatenated here (in tracker_inference_states list order, then ascending
    obj_idx within each) rather than kept as separate files, and "rank_sizes"
    in the index records each tracker_state's current obj_ids length so the
    combined batch can be split back apart on restore.

    Returns number of cond frame files written.
    """
    inference_state = sam3_model._all_inference_states[session_id]["state"]
    tracker_states = inference_state.get("tracker_inference_states", [])

    if not tracker_states:
        print("save_cond_frame_states: no tracker states found — nothing to save.")
        return 0

    cond_states_dir = os.path.join(concept_dir, "cond_states")
    os.makedirs(cond_states_dir, exist_ok=True)

    # Group cond frame outputs across ranks/objects by frame index.
    # entries: frame_t (int) -> [(rank, obj_idx, obj_id, out), ...] in
    # (rank, obj_idx) order, matching what restore_cond_frame_states expects
    # to split back apart via per-rank offsets.
    frames_data: dict = {}
    for rank, tracker_state in enumerate(tracker_states):
        obj_id_to_idx = tracker_state.get("obj_id_to_idx", {})
        output_dict_per_obj = tracker_state.get("output_dict_per_obj", {})
        for obj_id, obj_idx in obj_id_to_idx.items():
            cond_outputs = output_dict_per_obj.get(obj_idx, {}).get("cond_frame_outputs", {})
            for frame_t, out in cond_outputs.items():
                if out.get("maskmem_features") is None and out.get("obj_ptr") is None:
                    continue
                frames_data.setdefault(int(frame_t), []).append((rank, obj_idx, int(obj_id), out))

    def _cat(entries, key, dtype):
        tensors = [e[3].get(key) for e in entries]
        if any(t is None for t in tensors):
            return None
        return torch.cat([t.detach().cpu().to(dtype) for t in tensors], dim=0).numpy()

    maskmem_pos_enc_saved = False
    total_saved = 0
    saved_frames = []

    for frame_t, entries in sorted(frames_data.items()):
        entries.sort(key=lambda e: (e[0], e[1]))  # (rank, obj_idx) ascending
        obj_ids = np.array([e[2] for e in entries], dtype=np.int64) if entries else np.array([], dtype=np.int64)

        save_dict = {"obj_ids": obj_ids}
        mf = _cat(entries, "maskmem_features", torch.float16)
        if mf is not None:
            save_dict["maskmem_features"] = mf
        obj_ptr = _cat(entries, "obj_ptr", torch.float32)
        if obj_ptr is not None:
            save_dict["obj_ptr"] = obj_ptr
        pred_masks = _cat(entries, "pred_masks", torch.float16)
        if pred_masks is not None:
            save_dict["pred_masks"] = pred_masks
        obj_score = _cat(entries, "object_score_logits", torch.float32)
        if obj_score is not None:
            save_dict["object_score_logits"] = obj_score
        iou_score = _cat(entries, "iou_score", torch.float32)
        if iou_score is not None:
            save_dict["iou_score"] = iou_score

        fname = os.path.join(cond_states_dir, f"frame{frame_t:06d}.npz")
        np.savez_compressed(fname, **save_dict)
        saved_frames.append(frame_t)
        total_saved += 1

        # maskmem_pos_enc is a model constant (same for all frames/objects); save once
        if not maskmem_pos_enc_saved:
            pos_enc = entries[0][3].get("maskmem_pos_enc")
            if pos_enc is not None:
                torch.save(pos_enc, os.path.join(cond_states_dir, "maskmem_pos_enc.pt"))
                maskmem_pos_enc_saved = True

    index = {
        "saved_at": datetime.now().isoformat(),
        "rank_sizes": [len(ts.get("obj_ids", [])) for ts in tracker_states],
        "frames": sorted(saved_frames),
    }
    with open(os.path.join(cond_states_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {total_saved} cond frame states → {cond_states_dir}")
    return total_saved


def save_refinement_obj_ptrs(sam3_model, session_id: str, concept: "SAM3Concept", concept_dir: str) -> int:
    """
    Save each instance's obj_ptr (256-d, non-spatial object-identity embedding) from
    its cond frames in this refinement round — not the full maskmem_features (~650x
    larger, a spatial map tied to one frame's own layout, not a natural fit for
    pooling across videos).

    Both the MEAN over cond frames (for a cheap single-vector descriptor) and the RAW
    per-cond-frame vectors (for a future consumer that wants the actual set — e.g.
    attention-pooling or nearest-neighbor matching the way SAM3's own tracker consumes
    obj_ptr, as a set of tokens, not a single prototype) are saved. Averaging is a
    one-way door — the mean can always be recomputed from the raw set later, but not
    vice versa — and the raw vectors are cheap (256-d each, single-digit to low-double
    -digit count per instance), so there's no real cost to keeping both.

    Cond frames where object_score_logits <= 0 are excluded from both. Those are
    frames where the model judged the instance absent — typically a negative-only
    correction ("this was wrongly detected here, remove it"), not a positive
    sighting. Their obj_ptr is gated toward the model's generic `no_obj_ptr`
    placeholder (see sam3_tracker_base.py's obj_ptr computation), so including them
    would dilute the signal with a non-appearance vector rather than reinforcing the
    instance's actual identity.

    Keyed by the user-assigned instance name (SAM3Instance.user_name), not the
    internal sam3_obj_id: obj_id is an ephemeral tracking artifact (reassigned
    across redetections), while user_name is the durable, meaningful identity a
    human gave the object — the thing worth pooling across runs and videos.
    Nothing currently stops two instances of the same concept from sharing a
    user_name (only concept names are duplicate-checked, see sam3_ui.py
    rename_instance), so when that happens their per-instance mean obj_ptr vectors
    are themselves averaged together into one row for that name in "obj_ptr" (each
    instance counted once, regardless of how many cond frames it had), while the raw
    vectors from both instances are simply concatenated under that name in
    "raw_obj_ptr" (nothing collapsed). Deleted instances are skipped, as are any
    obj_ids with no matching live instance.

    Intended to be called only from refinement rounds (human-corrected anchors),
    not initial detection. Writes a single file per concept, OVERWRITTEN on every
    call — each new round supersedes the last, since it replays every historical
    annotation and re-derives fresh (presumably improved) anchors, so an older
    round's obj_ptr is stale, not complementary.

    Must be called BEFORE close_session() while tracker states are still live.
    Returns number of named instance groups whose obj_ptr was saved.
    """
    inference_state = sam3_model._all_inference_states[session_id]["state"]
    tracker_states = inference_state.get("tracker_inference_states", [])

    if not tracker_states:
        print("save_refinement_obj_ptrs: no tracker states found — nothing to save.")
        return 0

    # obj_id -> list of (frame_idx, obj_ptr vector) from cond frames judged "object present"
    by_obj_id: dict = {}
    for tracker_state in tracker_states:
        # Read each object's own slice from output_dict_per_obj, keyed by that
        # object's stable obj_idx — NOT sliced by position out of the shared
        # batched cond_frame_outputs tensor. That batched tensor is frozen at
        # whatever object count existed at THAT frame's own consolidation time,
        # which can be smaller than the tracker_state's current/final object
        # count (SAM3 keeps registering new objects into the same tracker_state
        # as detection/refinement continues, and never retroactively
        # re-consolidates older frames to match) — using a current-count-sized
        # id list to index that stale tensor both crashed (index out of
        # bounds) and would have silently mispaired ids with the wrong obj_ptr
        # vectors had it not crashed first. output_dict_per_obj has no such
        # problem: it always stores each object's own [1, ...] slice under its
        # own obj_idx, populated fresh whenever that frame is consolidated for
        # that object, regardless of how many other objects exist.
        obj_id_to_idx = tracker_state.get("obj_id_to_idx", {})
        output_dict_per_obj = tracker_state.get("output_dict_per_obj", {})
        for obj_id, obj_idx in obj_id_to_idx.items():
            obj_id = int(obj_id)
            cond_outputs = output_dict_per_obj.get(obj_idx, {}).get("cond_frame_outputs", {})
            for frame_t, out in cond_outputs.items():
                obj_ptr = out.get("obj_ptr")
                obj_score = out.get("object_score_logits")
                if obj_ptr is None or obj_score is None:
                    continue
                obj_ptr = obj_ptr.detach().cpu().to(torch.float32).numpy()  # [1, 256]
                obj_score = obj_score.detach().cpu().to(torch.float32).numpy().reshape(-1)  # [1]
                if obj_score[0] <= 0:
                    continue  # object judged absent at this frame — skip
                by_obj_id.setdefault(obj_id, []).append((int(frame_t), obj_ptr[0]))

    if not by_obj_id:
        print("save_refinement_obj_ptrs: no present-object obj_ptr found in cond frames — nothing to save.")
        return 0

    # Group by user-assigned name (obj_id is not durable/meaningful); keep both the
    # per-instance mean and the raw (frame_idx, vector) pairs.
    by_name: dict = {}         # name -> list of per-instance mean obj_ptr vectors
    raw_by_name: dict = {}     # name -> list of (obj_id, frame_idx, vector)
    frames_by_name: dict = {}  # name -> total cond frames pooled across contributing instances
    skipped_obj_ids = []
    for obj_id, entries in by_obj_id.items():
        inst = concept.get_instance_by_sam3_id(obj_id)
        if inst is None or inst.deleted:
            skipped_obj_ids.append(obj_id)
            continue
        vectors = [v for _t, v in entries]
        instance_mean = np.mean(vectors, axis=0)
        by_name.setdefault(inst.user_name, []).append(instance_mean)
        frames_by_name[inst.user_name] = frames_by_name.get(inst.user_name, 0) + len(vectors)
        for frame_t, v in sorted(entries, key=lambda e: e[0]):
            raw_by_name.setdefault(inst.user_name, []).append((obj_id, frame_t, v))

    if skipped_obj_ids:
        print(f"save_refinement_obj_ptrs: skipping {len(skipped_obj_ids)} obj_id(s) with no "
              f"live (non-deleted) instance in this concept: {skipped_obj_ids}")

    if not by_name:
        print("save_refinement_obj_ptrs: no named, non-deleted instances to save.")
        return 0

    names_sorted = sorted(by_name.keys())

    # Raw vectors are ragged per name (different cond-frame counts), so they're
    # stored flat with parallel index arrays rather than a jagged/object array
    # (keeps the file plain float/int dtypes, no allow_pickle needed to read it back).
    raw_names, raw_obj_ids, raw_frame_idx, raw_vectors = [], [], [], []
    for name in names_sorted:
        for obj_id, frame_t, v in raw_by_name[name]:
            raw_names.append(name)
            raw_obj_ids.append(obj_id)
            raw_frame_idx.append(frame_t)
            raw_vectors.append(v)

    save_dict = {
        "names": np.array(names_sorted),
        "num_instances": np.array([len(by_name[n]) for n in names_sorted], dtype=np.int64),
        "num_frames": np.array([frames_by_name[n] for n in names_sorted], dtype=np.int64),
        "obj_ptr": np.stack([np.mean(by_name[n], axis=0) for n in names_sorted], axis=0),
        "raw_names": np.array(raw_names),
        "raw_obj_id": np.array(raw_obj_ids, dtype=np.int64),
        "raw_frame_idx": np.array(raw_frame_idx, dtype=np.int64),
        "raw_obj_ptr": np.stack(raw_vectors, axis=0),
    }

    fname = os.path.join(concept_dir, "obj_ptr_priors.npz")
    np.savez_compressed(fname, **save_dict)

    dup_names = [n for n in names_sorted if len(by_name[n]) > 1]
    if dup_names:
        print(f"save_refinement_obj_ptrs: averaged {len(dup_names)} duplicate-named "
              f"instance group(s) together in 'obj_ptr' (raw vectors kept separate "
              f"in 'raw_obj_ptr'): {dup_names}")
    print(f"Saved obj_ptr for {len(names_sorted)} named instance(s) "
          f"({len(raw_vectors)} raw cond-frame vector(s), plus mean per name) → {fname}")
    return len(names_sorted)


def restore_cond_frame_states(sam3_model, session_id: str, concept_dir: str, device: str) -> int:
    """
    Inject saved cond frame states into a fresh session's tracker states.

    Call AFTER add_prompt() (text/box detection creates tracker states) but BEFORE
    propagate_in_video().  Overwrites the freshly-computed cond frame maskmem with
    the saved round-1 values, giving the tracker the prior session's anchors.

    Matching: the saved total object count must equal the current session's total
    object count (same text prompt on same video → deterministic detection order →
    same total).  The combined batch is then split back across the current
    session's ranks using "rank_sizes" from the index, in order.  Mismatched totals
    are skipped with a warning.

    Returns number of cond frame files injected.
    """
    cond_states_dir = os.path.join(concept_dir, "cond_states")
    index_path = os.path.join(cond_states_dir, "index.json")

    if not os.path.exists(index_path):
        print(f"restore_cond_frame_states: no saved states at {cond_states_dir}, skipping.")
        return 0

    with open(index_path) as f:
        index = json.load(f)

    inference_state = sam3_model._all_inference_states[session_id]["state"]
    tracker_states = inference_state.get("tracker_inference_states", [])

    if not tracker_states:
        print("restore_cond_frame_states: no tracker states in session. Call add_prompt() first.")
        return 0

    # Load maskmem_pos_enc (model constant, same values for all frames and objects)
    maskmem_pos_enc = None
    pos_enc_path = os.path.join(cond_states_dir, "maskmem_pos_enc.pt")
    if os.path.exists(pos_enc_path):
        maskmem_pos_enc = torch.load(pos_enc_path, map_location=device, weights_only=False)

    cur_rank_sizes = [len(ts.get("obj_ids", [])) for ts in tracker_states]
    total_cur = sum(cur_rank_sizes)
    saved_frames = index.get("frames", [])
    offload_to_cpu = inference_state.get("offload_state_to_cpu", False)

    restored_count = 0

    for frame_t in saved_frames:
        fpath = os.path.join(cond_states_dir, f"frame{frame_t:06d}.npz")
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}")
            continue

        data = np.load(fpath, allow_pickle=False)
        b = int(data["obj_ids"].shape[0]) if "obj_ids" in data else total_cur
        if b != total_cur:
            print(f"  restore frame {frame_t}: obj count mismatch "
                  f"(saved={b}, current={total_cur}), skipping.")
            continue

        def _load(key, dtype):
            if key not in data:
                return None
            t = torch.from_numpy(data[key]).to(dtype)
            return t.cpu() if (offload_to_cpu and key == "maskmem_features") else t.to(device)

        iou_score = _load("iou_score", torch.float32)
        if iou_score is None:
            # Not saved in older format; default to zeros (shape [B, 1])
            iou_score = torch.zeros((b, 1), dtype=torch.float32, device=device)
        out = {
            "maskmem_features": _load("maskmem_features", torch.bfloat16),
            "obj_ptr":          _load("obj_ptr",          torch.float32),
            "pred_masks":       _load("pred_masks",       torch.bfloat16),
            "object_score_logits": _load("object_score_logits", torch.float32),
            "iou_score":        iou_score,
            "maskmem_pos_enc":  maskmem_pos_enc,
        }

        # Split the combined batch back across the current session's ranks.
        offset = 0
        for rank, tracker_state in enumerate(tracker_states):
            n = cur_rank_sizes[rank]
            if n == 0:
                continue
            combined_cond = tracker_state["output_dict"]["cond_frame_outputs"]
            consolidated_inds = tracker_state["consolidated_frame_inds"]["cond_frame_outputs"]
            cond_per_obj = tracker_state.get("output_dict_per_obj", {})

            rank_out = {}
            for key, val in out.items():
                if val is not None and torch.is_tensor(val) and val.shape[0] == b:
                    rank_out[key] = val[offset : offset + n]
                else:
                    rank_out[key] = val
            combined_cond[frame_t] = rank_out
            consolidated_inds.add(frame_t)

            # Update per-object slices (each B=1 view into the rank's batch)
            for obj_idx in range(n):
                if obj_idx not in cond_per_obj:
                    continue
                obj_out = {"maskmem_pos_enc": maskmem_pos_enc}
                for key in ("maskmem_features", "obj_ptr", "pred_masks", "object_score_logits", "iou_score"):
                    val = rank_out.get(key)
                    if val is not None and torch.is_tensor(val) and val.shape[0] == n:
                        obj_out[key] = val[obj_idx : obj_idx + 1]
                    else:
                        obj_out[key] = val
                cond_per_obj[obj_idx]["cond_frame_outputs"][frame_t] = obj_out

            offset += n

        restored_count += 1

    print(f"Restored {restored_count} cond frame states from {cond_states_dir}")
    return restored_count


def process_concept_detection(
    sam3_model,
    project: SAM3Project,
    concept: SAM3Concept,
    device: str = "cuda:0",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    keep_session_alive: bool = False,
    save_cond_states: bool = False,
    cache_size: int = 50,
) -> SAM3Concept:
    """
    Run initial detection for a concept.

    Steps:
    1. Enable lazy loading (reuse sam_lazy_loader.py)
    2. Initialize inference state with CPU offloading (OOM mitigation)
    3. Add text prompt on detection frame
    4. Propagate (propagation_full = detector on every frame)
    5. Stream masks to disk incrementally (don't accumulate in memory)
    6. Save serializable inference state
    7. Create SAM3Instance objects with metadata
    8. Clear memory before next concept

    Args:
        sam3_model: SAM3 model instance
        project: SAM3 project
        concept: Concept to process
        device: Device to use
        progress_callback: Optional callback(frame_idx, num_frames)
        keep_session_alive: If True, do NOT close the session after processing.
            Caller is responsible for closing it. On success, session_id is stored
            in concept._live_session_id for retrieval.
        save_cond_states: If True, save maskmem cond frame states to disk for a
            future --restore-cond refinement round. Off by default since these
            states are only consumed by that opt-in path and can be sizeable.

    concept.max_instances (set by sam3_process.py, from --max-instances or the JSON
    "max_instances" field) caps SAM3's model.max_num_objects for the duration of this
    call only, then restores it. This governs THIS initial-detection propagation and
    nothing else — a later refinement round (replay_concept_refinements) never reads
    concept.max_instances and always replays every instance the user has kept, even if
    that exceeds this cap. See _cap handling below for what happens when detection would
    otherwise exceed the cap.

    Returns:
        Updated concept with instances populated
    """
    from sam_lazy_loader import enable_lazy_loading

    # 1. Enable lazy loading
    print("Enabling lazy loading for SAM3...")
    enable_lazy_loading(cache_size=cache_size, enable_sam3=True)

    # 2. Extract frames (reuse if already complete; resume if a prior extraction was killed)
    frames_dir = ensure_frames_extracted(
        project.video_path, project.get_frames_dir(), project.num_frames)

    # 3. Start SAM3 session (SAM3 uses session-based API)
    print(f"Starting SAM3 session for concept '{concept.name}'...")
    concept.status = ConceptStatus.PROCESSING

    # Create unique session ID for this concept
    import uuid
    session_id = str(uuid.uuid4())

    # Apply per-concept instance cap. SAM3 stores max_num_objects on the model object;
    # -1 in our convention means no limit, which SAM3 represents internally as 10000.
    _inner_model = getattr(sam3_model, "model", sam3_model)
    _prev_max_num_objects = getattr(_inner_model, "max_num_objects", 10000)
    _cap = getattr(concept, "max_instances", -1)
    _inner_model.max_num_objects = _cap if _cap > 0 else 10000
    if _cap > 0:
        print(f"Instance cap for '{concept.name}': {_cap}")

    # Encode/write masks in a background thread so the GPU propagation loop
    # is not stalled on PNG/NPZ encoding and disk I/O each frame.
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=project.mask_format)

    # start_session must be INSIDE the try: a failure here (bad frames dir, OOM
    # allocating state) must still set ConceptStatus.ERROR and restore
    # max_num_objects in the finally, or the concept is left stuck at PROCESSING
    # and the instance cap silently leaks to every subsequent concept.
    try:
        # offload_state_to_cpu: moves maskmem_features to CPU after each frame so GPU
        # doesn't accumulate the full video's worth of 648 KB/frame/instance tensors.
        sam3_model.start_session(
            resource_path=frames_dir,
            session_id=session_id,
            offload_state_to_cpu=True,
        )
        print(f"Session started: {session_id}")

        # 4. Add text prompt.
        # The frame_idx here is just for showing initial results on that frame.
        # Text prompts apply to all frames, and propagation will scan the entire video.
        print(f"Adding text prompt: '{concept.text_prompt}' (initial results on frame {concept.detection_frame})")
        sam3_model.add_prompt(
            session_id=session_id,
            frame_idx=concept.detection_frame,
            text=concept.text_prompt
        )

        # 5. Propagate and stream masks to disk
        print("Propagating masks through video...")
        instance_masks = defaultdict(dict)  # instance_id -> {frame_idx -> mask_path}
        pixel_counts_by_obj: dict = defaultdict(dict)  # obj_id -> {frame_idx -> pixel_count}
        frame_count = 0
        mask_count = 0

        # Hoist per-loop constants: concept dir and total frame size for pixel-ratio.
        concept_dir = project.get_concept_dir(concept.name)
        _frame_w, _frame_h = project.frame_dimensions
        total_pixels = _frame_w * _frame_h
        output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)

        # Inner inference_state for cache eviction.
        # SAM3 accumulates full-resolution bool masks in cached_frame_outputs for every
        # yielded frame — ~2MB/obj/frame at 1080p.  Once masks are saved to disk the
        # entry is no longer needed (forward-only propagation never looks backwards).
        inner_state = sam3_model._all_inference_states[session_id]["state"]
        _keep = _non_cond_keep_window(sam3_model)

        # Scan entire video forward from frame 0.
        # "both" with start_frame_idx=0 was a no-op for backward (range(0,-1,-1)=[0]),
        # so "forward" is correct and equivalent while being explicit.
        # Future improvement: keep the session alive across UI interactions so users
        # can add correction clicks on multiple frames and re-propagate once with all
        # cond-frame anchors — SAM3 attends to all cond frames simultaneously (no cap
        # with max_cond_frames_in_attn=-1), so multi-frame annotation before a single
        # propagate is both supported and more efficient than repeated single-frame runs.
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=None
        ):
            frame_idx = out["frame_index"]
            outputs = out["outputs"]
            frame_count += 1

            # Skip frames with no outputs
            if outputs is None:
                # Still evict from cache even when output is empty
                inner_state["cached_frame_outputs"].pop(frame_idx, None)
                continue

            # SAM3 outputs: out_obj_ids (array), out_binary_masks (array)
            # These are parallel arrays, not a dictionary
            out_obj_ids = outputs.get("out_obj_ids", [])
            out_binary_masks = outputs.get("out_binary_masks", [])

            # Debug: show what we got on first few frames
            if frame_count <= 3:
                print(f"  Frame {frame_idx}: {len(out_obj_ids)} objects - {list(out_obj_ids)}")

            for i, (obj_id, mask) in enumerate(zip(out_obj_ids, out_binary_masks)):
                mask_count += 1
                # Convert obj_id to Python int
                obj_id_int = int(obj_id)

                # Convert mask to numpy if it's a tensor
                # (bool→uint8 conversion happens in the writer thread)
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask

                # Cache output_dir per unique obj_id (os.path.join called once per object,
                # not once per object per frame).
                if obj_id_int not in output_dirs:
                    output_dirs[obj_id_int] = os.path.join(
                        concept_dir, "instances", str(obj_id_int), "masks"
                    )
                output_dir = output_dirs[obj_id_int]

                mask_path = mask_writer.submit(mask_np, output_dir, frame_idx)
                instance_masks[obj_id_int][frame_idx] = mask_path
                pixel_counts_by_obj[obj_id_int][frame_idx] = int((mask_np > 0).sum())

            # Evict this frame's full-res masks from SAM3's internal cache now that
            # they're on disk — forward propagation never needs to revisit past frames.
            inner_state["cached_frame_outputs"].pop(frame_idx, None)

            # Prune stale non_cond_frame_outputs inside each tracker state.
            # (SAM3 stores maskmem tensors per-GPU-rank, not at top-level inner_state.)
            _prune_non_cond_outputs(inner_state, frame_idx, keep=_keep)
            _offload_unselected_cond_frames(inner_state, frame_idx, sam3_model.model.tracker)

            # Clear GPU cache periodically
            if frame_idx % 100 == 0:
                torch.cuda.empty_cache()

            # Progress callback
            if progress_callback:
                progress_callback(frame_idx, project.num_frames)

        # Drain the writer before reading mask dirs below (compute_continuous_periods);
        # re-raises any write error from the background thread.
        mask_writer.close()

        print(f"\nPropagation summary: {frame_count} frames, {mask_count} masks total")

        # 6. Save cond frame maskmem states for future refinement rounds (must be before close_session)
        if save_cond_states:
            print("Saving cond frame states for future refinement...")
            concept_dir = project.get_concept_dir(concept.name)
            save_cond_frame_states(sam3_model, session_id, concept_dir)

        # 7. Get inference state for serialization
        print("Saving inference state...")
        inference_state = sam3_model._all_inference_states[session_id]["state"]
        inference_state["video_path"] = project.video_path
        save_concept_state(concept, inference_state, project.project_dir,
                           project.get_frames_dir())

        # 7. Create SAM3Instance objects with distinct colors
        print(f"Creating instance metadata ({len(instance_masks)} instances detected)...")
        instances = []
        num_instances = len(instance_masks)

        from sam3_utils import generate_instance_color

        for idx, obj_id in enumerate(sorted(instance_masks.keys())):
            # Convert numpy int to Python int
            obj_id_int = int(obj_id)

            # Score = fraction of video frames where this instance was detected
            score = len(instance_masks[obj_id]) / max(frame_count, 1)

            # Generate distinct color for this instance
            # Use full color spectrum to maximize visual distinction
            instance_color = generate_instance_color(idx, num_instances)

            mask_dir = os.path.join(
                project.get_concept_dir(concept.name),
                "instances", str(obj_id_int), "masks"
            )
            periods = compute_continuous_periods(mask_dir)
            _pcounts = pixel_counts_by_obj.get(obj_id_int, {})
            instance = SAM3Instance(
                sam3_obj_id=obj_id_int,
                user_name=f"{concept.name}_{obj_id_int}",  # Default name
                concept_name=concept.name,
                color_rgb=instance_color,  # Assign distinct color
                score=score,
                first_detection_frame=min(instance_masks[obj_id].keys()),
                last_detection_frame=max(instance_masks[obj_id].keys()),
                num_frames_with_mask=len(instance_masks[obj_id]),
                continuous_periods=periods,
                period_peaks=compute_period_peaks(periods, _pcounts, total_pixels),
                presence_norm=compute_presence_norm(_pcounts, total_pixels),
            )
            instances.append(instance)

        concept.instances = instances
        # This is a fresh detection pass (first-ever run, or a user-requested reset +
        # re-detect), so the id high-water mark is reset to exactly what was just found —
        # any manually-added instances from a prior round are gone along with `instances`.
        concept.highest_obj_id = max((inst.sam3_obj_id for inst in instances), default=-1)
        concept.status = ConceptStatus.COMPLETED
        concept.completed_at = datetime.now().isoformat()

        print(f"Concept '{concept.name}' processing complete: {len(instances)} instances detected")

    except Exception as e:
        concept.status = ConceptStatus.ERROR
        print(f"Error processing concept '{concept.name}': {e}")
        raise

    finally:
        # Stop the writer thread if propagation failed mid-loop (idempotent;
        # errors already surfaced via the close() call in the happy path).
        try:
            mask_writer.close()
        except Exception:
            pass
        # 8. Close session and clear memory
        if keep_session_alive and concept.status == ConceptStatus.COMPLETED:
            # Caller owns this session; store session_id for retrieval
            concept._live_session_id = session_id
        else:
            try:
                sam3_model.close_session(session_id=session_id)
            except Exception:
                pass
        _inner_model.max_num_objects = _prev_max_num_objects
        torch.cuda.empty_cache()

    return concept


def _add_box_refinement(sam3_model, session_id, frame_idx, obj_id, boxes_xywh_rel):
    """Add box refinement(s) to an EXISTING tracked object.

    SAM3's top-level add_prompt(bounding_boxes=...) is only safe when called
    WITHOUT obj_id: internally it always routes through
    Sam3VideoInferenceWithInstanceInteractivity.add_prompt's `else` branch
    (points is None), which calls the base class's add_prompt — a "semantic
    prompt" path that (a) calls reset_state() on the WHOLE session and (b)
    only accepts exactly one box, via _get_visual_prompt's "visual prompts
    should only have one box" assert. obj_id is silently dropped since the
    base method doesn't accept it. This is fine for an *initial* box-driven
    detection but wrong for per-object refinement replay, where obj_id must
    be honored and existing tracker state must survive.

    Instead we submit the box as two points with SAM2/SAM3's box-as-points
    convention (label 2 = top-left corner, label 3 = bottom-right corner),
    which flows through the working obj_id + points tracker-refinement path
    (add_tracker_new_points) — same path already used for point refinements.
    One add_prompt call per box, since each call treats its two corner
    points as a single box anchor.
    """
    for x, y, w, h in boxes_xywh_rel:
        sam3_model.add_prompt(
            session_id=session_id, frame_idx=frame_idx, obj_id=obj_id,
            points=[[x, y], [x + w, y + h]], point_labels=[2, 3],
        )


def _clear_stale_mask(output_dir: str, frame_idx: int) -> None:
    """Delete an instance's on-disk mask file for one frame, if present (checks both
    supported extensions).

    Used instead of overwriting with an explicit all-zero mask when a frame that
    previously had real content should no longer show this instance. compute_continuous_periods()
    only checks file *existence*, not pixel content, so a zero-content file gets counted
    as "present" and silently merged into a neighboring real period — inflating the
    presence bar with frames where the instance is actually absent. Deleting instead
    makes the gap a real gap, matching every other consumer in this codebase
    (DynamicFrameCompositor, the SAM2-covered-id mask union, the gaze-alignment
    scripts) — they all already treat a missing mask file as "this instance
    contributes nothing here", identical to how they'd treat an all-zero one.
    Mask anchors live in a separate mask_anchors/ directory and are never touched here.
    """
    for ext in (".png", ".npz"):
        p = os.path.join(output_dir, f"{frame_idx:06d}{ext}")
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def replay_concept_refinements(
    concept: SAM3Concept,
    instances_with_pending: List[Tuple["SAM3Instance", List[dict]]],
    resource_path: str,
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    device: str = "cuda:0",
    restore_cond_states: bool = False,
    save_cond_states: bool = False,
    save_obj_ptr_prior: bool = False,
    mask_format: str = "png",
    cache_size: int = 50,
    preload_original_masks: bool = False,
    new_instance_policy: str = "disallow",
) -> bool:
    """
    Replay saved refinements for ALL instances of a concept in ONE session.

    All instances are tracked simultaneously so SAM3's non-overlapping constraints
    are applied across the full concept — the same way initial detection works.
    Running each instance in its own isolated session would break non-overlapping.

    After propagation, masks for EVERY detected instance are written to disk (not
    just the ones with new annotations), and saved cond states are updated so the
    next refinement round can restore the latest anchors.

    By default, the only conditioning frames added for an instance are ones the
    user actually created (points, boxes, mask anchors) — an instance with no such
    annotation this round gets no extra anchor beyond whatever the fresh text/box
    redetection itself gave it. It IS still force-included in this round's joint
    Tracker propagation (see the untouched_live_ids block below) so the
    non-overlapping guarantee above actually holds for it — only the *reinforcement*
    is opt-in, not participation in propagation. Set preload_original_masks=True to
    additionally reinforce every live instance from
    its own on-disk mask (from a *previous* round) at its first detection frame —
    see _preload_original_masks(). That reinforcement is invisible to the user (not
    something they placed) and, since it's keyed by id against whatever the fresh
    text redetection happens to find, can silently bind stale content to the wrong
    sub-state if that id assignment doesn't line up with history — hence off by
    default.

    Args:
        concept: The concept whose instances are being refined.
        instances_with_pending: List of (instance, pending_entries) — every instance
            that has at least one entry with "propagated": false.  This acts as the
            TRIGGER only; the replay itself re-adds ALL saved annotations (historical
            and pending) of every non-deleted instance in the concept, so earlier
            rounds' corrections stay anchored without needing --restore-cond.
        resource_path: Path to extracted frames directory.
        orig_width / orig_height: Video dimensions in pixels.
        num_frames: Total frame count.
        sam3_model: Loaded SAM3 model.
        project_dir: Project root directory.
        progress_callback: Optional callback(frame_idx, num_frames).
        device: CUDA device string for tensor placement.
        preload_original_masks: Opt-in reinforcement from previous-round on-disk
            masks (see above). Default False.
        new_instance_policy: What to do with an object first observed (nonzero mask)
            during propagation that matches none of the sub-states present right
            before propagate_in_video() started (i.e. a genuinely new detection, not
            a live/deleted/absorbed instance under a coincidentally-matching id):
              "allow": persist its masks under a freshly re-indexed id
                  (concept.highest_obj_id + 1, ...) and promote it to a real
                  SAM3Instance after propagation. Note that user-deleted instances are
                  removed from the session first, so they reappear here as "new".
              "latent": leave its sub-state alone (it keeps existing/competing for
                  the rest of this run) but never persist or promote it.
              "disallow" (default): cap SAM3's max_num_objects so it can never be
                  created. Refinement never adds instances beyond what was annotated.

    Returns True if propagation ran; False if no pending entries found.
    """
    import uuid

    if not instances_with_pending:
        return False

    from sam_lazy_loader import enable_lazy_loading
    enable_lazy_loading(cache_size=cache_size, enable_sam3=True)

    # Load refinements.json for EVERY non-deleted instance of the concept — not just
    # the ones with pending entries.  All historical clicks are replayed each round so
    # corrections from earlier rounds stay anchored in the fresh session (without this,
    # a round-2 propagation would silently undo round-1 fixes unless --restore-cond
    # was used).  Pending entries only act as the trigger; the replay set is everything.
    refinements_by_instance: dict = {}  # obj_id -> (path, all_entries, inst)
    for inst in concept.instances:
        if inst.deleted:
            continue
        rpath = os.path.join(
            project_dir, "concepts", concept.name,
            "instances", str(inst.sam3_obj_id), "refinements.json"
        )
        if not os.path.exists(rpath):
            continue
        with open(rpath) as f:
            all_entries = json.load(f).get("refinements", [])
        if all_entries:
            refinements_by_instance[inst.sam3_obj_id] = (rpath, all_entries, inst)

    # Build (obj_id, frame_idx) -> [points] from ALL entries in file order with
    # last-entry-wins per frame: each UI save writes a frame's complete point set as
    # one entry, so the newest entry supersedes older ones for that frame (and an
    # entry with no points clears the frame).  Merging entries instead would
    # resurrect points the user deleted in the UI.
    # "mask_anchor" entries are skipped here — they carry no point prompts and
    # must not clear other annotations at the same frame.  Their effect is applied
    # via _inject_mask_anchors() which reads directly from mask_anchor_frames.
    obj_frame_points: dict = defaultdict(dict)   # obj_id -> {frame_idx -> [points]}
    obj_frame_boxes: dict = defaultdict(dict)    # obj_id -> {frame_idx -> [boxes xywh rel]}
    for obj_id, (rpath, all_entries, inst) in refinements_by_instance.items():
        for r in all_entries:
            if r.get("type") == "mask_anchor":
                continue
            frame_idx = r.get("frame_idx")
            if frame_idx is None:
                continue
            pts = [(p["x"], p["y"], p["is_positive"]) for p in r.get("points", [])]
            if pts:
                obj_frame_points[obj_id][frame_idx] = pts
            else:
                obj_frame_points[obj_id].pop(frame_idx, None)
            # Boxes stored as pixel coords; convert to xywh relative on load.
            raw_boxes = r.get("boxes", [])
            boxes_xywh = [(b["x1"] / orig_width,
                           b["y1"] / orig_height,
                           (b["x2"] - b["x1"]) / orig_width,
                           (b["y2"] - b["y1"]) / orig_height)
                          for b in raw_boxes]
            if boxes_xywh:
                obj_frame_boxes[obj_id][frame_idx] = boxes_xywh
            else:
                obj_frame_boxes[obj_id].pop(frame_idx, None)

    n_pending = sum(len(p) for _, p in instances_with_pending)
    total_frames_replayed = sum(len(v) for v in obj_frame_points.values())
    print(f"Replaying ALL saved annotations for concept '{concept.name}': "
          f"{total_frames_replayed} annotated frame(s) across "
          f"{len(obj_frame_points)} instance(s) "
          f"({n_pending} new pending entr{'y' if n_pending == 1 else 'ies'}) "
          f"in a single joint session.")

    session_id = str(uuid.uuid4())
    sam3_model.start_session(
        resource_path=resource_path,
        session_id=session_id,
        offload_state_to_cpu=True,
    )

    try:
        # Re-run text detection anchored to the same frame used during initial detection.
        # frame_idx only determines where initial results are displayed; the prompt
        # applies to all frames.  Using concept.detection_frame (not hard-coded 0) keeps
        # the API call consistent with process_concept_detection, which matters if the
        # user explicitly set a non-zero detection_frame in their concepts JSON.
        sam3_model.add_prompt(
            session_id=session_id,
            frame_idx=concept.detection_frame,
            text=concept.text_prompt,
        )

        live_obj_ids = {inst.sam3_obj_id for inst in concept.instances if not inst.deleted}

        # Pre-initialise LIVE instances from their previous-round on-disk masks — opt-in
        # only (see preload_original_masks docstring above). Off by default: an instance
        # with no annotation this round and no fresh redetection under its own id simply
        # isn't touched, rather than being silently reinforced from stale prior-round data.
        if preload_original_masks:
            _preload_original_masks(
                sam3_model, session_id, concept, project_dir, orig_width, orig_height
            )

        # Three categories of non-live obj_ids all get remove_object()'d before propagation,
        # so no dead sub-state is ever kept alive purely to "block" its old region — that job
        # is now the new_instance_policy's, applied uniformly to anything that reappears there:
        #
        # (a) ABSORBED sources — instances with deleted=True AND absorbed_source=True.
        #     The user said "this duplicate is the same object as another one I'm keeping."
        #     Their pixels are freed for the absorbing target to claim via points/anchors.
        #
        # (b) DELETED instances — deleted=True, absorbed_source=False (user said "I don't
        #     want this object"). Previously these were kept alive deliberately to block
        #     re-detection via non-overlapping constraints; now removed like absorbed
        #     sources, since new_instance_policy governs whether anything reappearing in
        #     that region is allowed, latent, or disallowed.
        #
        # (c) Phantom obj_ids — completely absent from concept.instances (not in live,
        #     deleted, or absorbed sets). Also call remove_object() for these.
        absorbed_source_ids = {inst.sam3_obj_id for inst in concept.instances
                               if inst.deleted and getattr(inst, 'absorbed_source', False)}
        deleted_obj_ids = {inst.sam3_obj_id for inst in concept.instances
                           if inst.deleted and not getattr(inst, 'absorbed_source', False)}
        inner_state = sam3_model._all_inference_states[session_id]["state"]
        session_obj_ids = set(
            int(x) for x in inner_state.get("tracker_metadata", {}).get("obj_ids_all_gpu", [])
        )
        phantom_obj_ids = session_obj_ids - live_obj_ids - deleted_obj_ids - absorbed_source_ids
        to_remove_ids = deleted_obj_ids | absorbed_source_ids | phantom_obj_ids
        for rem_id in to_remove_ids:
            try:
                sam3_model.remove_object(session_id=session_id, obj_id=rem_id)
                if rem_id in absorbed_source_ids:
                    label = "absorbed source"
                elif rem_id in deleted_obj_ids:
                    label = "deleted"
                else:
                    label = "phantom"
                print(f"  Removed {label} obj_id={rem_id} from session (pixels freed).")
            except Exception as e:
                print(f"  Warning: could not remove obj_id={rem_id}: {e}")

        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        if restore_cond_states:
            n = restore_cond_frame_states(sam3_model, session_id, concept_dir, device)
            if n > 0:
                print(f"  Restored {n} cond frame(s) from prior round.")
        else:
            print("  Skipping cond state restore (default); propagation uses text detection + correction points only.")

        # Raw per-frame backbone features (image + FPN) live in feature_cache, keyed by
        # frame_idx. run_backbone_and_detection() only evicts the *adjacent* frame_idx
        # (sam3_video_base.py), which assumes sequential propagation order. The replay
        # loop below visits frames out of order (whatever frames were actually annotated,
        # globally descending — not consecutive integers), so that eviction never matches
        # and entries would otherwise pile up: one full backbone feature map per annotated
        # frame, for the entire replay set, all resident at once. This used to be tolerated
        # as "bounded by the replay set size," but for concepts with hundreds of annotated
        # frames (e.g. 177 for one project's 'body' concept) that bound alone was enough to
        # OOM. _consolidate_and_evict_feature_cache (called in batches inside the loop
        # below) fixes this properly: it forces early, incremental
        # propagate_in_video_preflight passes so each batch's frames get consolidated into
        # output_dict — no longer dependent on feature_cache — and are evicted immediately,
        # instead of deferring everything to the single automatic preflight call inside
        # propagate_in_video() at the end.
        inner_state = sam3_model._all_inference_states[session_id]["state"]
        _keep = _non_cond_keep_window(sam3_model)

        # Initialize manually-added instances (never text-detected) via ALL their pending
        # point sets, and add refinement prompts for text-detected instances on each
        # annotated frame.  The first add_prompt(obj_id=X, points=...) call for a
        # manually-added obj_id creates its tracker slot; subsequent calls add further
        # cond anchors — same add_prompt() call either way, so both categories are
        # processed together below.
        #
        # Ordering is NOT per-object-then-per-frame here: it must be a single GLOBAL pass
        # over frame_idx in DESCENDING order across ALL objects AND across both prompt
        # types (point/box conditioning AND mask-anchor injection). Reason: SAM3's
        # run_backbone_and_detection() unconditionally evicts feature_cache[frame_idx - 1]
        # every time it computes a new frame's features (sam3_video_base.py) — reached by
        # add_prompt(points=...) via add_tracker_new_points's own _prepare_backbone_feats
        # call, and by _inject_mask_anchor_at_frame's explicit _prepare_backbone_feats call
        # before add_new_mask, so both prompt types are subject to the identical eviction
        # rule. The tracker-only submodule used by propagate_in_video_preflight has no
        # backbone of its own to recompute a cache miss (_get_image_feature raises "Image
        # features for frame N are not cached"). Preflight consolidates every cond frame
        # across every object at the end of this function, so ALL of them must still be
        # cached then. Interleaving objects/prompt-types in arbitrary order, or running
        # mask-anchor injection as a separate pass after point/box prompts, could evict an
        # earlier-processed frame before preflight reads it (e.g. a point at frame 1 gets
        # computed, then a mask anchor at frame 2 pops frame 2-1=1). Visiting frames
        # highest-to-lowest across BOTH categories together guarantees each processed
        # frame's features, once computed, are never evicted by a later (lower) frame in
        # this pass.
        manually_added_ids = {inst.sam3_obj_id for inst in concept.instances
                              if getattr(inst, 'manually_added', False) and not inst.deleted}
        all_annotated_ids = set(obj_frame_points) | set(obj_frame_boxes)
        # frame_idx -> [("point_box", obj_id), ...] | [("mask_anchor", inst), ...]
        frame_tasks: dict = defaultdict(list)
        for obj_id in all_annotated_ids:
            frames = set(obj_frame_points.get(obj_id, {})) | set(obj_frame_boxes.get(obj_id, {}))
            for frame_idx in frames:
                frame_tasks[frame_idx].append(("point_box", obj_id))
        for inst in concept.instances:
            if inst.deleted or not inst.mask_anchor_frames:
                continue
            for frame_idx in inst.mask_anchor_frames:
                frame_tasks[frame_idx].append(("mask_anchor", inst))

        # Batch size for _consolidate_and_evict_feature_cache below: how many annotated
        # frames' worth of raw backbone features (image + FPN, much larger per-entry than
        # a cond frame's maskmem_features) are allowed to pile up in feature_cache before
        # we force an early consolidation + eviction. Smaller bounds memory tighter at the
        # cost of more propagate_in_video_preflight calls (each one a real, if small,
        # compute pass); this is an internal memory/overhead tradeoff, not something worth
        # exposing as a CLI flag yet.
        _FEATURE_CACHE_BATCH = 8
        _batch_frames: List[int] = []

        for frame_idx in sorted(frame_tasks, reverse=True):
            for kind, payload in frame_tasks[frame_idx]:
                if kind == "mask_anchor":
                    _inject_mask_anchor_at_frame(
                        sam3_model, session_id, concept, project_dir,
                        orig_width, orig_height, frame_idx, payload,
                    )
                    continue
                obj_id = payload
                pts = obj_frame_points.get(obj_id, {}).get(frame_idx, [])
                boxes = obj_frame_boxes.get(obj_id, {}).get(frame_idx, [])
                pts_normalized = [[x / orig_width, y / orig_height] for x, y, _ in pts]
                pt_labels = [1 if is_positive else 0 for _, _, is_positive in pts]
                # SAM3's add_prompt() rejects points and boxes in the same call
                # ("When points are provided, text_str and boxes_xywh must be
                # None.") — issue them as separate cond-anchor calls instead.
                if pts_normalized:
                    sam3_model.add_prompt(
                        session_id=session_id, frame_idx=frame_idx, obj_id=obj_id,
                        points=pts_normalized, point_labels=pt_labels,
                    )
                if boxes:
                    _add_box_refinement(sam3_model, session_id, frame_idx, obj_id, boxes)

            # This loop is itself the OOM source seen in practice: it calls add_prompt/
            # add_new_mask once per annotated frame (up to hundreds for a heavily-corrected
            # concept), and EACH call creates a new cond frame whose maskmem_features stay
            # GPU-resident with nothing evicting them until propagate_in_video starts —
            # unlike the propagation loop below, which already calls this every frame via
            # _prune_non_cond_outputs. Offload here too, using this loop's own frame_idx as
            # the reference point, so cond frames outside the max-cond-frames-in-attn window
            # around wherever we currently are in this descending pass get moved to CPU
            # immediately instead of accumulating for the remainder of the loop.
            _offload_unselected_cond_frames(inner_state, frame_idx, sam3_model.model.tracker)

            # Clear every frame — cheaper than the feature_cache batch below (no
            # propagate_in_video_preflight pass, just dropping dict entries) and each
            # individual add_prompt/add_new_mask call can strand up to 2 entries here
            # (see _clear_multigpu_buffer's docstring), so waiting a full batch would
            # still let ~16 stranded entries accumulate between flushes.
            _clear_multigpu_buffer(inner_state)

            # Bound feature_cache (raw per-frame backbone+FPN features) the same way,
            # batched rather than per-frame since each flush costs a real (if small)
            # propagate_in_video_preflight pass — see _consolidate_and_evict_feature_cache.
            _batch_frames.append(frame_idx)
            if len(_batch_frames) >= _FEATURE_CACHE_BATCH:
                _consolidate_and_evict_feature_cache(inner_state, sam3_model, _batch_frames)
                _batch_frames = []

        _consolidate_and_evict_feature_cache(inner_state, sam3_model, _batch_frames)

        for obj_id in manually_added_ids:
            if obj_id not in obj_frame_points and obj_id not in obj_frame_boxes:
                continue  # no pending annotations → nothing to initialize
            n_frames = len(obj_frame_points.get(obj_id, {}))
            print(f"  Initialized manually-added instance obj_id={obj_id} "
                  f"via {n_frames} annotated frame(s).")

        # Force every remaining live instance into this round's Tracker propagation,
        # even ones with zero points/boxes/mask-anchors this session (a "clean" instance
        # VG has always tracked correctly on its own, with nothing in its
        # refinements.json). Without this, parse_action_history_for_propagation()
        # (sam3_video_inference.py) decides propagation scope purely from
        # inference_state["action_history"]: it returns "propagation_full" (proper,
        # real propagation for every object) ONLY when history is completely empty; the
        # moment ANY object gets an add/refine action this round (which happens as soon
        # as a single instance has a pending correction), the WHOLE session downgrades to
        # "propagation_partial" scoped ONLY to objects with a logged action — every other
        # live instance, no matter how well it was tracking before, is silently excluded
        # from real Tracker propagation and only ever shows up on whatever cond frame(s)
        # it happens to already have (the same bug just fixed for mask-anchor-only
        # instances, but it applies equally to a plain text-detected instance with no
        # corrections at all). This directly contradicts this function's own stated
        # guarantee that "All instances of a concept must be refined together in ONE
        # session so SAM3's non-overlapping constraints are applied across all of them
        # jointly" — an excluded instance doesn't compete for pixels at all outside its
        # cond frames. Registering a no-op "refine" action per untouched live instance
        # costs nothing (add_action_history only appends bookkeeping, no tracker state
        # mutation) and guarantees full joint propagation regardless of how many other
        # instances have pending corrections this round.
        touched_ids = all_annotated_ids | {
            inst.sam3_obj_id for inst in concept.instances
            if not inst.deleted and inst.mask_anchor_frames
        }
        untouched_live_ids = live_obj_ids - touched_ids
        for obj_id in untouched_live_ids:
            sam3_model.model.add_action_history(
                inner_state, "refine", frame_idx=concept.detection_frame, obj_ids=[obj_id]
            )
        if untouched_live_ids:
            print(f"  Included {len(untouched_live_ids)} instance(s) with no annotation "
                  f"this round in joint propagation (no corrections to replay): "
                  f"{sorted(untouched_live_ids)}")

        # Single propagation: all instances in one batched forward pass.
        # Non-overlapping constraints are enforced across all instances jointly.
        # Only write masks for live instances; ghost IDs from deleted instances'
        # physical counterparts (IDs above max_obj_id) are silently suppressed.
        from sam3_utils import AsyncMaskWriter
        mask_writer = AsyncMaskWriter(mask_format=mask_format)
        pixel_counts_by_obj: dict = defaultdict(dict)
        total_pixels = orig_width * orig_height
        instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
        output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)

        # Record which frames already have mask data for each live instance, so we know
        # which stale files to delete (via _clear_stale_mask) when refinement produces no
        # mask at a frame this round (rather than leaving the old file in place).
        original_mask_frames: dict = {}
        for obj_id_int in live_obj_ids:
            mask_dir = os.path.join(instances_root, str(obj_id_int), "masks")
            if not os.path.isdir(mask_dir):
                continue
            frames = set()
            for fname in os.listdir(mask_dir):
                stem, ext = os.path.splitext(fname)
                if ext in (".npz", ".png"):
                    try:
                        frames.add(int(stem))
                    except ValueError:
                        pass
            if frames:
                original_mask_frames[obj_id_int] = frames

        # Real-time snapshot of every sub-state actually present right before propagation
        # starts (after initial detection, removal of deleted/absorbed/phantom ids, and all
        # manual/point/box/mask-anchor conditioning above). Any id NOT in this set that shows
        # up in propagate_in_video()'s output is unambiguously a brand-new detection — ids are
        # unique within a live session, so this can never collide with a live/deleted/absorbed
        # instance's id, unlike comparing against the historical (cross-round) live_obj_ids set.
        pre_propagation_ids = set(
            int(x) for x in inner_state.get("tracker_metadata", {}).get("obj_ids_all_gpu", [])
        )
        new_id_remap: dict = {}  # raw session id -> persistent id (assigned first time seen)

        # present_targets_this_frame (built per-frame below) drives stale on-disk mask
        # clearing: when a once-live instance's sub-state is absent from a frame's output
        # (e.g. hotstart removal mid-propagation, or simply predicted empty), any mask
        # file left over from a prior round at that frame is deleted so a ghost mask
        # doesn't linger and overlap whatever object legitimately claims the region now.
        #
        # frames_iterated tracks every frame_idx propagation actually walked through,
        # independent of which objects appear in that frame's output. A pending
        # refinement entry is "reached" once its frame is in this set — an object being
        # absent from out_obj_ids at a frame usually just means the correction succeeded
        # (e.g. a negative point drove its mask to zero, which SAM3 omits from
        # out_obj_ids entirely), not that the frame was never visited, so per-object
        # presence is the wrong signal to gate "propagated" on.
        frames_iterated: set = set()

        _inner_model = getattr(sam3_model, "model", sam3_model)
        _prev_max_num_objects = getattr(_inner_model, "max_num_objects", 10000)
        if new_instance_policy == "disallow":
            _inner_model.max_num_objects = len(pre_propagation_ids)
            print(f"  new_instance_policy=disallow: capped max_num_objects to "
                  f"{len(pre_propagation_ids)} (no new objects can be created).")

        try:
            for out in sam3_model.propagate_in_video(
                session_id=session_id,
                start_frame_idx=0,
                propagation_direction="forward",
            ):
                out_frame_idx = out["frame_index"]
                outputs = out["outputs"]
                if outputs is None:
                    inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
                    continue
                frames_iterated.add(out_frame_idx)

                out_obj_ids = outputs.get("out_obj_ids", [])
                out_binary_masks = outputs.get("out_binary_masks", [])

                present_targets_this_frame = set()
                for obj_id, mask in zip(out_obj_ids, out_binary_masks):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                        mask_np = mask_np[0]
                    obj_id_int = int(obj_id)

                    if obj_id_int not in pre_propagation_ids:
                        # Genuinely new detection — not any known sub-state.
                        if new_instance_policy != "allow":
                            continue  # "latent" (or a stray "disallow" leak): don't persist
                        if obj_id_int not in new_id_remap:
                            concept.highest_obj_id += 1
                            new_id_remap[obj_id_int] = concept.highest_obj_id
                            print(f"  New object detected mid-video (raw id={obj_id_int}) — "
                                  f"assigning persistent id={new_id_remap[obj_id_int]}.")
                        target_id = new_id_remap[obj_id_int]
                    elif obj_id_int not in live_obj_ids:
                        continue  # deleted/absorbed physical counterpart — suppress output
                    else:
                        target_id = obj_id_int

                    present_targets_this_frame.add(target_id)
                    if target_id not in output_dirs:
                        output_dirs[target_id] = os.path.join(
                            instances_root, str(target_id), "masks"
                        )
                    pixel_count = int((mask_np > 0).sum())
                    if pixel_count == 0:
                        # Clear any previous non-zero mask at this frame by deleting the
                        # file (see _clear_stale_mask); frames that never had data are
                        # skipped — nothing to clear.
                        if out_frame_idx in original_mask_frames.get(target_id, set()):
                            _clear_stale_mask(output_dirs[target_id], out_frame_idx)
                        continue
                    mask_writer.submit(mask_np, output_dirs[target_id], out_frame_idx)
                    pixel_counts_by_obj[target_id][out_frame_idx] = pixel_count

                # Any instance that HAD an active sub-state at the start of this round's
                # propagation (pre_propagation_ids) but is entirely absent from THIS frame's
                # output — not merely predicted empty, which is the pixel_count==0 branch
                # above, but genuinely not a candidate at all (e.g. removed by hotstart
                # mid-propagation) — still needs its stale mask cleared here, or a ghost mask
                # from an earlier round lingers forever. Restricted to pre_propagation_ids so
                # instances deliberately left untouched this round (e.g. manually-added with
                # no pending annotation) are never swept — they were never in the session to
                # begin with, so absence here says nothing about them.
                for obj_id_int in live_obj_ids & pre_propagation_ids:
                    if obj_id_int in present_targets_this_frame:
                        continue
                    if out_frame_idx in original_mask_frames.get(obj_id_int, set()):
                        if obj_id_int not in output_dirs:
                            output_dirs[obj_id_int] = os.path.join(
                                instances_root, str(obj_id_int), "masks"
                            )
                        _clear_stale_mask(output_dirs[obj_id_int], out_frame_idx)

                # Evict after saving — forward propagation never revisits past frames
                inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
                _prune_non_cond_outputs(inner_state, out_frame_idx, keep=_keep)
                _offload_unselected_cond_frames(inner_state, out_frame_idx, sam3_model.model.tracker)

                if progress_callback:
                    progress_callback(out_frame_idx, num_frames)
                if out_frame_idx % 100 == 0:
                    torch.cuda.empty_cache()
        finally:
            # Drain before the deleted-instance wipe below touches mask dirs;
            # re-raises any write error from the background thread.
            mask_writer.close()
            _inner_model.max_num_objects = _prev_max_num_objects

        # Wipe mask files for deleted instances — they were excluded from propagation so
        # their on-disk masks are now stale and should not appear in any future compositing.
        #
        # Two sub-cases:
        #   • User-deleted (absorbed_source=False): recreate empty dir so subsequent runs
        #     don't mistake an absent dir for "not yet detected".
        #   • Absorbed sources (absorbed_source=True): their original masks were kept until
        #     now for ghost rendering / union-check in the UI.  This refinement run has
        #     incorporated them into the target via mask anchors, so the originals are no
        #     longer needed.  remove_object() was called for these IDs above, so they
        #     produced NO new masks during propagation — deleting here is safe.
        import shutil
        for inst in concept.instances:
            if not inst.deleted:
                continue
            mask_dir = os.path.join(
                project_dir, "concepts", concept.name,
                "instances", str(inst.sam3_obj_id), "masks"
            )
            is_absorbed = getattr(inst, 'absorbed_source', False)
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir)
                if not is_absorbed:
                    os.makedirs(mask_dir, exist_ok=True)
            label = "absorbed source" if is_absorbed else "deleted"
            print(f"  Cleared masks for {label} instance obj_id={inst.sam3_obj_id}.")

        if new_instance_policy == "allow" and new_id_remap:
            _promote_new_instances(
                concept, set(new_id_remap.values()), pixel_counts_by_obj,
                instances_root, total_pixels, num_frames,
            )

        # Report instance directories on disk that weren't covered by this refinement run.
        # This happens when a prior run detected more objects than this run (e.g., the text
        # prompt missed them, or they weren't visible at the new prompt frame).
        # We do NOT auto-delete — stale masks may still be valid from a prior round.
        instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
        if os.path.isdir(instances_root):
            on_disk = set()
            for d in os.listdir(instances_root):
                try:
                    on_disk.add(int(d))
                except ValueError:
                    pass
            # Manually-added instances that had no pending points this run were never in
            # the session, so their existing masks are valid — don't flag them as stale.
            manually_added_no_pending = {
                inst.sam3_obj_id for inst in concept.instances
                if getattr(inst, 'manually_added', False)
                and not inst.deleted
                and inst.sam3_obj_id not in obj_frame_points
            }
            stale = on_disk - set(pixel_counts_by_obj.keys()) - deleted_obj_ids - absorbed_source_ids - manually_added_no_pending
            if stale:
                registered_ids = {inst.sam3_obj_id for inst in concept.instances}
                # Orphan dirs: numeric ID not referenced by any instance in project.json.
                # These are left-over from a prior detection run and are pure garbage.
                orphan_ids = stale - registered_ids
                # Registered but not re-tracked: real instances SAM3 didn't output this run.
                # Prior masks may still be valid — preserve them.
                skipped_ids = stale - orphan_ids
                if orphan_ids:
                    for oid in sorted(orphan_ids):
                        orphan_dir = os.path.join(instances_root, str(oid))
                        shutil.rmtree(orphan_dir, ignore_errors=True)
                    print(f"  Removed {len(orphan_ids)} orphan instance dir(s) not in "
                          f"project.json: {sorted(orphan_ids)}")
                if skipped_ids:
                    print(f"  NOTE: {len(skipped_ids)} instance dir(s) registered but not "
                          f"re-tracked by SAM3 this run: {sorted(skipped_ids)}")
                    print(f"  Their existing masks are preserved (SAM3 may not have detected "
                          f"them this round).")

        # Update cond states for next refinement round (multi-object tensor, all instances).
        if save_cond_states:
            save_cond_frame_states(sam3_model, session_id, concept_dir)
        if save_obj_ptr_prior:
            save_refinement_obj_ptrs(sam3_model, session_id, concept, concept_dir)

        # Recompute periods and frame-count metadata for every instance from actual mask files.
        # pixel_counts_by_obj has fresh counts for instances tracked in this run; stale instances
        # (not output by SAM3) still get their periods refreshed from disk, just with empty pixel
        # counts (so period_peaks may be incomplete for them — acceptable).
        for inst in concept.instances:
            if inst.deleted:
                continue
            mask_dir = os.path.join(
                instances_root, str(inst.sam3_obj_id), "masks"
            )
            periods = compute_continuous_periods(mask_dir)
            _pcounts = pixel_counts_by_obj.get(inst.sam3_obj_id, {})
            _pcounts = _backfill_stale_period_counts(periods, _pcounts, mask_dir)
            inst.continuous_periods = periods
            inst.period_peaks = compute_period_peaks(periods, _pcounts, total_pixels)
            inst.presence_norm = compute_presence_norm(_pcounts, total_pixels)
            inst.num_frames_with_mask = sum(e - s + 1 for s, e in periods)
            if periods:
                inst.first_detection_frame = periods[0][0]
                inst.last_detection_frame = periods[-1][1]

    finally:
        try:
            sam3_model.close_session(session_id=session_id)
        except Exception:
            pass

    # Mark pending entries as propagated for every frame this round's propagation actually
    # walked through (frames_iterated), regardless of whether the object appears in that
    # frame's output — absence from out_obj_ids usually just means the correction
    # succeeded (e.g. a negative point drove the mask to zero, which SAM3 omits from
    # out_obj_ids entirely), not that the frame was never reached. An entry only stays
    # pending if propagation never got as far as its frame at all (e.g. an error cut the
    # run short), in which case it's silently retried on the next --refine.
    stuck_entries = []  # (user_name, obj_id, frame_idx) left pending for a future retry
    for obj_id, (rpath, all_entries, inst) in refinements_by_instance.items():
        changed = False
        for r in all_entries:
            if r.get("propagated", True):
                continue
            frame_idx = r.get("frame_idx")
            if frame_idx in frames_iterated:
                r["propagated"] = True
                changed = True
            else:
                stuck_entries.append((inst.user_name, obj_id, frame_idx))
        if changed:
            with open(rpath, "w") as f:
                json.dump({"refinements": all_entries}, f, indent=2)

    if stuck_entries:
        print(f"  WARNING: {len(stuck_entries)} correction(s) left PENDING — the instance's "
              f"sub-state never reached that frame in this round's output (likely removed/"
              f"lost track earlier in propagation). Will retry on the next --refine: "
              f"{stuck_entries}")

    print(f"  Concept-level replay complete for '{concept.name}'.")
    return True


def online_replay_concept_refinements(
    concept: SAM3Concept,
    instances_with_pending: List[Tuple["SAM3Instance", List[dict]]],
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    session_id: str,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_cond_states: bool = False,
    save_obj_ptr_prior: bool = False,
    mask_format: str = "png",
    new_instance_policy: str = "disallow",
):
    """
    Apply pending refinements for multiple instances through a live session in one propagation pass.

    Unlike adding correction anchors and re-propagating per-instance (which would break
    non-overlapping constraints across instances), this function adds
    all correction anchors for all instances FIRST, then propagates ONCE.  SAM3's
    non-overlapping constraint is then enforced jointly — required for cases like swapping
    a region between two instances (positive on A + negative on B).

    Args:
        concept: The concept whose instances are being refined.
        instances_with_pending: List of (instance, [refinement_entry, ...]) where each
            entry is a dict with 'frame_idx' and 'points' ({'x', 'y', 'is_positive'}).
        orig_width / orig_height: Video frame dimensions for coordinate normalization.
        num_frames: Total frame count for progress reporting.
        sam3_model: SAM3 model (session must be alive).
        session_id: Live session ID.
        project_dir: Project root directory.
        progress_callback: Optional callback(frame_idx, num_frames).
    """
    if not instances_with_pending:
        return

    # Add correction anchors for ALL instances before propagating.
    # Each add_prompt(obj_id=X) call routes to add_tracker_new_points — no reset_state.
    for instance, pending_entries in instances_with_pending:
        for entry in pending_entries:
            if entry.get("type") == "mask_anchor":
                continue  # mask is injected via _inject_mask_anchors
            frame_idx = entry["frame_idx"]
            raw_pts = entry.get("points", [])
            raw_boxes = entry.get("boxes", [])
            if not raw_pts and not raw_boxes:
                continue
            print(f"Online refinement: adding {len(raw_pts)} pt(s) + {len(raw_boxes)} box(es) "
                  f"at frame {frame_idx} for obj {instance.sam3_obj_id}")
            # SAM3's add_prompt() rejects points and boxes in the same call
            # ("When points are provided, text_str and boxes_xywh must be
            # None.") — issue them as separate cond-anchor calls instead.
            if raw_pts:
                sam3_model.add_prompt(
                    session_id=session_id, frame_idx=frame_idx, obj_id=instance.sam3_obj_id,
                    points=torch.tensor(
                        [[p["x"] / orig_width, p["y"] / orig_height] for p in raw_pts],
                        dtype=torch.float32,
                    ),
                    point_labels=torch.tensor(
                        [1 if p["is_positive"] else 0 for p in raw_pts],
                        dtype=torch.int32,
                    ),
                )
            if raw_boxes:
                boxes_xywh_rel = [
                    (b["x1"] / orig_width, b["y1"] / orig_height,
                     (b["x2"] - b["x1"]) / orig_width,
                     (b["y2"] - b["y1"]) / orig_height)
                    for b in raw_boxes
                ]
                _add_box_refinement(
                    sam3_model, session_id, frame_idx, instance.sam3_obj_id, boxes_xywh_rel
                )

    # Inject mask-conditioning anchors (from absorb wizard) before propagation.
    _inject_mask_anchors(sam3_model, session_id, concept, project_dir, orig_width, orig_height)

    # Remove absorbed sources from the session so their pixels are freed for the absorbing
    # target.  Unlike truly-deleted instances (kept alive to block re-detection), absorbed
    # sources competing via non-overlapping constraints actively prevent the absorbing
    # instance from claiming the merged region.
    live_obj_ids = {inst.sam3_obj_id for inst in concept.instances if not inst.deleted}
    for inst in concept.instances:
        if inst.deleted and getattr(inst, 'absorbed_source', False):
            try:
                sam3_model.remove_object(session_id=session_id, obj_id=inst.sam3_obj_id)
                print(f"  Removed absorbed source obj_id={inst.sam3_obj_id} from live session.")
            except Exception as e:
                print(f"  Warning: could not remove absorbed source obj_id={inst.sam3_obj_id}: {e}")

    # Single propagation pass — non-overlapping applied across ALL instances simultaneously
    inst_names = [inst.user_name for inst, _ in instances_with_pending]
    print(f"Re-propagating with corrections for {len(instances_with_pending)} instance(s): {inst_names}")
    _keep = _non_cond_keep_window(sam3_model)
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=mask_format)
    pixel_counts_by_obj: dict = defaultdict(dict)
    total_pixels = orig_width * orig_height
    instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
    output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)

    # Record which frames already have mask data for each live instance, so we know
    # which stale files to delete (via _clear_stale_mask) when refinement produces no
    # mask at a frame this round (rather than leaving the old file in place). Mirrors
    # the offline replay_concept_refinements.
    original_mask_frames: dict = {}
    for _obj_id_int in live_obj_ids:
        _mask_dir = os.path.join(instances_root, str(_obj_id_int), "masks")
        if not os.path.isdir(_mask_dir):
            continue
        _frames = set()
        for _fname in os.listdir(_mask_dir):
            _stem, _fext = os.path.splitext(_fname)
            if _fext in (".npz", ".png"):
                try:
                    _frames.add(int(_stem))
                except ValueError:
                    pass
        if _frames:
            original_mask_frames[_obj_id_int] = _frames

    # Frames each instance actually appeared in this round's propagation output —
    # used below for stale-mask clearing (see original_mask_frames above; done via
    # present_targets_this_frame, populated per-frame below).
    #
    # frames_iterated tracks every frame_idx propagation actually walked through,
    # independent of which objects appear in that frame's output. Used below to mark
    # reachable refinement entries as propagated — see the matching comment in
    # replay_concept_refinements for why per-object absence from out_obj_ids is the
    # wrong signal for that: it usually just means the correction succeeded.
    frames_iterated: set = set()

    # In the live session, deleted instances' sub-states are still present (remove_object
    # is not called from the UI on delete to preserve undo — unlike the offline CLI replay,
    # this session stays alive across UI interactions, so removing a sub-state here would
    # make Ctrl+Z unable to resume tracking it). Gate output so their masks are not written
    # to disk.  (live_obj_ids already computed above.)
    inner_state = sam3_model._all_inference_states[session_id]["state"]
    # Real-time snapshot of every sub-state present right before this propagation pass
    # (after absorbed-source removal, correction points, and mask-anchor injection above).
    # An id NOT in this set that shows up in output is unambiguously new — see the matching
    # comment in replay_concept_refinements for why this is safer than comparing against the
    # historical live_obj_ids set.
    pre_propagation_ids = set(
        int(x) for x in inner_state.get("tracker_metadata", {}).get("obj_ids_all_gpu", [])
    )
    new_id_remap: dict = {}  # raw session id -> persistent id (assigned first time seen)

    _inner_model = getattr(sam3_model, "model", sam3_model)
    _prev_max_num_objects = getattr(_inner_model, "max_num_objects", 10000)
    if new_instance_policy == "disallow":
        _inner_model.max_num_objects = len(pre_propagation_ids)
        print(f"  new_instance_policy=disallow: capped max_num_objects to "
              f"{len(pre_propagation_ids)} (no new objects can be created).")

    try:
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward",
        ):
            out_frame_idx = out["frame_index"]
            outputs = out["outputs"]
            if outputs is None:
                inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
                continue
            frames_iterated.add(out_frame_idx)
            present_targets_this_frame = set()
            for obj_id, mask in zip(outputs.get("out_obj_ids", []), outputs.get("out_binary_masks", [])):
                obj_id_int = int(obj_id)

                if obj_id_int not in pre_propagation_ids:
                    if new_instance_policy != "allow":
                        continue  # "latent" (or a stray "disallow" leak): don't persist
                    if obj_id_int not in new_id_remap:
                        concept.highest_obj_id += 1
                        new_id_remap[obj_id_int] = concept.highest_obj_id
                        print(f"  New object detected mid-video (raw id={obj_id_int}) — "
                              f"assigning persistent id={new_id_remap[obj_id_int]}.")
                    target_id = new_id_remap[obj_id_int]
                elif obj_id_int not in live_obj_ids:
                    continue  # deleted instance — suppress output
                else:
                    target_id = obj_id_int

                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                present_targets_this_frame.add(target_id)
                if target_id not in output_dirs:
                    output_dirs[target_id] = os.path.join(
                        instances_root, str(target_id), "masks"
                    )
                pixel_count = int((mask_np > 0).sum())
                if pixel_count == 0:
                    # Refinement removed this instance from the frame: delete any
                    # previous round's mask file instead of leaving a stale ghost.
                    if out_frame_idx in original_mask_frames.get(target_id, set()):
                        _clear_stale_mask(output_dirs[target_id], out_frame_idx)
                    continue
                mask_writer.submit(mask_np, output_dirs[target_id], out_frame_idx)
                pixel_counts_by_obj[target_id][out_frame_idx] = pixel_count

            # Instances with an active sub-state at propagation start that are entirely
            # absent from this frame's output (e.g. removed by hotstart mid-propagation)
            # also need their stale masks cleared — see replay_concept_refinements.
            for obj_id_int in live_obj_ids & pre_propagation_ids:
                if obj_id_int in present_targets_this_frame:
                    continue
                if out_frame_idx in original_mask_frames.get(obj_id_int, set()):
                    if obj_id_int not in output_dirs:
                        output_dirs[obj_id_int] = os.path.join(
                            instances_root, str(obj_id_int), "masks"
                        )
                    _clear_stale_mask(output_dirs[obj_id_int], out_frame_idx)

            # Evict after saving — forward propagation never revisits past frames.
            # Without this the session accumulates full-resolution masks for every
            # propagated frame (OOM on long videos).
            inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
            _prune_non_cond_outputs(inner_state, out_frame_idx, keep=_keep)
            _offload_unselected_cond_frames(
                inner_state, out_frame_idx, sam3_model.model.tracker,
            )
            if progress_callback:
                progress_callback(out_frame_idx, num_frames)
            if out_frame_idx % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        # Drain before compute_continuous_periods below reads the mask dirs;
        # re-raises any write error from the background thread.
        mask_writer.close()
        _inner_model.max_num_objects = _prev_max_num_objects

    if new_instance_policy == "allow" and new_id_remap:
        _promote_new_instances(
            concept, set(new_id_remap.values()), pixel_counts_by_obj,
            instances_root, total_pixels, num_frames,
        )

    # Save updated cond states for the next refinement round
    if save_cond_states or save_obj_ptr_prior:
        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        if save_cond_states:
            save_cond_frame_states(sam3_model, session_id, concept_dir)
        if save_obj_ptr_prior:
            save_refinement_obj_ptrs(sam3_model, session_id, concept, concept_dir)

    # Refresh metadata for ALL concept instances from actual mask files.
    # The propagation may have rewritten masks for instances beyond instances_with_pending,
    # so we scan the full concept rather than only the subset that had pending annotations.
    for inst in concept.instances:
        if inst.deleted:
            continue
        mask_dir = os.path.join(
            instances_root, str(inst.sam3_obj_id), "masks"
        )
        periods = compute_continuous_periods(mask_dir)
        _pcounts = pixel_counts_by_obj.get(inst.sam3_obj_id, {})
        _pcounts = _backfill_stale_period_counts(periods, _pcounts, mask_dir)
        inst.continuous_periods = periods
        inst.period_peaks = compute_period_peaks(periods, _pcounts, total_pixels)
        inst.presence_norm = compute_presence_norm(_pcounts, total_pixels)
        inst.num_frames_with_mask = sum(e - s + 1 for s, e in periods)
        if periods:
            inst.first_detection_frame = periods[0][0]
            inst.last_detection_frame = periods[-1][1]

    # Mark pending entries as propagated for every frame this round's propagation
    # actually walked through (frames_iterated), regardless of whether the object
    # appears in that frame's output — see the matching comment in
    # replay_concept_refinements for why per-object absence is the wrong signal here.
    # An entry only stays pending if propagation never reached its frame at all.
    stuck_entries = []  # (user_name, obj_id, frame_idx) left pending for a future retry
    for instance, _ in instances_with_pending:
        refinements_path = os.path.join(
            project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "refinements.json"
        )
        if not os.path.exists(refinements_path):
            continue
        with open(refinements_path) as f:
            data = json.load(f)
        changed = False
        for r in data.get("refinements", []):
            if r.get("propagated", True):
                continue
            if r.get("frame_idx") in frames_iterated:
                r["propagated"] = True
                r["online"] = True
                changed = True
            else:
                stuck_entries.append(
                    (instance.user_name, instance.sam3_obj_id, r.get("frame_idx")))
        if changed:
            with open(refinements_path, "w") as f:
                json.dump(data, f, indent=2)

    if stuck_entries:
        print(f"  WARNING: {len(stuck_entries)} correction(s) left PENDING — the instance's "
              f"sub-state never reached that frame in this round's output (likely removed/"
              f"lost track earlier in propagation). Will retry on the next refinement: "
              f"{stuck_entries}")

    print(f"Online concept-level refinement complete for '{concept.name}'.")


def load_sam3_model(model_name: str = "sam3", device: str = "cuda:0",
                    use_fa3: bool = False, max_cond_frames_in_attn: int = -1):
    """
    Load SAM3 model with proper configuration.

    Args:
        model_name: "sam3" (default) or "sam3.1" (Object Multiplex —
            bucketed joint multi-object tracking, much faster for many objects)
        device: Device to use
        use_fa3: Enable FlashAttention-3 fp8 kernels (SAM3.1 only). Requires a
            Hopper GPU (H100/H200) and the flash_attn_interface package;
            crashes on Ampere/Ada (e.g. L40S). Ignored for SAM3.
        max_cond_frames_in_attn: How many conditioning frames the tracker attends
            to per forward pass (default: -1 = no limit, matching SAM3's built-in
            default).  All user correction anchors are always attended to.
            Set to a small positive value (e.g. 4) only if attention compute is a
            bottleneck with many correction frames.

    Returns:
        SAM3 model instance (session-based predictor; same API for both versions)
    """
    import sys
    import os

    # Find project root and add SAM3 to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sam3_path = os.path.join(project_root, "sam_models", "sam3")

    if not os.path.exists(sam3_path):
        raise RuntimeError(
            "SAM3 not found. Please run setup.py to install SAM3.\n"
            f"Expected path: {sam3_path}"
        )

    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    use_multiplex = model_name in ("sam3.1", "3.1", "sam3_multiplex")
    if use_fa3 and not use_multiplex:
        print("WARNING: use_fa3 only applies to SAM3.1 (multiplex); ignored for SAM3.")

    # Load model
    ckpt_name = "sam3.1_multiplex.pt" if use_multiplex else "sam3.pt"
    checkpoint_path = os.path.join(sam3_path, "checkpoints", ckpt_name)
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"SAM3 checkpoint not found: {checkpoint_path}\n"
            "Please download checkpoint using setup.py"
        )

    print(f"Loading {'SAM3.1 multiplex' if use_multiplex else 'SAM3'} model "
          f"from {checkpoint_path} on {device}...")

    # Determine which GPU to use.
    # Sam3VideoPredictorMultiGPU accepts gpus_to_use=[int] to control placement.
    # Its __init__ calls torch.cuda.set_device(device) BEFORE super().__init__(),
    # so all internal .cuda() calls inside Sam3VideoPredictor land on the right GPU.
    if device.startswith("cuda"):
        # device may be "cuda:0" or a comma-separated list "cuda:0,cuda:1,cuda:2"
        parts = [d.strip() for d in device.split(",")]
        if any(":" in p for p in parts):
            gpus_to_use = [int(p.split(":")[1]) for p in parts if ":" in p]
        else:
            gpus_to_use = [torch.cuda.current_device()]
        gpu_id = gpus_to_use[0]
        # Set global current device so any module-level .cuda() calls also land here
        torch.cuda.set_device(gpu_id)
    else:
        # CPU: SAM3 requires CUDA; this will likely fail, but let SAM3 raise its own error
        print(f"WARNING: SAM3 requires CUDA. CPU mode ({device}) is not officially supported.")
        gpus_to_use = None

    if use_multiplex:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        # FlashAttention-3 runs fp8 kernels that only exist on Hopper (H100);
        # on Ampere/Ada (e.g. L40S) it must stay disabled so attention falls
        # back to PyTorch SDPA.
        # The builder hardcodes .cuda(), which lands on the device selected by
        # torch.cuda.set_device() above.
        if use_fa3:
            if gpus_to_use is None:
                raise RuntimeError("--use-fa3 requires a CUDA device.")
            cap = torch.cuda.get_device_capability(gpus_to_use[0])
            if cap[0] < 9:
                raise RuntimeError(
                    f"--use-fa3 requires a Hopper GPU (compute capability 9.0+), "
                    f"but {torch.cuda.get_device_name(gpus_to_use[0])} is sm_{cap[0]}{cap[1]}. "
                    "Run without --use-fa3 on this GPU."
                )
            print("FlashAttention-3 (fp8) enabled")
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            use_fa3=use_fa3,
            compile=False,
        )

        # Sam3BasePredictor.start_session always forwards offload_state_to_cpu to
        # init_state, but the multiplex init_state doesn't accept it (upstream
        # signature mismatch). Filter kwargs to what init_state actually takes.
        import inspect
        orig_init_state = predictor.model.init_state
        valid_params = set(inspect.signature(orig_init_state).parameters)

        def _filtered_init_state(**kwargs):
            return orig_init_state(
                **{k: v for k, v in kwargs.items() if k in valid_params}
            )

        predictor.model.init_state = _filtered_init_state
    else:
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=gpus_to_use,
        )

        # Offload per-frame maskmem_features to CPU so they don't accumulate on GPU.
        # predictor.model is Sam3VideoInferenceWithInstanceInteractivity — it does NOT have
        # this attribute.  The real flag lives on predictor.model.tracker (Sam3TrackerPredictor).
        # (Multiplex bounds GPU memory via _prune_non_cond_outputs instead; its bucketized
        # tracker states are not safe to offload through this flag.)
        tracker = getattr(getattr(predictor, "model", None), "tracker", None)
        if tracker is not None and hasattr(tracker, "offload_output_to_cpu_for_eval"):
            tracker.offload_output_to_cpu_for_eval = True
            print("SAM3: offload_output_to_cpu_for_eval=True on tracker")
        # NOTE: tracker.non_overlap_masks_for_output (set False by sam3.model_builder's
        # build_tracker(), overriding Sam3TrackerPredictor's own default of True) only
        # gates _get_orig_video_res_output() in sam3_tracking_predictor.py — a separate,
        # image-editing-style API this pipeline never reads output from. The actual
        # output path for propagate_in_video() (used by every function in this file) is
        # Sam3VideoInference._postprocess_output() in sam3_video_inference.py, which
        # calls tracker._apply_object_wise_non_overlapping_constraints() UNCONDITIONALLY
        # whenever more than one object mask is present (no flag check at all) — so
        # instances within one concept's joint session are already pixel-exclusive by
        # construction, and toggling that flag here would be a no-op for this codebase.
        # Disable prev-mask-logit bias so user correction points segment from scratch
        # rather than refining the (possibly wrong) VG detector prediction at that frame.
        # With iter_use_prev_mask_pred=True (the SAM3 default), a correction at frame F
        # receives the VG mask as prev_sam_mask_logits, biasing the decoder toward the
        # wrong region even when the correction point is on a different object.
        # SAM3.1 multiplex already defaults to False, so this only affects SAM3.
        if tracker is not None and hasattr(tracker, "iter_use_prev_mask_pred"):
            tracker.iter_use_prev_mask_pred = False
            print("SAM3: iter_use_prev_mask_pred=False on tracker (correction points start fresh)")

    # Patch max_cond_frames_in_attn on the tracker (lives on SAM2Base which
    # Sam3TrackerPredictor inherits from). Works for both SAM3 and SAM3.1.
    _tracker = getattr(getattr(predictor, "model", None), "tracker", None)
    if _tracker is None:
        # SAM3.1 multiplex may expose tracker differently
        _tracker = getattr(predictor, "model", None)
    if _tracker is not None and hasattr(_tracker, "max_cond_frames_in_attn"):
        _tracker.max_cond_frames_in_attn = max_cond_frames_in_attn
        print(f"SAM3: max_cond_frames_in_attn={max_cond_frames_in_attn}")

    print(f"SAM3 model loaded on device: {device}")

    return predictor


def get_video_info(video_path: str) -> Tuple[int, Tuple[int, int], float]:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        (num_frames, (width, height), fps)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return num_frames, (width, height), fps
