#!/usr/bin/env python3
"""
SAM3 Batch Processing Script - Process concepts offline without UI.

Usage:
    python sam3_process.py --video video.mp4 --project project_dir --concepts concepts.json
    python sam3_process.py --project project_dir --concepts person,car
    python sam3_process.py --project project_dir --concepts batch.json --filter person,car
    python sam3_process.py --project project_dir --refine
    python sam3_process.py --project project_dir --refine person,car
    python sam3_process.py --project project_dir --delete person
    python sam3_process.py --project project_dir --delete person:inst1,inst2
    python sam3_process.py --project project_dir --export output.mp4

concepts.json format:
{
  "concepts": [
    {
      "name": "person",
      "text_prompt": "person wearing blue shirt",
      "detection_frame": 0,
      "max_instances": 5
    },
    {
      "name": "car",
      "text_prompt": "red car",
      "detection_frame": 10
    }
  ]
}
Per-concept "max_instances" above is used only if --max-instances is not passed on the
command line (or passed as -1). An explicit --max-instances N (N > 0) overrides it, and
every other concept about to run initial detection this invocation, applying uniformly
in propagation. This only affects initial detection; a later --refine run always keeps
every instance the user has annotated, even past this cap.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sam3_project import SAM3Project, SAM3Concept, cleanup_frames_dir
from sam3_pipeline import process_concept_detection, load_sam3_model, get_video_info, replay_concept_refinements
from sam3_utils import (
    generate_concept_color, validate_text_prompt,
    check_and_reencode_video, force_reencode_video_mjpeg, DynamicFrameCompositor
)


def load_concepts_from_json(json_path: str) -> list:
    """Load concept definitions from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    concepts = []
    for i, concept_data in enumerate(data.get("concepts", [])):
        name = concept_data["name"]
        text_prompt = concept_data["text_prompt"]
        detection_frame = concept_data.get("detection_frame", 0)

        if not validate_text_prompt(text_prompt):
            print(f"Warning: Invalid text prompt for concept '{name}': {text_prompt}")
            continue

        color = generate_concept_color(i)
        concept = SAM3Concept(
            name=name,
            text_prompt=text_prompt,
            color_rgb=color,
            detection_frame=detection_frame,
            max_instances=concept_data.get("max_instances", -1),
        )
        concepts.append(concept)

    return concepts


def _load_project(project_dir: str):
    """Load project or print error and return None."""
    if not os.path.exists(os.path.join(project_dir, "project.json")):
        print(f"Error: Project not found at {project_dir}")
        return None
    return SAM3Project.load(project_dir)


def _confirm(prompt: str, force: bool) -> bool:
    """Return True if the action should proceed."""
    if force:
        return True
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer != 'y':
        print("Aborted.")
        return False
    return True


def _delete_single_instance(project, concept, instance, force: bool) -> int:
    inst_dir = os.path.join(
        project.get_concept_dir(concept.name),
        "instances", str(instance.sam3_obj_id)
    )
    print(f"Concept:   '{concept.name}'")
    print(f"Instance:  '{instance.user_name}' (obj_id {instance.sam3_obj_id})")
    print(f"Directory: {inst_dir}")

    if not _confirm("\nDelete this instance and its masks?", force):
        return 0

    if os.path.exists(inst_dir):
        shutil.rmtree(inst_dir)
        print(f"Removed directory: {inst_dir}")
    else:
        print(f"(Directory not found, skipping: {inst_dir})")

    instance.deleted = True
    instance.visible = False
    project.save()
    print(f"Instance '{instance.user_name}' marked deleted in project.")
    return 0


def handle_delete(args):
    """Handle --delete CONCEPT or --delete CONCEPT:INST1,INST2,..."""
    project = _load_project(args.project)
    if project is None:
        return 1

    delete_arg = args.delete
    if ':' in delete_arg:
        # Delete specific instances: CONCEPT:inst1,inst2
        concept_name, instances_str = delete_arg.split(':', 1)
        concept_name = concept_name.strip()
        instance_names = [s.strip() for s in instances_str.split(',') if s.strip()]

        concept = project.get_concept_by_name(concept_name)
        if concept is None:
            print(f"Error: Concept '{concept_name}' not found in project.")
            print(f"Available concepts: {[c.name for c in project.concepts]}")
            return 1

        live_by_name = {
            i.user_name: i for i in concept.instances if not i.deleted
        }
        live_by_id = {
            str(i.sam3_obj_id): i for i in concept.instances if not i.deleted
        }

        not_found = [n for n in instance_names if n not in live_by_name and n not in live_by_id]
        if not_found:
            live = [(i.user_name, i.sam3_obj_id) for i in concept.instances if not i.deleted]
            print(f"Error: Instance(s) not found in concept '{concept_name}': {not_found}")
            print(f"Live instances: {live}")
            return 1

        for name in instance_names:
            instance = live_by_name.get(name) or live_by_id.get(name)
            rc = _delete_single_instance(project, concept, instance, args.force)
            if rc != 0:
                return rc
        return 0

    else:
        # Delete entire concept
        concept_name = delete_arg.strip()
        concept = project.get_concept_by_name(concept_name)
        if concept is None:
            print(f"Error: Concept '{concept_name}' not found in project.")
            print(f"Available concepts: {[c.name for c in project.concepts]}")
            return 1

        concept_dir = project.get_concept_dir(concept_name)
        num_instances = len([i for i in concept.instances if not i.deleted])
        print(f"Concept:    '{concept_name}'")
        print(f"Status:     {concept.status.value}")
        print(f"Instances:  {num_instances}")
        print(f"Directory:  {concept_dir}")

        if not _confirm("\nDelete this concept and all its data?", args.force):
            return 0

        if os.path.exists(concept_dir):
            shutil.rmtree(concept_dir)
            print(f"Removed directory: {concept_dir}")
        else:
            print(f"(Directory not found, skipping: {concept_dir})")

        project.remove_concept(concept_name)
        project.save()
        print(f"Concept '{concept_name}' removed from project.")
        return 0


def _model_name_for(args) -> str:
    """Map --sam-version to the load_sam3_model model_name."""
    return "sam3.1" if args.sam_version == "3.1" else "sam3"


def _parallel_worker_main(worker_id, device, project_dir, model_name, use_fa3,
                          max_cond_frames_in_attn, save_cond_states, cache_size,
                          task_queue, result_queue):
    """
    Worker process for parallel concept processing.

    Loads the SAM3 model ONCE, then pulls concepts from the task queue until
    a None sentinel arrives. Writes only to its concepts' own directories
    (masks, cond states, concept_metadata.json) — never to project.json,
    which is owned exclusively by the parent orchestrator.
    """
    import traceback
    from sam3_project import SAM3Project, SAM3Concept
    from sam3_pipeline import load_sam3_model, process_concept_detection

    tag = f"[worker {worker_id}/{device}]"
    try:
        # Read-only view of the project: paths, num_frames, mask_format.
        project = SAM3Project.load(project_dir)
        sam3_model = load_sam3_model(model_name=model_name, device=device,
                                     use_fa3=use_fa3,
                                     max_cond_frames_in_attn=max_cond_frames_in_attn)
    except Exception as e:
        traceback.print_exc()
        result_queue.put(("fatal", worker_id, str(e)))
        return
    result_queue.put(("ready", worker_id, None))

    while True:
        item = task_queue.get()
        if item is None:
            break
        concept_dict, task_idx, task_total = item
        concept = SAM3Concept.from_dict(concept_dict)
        print(f"{tag} processing concept '{concept.name}' [{task_idx}/{task_total}] "
              f"({concept.prompt_label()})")

        def progress_callback(frame_idx, num_frames, _name=concept.name,
                               _ci=task_idx, _n=task_total):
            if frame_idx % 200 == 0:
                pct = (frame_idx + 1) / num_frames * 100
                print(f"{tag} [{_name}] [{_ci}/{_n}] {frame_idx+1}/{num_frames} frames ({pct:.1f}%)")

        try:
            process_concept_detection(
                sam3_model=sam3_model,
                project=project,
                concept=concept,
                device=device,
                progress_callback=progress_callback,
                save_cond_states=save_cond_states,
                cache_size=cache_size,
            )
            # Persist this concept's metadata (disjoint per-concept file).
            project._save_concept_metadata(concept)
            result_queue.put(("done", worker_id, concept.name))
        except Exception as e:
            traceback.print_exc()
            result_queue.put(("error", worker_id, (concept.name, str(e))))


def process_project_parallel(args, project, concepts_to_process, devices):
    """
    Process concepts with a pool of worker processes.

    Orchestrator pattern: the parent owns project.json exclusively. Workers
    write only to their disjoint concepts/<name>/ directories; the parent
    merges each finished concept's metadata back and saves project.json.
    Workers are assigned round-robin to the given devices, so multiple
    workers on one GPU (utilization packing) and multiple GPUs both work.
    """
    import multiprocessing as mp
    from sam3_project import ConceptStatus

    # Register all concepts and persist BEFORE spawning so workers see them.
    for concept in concepts_to_process:
        if project.get_concept_by_name(concept.name) is None:
            project.add_concept(concept)
    project.save()

    # Pre-extract frames once — eliminates the extraction race between workers.
    # Completeness-checked: resumes if a previous extraction was killed midway.
    from sam3_utils import ensure_frames_extracted
    frames_dir = ensure_frames_extracted(
        project.video_path, project.get_frames_dir(), project.num_frames)

    n_workers = min(args.parallel, len(concepts_to_process))
    print(f"\nStarting {n_workers} worker(s) on device(s): {', '.join(devices)}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    total = len(concepts_to_process)
    for idx, concept in enumerate(concepts_to_process, start=1):
        task_queue.put((concept.to_dict(), idx, total))
    for _ in range(n_workers):
        task_queue.put(None)  # one stop sentinel per worker

    workers = []
    for i in range(n_workers):
        device = devices[i % len(devices)]
        p = ctx.Process(
            target=_parallel_worker_main,
            args=(i, device, project.project_dir, _model_name_for(args),
                  args.use_fa3, args.max_cond_frames_in_attn, args.save_cond_states,
                  args.cache_size, task_queue, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Merge results as they arrive — parent is the only writer of project.json.
    import queue as _queue_mod
    remaining = len(concepts_to_process)
    errors = []
    while remaining > 0:
        try:
            # Timeout so the parent re-checks worker liveness periodically: a
            # blocking get() can hang forever if the last worker emits "fatal"
            # while still marginally alive (is_alive() below races process
            # teardown) and then no message ever arrives again.
            kind, worker_id, payload = result_queue.get(timeout=15)
        except _queue_mod.Empty:
            if not any(p.is_alive() for p in workers):
                print("All workers died with tasks unfinished — aborting.")
                errors.append(("workers", "all workers died before finishing"))
                break
            continue
        if kind == "ready":
            print(f"[worker {worker_id}] model loaded, ready")
            continue
        if kind == "fatal":
            # Worker died before processing anything; its queued sentinel is
            # unconsumed but other workers will still drain the task queue.
            errors.append((f"worker {worker_id}", payload))
            print(f"[worker {worker_id}] FATAL: {payload}")
            # If ALL workers died, the remaining tasks will never finish.
            # (If the last one is still mid-teardown here, the get() timeout
            # above catches it on the next iteration.)
            if not any(p.is_alive() for p in workers):
                print("All workers died — aborting.")
                break
            continue

        remaining -= 1
        if kind == "done":
            cname = payload
            updated = project._load_concept_metadata(cname)
            if updated is not None:
                for idx, c in enumerate(project.concepts):
                    if c.name == cname:
                        project.concepts[idx] = updated
                        break
                else:
                    project.add_concept(updated)
            # Skip concept metadata rewrite: workers own those files and a
            # stale in-memory copy must not clobber an in-flight concept.
            project.save(write_concept_metadata=False)
            print(f"\nConcept '{cname}' complete and merged "
                  f"({remaining} remaining)\n")
        else:  # "error"
            cname, err = payload
            errors.append((cname, err))
            # Re-read the concept's on-disk metadata first (the worker may have
            # written partial state before failing), then persist ERROR into its
            # concept_metadata.json — load() reads status from there, not from
            # the project.json summary, so without this the failure is invisible
            # on reload. Safe: the worker that owned this concept has moved on.
            updated = project._load_concept_metadata(cname)
            if updated is not None:
                for idx, pc in enumerate(project.concepts):
                    if pc.name == cname:
                        project.concepts[idx] = updated
                        break
            c = project.get_concept_by_name(cname)
            if c is not None:
                c.status = ConceptStatus.ERROR
                try:
                    project._save_concept_metadata(c)
                except Exception as save_err:
                    print(f"WARNING: could not persist error status for "
                          f"'{cname}': {save_err}")
            project.save(write_concept_metadata=False)
            print(f"\nConcept '{cname}' FAILED: {err} ({remaining} remaining)\n")

    for p in workers:
        p.join(timeout=60)

    print(f"\n{'='*60}")
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("All concepts processed!")
    print(f"{'='*60}")
    return 1 if errors else 0


def _finalize_concept_refinement(concept, project_dir):
    """
    Post-processing shared by the serial (_refine_project_impl) and parallel
    (_parallel_refine_worker_main) `--refine` paths, run after
    replay_concept_refinements() succeeds for a concept.

    1. Drops merge-absorbed source instances from metadata — their masks were
       already deleted by the pipeline, so keeping stale entries around risks
       obj_id collisions with SAM3's own detector on a future redetect.
    2. Marks the concept COMPLETED once no live instance has any unpropagated
       refinements.json entry left. process_concept_detection is otherwise the
       only place that ever sets ConceptStatus.COMPLETED, so a concept whose
       instances were all manually added (never ran text-prompt detection) or
       that errored out after an earlier detection run would otherwise stay
       "pending"/"error" forever even once fully refined. This mirrors the
       same check handle_status() already uses for its [REFINEMENT NEEDED]
       listing, so --status agrees with what actually happened.

    Mutates `concept` in place. Does not save/persist anything — callers own
    that (project.save() vs. project._save_concept_metadata()) and their own
    logging style (plain print vs. worker-tagged print).

    Returns (absorbed_ids_removed: set[int], marked_completed: bool).
    """
    absorbed_ids_removed = {
        inst.sam3_obj_id for inst in concept.instances
        if inst.deleted and getattr(inst, 'absorbed_source', False)
    }
    if absorbed_ids_removed:
        concept.instances = [
            inst for inst in concept.instances
            if not (inst.deleted and getattr(inst, 'absorbed_source', False))
        ]
        for inst in concept.instances:
            inst.absorbed_source_ids = [
                sid for sid in inst.absorbed_source_ids
                if sid not in absorbed_ids_removed
            ]

    still_pending = False
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
            refs = json.load(f).get("refinements", [])
        if any(not r.get("propagated", True) for r in refs):
            still_pending = True
            break

    marked_completed = False
    if not still_pending and concept.status.value != "completed":
        from sam3_project import ConceptStatus
        from datetime import datetime
        concept.status = ConceptStatus.COMPLETED
        concept.completed_at = datetime.now().isoformat()
        marked_completed = True

    return absorbed_ids_removed, marked_completed


def _parallel_refine_worker_main(
    worker_id, device, project_dir, model_name, use_fa3, max_cond_frames_in_attn,
    restore_cond_states, save_cond_states, save_obj_ptr_prior, preload_original_masks,
    cache_size, new_instance_policy, task_queue, result_queue
):
    """
    Worker process for parallel refinement.

    Handles two task types:
      {"type": "redetect", "concept": concept_dict}
      {"type": "refine", "concept_name": str}

    Rebuilds instances_with_pending from refinements.json on disk so no large
    objects need to cross the process boundary.
    """
    import json
    import traceback
    from sam3_project import SAM3Project, SAM3Concept
    from sam3_pipeline import load_sam3_model, process_concept_detection, replay_concept_refinements

    tag = f"[refine-worker {worker_id}/{device}]"
    try:
        project = SAM3Project.load(project_dir)
        sam3_model = load_sam3_model(model_name=model_name, device=device,
                                     use_fa3=use_fa3,
                                     max_cond_frames_in_attn=max_cond_frames_in_attn)
    except Exception as e:
        traceback.print_exc()
        result_queue.put(("fatal", worker_id, str(e)))
        return
    result_queue.put(("ready", worker_id, None))

    resource_path = project.get_frames_dir()
    orig_width, orig_height = project.frame_dimensions
    num_frames = project.num_frames

    while True:
        item = task_queue.get()
        if item is None:
            break

        task_type = item["type"]
        cname = item["concept_name"]
        task_idx = item.get("task_idx")
        task_total = item.get("task_total")

        def progress_callback(frame_idx, nf, _cname=cname, _ci=task_idx, _n=task_total):
            if frame_idx % 200 == 0:
                pct = (frame_idx + 1) / nf * 100
                print(f"{tag} [{_cname}] [{_ci}/{_n}] {frame_idx+1}/{nf} frames ({pct:.1f}%)")

        try:
            if task_type == "redetect":
                concept = SAM3Concept.from_dict(item["concept"])
                print(f"{tag} re-detecting concept '{concept.name}' [{task_idx}/{task_total}]")
                process_concept_detection(
                    sam3_model=sam3_model,
                    project=project,
                    concept=concept,
                    device=device,
                    progress_callback=progress_callback,
                    save_cond_states=save_cond_states,
                    cache_size=cache_size,
                )
                project._save_concept_metadata(concept)
            else:
                concept = project.get_concept_by_name(cname)
                # Rebuild pending list from disk — avoids large object crossing process boundary
                instances_with_pending = []
                for inst in concept.instances:
                    rpath = os.path.join(
                        project_dir, "concepts", cname,
                        "instances", str(inst.sam3_obj_id), "refinements.json"
                    )
                    if not os.path.exists(rpath):
                        continue
                    with open(rpath) as f:
                        refinements = json.load(f).get("refinements", [])
                    pending = [r for r in refinements if not r.get("propagated", True)]
                    if pending:
                        instances_with_pending.append((inst, pending))
                print(f"{tag} refining concept '{cname}' [{task_idx}/{task_total}] "
                      f"({len(instances_with_pending)} instance(s) with pending)")
                replay_concept_refinements(
                    concept=concept,
                    instances_with_pending=instances_with_pending,
                    resource_path=resource_path,
                    orig_width=orig_width,
                    orig_height=orig_height,
                    num_frames=num_frames,
                    sam3_model=sam3_model,
                    project_dir=project_dir,
                    progress_callback=progress_callback,
                    device=device,
                    restore_cond_states=restore_cond_states,
                    save_cond_states=save_cond_states,
                    save_obj_ptr_prior=save_obj_ptr_prior,
                    mask_format=project.mask_format,
                    cache_size=cache_size,
                    preload_original_masks=preload_original_masks,
                    new_instance_policy=new_instance_policy,
                )
                absorbed_ids_removed, marked_completed = _finalize_concept_refinement(
                    concept, project_dir)
                if absorbed_ids_removed:
                    print(f"{tag} removed {len(absorbed_ids_removed)} absorbed source(s) from "
                          f"project metadata: obj_ids {sorted(absorbed_ids_removed)}")
                if marked_completed:
                    print(f"{tag} concept '{cname}' has no pending annotations left — marking completed.")
                # Persist updated presence fields and cleaned-up instance list.
                project._save_concept_metadata(concept)
            result_queue.put(("done", worker_id, cname))
        except Exception as e:
            traceback.print_exc()
            result_queue.put(("error", worker_id, (cname, str(e))))


def refine_project_parallel(args, project, redetect_concepts, refine_concepts, devices):
    """
    Parallel dispatch for --refine: runs redetect and refine tasks across worker processes.

    Tasks are assigned round-robin across devices, matching process_project_parallel's model.
    Workers write only to their concepts' own directories; project.json is saved by the
    parent after all workers finish.

    Returns (rc, failed_cnames) where failed_cnames is the set of concept names whose
    redetect/refine task raised an error — the caller uses this to avoid clearing state
    (e.g. the redetect sentinel) for concepts that didn't actually finish.
    """
    import multiprocessing as mp

    all_tasks = []
    for cname, concept in redetect_concepts.items():
        all_tasks.append({"type": "redetect", "concept": concept.to_dict(), "concept_name": cname})
    for cname in refine_concepts:
        all_tasks.append({"type": "refine", "concept_name": cname})

    if not all_tasks:
        return 0, set()

    total_tasks = len(all_tasks)
    for idx, task in enumerate(all_tasks, start=1):
        task["task_idx"] = idx
        task["task_total"] = total_tasks

    n_workers = min(args.parallel, len(all_tasks))
    print(f"\nStarting {n_workers} refine worker(s) on device(s): {', '.join(devices)}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in all_tasks:
        task_queue.put(task)
    for _ in range(n_workers):
        task_queue.put(None)

    workers = []
    for i in range(n_workers):
        device = devices[i % len(devices)]
        p = ctx.Process(
            target=_parallel_refine_worker_main,
            args=(i, device, project.project_dir, _model_name_for(args),
                  args.use_fa3, args.max_cond_frames_in_attn,
                  args.restore_cond, args.save_cond_states, args.save_obj_ptr_prior,
                  args.preload_original_masks, args.cache_size,
                  args.new_instance_policy, task_queue, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Wait for all workers to be ready
    n_ready = 0
    errors = []
    while n_ready < n_workers:
        msg_type, worker_id, payload = result_queue.get()
        if msg_type == "ready":
            n_ready += 1
            print(f"[refine-worker {worker_id}] ready")
        elif msg_type == "fatal":
            errors.append((f"worker-{worker_id}", payload))
            n_ready += 1

    if errors:
        for p in workers:
            p.terminate()
        for name, err in errors:
            print(f"  {name} failed to start: {err}")
        # No tasks were processed — every concept's redetect/refine is still pending.
        return 1, {t["concept_name"] for t in all_tasks}

    remaining = len(all_tasks)
    while remaining > 0:
        msg_type, worker_id, payload = result_queue.get()
        if msg_type == "done":
            remaining -= 1
            cname = payload
            # Workers write the authoritative concepts/<name>/concept_metadata.json
            # directly but never send the mutated concept back over the queue, so
            # the parent's in-memory copy is still the stale pre-refine version.
            # Re-read it now (mirroring process_project_parallel's merge-back)
            # so the project.json summary this function's caller saves below
            # reflects what the worker actually wrote, not stale status/counts.
            updated = project._load_concept_metadata(cname)
            if updated is not None:
                for idx, c in enumerate(project.concepts):
                    if c.name == cname:
                        project.concepts[idx] = updated
                        break
                else:
                    project.add_concept(updated)
            print(f"[refine-worker {worker_id}] done: '{cname}' ({remaining} remaining)")
        elif msg_type == "error":
            remaining -= 1
            cname, err = payload
            errors.append((cname, err))
            print(f"\n[refine-worker {worker_id}] FAILED: '{cname}': {err} ({remaining} remaining)\n")
        elif msg_type == "fatal":
            errors.append((f"worker-{worker_id}", payload))
            remaining -= 1

    for p in workers:
        p.join(timeout=60)

    print(f"\n{'='*60}")
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("All refinements processed!")
    print(f"{'='*60}")
    failed_cnames = {name for name, _err in errors if name in redetect_concepts or name in refine_concepts}
    return (1 if errors else 0), failed_cnames


def process_project(args):
    """Process concepts in a project"""
    # --device accepts a comma-separated list (e.g. cuda:2,cuda:3).
    # Parallel workers are assigned round-robin; sequential mode uses the first.
    devices = [d.strip() for d in args.device.split(",") if d.strip()]

    # Load or create project
    if os.path.exists(os.path.join(args.project, "project.json")):
        print(f"Loading existing project from {args.project}")
        project = SAM3Project.load(args.project)
    else:
        if not args.video:
            print("Error: --video required for new project")
            return 1

        print(f"Creating new project at {args.project}")

        num_frames, frame_dims, fps = get_video_info(args.video)
        print(f"Video: {num_frames} frames, {frame_dims[0]}x{frame_dims[1]}, {fps:.2f} fps")

        os.makedirs(args.project, exist_ok=True)
        mjpeg_video = check_and_reencode_video(args.video, args.project)

        project = SAM3Project.create_new(
            project_dir=args.project,
            video_path=args.video,
            num_frames=num_frames,
            frame_dimensions=frame_dims,
            fps=fps,
            device=devices[0],
            mask_format=args.mask_format,
        )

        if mjpeg_video != args.video:
            project.mjpeg_video_path = mjpeg_video
            project.save()

    from sam3_project import acquire_refine_lock, release_refine_lock
    acquire_refine_lock(project.project_dir)
    try:
        return _process_project_impl(args, project, devices)
    finally:
        release_refine_lock(project.project_dir)


def _process_project_impl(args, project, devices):
    # Pin frame-dir if user specified one (persists across runs; stored in project.json).
    frame_dir = getattr(args, "frame_dir", None)
    if frame_dir and project.frames_dir != frame_dir:
        project.frames_dir = frame_dir
        project.save()
        print(f"Frame cache pinned to: {frame_dir}")

    # Build the list of concepts to process
    if args.concepts:
        if args.concepts.endswith(".json") or os.path.isfile(args.concepts):
            loaded = load_concepts_from_json(args.concepts)
            # Skip concepts the project already completed (resume support).
            # Use --reset to discard and redetect a completed concept.
            concepts_to_process = []
            skipped = []
            for c in loaded:
                existing = project.get_concept_by_name(c.name)
                if existing is not None and existing.status.value == "completed":
                    skipped.append(c.name)
                    continue
                concepts_to_process.append(existing if existing is not None else c)
            if skipped:
                print(f"Skipping {len(skipped)} already-completed concept(s): "
                      f"{', '.join(skipped)}  (use --reset to discard and redetect)")
            print(f"Loaded {len(concepts_to_process)} concept(s) to process "
                  f"from {args.concepts}")
        else:
            names = [n.strip() for n in args.concepts.split(",") if n.strip()]
            concepts_to_process = []
            skipped = []
            for i, name in enumerate(names):
                existing = next((c for c in project.concepts if c.name == name), None)
                if existing and existing.status.value in ("pending", "error"):
                    concepts_to_process.append(existing)
                elif existing:
                    skipped.append(name)
                elif not existing:
                    color = generate_concept_color(len(project.concepts) + i)
                    concepts_to_process.append(SAM3Concept(
                        name=name, text_prompt=name, color_rgb=color, detection_frame=0,
                        max_instances=args.max_instances,
                    ))
            if skipped:
                print(f"Skipping {len(skipped)} already-completed concept(s): "
                      f"{', '.join(skipped)}  (use --reset to discard and redetect)")
            print(f"Using {len(concepts_to_process)} concept(s) from list: {', '.join(names)}")
    else:
        concepts_to_process = [
            c for c in project.concepts
            if c.status.value in ("pending", "error")
        ]
        print(f"Found {len(concepts_to_process)} concept(s) to process in project")

    # Apply --filter
    if args.filter:
        filter_names = {n.strip() for n in args.filter.split(",") if n.strip()}
        before = len(concepts_to_process)
        concepts_to_process = [c for c in concepts_to_process if c.name in filter_names]
        print(f"Filter '{args.filter}': {before} -> {len(concepts_to_process)} concept(s)")

    # --max-instances, when explicitly capping (>0), overrides every concept about to
    # run initial detection here — including a JSON file's per-concept "max_instances"
    # field and a pre-existing pending/error concept's previously-stored value. This is
    # the initial-detection path only (--refine never reaches _process_project_impl); a
    # cap applied here never affects a later --refine run, which replays whatever the
    # user has kept regardless of this number (see replay_concept_refinements).
    if args.max_instances > 0:
        for c in concepts_to_process:
            if c.max_instances != args.max_instances:
                print(f"  --max-instances {args.max_instances}: overriding '{c.name}' "
                      f"(was {c.max_instances})")
                c.max_instances = args.max_instances

    if not concepts_to_process:
        print("No concepts to process")
        return 0

    if args.parallel > 1:
        rc = process_project_parallel(args, project, concepts_to_process, devices)
        cleanup_frames_dir(project)
        return rc

    sam3_model = load_sam3_model(model_name=_model_name_for(args), device=devices[0],
                                 use_fa3=args.use_fa3,
                                 max_cond_frames_in_attn=args.max_cond_frames_in_attn)

    failed_concepts = []
    for i, concept in enumerate(concepts_to_process):
        print(f"\n{'='*60}")
        print(f"Processing concept {i+1}/{len(concepts_to_process)}: '{concept.name}'")
        print(f"Prompt: {concept.prompt_label()}")
        print(f"{'='*60}\n")

        existing = project.get_concept_by_name(concept.name)
        if existing is None:
            project.add_concept(concept)
        else:
            concept = existing

        def progress_callback(frame_idx, num_frames, _name=concept.name, _ci=i, _n=len(concepts_to_process)):
            if frame_idx % 200 == 0:
                progress = (frame_idx + 1) / num_frames * 100
                print(f"[concept '{_name}' {_ci+1}/{_n}] Progress: {frame_idx+1}/{num_frames} frames ({progress:.1f}%)")

        try:
            process_concept_detection(
                sam3_model=sam3_model,
                project=project,
                concept=concept,
                device=devices[0],
                progress_callback=progress_callback,
                save_cond_states=args.save_cond_states,
                cache_size=args.cache_size,
            )
            project.save()
            print(f"\nConcept '{concept.name}' complete!")

        except Exception as e:
            print(f"\nError processing concept '{concept.name}': {e}")
            import traceback
            traceback.print_exc()
            # Persist the ERROR status set by process_concept_detection so the
            # failure is visible to --status and the concept is re-selected
            # (pending/error) on the next run.
            failed_concepts.append(concept.name)
            try:
                project.save()
            except Exception as save_err:
                print(f"WARNING: could not persist error status: {save_err}")
            continue

    print(f"\n{'='*60}")
    if failed_concepts:
        print(f"Finished with {len(failed_concepts)} FAILED concept(s): {failed_concepts}")
    else:
        print("All concepts processed!")
    print(f"{'='*60}")
    cleanup_frames_dir(project)
    return 1 if failed_concepts else 0


def handle_delete_frames(args):
    """Handle --delete-frames: show path then remove the persistent frame cache."""
    project = _load_project(args.project)
    if project is None:
        return 1

    frames_dir = project.get_frames_dir()
    if not os.path.isdir(frames_dir):
        print(f"No frame cache found for project at {args.project}")
        print(f"(Expected path: {frames_dir})")
        return 0

    n_files = sum(1 for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    size_mb = sum(
        os.path.getsize(os.path.join(frames_dir, f))
        for f in os.listdir(frames_dir) if f.endswith('.jpg')
    ) / (1024 ** 2)
    kind = "persistent" if project.frames_dir else "temporary"
    print(f"Frame cache directory : {frames_dir}  ({kind})")
    print(f"Files                 : {n_files} JPEGs  ({size_mb:.1f} MB)")

    if not _confirm(f"\nDelete frame cache at {frames_dir}?", args.force):
        return 0

    shutil.rmtree(frames_dir)
    print(f"Deleted: {frames_dir}")

    project.frames_dir = None
    project.save()
    print("Project updated (frames_dir cleared).")
    return 0


def handle_quality_metrics(args):
    """Compute and save quality_metrics.npz for the project.

    Runs by default after detection / --refine / --redo / --reset;
    suppressed by --no-quality-metrics.
    """
    project = _load_project(args.project)
    if project is None:
        return 1

    from sam3_utils import build_sam3_masks_metadata
    from utils import calculate_quality_metrics_grouped, save_quality_metrics_grouped

    print(f"Calculating quality metrics for project: {args.project}")
    masks_by_frame, load_fn, obj_id_to_info = build_sam3_masks_metadata(project)

    if not masks_by_frame:
        print("No masks found in project. Run concept detection first.")
        return 1

    print(f"Found {len(obj_id_to_info)} object(s) across {len(masks_by_frame)} frame(s).")
    width, height = project.frame_dimensions

    # Inter-frame change is computed per concept; overlap is a per-concept
    # one-vs-rest cross-concept ratio (fraction of the concept's area also claimed
    # by another concept). Background ratio stays global.
    obj_id_to_group = {oid: info["concept"] for oid, info in obj_id_to_info.items()}
    result = calculate_quality_metrics_grouped(
        masks=masks_by_frame,
        load_mask_func=load_fn,
        obj_id_to_group=obj_id_to_group,
        frame_dimensions=(height, width),
        num_frames=project.num_frames,
    )
    save_quality_metrics_grouped(args.project, result)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print(f"Background ratio         (mean, global): {_mean(result['background_ratios']):.3f}")
    for g in result['group_names']:
        print(f"Concept '{g}':")
        print(f"  Inter-frame change     (mean): {_mean(result['inter_frame_changes_by_group'][g]):.3f}")
        print(f"  Overlap vs other concepts (mean): {_mean(result['overlap_ratios_by_group'][g]):.3f}")
    _grp_overlap_means = [_mean(result['overlap_ratios_by_group'][g]) for g in result['group_names']]
    print(f"Inter-frame change (mean, all concepts combined): {_mean(result['inter_frame_changes']):.3f}")
    print(f"Overlap vs other concepts (mean over concepts): {_mean(_grp_overlap_means):.3f}")
    print(f"Global excess-assignment overlap (backward-compat): {_mean(result['overlap_ratios']):.3f}")
    print(f"Saved: {os.path.join(args.project, 'quality_metrics.npz')}")
    return 0


def _load_sam2_object_list(csv_path: str) -> dict:
    """
    Load a SAM2 object-list CSV (exported from sam2_ui.py) and return
    {name: {"id": int, "color": [R, G, B]}} for name-based matching.
    Warns about duplicate names in the CSV (last entry wins).
    """
    import csv as _csv
    name_to_entry: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue
            try:
                obj_id = int(row["id"])
                color = [int(row["color_r"]), int(row["color_g"]), int(row["color_b"])]
            except (KeyError, ValueError):
                print(f"WARNING: skipping malformed CSV row: {row}")
                continue
            if name in name_to_entry:
                print(f"WARNING: duplicate name '{name}' in CSV — last entry (id={obj_id}) wins.")
            name_to_entry[name] = {"id": obj_id, "color": color}
    return name_to_entry


def _resolve_unmatched_interactively(inst_label: str, existing_mapping: dict,
                                     next_id: int) -> tuple:
    """
    Interactively resolve a new SAM3 instance that had no name match in the CSV.
    Returns (sam2_id, action) where action is 'new', 'merge', or 'discard'.
    sam2_id is None for 'discard'.
    """
    print(f"\n  Unmatched instance: {inst_label}")
    print(f"  Options:")
    print(f"    n  — assign new ID {next_id}")
    print(f"    m  — merge into an existing SAM2 object ID")
    print(f"    d  — discard (skip this instance)")
    if existing_mapping:
        print(f"  Existing IDs: " +
              ", ".join(f"{k}={v.get('name','?')}" for k, v in sorted(
                  existing_mapping.items(), key=lambda x: int(x[0]))))
    while True:
        try:
            choice = input("  Choice [n/m/d]: ").strip().lower()
        except EOFError:
            print("  (non-interactive — defaulting to new ID)")
            return next_id, "new"
        if choice == "n" or choice == "":
            return next_id, "new"
        elif choice == "d":
            return None, "discard"
        elif choice == "m":
            try:
                target = int(input(f"  Merge into SAM2 object ID: ").strip())
                if str(target) not in existing_mapping:
                    print(f"  ID {target} not in existing mapping. Try again.")
                    continue
                return target, "merge"
            except (ValueError, EOFError):
                print("  Invalid ID.")
                continue
        else:
            print("  Please enter n, m, or d.")


def handle_export_sam2(args):
    """
    Handle --export-sam2: write/update sam2_handoff.json so sam2_ui.py can load the project.

    IDs are stable across runs: existing assignments are preserved; new instances get new IDs.
    With --sam2-object-list: new instances are matched by name to the CSV; unmatched ones are
    resolved interactively.  Without --sam2-object-list: all new instances are auto-assigned.
    """
    project = _load_project(args.project)
    if project is None:
        return 1

    import time as _time_mod
    handoff_path = os.path.join(args.project, "sam2_handoff.json")

    # Load CSV for name-matching (only if provided)
    csv_name_map: dict = {}
    if getattr(args, "sam2_object_list", None):
        try:
            csv_name_map = _load_sam2_object_list(args.sam2_object_list)
            print(f"Loaded {len(csv_name_map)} entries from {args.sam2_object_list}")
        except Exception as e:
            print(f"ERROR: Could not load --sam2-object-list: {e}")
            return 1

    # Load existing handoff to preserve stable IDs
    existing_handoff: dict = {}
    if os.path.exists(handoff_path):
        try:
            with open(handoff_path) as f:
                existing_handoff = json.load(f)
            print(f"Updating existing sam2_handoff.json (preserving stable IDs)…")
        except Exception:
            print("WARNING: Could not read existing sam2_handoff.json — creating fresh.")

    existing_mapping: dict = existing_handoff.get("object_mapping", {})
    # Build reverse lookup: (concept_name, sam3_obj_id) → sam2_obj_id
    reverse_lookup: dict = {}
    for sam2_id_str, entry in existing_mapping.items():
        key = (entry.get("concept"), entry.get("instance_id"))
        reverse_lookup[key] = int(sam2_id_str)

    # IDs reserved in the handoff or the CSV — auto-assign always stays above max(csv_ids)
    # so gaps in the CSV (unassigned SAM2 objects) remain available for SAM2 to use natively.
    reserved_ids: set = {int(k) for k in existing_mapping}
    if csv_name_map:
        reserved_ids.update(e["id"] for e in csv_name_map.values())

    def _next_free_id(reserved: set) -> int:
        # Start above the maximum reserved ID so new SAM3-derived IDs never collide
        # with CSV IDs (even unassigned gaps).
        i = max(reserved, default=0) + 1
        while i in reserved:
            i += 1
        return i

    next_id = _next_free_id(reserved_ids)
    new_mapping: dict = dict(existing_mapping)
    new_colors: dict = dict(existing_handoff.get("object_colors", {}))
    # Track which CSV IDs were consumed (by direct name-match or interactive merge-into)
    used_csv_ids: set = set()
    # Preserve existing covered_ids across runs
    existing_covered: dict = existing_handoff.get("sam2_covered_ids", {})
    new_covered: dict = dict(existing_covered)

    mask_format = getattr(project, "mask_format", "png")

    # Cross-concept name tracking for duplicate warnings
    seen_names_global: dict = {}

    for concept in project.concepts:
        seen_names_concept: set = set()
        for inst in concept.instances:
            if inst.deleted:
                continue

            mask_dir_rel = os.path.join(
                "concepts", concept.name, "instances", str(inst.sam3_obj_id), "masks"
            )
            abs_mask_dir = os.path.join(project.project_dir, mask_dir_rel)
            if not os.path.isdir(abs_mask_dir):
                continue

            # Duplicate within concept
            if inst.user_name in seen_names_concept:
                print(f"WARNING: duplicate name '{inst.user_name}' in concept '{concept.name}' "
                      f"— exported as separate SAM2 objects.")
            seen_names_concept.add(inst.user_name)

            seen_names_global.setdefault(inst.user_name, []).append(
                (concept.name, inst.sam3_obj_id))

            key = (concept.name, inst.sam3_obj_id)
            if key in reverse_lookup:
                # Already assigned in a previous run — keep stable
                sam2_id = reverse_lookup[key]
                color = inst.get_effective_color(concept.color_rgb or (200, 200, 200))
            elif csv_name_map and inst.user_name in csv_name_map:
                # Name-matched to CSV entry
                csv_entry = csv_name_map[inst.user_name]
                sam2_id = csv_entry["id"]
                color = tuple(csv_entry["color"])
                print(f"  Matched '{inst.user_name}' → SAM2 id {sam2_id} (from CSV)")
                reserved_ids.add(sam2_id)
                used_csv_ids.add(sam2_id)
            elif csv_name_map:
                # CSV provided but no match — interactive resolution
                inst_label = f"concept={concept.name}, name='{inst.user_name}'"
                sam2_id, action = _resolve_unmatched_interactively(
                    inst_label, new_mapping, next_id)
                if action == "discard":
                    print(f"  Discarded '{inst.user_name}'")
                    continue
                color = inst.get_effective_color(concept.color_rgb or (200, 200, 200))
                if action == "new":
                    reserved_ids.add(sam2_id)
                    next_id = _next_free_id(reserved_ids)
                elif action == "merge":
                    used_csv_ids.add(sam2_id)  # merged into an existing (possibly CSV) ID
            else:
                # No CSV — auto-assign
                sam2_id = next_id
                color = inst.get_effective_color(concept.color_rgb or (200, 200, 200))
                reserved_ids.add(sam2_id)
                next_id = _next_free_id(reserved_ids)

            new_mapping[str(sam2_id)] = {
                "concept": concept.name,
                "instance_id": inst.sam3_obj_id,
                "name": inst.user_name,
                "mask_dir_rel": mask_dir_rel,
                "mask_filename_pattern": "{frame:06d}." + mask_format,
            }
            new_colors[str(sam2_id)] = list(color)

    # Warn about cross-concept name duplicates
    for name, locs in seen_names_global.items():
        if len(locs) > 1:
            locs_str = ", ".join(f"{c}/{i}" for c, i in locs)
            print(f"WARNING: name '{name}' appears in multiple concepts ({locs_str}) "
                  f"— these are separate SAM2 objects with the same display name.")

    # Prompt about uncovered CSV entries (CSV IDs that were never matched or merged into).
    # These are SAM2 objects the user may want to declare "done by SAM3" (read-only in UI,
    # no re-segmentation) even though no SAM3 instance was directly assigned to them.
    if csv_name_map:
        csv_id_to_name = {e["id"]: name for name, e in csv_name_map.items()}
        csv_id_to_color = {e["id"]: e["color"] for e in csv_name_map.values()}
        uncovered_csv_ids = {
            cid for cid in csv_id_to_name
            if cid not in used_csv_ids and str(cid) not in existing_covered
        }
        if uncovered_csv_ids:
            # Build a list of all SAM3-derived IDs for the user to reference
            sam3_derived_ids = sorted(
                int(k) for k in new_mapping
                if int(k) not in {e["id"] for e in csv_name_map.values()}
            )
            print(f"\n{'='*60}")
            print(f"The following SAM2 object(s) from your CSV were not matched to "
                  f"any SAM3 instance:")
            for cid in sorted(uncovered_csv_ids):
                print(f"  id={cid}  name='{csv_id_to_name[cid]}'")
            if sam3_derived_ids:
                print(f"SAM3-derived IDs assigned in this run: {sam3_derived_ids}")
            print(f"{'='*60}")

        for cid in sorted(uncovered_csv_ids):
            cname = csv_id_to_name[cid]
            ccolor = csv_id_to_color[cid]
            print(f"\nSAM2 object '{cname}' (id {cid}) was not matched.")
            print(f"  Mark as 'covered by SAM3'?")
            print(f"  (read-only in sam2_ui; excluded from re-segmentation)")
            try:
                ans = input("  [y/n, default n]: ").strip().lower()
            except EOFError:
                ans = "n"
            if ans != "y":
                continue

            sub_ids_str = ""
            if sam3_derived_ids:
                print(f"  Which SAM3 sub-IDs represent this object's mask as a union?")
                print(f"  (comma-separated from {sam3_derived_ids}, blank to skip union)")
                try:
                    sub_ids_str = input("  Sub-IDs: ").strip()
                except EOFError:
                    sub_ids_str = ""

            sub_ids: list = []
            for tok in sub_ids_str.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    sub_ids.append(int(tok))

            new_covered[str(cid)] = {
                "name": cname,
                "color": ccolor,
                "sam3_sub_ids": sub_ids,
            }
            print(f"  Marked '{cname}' (id {cid}) as covered."
                  + (f" Union of sub-IDs: {sub_ids}" if sub_ids else ""))

    now = _time_mod.strftime("%Y-%m-%d %H:%M:%S")
    handoff = {
        "version": 1,
        "created": existing_handoff.get("created", now),
        "updated": now,
        "sam3_project_dir": os.path.abspath(args.project),
        "original_video_path": project.video_path,
        "num_frames": project.num_frames,
        "object_mapping": new_mapping,
        "object_colors": new_colors,
        "sam2_covered_ids": new_covered,
        "sam2_results_subdir": "sam2_results",
    }
    with open(handoff_path, "w") as f:
        json.dump(handoff, f, indent=2)

    n = len(new_mapping)
    print(f"sam2_handoff.json written: {n} SAM2 object(s) → {handoff_path}")
    print(f"Open in sam2_ui.py: File → Import Masks → {args.project}")
    return 0


def handle_reencode(args):
    """Handle --reencode: re-encode the project video to MJPEG."""
    project = _load_project(args.project)
    if project is None:
        return 1

    print(f"Re-encoding video to MJPEG for fast random access...")
    print(f"Source: {project.video_path}")
    result = force_reencode_video_mjpeg(project.video_path, project.project_dir)
    project.mjpeg_video_path = result
    project.save()
    print(f"MJPEG video saved to: {result}")
    print("Project updated (mjpeg_video_path set).")
    return 0


def reset_project(args):
    """Handle --reset: flag completed concept(s) for full re-detection, then hand off
    to refine_project so the existing sentinel-driven redetect pass (deletion summary,
    confirmation, instance-dir wipe, re-detection) does the actual work. This is the
    CLI equivalent of the UI's 'Reset & Re-detect' button — unlike plain process mode's
    old -f behavior, it never runs silently: it always lists what will be lost and
    always confirms unless -f is given."""
    project = _load_project(args.project)
    if project is None:
        return 1

    reset_filter = None
    if args.reset:
        reset_filter = {n.strip() for n in args.reset.split(",") if n.strip()}
        targets = [c for c in project.concepts if c.name in reset_filter]
        missing = reset_filter - {c.name for c in targets}
        if missing:
            print(f"Warning: concept(s) not found, skipping: {', '.join(sorted(missing))}")
    else:
        targets = list(project.concepts)

    written = []
    for c in targets:
        instances_dir = os.path.join(project.project_dir, "concepts", c.name, "instances")
        has_data = os.path.isdir(instances_dir) and any(os.scandir(instances_dir))
        if not has_data:
            continue  # nothing on disk to reset; a plain run will detect it fresh anyway
        sentinel = os.path.join(project.project_dir, "concepts", c.name, "redetect")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        open(sentinel, "w").close()
        written.append(c.name)

    if not written:
        print("No concept(s) with existing instance data to reset.")
        return 0

    print(f"Flagged {len(written)} concept(s) for reset: {', '.join(written)}")
    # Scope the handoff to exactly these concepts, whether --reset was bare or filtered,
    # so unrelated concepts' pending refinements aren't swept in by this call.
    args.refine = ",".join(written)
    return refine_project(args)


def _iter_refinement_files(project, concept):
    """Yield (instance, refinements.json path, loaded dict) for every non-deleted
    instance of `concept` that has a refinements.json on disk. Skips unreadable files."""
    for inst in concept.instances:
        if inst.deleted:
            continue
        rpath = os.path.join(project.project_dir, "concepts", concept.name,
                             "instances", str(inst.sam3_obj_id), "refinements.json")
        if not os.path.exists(rpath):
            continue
        try:
            with open(rpath) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        yield inst, rpath, data


def _concept_refinement_count(project, concept) -> int:
    """Total saved refinement entries across all non-deleted instances of a concept."""
    return sum(len(data.get("refinements", []))
               for _inst, _rpath, data in _iter_refinement_files(project, concept))


def _mark_all_refinements_pending(project, concept) -> int:
    """Flip every saved refinement entry to propagated=False so the refine replay path
    re-applies ALL of them (not only newly-added ones) on top of fresh detection.
    Returns the number of entries re-flagged."""
    total = 0
    for _inst, rpath, data in _iter_refinement_files(project, concept):
        entries = data.get("refinements", [])
        if not entries:
            continue
        for r in entries:
            r["propagated"] = False
        with open(rpath, "w") as f:
            json.dump(data, f, indent=2)
        total += len(entries)
    return total


def _delete_concept_masks_keep_refinements(project, concept) -> None:
    """Remove regenerable outputs (per-instance masks/, concept-level cond_states/ and
    obj_ptr_priors.npz) but KEEP each instance's refinements.json and metadata, so the
    refine replay can regenerate masks while re-applying the saved refinement points."""
    concept_dir = os.path.join(project.project_dir, "concepts", concept.name)
    instances_dir = os.path.join(concept_dir, "instances")
    if os.path.isdir(instances_dir):
        for entry in os.scandir(instances_dir):
            if not entry.is_dir():
                continue
            masks_dir = os.path.join(entry.path, "masks")
            if os.path.isdir(masks_dir):
                shutil.rmtree(masks_dir)
    stale_cond = os.path.join(concept_dir, "cond_states")
    if os.path.isdir(stale_cond):
        shutil.rmtree(stale_cond)
    stale_prior = os.path.join(concept_dir, "obj_ptr_priors.npz")
    if os.path.exists(stale_prior):
        os.remove(stale_prior)


def _redo_choice(force: bool) -> str:
    """Ask how to handle existing refinements for a --redo run.
    Returns 'reapply', 'fresh', or 'cancel'. -f/--force auto-picks 'reapply'."""
    if force:
        return "reapply"
    print("\nThis will re-run detection. How should the saved refinement points be handled?")
    print("  [1] Reapply  — fresh detection, then replay ALL saved refinements on top (default)")
    print("  [2] Fresh    — fresh detection only; keep refinement files on disk but ignore")
    print("                 them this run (still viewable in sam3_ui.py)")
    print("  [3] Cancel")
    answer = input("Choose [1/2/3] (default 1): ").strip().lower()
    if answer in ("", "1", "reapply"):
        return "reapply"
    if answer in ("2", "fresh"):
        return "fresh"
    print("Aborted.")
    return "cancel"


def redo_project(args):
    """Handle --redo: re-run text-prompt detection like --reset, but respect existing
    refinement points and mask anchors instead of discarding them. Delegates the heavy
    lifting (frame extraction, model load, re-detection, replay, parallel dispatch) to
    the shared refine path; this function only sets up which concepts redetect fresh vs.
    redetect-and-reapply, and performs the destructive wipe after confirmation."""
    project = _load_project(args.project)
    if project is None:
        return 1

    # Resolve targets (bare --redo = every concept; --redo a,b = those).
    if args.redo:
        wanted = {n.strip() for n in args.redo.split(",") if n.strip()}
        targets = [c for c in project.concepts if c.name in wanted]
        missing = wanted - {c.name for c in targets}
        if missing:
            print(f"Warning: concept(s) not found, skipping: {', '.join(sorted(missing))}")
    else:
        targets = list(project.concepts)

    if not targets:
        print("No concept(s) to redo.")
        return 0

    ref_counts = {c.name: _concept_refinement_count(project, c) for c in targets}
    with_ref = [c for c in targets if ref_counts[c.name] > 0]
    without_ref = [c for c in targets if ref_counts[c.name] == 0]

    # A concept can have a "redetect" sentinel already on disk for a legitimate reason
    # unrelated to this run: the UI's "Reset & Re-detect" button writes the sentinel on
    # Save but deliberately leaves instances/refinements.json untouched until some future
    # --refine/--redo actually consumes it (see sam3_ui.py's flush comment). So a with_ref
    # concept that ALSO has a pending sentinel is genuinely ambiguous — it could be that
    # real, not-yet-run reset request, or a stale leftover from an earlier interrupted
    # redetect. Either way, _refine_project_impl checks the sentinel BEFORE checking for
    # pending refinements, so if the user now picks "reapply" for this concept, the
    # sentinel will silently win and the concept will be fully re-detected (fresh obj_ids,
    # losing names/history) instead of reapplying the listed refinements. We cannot safely
    # guess which the user wants, so just surface it here rather than deleting or ignoring
    # the sentinel automatically.
    sentinel_conflicts = [
        c for c in with_ref
        if os.path.exists(os.path.join(project.project_dir, "concepts", c.name, "redetect"))
    ]

    # Summarize and confirm.
    print(f"\n{'='*60}")
    print(f"--redo: re-detecting {len(targets)} concept(s); existing masks will be "
          f"regenerated.")
    for c in targets:
        n = ref_counts[c.name]
        note = f"{n} refinement point(s) will be reapplied" if n else "no refinements"
        print(f"  {c.name}  ({note})")
    if sentinel_conflicts:
        print(f"{'='*60}")
        print("WARNING: the following concept(s) already have a pending 'redetect' "
              "sentinel on disk (e.g. from the UI's 'Reset & Re-detect' button, or left "
              "over from an earlier interrupted run). If you choose 'reapply' below, "
              "that sentinel takes priority and these concept(s) will be FULLY "
              "re-detected instead — refinements will NOT be reapplied, and current "
              "instance names/ids will be lost:")
        for c in sentinel_conflicts:
            print(f"  {c.name}")
        print("If that's not what you want, cancel now, decide per-concept "
              "(--redo person,car for just the ones without a sentinel), or run "
              "--refine first to resolve the pending sentinel(s).")
    print(f"{'='*60}")

    if with_ref:
        choice = _redo_choice(args.force)
        if choice == "cancel":
            return 0
    else:
        if not _confirm("Delete existing masks and re-detect these concept(s)?", args.force):
            return 0
        choice = "fresh"

    # Route each concept through the shared refine path:
    #   - reapply: mark refinements pending + wipe masks (keep refinements.json) so the
    #     concept lands in the refine pass (fresh text detection + replay of all points).
    #   - fresh (or no refinements): write the redetect sentinel so the concept lands in
    #     the redetect pass. --redo's fresh mode preserves refinements.json across the
    #     wipe; --reset does not.
    #
    # Unlike --reset, --redo must NEVER discard mask_anchors/ (user-created anchors are
    # precious, unlike regenerable masks/). Set this unconditionally, before the sentinel
    # loop below, so it also covers `without_ref` concepts: they always get a redetect
    # sentinel regardless of `choice`, so without this flag a "reapply" run would route
    # them through _refine_project_impl's full `shutil.rmtree(instances_dir)` branch,
    # which deletes mask_anchors/ along with masks/.
    args._redo_keep_refinements = True

    for c in without_ref:
        sentinel = os.path.join(project.project_dir, "concepts", c.name, "redetect")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        open(sentinel, "w").close()

    if choice == "reapply":
        for c in with_ref:
            n = _mark_all_refinements_pending(project, c)
            _delete_concept_masks_keep_refinements(project, c)
            # NOTE: if this concept has a pending "redetect" sentinel (see
            # sentinel_conflicts warning above), it deliberately is NOT touched here —
            # we can't safely tell a stale leftover sentinel apart from a real,
            # not-yet-run UI reset request, so _refine_project_impl's sentinel-first
            # check is left to decide, and the user was warned above.
            print(f"  '{c.name}': flagged {n} refinement point(s) for replay, "
                  f"cleared old masks.")
    else:  # fresh
        for c in with_ref:
            sentinel = os.path.join(project.project_dir, "concepts", c.name, "redetect")
            os.makedirs(os.path.dirname(sentinel), exist_ok=True)
            open(sentinel, "w").close()

    # Scope the delegated run to exactly these concepts. We already confirmed above, so
    # suppress the inner deletion prompt.
    args.refine = ",".join(c.name for c in targets)
    args.force = True
    return refine_project(args)


def refine_project(args):
    """Replay saved refinement annotations, grouped by concept for correct non-overlapping."""
    project = _load_project(args.project)
    if project is None:
        return 1

    print(f"Loading project from {args.project}")

    from sam3_project import acquire_refine_lock, release_refine_lock
    acquire_refine_lock(project.project_dir)
    try:
        return _refine_project_impl(args, project)
    finally:
        release_refine_lock(project.project_dir)


def _refine_project_impl(args, project):
    # Purge mask dirs for user-deleted instances (deleted=True, absorbed_source=False).
    # Handles the rsync case: annotation machine marks deletions in project.json,
    # processing machine syncs the JSON but still has the old mask files on disk.
    # Absorbed sources are skipped here — their masks are deleted post-refinement by
    # replay_concept_refinements once the absorb anchor has been propagated.
    _purge_count = 0
    for _concept in project.concepts:
        for _inst in _concept.instances:
            if not _inst.deleted:
                continue
            if getattr(_inst, 'absorbed_source', False):
                continue  # kept until post-refinement cleanup
            _masks_dir = os.path.join(
                project.project_dir, "concepts", _concept.name,
                "instances", str(_inst.sam3_obj_id), "masks"
            )
            if os.path.isdir(_masks_dir):
                shutil.rmtree(_masks_dir)
                _purge_count += 1
    if _purge_count:
        print(f"Purged mask dirs for {_purge_count} deleted instance(s) "
              f"(marked deleted in project.json but still present on disk).")

    # Parse optional concept filter from --refine person,car
    refine_filter = None
    if args.refine:  # non-empty string = filter provided
        refine_filter = {n.strip() for n in args.refine.split(",") if n.strip()}

    # Apply --frame-dir if provided (pins frame cache, stored in project.json)
    frame_dir = getattr(args, "frame_dir", None)
    if frame_dir and project.frames_dir != frame_dir:
        project.frames_dir = frame_dir
        project.save()
        print(f"Frame cache pinned to: {frame_dir}")

    # --- Pass 1: detect concepts flagged for full re-detection ----------------
    # The UI's "Reset & Re-detect" button writes a sentinel file
    # `concepts/<name>/redetect` and clears all instances/masks.  We run
    # process_concept_detection (text-prompt re-detection) for these first,
    # then handle normal pending-annotation refinements below.
    #
    # We also pick up brand-new concepts here (status pending/error with zero
    # instances, e.g. saved via the UI's "Save as pending" box/text dialogs) —
    # otherwise --refine silently drops them since they have no instances to
    # scan for pending refinement points and no redetect sentinel either.
    concepts_to_redetect = {}  # concept.name -> concept
    for concept in project.concepts:
        if refine_filter and concept.name not in refine_filter:
            continue
        sentinel = os.path.join(project.project_dir, "concepts", concept.name, "redetect")
        if os.path.exists(sentinel):
            concepts_to_redetect[concept.name] = concept
        elif concept.status.value in ("pending", "error") and not concept.instances:
            concepts_to_redetect[concept.name] = concept

    # --- Pass 2: concepts with pending point annotations ---------------------
    # All instances of a concept must be refined together in one session so
    # SAM3's non-overlapping constraints are applied across all of them jointly.
    concepts_to_refine = {}  # concept.name -> (concept, [(instance, pending_entries)])
    for concept in project.concepts:
        if refine_filter and concept.name not in refine_filter:
            continue
        if concept.name in concepts_to_redetect:
            continue  # will be fully re-detected; no point in replaying old annotations
        pending_instances = []
        for inst in concept.instances:
            if inst.deleted:
                continue
            rpath = os.path.join(
                project.project_dir, "concepts", concept.name,
                "instances", str(inst.sam3_obj_id), "refinements.json"
            )
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                refinements = json.load(f).get("refinements", [])
            pending = [r for r in refinements if not r.get("propagated", True)]
            if pending:
                pending_instances.append((inst, pending))
        if pending_instances:
            concepts_to_refine[concept.name] = (concept, pending_instances)

    if not concepts_to_redetect and not concepts_to_refine:
        print("No pending annotations found. Add points in sam3_ui.py using 'Save for Batch'.")
        return 0

    from sam3_utils import ensure_frames_extracted
    resource_path = ensure_frames_extracted(
        project.video_path, project.get_frames_dir(), project.num_frames)

    if concepts_to_redetect:
        print(f"\nFound {len(concepts_to_redetect)} concept(s) flagged for full re-detection:")
        for cname in concepts_to_redetect:
            print(f"  {cname}")
    if concepts_to_refine:
        total_instances = sum(len(v[1]) for v in concepts_to_refine.values())
        print(f"\nFound {total_instances} instance(s) with pending annotations "
              f"across {len(concepts_to_refine)} concept(s):")
        for cname, (concept, pinsts) in concepts_to_refine.items():
            for inst, pending in pinsts:
                print(f"  {cname} / {inst.user_name}  "
                      f"(obj_id {inst.sam3_obj_id}, {len(pending)} annotation(s))")

    # Pre-flight: collect which concepts have existing mask files that will be deleted.
    deletion_summary = {}  # cname -> (n_inst_dirs, n_pending_annotations)
    for cname in concepts_to_redetect:
        instances_dir = os.path.join(project.project_dir, "concepts", cname, "instances")
        if os.path.isdir(instances_dir):
            inst_dirs = [e.name for e in os.scandir(instances_dir) if e.is_dir()]
            if inst_dirs:
                pending_total = 0
                for inst_id in inst_dirs:
                    rpath = os.path.join(instances_dir, inst_id, "refinements.json")
                    if os.path.exists(rpath):
                        with open(rpath) as _f:
                            refs = json.load(_f).get("refinements", [])
                        pending_total += sum(
                            1 for r in refs if not r.get("propagated", True))
                deletion_summary[cname] = (len(inst_dirs), pending_total)

    if deletion_summary:
        print(f"\n{'='*60}")
        print("The following concept(s) have existing mask files that will be")
        print("permanently deleted before re-detection runs:")
        for cname, (n_dirs, n_pending) in deletion_summary.items():
            line = f"  {cname}  ({n_dirs} instance dir(s))"
            if n_pending:
                line += f"  — {n_pending} unsaved annotation point(s) will also be lost"
            print(line)
        print(f"{'='*60}")
        if not _confirm("Delete these files and proceed with re-detection?", args.force):
            return 0

    devices = [d.strip() for d in args.device.split(",") if d.strip()]

    # Delete stale instance dirs for redetect concepts before dispatching workers
    # (user already confirmed above when deletion_summary was shown).
    # --redo's "fresh" mode sets _redo_keep_refinements: wipe regenerable outputs
    # (masks/) but preserve each instance's refinements.json so it stays viewable in
    # sam3_ui.py — old obj_ids usually recur on deterministic re-detection.
    keep_refinements = getattr(args, "_redo_keep_refinements", False)
    for cname in concepts_to_redetect:
        instances_dir = os.path.join(project.project_dir, "concepts", cname, "instances")
        if os.path.isdir(instances_dir):
            inst_dirs = [e.name for e in os.scandir(instances_dir) if e.is_dir()]
            if inst_dirs:
                if keep_refinements:
                    print(f"  Clearing masks for {len(inst_dirs)} instance(s) of '{cname}' "
                          f"(keeping refinement files)...")
                    for e in os.scandir(instances_dir):
                        if not e.is_dir():
                            continue
                        masks_dir = os.path.join(e.path, "masks")
                        if os.path.isdir(masks_dir):
                            shutil.rmtree(masks_dir)
                else:
                    print(f"  Deleting {len(inst_dirs)} stale instance dir(s) for '{cname}'...")
                    shutil.rmtree(instances_dir)

    total_tasks = len(concepts_to_redetect) + len(concepts_to_refine)
    if args.parallel > 1 and total_tasks > 1:
        rc, failed_cnames = refine_project_parallel(args, project, concepts_to_redetect, concepts_to_refine, devices)
        if failed_cnames:
            print(f"\nNote: leaving redetect sentinel/pending state in place for failed concept(s) "
                  f"so a re-run of --refine will retry them: {sorted(failed_cnames)}")
        # Remove redetect sentinels only for concepts that actually completed — workers
        # that errored (e.g. OOM) never finished, so clearing the sentinel here would
        # silently drop the concept from future --refine runs (it has no pending
        # point-annotations either, so nothing would ever re-trigger it).
        for cname in concepts_to_redetect:
            if cname in failed_cnames:
                continue
            sentinel = os.path.join(project.project_dir, "concepts", cname, "redetect")
            if os.path.exists(sentinel):
                os.remove(sentinel)
        # Workers own concept_metadata.json — use write_concept_metadata=False so the
        # parent's stale in-memory concept copy doesn't clobber the fresh data workers wrote.
        project.save(write_concept_metadata=False)
        cleanup_frames_dir(project)
        return rc

    sam3_model = load_sam3_model(model_name=_model_name_for(args), device=devices[0],
                                 use_fa3=args.use_fa3,
                                 max_cond_frames_in_attn=args.max_cond_frames_in_attn)
    orig_width, orig_height = project.frame_dimensions
    num_frames = project.num_frames
    errors = []

    # Re-detection pass
    for i, (cname, concept) in enumerate(concepts_to_redetect.items()):
        print(f"\n[redetect {i+1}/{len(concepts_to_redetect)}] Re-detecting concept '{cname}'...")

        def redetect_progress(frame_idx, nf, _cname=cname, _ci=i, _n=len(concepts_to_redetect)):
            if frame_idx % 200 == 0:
                pct = (frame_idx + 1) / nf * 100
                print(f"  [{_ci+1}/{_n}] {frame_idx+1}/{nf} frames ({pct:.1f}%)")

        sentinel = os.path.join(project.project_dir, "concepts", cname, "redetect")

        try:
            process_concept_detection(
                sam3_model=sam3_model,
                project=project,
                concept=concept,
                device=devices[0],
                progress_callback=redetect_progress,
                save_cond_states=args.save_cond_states,
                cache_size=args.cache_size,
            )
            if os.path.exists(sentinel):
                os.remove(sentinel)
            project.save()
            print(f"  Concept '{cname}' re-detected: {len(concept.instances)} instance(s).")
        except Exception as e:
            errors.append((cname, str(e)))
            print(f"  Error re-detecting concept '{cname}': {e}")
            import traceback
            traceback.print_exc()

    # Refinement pass (serial)
    for i, (cname, (concept, instances_with_pending)) in enumerate(concepts_to_refine.items()):
        print(f"\n[refine {i+1}/{len(concepts_to_refine)}] Replaying refinements for concept "
              f"'{cname}' ({len(instances_with_pending)} instance(s))...")

        def progress_callback(frame_idx, nf, _cname=cname, _ci=i, _n=len(concepts_to_refine)):
            if frame_idx % 200 == 0:
                pct = (frame_idx + 1) / nf * 100
                print(f"  [{_ci+1}/{_n}] {frame_idx+1}/{nf} frames ({pct:.1f}%)")

        try:
            replay_concept_refinements(
                concept=concept,
                instances_with_pending=instances_with_pending,
                resource_path=resource_path,
                orig_width=orig_width,
                orig_height=orig_height,
                num_frames=num_frames,
                sam3_model=sam3_model,
                project_dir=project.project_dir,
                progress_callback=progress_callback,
                device=devices[0],
                restore_cond_states=args.restore_cond,
                save_cond_states=args.save_cond_states,
                save_obj_ptr_prior=args.save_obj_ptr_prior,
                mask_format=project.mask_format,
                cache_size=args.cache_size,
                preload_original_masks=args.preload_original_masks,
                new_instance_policy=args.new_instance_policy,
            )
            absorbed_ids_removed, marked_completed = _finalize_concept_refinement(
                concept, project.project_dir)
            if absorbed_ids_removed:
                print(f"  Removed {len(absorbed_ids_removed)} absorbed source(s) from "
                      f"project metadata: obj_ids {sorted(absorbed_ids_removed)}")
            if marked_completed:
                print(f"  Concept '{cname}' has no pending annotations left — marking completed.")
        except Exception as e:
            errors.append((cname, str(e)))
            print(f"  Error refining concept '{cname}': {e}")
            import traceback
            traceback.print_exc()

    project.save()
    print(f"\n{'='*60}")
    if errors:
        print(f"Done with {len(errors)} error(s):")
        for name, err in errors:
            print(f"  {name}: {err}")
    else:
        print("All done.")
    print(f"{'='*60}")
    cleanup_frames_dir(project)
    return 1 if errors else 0


def export_project(args):
    """Export project to video"""
    project = _load_project(args.project)
    if project is None:
        return 1

    print(f"Exporting video to {args.export}")
    compositor = DynamicFrameCompositor(project, use_mjpeg=True)

    def progress_callback(frame_idx, num_frames):
        if frame_idx % 200 == 0:
            progress = (frame_idx + 1) / num_frames * 100
            print(f"Exporting: {frame_idx+1}/{num_frames} frames ({progress:.1f}%)")

    try:
        compositor.export_video(args.export, progress_callback=progress_callback)
        print(f"\nVideo exported to {args.export}")
    finally:
        compositor.release()

    return 0


def handle_status(args):
    """Handle --status: print a summary of the project and suggest next commands."""
    project = _load_project(args.project)
    if project is None:
        return 1

    W = 60
    sep = "=" * W
    thin = "-" * W

    # ── Running job check ─────────────────────────────────────────
    from sam3_project import check_refine_lock
    import time as _time_mod
    lock_info = check_refine_lock(args.project)

    # ── Project header ────────────────────────────────────────────
    print(sep)
    print("PROJECT STATUS")
    print(sep)
    print(f"  Directory  : {os.path.abspath(args.project)}")
    if getattr(project, 'last_saved', None):
        print(f"  Last saved : {project.last_saved}")
    print(f"  Video      : {project.video_path}")
    w, h = project.frame_dimensions
    print(f"  Resolution : {w}x{h}  ({project.num_frames} frames @ {project.fps:.2f} fps)")
    print(f"  Mask format: {getattr(project, 'mask_format', 'png')}")

    # Frame cache
    effective_frames_dir = project.get_frames_dir()
    if os.path.isdir(effective_frames_dir):
        n_jpg = sum(1 for f in os.listdir(effective_frames_dir) if f.endswith('.jpg'))
        kind = "persistent" if project.frames_dir else "temporary"
        print(f"  Frame cache: {effective_frames_dir}  ({n_jpg} JPEGs, {kind})")
    else:
        print(f"  Frame cache: (none — will be extracted on next run)")

    mjpeg = getattr(project, 'mjpeg_video_path', None)
    if mjpeg and os.path.exists(mjpeg):
        print(f"  MJPEG video: {mjpeg}")

    # Auxiliary files
    handoff = os.path.join(args.project, "sam2_handoff.json")
    metrics = os.path.join(args.project, "quality_metrics.npz")
    print(f"  sam2_handoff.json : {'YES' if os.path.exists(handoff) else 'no'}")
    print(f"  quality_metrics   : {'YES' if os.path.exists(metrics) else 'no'}")

    # ── Per-concept breakdown ─────────────────────────────────────
    print()
    print(thin)
    print(f"  CONCEPTS ({len(project.concepts)} total)")
    print(thin)

    # Collect pending work while iterating
    concepts_pending_detection = []   # status != completed
    concepts_to_redetect = []         # have redetect sentinel
    concepts_with_pending_refine = [] # have unpropagated refinements

    for concept in project.concepts:
        status_val = concept.status.value
        live_instances = [i for i in concept.instances if not i.deleted]
        total_masks = sum(i.num_frames_with_mask for i in live_instances)

        print(f"\n  Concept: '{concept.name}'")
        print(f"    Prompt         : {concept.prompt_label()}")
        print(f"    Status         : {status_val}")
        if status_val == "completed" and getattr(concept, 'completed_at', None):
            print(f"    Completed at   : {concept.completed_at}")
        print(f"    Detection frame: {concept.detection_frame}")
        print(f"    Instances (live): {len(live_instances)}")
        if live_instances:
            print(f"    Total mask frames: {total_masks}")
            for inst in live_instances:
                periods = f"{len(inst.continuous_periods)} period(s)" if inst.continuous_periods else ""
                print(f"      [{inst.sam3_obj_id}] '{inst.user_name}'  "
                      f"frames={inst.num_frames_with_mask}  score={inst.score:.3f}  {periods}")

        # Check redetect sentinel
        sentinel = os.path.join(args.project, "concepts", concept.name, "redetect")
        if os.path.exists(sentinel):
            print(f"    [!] Flagged for re-detection (redetect sentinel present)")
            concepts_to_redetect.append(concept.name)

        # Check pending refinements per instance
        pending_count = 0
        for inst in live_instances:
            rpath = os.path.join(
                args.project, "concepts", concept.name,
                "instances", str(inst.sam3_obj_id), "refinements.json"
            )
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                refs = json.load(f).get("refinements", [])
            n_pending = sum(1 for r in refs if not r.get("propagated", True))
            if n_pending:
                print(f"    [!] Instance '{inst.user_name}': "
                      f"{n_pending} pending annotation(s) (not yet propagated)")
                pending_count += n_pending
        if pending_count:
            concepts_with_pending_refine.append(concept.name)

        if status_val != "completed":
            concepts_pending_detection.append(concept.name)

    # ── CUDA devices ──────────────────────────────────────────────
    print()
    print(thin)
    print("  CUDA DEVICES")
    print(thin)
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            print("  No CUDA GPUs detected (CPU only)")
            gpu_info = []
        else:
            gpu_info = []
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024 ** 3)
                try:
                    free_bytes = torch.cuda.mem_get_info(i)[0]
                    free_gb = free_bytes / (1024 ** 3)
                    mem_str = f"{free_gb:.1f}/{total_gb:.1f} GB free"
                except Exception:
                    mem_str = f"{total_gb:.1f} GB total"
                gpu_info.append((f"cuda:{i}", props.name, mem_str))
                print(f"  cuda:{i}  {props.name}  {mem_str}")
    except ImportError:
        print("  (torch not importable — cannot query GPUs)")
        gpu_info = []
        n_gpus = 0

    # ── Action summary ────────────────────────────────────────────
    print()
    print(sep)
    print("ACTION SUMMARY")
    print(sep)

    script = f"python {os.path.basename(__file__)}"
    proj_arg = f'--project "{os.path.abspath(args.project)}"'

    # Build device/parallel recommendation
    if n_gpus == 0:
        device_arg = "--device cpu"
        parallel_note = ""
    elif n_gpus == 1:
        device_arg = "--device cuda:0"
        parallel_note = "  (consider --parallel 2 if GPU has >8 GB free; each worker uses ~4 GB)"
    else:
        device_arg = "--device all  # uses all {} GPU(s): {}".format(
            n_gpus, ", ".join(g[0] for g in gpu_info))
        suggested_parallel = n_gpus * 2
        parallel_note = (
            f"  --parallel {suggested_parallel}  "
            f"# {n_gpus} GPU(s) × 2 workers each — reduce if GPU memory is tight (~4 GB per worker)"
        )

    needs_action = False

    if concepts_pending_detection:
        needs_action = True
        print(f"\n  [DETECTION NEEDED] {len(concepts_pending_detection)} concept(s) not completed:")
        for name in concepts_pending_detection:
            print(f"    - {name}")
        cmd = f"    {script} {proj_arg} \\\n      {device_arg}"
        if n_gpus > 1:
            cmd += f" \\\n      {parallel_note.strip()}"
        print(f"\n  Suggested command:\n{cmd}")
        if parallel_note and n_gpus == 1:
            print(f"  Note: {parallel_note.strip()}")

    if concepts_to_redetect:
        needs_action = True
        print(f"\n  [RE-DETECTION NEEDED] {len(concepts_to_redetect)} concept(s) flagged:")
        for name in concepts_to_redetect:
            print(f"    - {name}")
        filter_arg = "--refine " + ",".join(concepts_to_redetect)
        cmd = f"    {script} {proj_arg} \\\n      {filter_arg} \\\n      {device_arg}"
        if n_gpus > 1:
            cmd += f" \\\n      {parallel_note.strip()}"
        print(f"\n  Suggested command:\n{cmd}")

    if concepts_with_pending_refine:
        needs_action = True
        print(f"\n  [REFINEMENT NEEDED] {len(concepts_with_pending_refine)} concept(s) "
              f"have pending point annotations:")
        for name in concepts_with_pending_refine:
            print(f"    - {name}")
        filter_arg = "--refine " + ",".join(concepts_with_pending_refine)
        # Refinement is single-GPU (sequential per concept by design); no --parallel tip
        cmd = f"    {script} {proj_arg} \\\n      {filter_arg} \\\n      {device_arg.split('#')[0].strip()}"
        print(f"\n  Suggested command:\n{cmd}")

    if not needs_action:
        print("\n  All concepts completed. No pending refinements.")
        if not os.path.exists(handoff):
            print(f"\n  [OPTIONAL] Export to SAM2 handoff for use in sam2_ui.py:")
            print(f"    {script} {proj_arg} --export-sam2")
        if not os.path.exists(metrics):
            print(f"\n  [OPTIONAL] (Re)compute quality metrics:")
            print(f"    {script} {proj_arg}")

    if lock_info:
        elapsed = int(_time_mod.time() - lock_info.get("started", _time_mod.time()))
        h_e, rem = divmod(elapsed, 3600)
        m_e, s_e = divmod(rem, 60)
        elapsed_str = (f"{h_e}h {m_e}m {s_e}s" if h_e else
                       f"{m_e}m {s_e}s" if m_e else f"{s_e}s")
        print(f"\n  [RUNNING JOB] PID {lock_info['pid']} has been running for {elapsed_str}")

    print(sep)
    return 0


def main():
    os.umask(0o002)  # ensure group-writable output for shared results dirs
    parser = argparse.ArgumentParser(
        description="SAM3 batch processing script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument("--concepts",
                        help="Input concepts: JSON file path or comma-separated names. "
                             "Names without a JSON file use the name as both name and text prompt.")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated concept names to process (subset of --concepts or project).")
    parser.add_argument("--max-instances", type=int, default=-1, dest="max_instances",
                        help="Cap on how many instances SAM3 may create per concept, "
                             "applied during initial detection only (default: -1 = no "
                             "limit). When explicitly set to a positive value, it "
                             "overrides every concept about to run initial detection "
                             "this invocation, including a JSON file's per-concept "
                             "'max_instances' field and a pre-existing pending/error "
                             "concept's previously-stored value. Has no effect on "
                             "--refine: a refinement run replays every instance the "
                             "user has kept regardless of this cap, even if that "
                             "exceeds it.")
    parser.add_argument("--export", help="Export segmented video to this path")
    parser.add_argument("--refine", nargs="?", const="", default=None,
                        metavar="CONCEPT,...",
                        help="Replay saved point annotations. "
                             "Bare --refine refines all concepts; "
                             "--refine person,car refines only those concepts.")
    parser.add_argument("--reset", nargs="?", const="", default=None,
                        metavar="CONCEPT,...",
                        help="Discard a completed concept's instances, renames, mask "
                             "anchors, and refinement points, then fully re-detect it "
                             "from scratch. Bare --reset targets every completed "
                             "concept; --reset person,car targets only those. Prints "
                             "what will be lost and asks for confirmation unless -f is "
                             "given. Equivalent to the UI's 'Reset & Re-detect' button.")
    parser.add_argument("--redo", nargs="?", const="", default=None,
                        metavar="CONCEPT,...",
                        help="Re-run text-prompt detection like --reset, but respect "
                             "existing refinement points and mask anchors instead of "
                             "discarding them. Bare --redo targets every concept; "
                             "--redo person,car targets only those. When refinements "
                             "exist it offers a choice: (1) redo + reapply all saved "
                             "refinements on top of fresh detection, or (2) redo fresh "
                             "while keeping the refinement files on disk (ignored this "
                             "run, still viewable in sam3_ui.py). Prompts for "
                             "confirmation unless -f is given (-f picks 'reapply').")
    parser.add_argument("--delete", default=None,
                        metavar="CONCEPT[:INST,...]",
                        help="Delete a concept or specific instances. "
                             "--delete person removes the whole concept. "
                             "--delete person:inst1,inst2 removes only those instances. "
                             "Prompts for confirmation unless -f is given.")
    parser.add_argument("--frame-dir", default=None, dest="frame_dir",
                        metavar="DIR",
                        help="Directory for extracted video frames (JPEGs). "
                             "If set, frames are kept after the session and the path is "
                             "stored in project metadata for reuse on the next run. "
                             "If omitted, a temporary directory under /tmp is used and "
                             "deleted automatically when the session ends.")
    parser.add_argument("--mask-format", default="png", choices=["png", "npz"],
                        dest="mask_format",
                        help="Mask storage format for new projects (default: png). "
                             "'npz' is compressed and faster to read/write. "
                             "Existing projects preserve their original format.")
    parser.add_argument("--sam-version", default="3", choices=["3", "3.1"],
                        dest="sam_version",
                        help="SAM3 model version (default: 3). '3.1' uses the Object "
                             "Multiplex checkpoint: joint multi-object tracking that is "
                             "much faster when a concept detects many instances.")
    parser.add_argument("--use-fa3", action="store_true", dest="use_fa3",
                        help="Enable FlashAttention-3 fp8 kernels for SAM3.1 "
                             "(--sam-version 3.1 only). Requires a Hopper GPU "
                             "(H100/H200, compute capability 9.0+); fails on "
                             "Ampere/Ada GPUs such as the L40S.")
    parser.add_argument("--cache-size", type=int, default=50,
                        dest="cache_size",
                        help="Number of decoded video frames held in the LRU cache during "
                             "propagation (default: 50). Higher values reduce re-decoding "
                             "on fast disks at the cost of more RAM (~100 MB per 10 frames "
                             "for 1600x1200 video). Lower values save RAM on slow machines.")
    parser.add_argument("--max-cond-frames", type=int, default=-1,
                        dest="max_cond_frames_in_attn",
                        help="Number of conditioning frames the tracker attends to per "
                             "forward pass (default: -1 = no limit, matching SAM3's "
                             "built-in default).  All user correction anchors are always "
                             "attended to.  Set to a small positive value (e.g. 4-8) to "
                             "bound attention compute AND GPU memory when a concept has "
                             "many annotation/mask-anchor frames: cond frames outside the "
                             "selected window are also offloaded to CPU between propagation "
                             "steps (see _offload_unselected_cond_frames in sam3_pipeline.py) "
                             "and transparently reloaded onto GPU only when propagation "
                             "comes back near them. At -1 this offload is a no-op, since "
                             "every cond frame is always in the selected window.")
    parser.add_argument("--device", default="cuda:0",
                        help="Device, comma-separated list, or 'all' to use every available GPU "
                             "(e.g. cuda:2, cuda:2,cuda:3, or all). Default: cuda:0")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent worker processes for concept "
                             "processing (default: 1 = sequential). Workers are "
                             "assigned round-robin to --device entries; multiple "
                             "workers per GPU pays off while a single job leaves "
                             "the GPU under-utilized (~2 per GPU currently). Each "
                             "worker loads its own model copy (~4 GB GPU memory).")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Skip confirmation prompts. Only meaningful alongside "
                             "--delete, --delete-frames, --refine, --reset, or --redo "
                             "— plain process mode never needs it, since it never "
                             "touches an already-completed concept's data.")
    parser.add_argument("--restore-cond", action="store_true", dest="restore_cond",
                        help="Restore saved maskmem cond-frame states from the prior refinement "
                             "round before propagating. Off by default: prior states may encode "
                             "stale/wrong segmentation and bias corrections. Enable only when the "
                             "prior round was good and you want continuity across sessions.")
    parser.add_argument("--preload-original-masks", action="store_true",
                        dest="preload_original_masks",
                        help="Reinforce every live instance from its own on-disk mask (saved at "
                             "the end of the *previous* round) as an extra conditioning frame at "
                             "its first detection frame. Off by default: only conditioning frames "
                             "the user actually created (points, boxes, mask anchors) are used, "
                             "so an instance with no annotation this round and no fresh "
                             "redetection under its own id is simply left untouched, rather than "
                             "silently reinforced with stale prior-round content the user never "
                             "sees.")
    parser.add_argument("--new-instance-policy", default="allow",
                        choices=["allow", "latent", "disallow"], dest="new_instance_policy",
                        help="What to do with an object first detected mid-video during "
                             "--refine (not present at the initial detection frame, not a "
                             "known live/deleted/absorbed instance). 'allow' (default): "
                             "promote it to a new instance under a freshly assigned id. "
                             "'latent': let it keep tracking/competing for the rest of this "
                             "run but never persist or promote it. 'disallow': cap SAM3's "
                             "max_num_objects so it can never be created in the first place.")
    parser.add_argument("--save-cond-states", action="store_true", dest="save_cond_states",
                        help="Save maskmem cond-frame states to concepts/<name>/cond_states/ "
                             "after detection/refinement, for a future --restore-cond round. "
                             "Off by default: these states are only consumed by --restore-cond "
                             "and can be sizeable (100s of MB per concept). "
                             "Planned to be phased out in favor of scoped flags such as "
                             "--save-obj-ptr-prior.")
    parser.add_argument("--save-obj-ptr-prior", action="store_true", dest="save_obj_ptr_prior",
                        help="After a refinement round only (never at initial detection), save "
                             "each instance's obj_ptr — a small 256-d object-identity embedding — "
                             "from cond frames where the object was judged present, to "
                             "concepts/<name>/obj_ptr_priors.npz. Saves BOTH the mean over those "
                             "cond frames ('obj_ptr', one row per instance name) and the raw "
                             "per-cond-frame vectors ('raw_obj_ptr', with parallel 'raw_names'/"
                             "'raw_obj_id'/'raw_frame_idx' index arrays). Overwritten each "
                             "refinement round (newer, human-corrected anchors supersede older "
                             "ones). Intended for a future cross-video object/category prior; "
                             "unlike --save-cond-states this does not save maskmem_features "
                             "(~650x larger) and has no effect on tracking or detection today.")
    parser.add_argument("--delete-frames", action="store_true", dest="delete_frames",
                        help="Delete the persistent frame cache registered for this project "
                             "(shows path and size, then prompts for confirmation unless -f).")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode the project video to MJPEG (all-keyframe) format for "
                             "fast random-access playback.  Output: <project>/video_mjpeg.avi. "
                             "Prefer the auto-extracted frame cache when disk space allows; "
                             "use --reencode when you want smaller storage than raw JPEGs.")
    parser.add_argument("--export-sam2", action="store_true", dest="export_sam2",
                        help="Write (or update) sam2_handoff.json in the project directory so "
                             "sam2_ui.py can load the SAM3 project directly.  Masks are read "
                             "from their original SAM3 paths — nothing is copied.  Running again "
                             "adds new instances without changing existing ID assignments.")
    parser.add_argument("--sam2-object-list", default=None, dest="sam2_object_list",
                        metavar="OBJECTS_CSV",
                        help="CSV file exported from sam2_ui.py ('Export Object List') containing "
                             "id, name, color_r, color_g, color_b columns.  When provided with "
                             "--export-sam2, new SAM3 instances are matched to entries by name "
                             "(exact match).  Unmatched instances are resolved interactively: "
                             "assign new ID, merge into an existing ID, or discard.  Without this "
                             "flag, all new instances are auto-assigned fresh IDs.")
    parser.add_argument("--no-quality-metrics", action="store_true", dest="no_quality_metrics",
                        help="Skip the quality-metrics computation that otherwise runs by "
                             "default after detection / --refine / --redo / --reset.  By "
                             "default those operations compute inter-frame-change / "
                             "background-ratio / overlap-ratio metrics for all masks in the "
                             "project and save quality_metrics.npz to the project directory.")
    parser.add_argument("--status", action="store_true",
                        help="Print a summary of the project: concepts, instances, pending "
                             "refinements/re-detection, available CUDA devices, and suggested "
                             "next commands. Does not load the SAM3 model.")

    args = parser.parse_args()

    _exclusive = [n for n, v in (("--refine", args.refine),
                                 ("--reset", args.reset),
                                 ("--redo", args.redo)) if v is not None]
    if len(_exclusive) > 1:
        parser.error(f"{' and '.join(_exclusive)} are mutually exclusive; "
                     "run them separately.")

    if args.force and not any([
        args.delete_frames, args.delete is not None,
        args.refine is not None, args.reset is not None, args.redo is not None,
    ]):
        parser.error("-f/--force has nothing to confirm-skip here: it only applies "
                     "to --delete, --delete-frames, --refine, --reset, or --redo.")

    if args.device.strip().lower() == "all":
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            print("Error: --device all specified but no CUDA devices found.")
            sys.exit(1)
        args.device = ",".join(f"cuda:{i}" for i in range(n))
        print(f"--device all: found {n} GPU(s): {args.device}")

    if args.status:
        return handle_status(args)

    if args.delete_frames:
        return handle_delete_frames(args)

    if args.reencode:
        return handle_reencode(args)

    if args.delete is not None:
        return handle_delete(args)

    if args.export:
        return export_project(args)

    if args.export_sam2:
        # No quality-metrics recompute here: --export-sam2 only writes sam2_handoff.json
        # and never changes SAM3 masks, so quality_metrics.npz from detection/refine
        # still applies.
        return handle_export_sam2(args)

    want_quality = not args.no_quality_metrics

    if args.redo is not None:
        rc = redo_project(args)
        if rc == 0 and want_quality:
            return handle_quality_metrics(args)
        return rc

    if args.reset is not None:
        rc = reset_project(args)
        if rc == 0 and want_quality:
            return handle_quality_metrics(args)
        return rc

    if args.refine is not None:
        rc = refine_project(args)
        if rc == 0 and want_quality:
            return handle_quality_metrics(args)
        return rc

    rc = process_project(args)
    if rc == 0 and want_quality:
        return handle_quality_metrics(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
