#!/usr/bin/env python3
"""
SAM3 UI - Hierarchical video object segmentation interface.

Features:
- Text-based concept detection (e.g., "person", "car")
- Hierarchical object management (concept → instances → user names)
- Interactive refinement with positive/negative points
- Dynamic frame compositing (no pre-rendering)
- Instance operations (rename, delete, merge)
"""

import colorsys
import copy
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, List
import threading
import time as _time
import torch
import gaze_overlay

# Lightweight performance logger — remove when profiling is done.
_PERF_LOG: Dict[str, float] = {}

class _T:
    """Context manager: accumulates elapsed time into _PERF_LOG."""
    __slots__ = ('label', '_t0')
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        self._t0 = _time.perf_counter()
        return self
    def __exit__(self, *_):
        _PERF_LOG[self.label] = _PERF_LOG.get(self.label, 0.0) + (_time.perf_counter() - self._t0)

def _perf_reset():
    _PERF_LOG.clear()

def _perf_report(action: str, total: float):
    parts = " | ".join(f"{k}={v*1000:.1f}ms" for k, v in _PERF_LOG.items())
    print(f"[PERF] {action} total={total*1000:.1f}ms | {parts}")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import SAM3 modules
from sam3_project import SAM3Project, SAM3Concept, SAM3Instance, ConceptStatus, check_refine_lock
from sam3_pipeline import (
    load_sam3_model, get_video_info,
    process_concept_detection,
    compute_continuous_periods, online_replay_concept_refinements,
    replay_concept_refinements, ensure_bf16_autocast,
)
from sam3_utils import DynamicFrameCompositor, generate_concept_color, validate_text_prompt, export_to_sam2_format, load_sam3_mask, retire_deleted_handoff_ids


# ── App-level settings (persisted across sessions, independent of any project) ─
_APP_SETTINGS_PATH = Path.home() / ".sam3_ui_settings.json"


def _load_app_settings() -> dict:
    try:
        if _APP_SETTINGS_PATH.exists():
            return json.loads(_APP_SETTINGS_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_app_settings(settings: dict) -> None:
    try:
        _APP_SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print(f"Warning: could not save app settings: {e}")


class SAM3VideoUI:
    """Main UI application for SAM3 hierarchical video segmentation"""

    def __init__(self, root):
        self.root = root
        self.root.title("SAM3 Video Segmentation")
        self.root.geometry("1600x900")

        # State
        self.project: Optional[SAM3Project] = None
        self.compositor: Optional[DynamicFrameCompositor] = None
        self.sam3_model = None
        self.available_devices = self._detect_devices()
        default_device = self.available_devices[0][0] if self.available_devices else "cpu"
        self.device = default_device
        self.device_var = tk.StringVar(value=default_device)
        # Live SAM3 sessions kept open for fast online refinement (concept_name -> session_id)
        self.active_sessions: Dict[str, str] = {}

        # Video state
        self.current_frame_idx = 0
        self.video_path: Optional[str] = None
        self.num_frames = 0
        self.frame_dimensions = (0, 0)
        self.fps = 0.0
        self.gaze_points_by_frame = None  # {frame_idx: [(x, y), ...]} from a loaded gaze CSV
        self.gaze_csv_path: Optional[str] = None
        self.gaze_world_path: Optional[str] = None

        # UI state
        self.selected_concept: Optional[SAM3Concept] = None
        self.selected_instance: Optional[SAM3Instance] = None
        # All currently selected (concept, instance) pairs; may have more than one.
        self.selected_instances: List[tuple] = []  # [(SAM3Concept, SAM3Instance), ...]
        self.point_removal_mode = False
        self.box_draw_mode = False          # left-click-drag draws a box instead of a point
        self.refinement_points: List[tuple] = []  # [(x, y, is_positive), ...]
        self.refinement_boxes: List[tuple] = []   # [(x1, y1, x2, y2), ...] in display coords
        self._box_draw_start = None         # (cx, cy) in display coords at drag start
        self._box_draw_canvas_id = None     # canvas item id for live-drag preview rectangle
        self.undo_stack: List[dict] = []  # {'type': 'point'|'box'|'remove_point'|'remove_box'|'delete_instance', ...}
        self.redo_stack: List[dict] = []
        # Absorb wizard state (None when no wizard is active)
        self._absorb_wizard: Optional[dict] = None
        self._wizard_pending_points: List[tuple] = []  # points placed in wizard point mode
        self._wizard_pending_boxes: List[tuple] = []   # boxes drawn in wizard point/box mode
        # True when instance renames/deletes/adds are in memory but not yet written to JSON.
        # Flushed by save_points_for_batch, apply_refinement, and the explicit Save Project.
        self._metadata_dirty: bool = False

        # Playback
        self.playing = False
        self.playback_speed = tk.DoubleVar(value=1.0)
        self.zoom_buttons: Dict[int, object] = {}
        self.speed_buttons: Dict[float, object] = {}
        # Pending after-id for debounced canvas-resize redraws
        self._canvas_resize_pending = None
        # True while a _update_frame_from_play call is queued but not yet executed.
        # Prevents the background thread from flooding the event queue when rendering
        # (compositor + mask I/O) is slower than the target frame period.
        self._render_pending: bool = False

        # Nav panel row references — stored so image mode can hide them.
        self._nav_row1: Optional[tk.Frame] = None   # play/pause + frame nav
        self._nav_row2: Optional[tk.Frame] = None   # presence bar
        self._nav_row3: Optional[tk.Frame] = None   # zoom + speed
        self._nav_row4: Optional[tk.Frame] = None   # quality colorbars (overlap + bg)
        self._nav_row5: Optional[tk.Frame] = None   # quality colorbar (inter-frame change)
        self._export_btn: Optional[tk.Button] = None  # "Export Video/Image" button

        # Timeline zoom (slider window)
        self.slider_zoom_level = tk.IntVar(value=1)
        self.slider_window_center = 0
        self.zoom_jump_scheduled = None

        # Presence bar frame-cache: (concept_name, obj_id) -> set of frame indices
        self._presence_cache: Dict[tuple, set] = {}
        # When True, next _update_presence_bar() regenerates the background image.
        # When False, only the playhead canvas items are moved — O(1).
        self._presence_bar_dirty: bool = True
        # Cached PhotoImage of the bar background (blocks only, no playhead).
        # Stored here to prevent GC; updated only when dirty.
        self._presence_bar_photo = None
        self._presence_bar_img_id = None
        # Same pattern for quality colorbars.
        self._quality_overlap_photo = None
        self._quality_overlap_img_id = None
        self._quality_bg_photo = None
        self._quality_bg_img_id = None
        self._quality_inter_photo = None
        self._quality_inter_img_id = None

        # Quality metrics (loaded from quality_metrics.npz when available)
        self.quality_bg: Optional[List[float]] = None      # background ratio per frame
        self.quality_overlap: Optional[List[float]] = None  # overlap ratio per frame
        self.quality_inter: Optional[List[float]] = None    # inter-frame change ratio per frame
        # Per-concept metrics ({concept_name: [per-frame ...]}) when the metrics
        # file was written by the per-concept version; None for older files.
        self.quality_overlap_by_group: Optional[dict] = None
        self.quality_inter_by_group: Optional[dict] = None
        # Global all-concepts-combined fallback tracks (older files / unknown concept).
        self._quality_overlap_global: Optional[List[float]] = None
        self._quality_inter_global: Optional[List[float]] = None
        # Which concept the Overlap/Change bars currently show (None = global).
        self._quality_bars_concept: Optional[str] = None
        # When True, _draw_quality_colorbars() redraws bars from scratch; otherwise
        # only the playhead is moved (same fast-path logic as _presence_bar_dirty).
        self._quality_bars_dirty: bool = True

        # In-memory annotation cache: (concept_name, obj_id) -> {frame_idx -> [(x, y, is_positive)]}
        # Populated on project load (propagated=False entries only) and updated on every point
        # add/remove.  Survives frame navigation; written to disk by save_points_for_batch /
        # apply_refinement.
        self._points_cache: Dict[tuple, Dict[int, List[tuple]]] = {}
        # Parallel box cache: same key structure, values are [(x1, y1, x2, y2), ...].
        self._boxes_cache: Dict[tuple, Dict[int, List[tuple]]] = {}
        # All annotations including propagated=True — same structure as _points_cache.
        # Used for "Next/Prev Ann" navigation so already-applied annotations remain navigable.
        # Updated live alongside _points_cache on every add/remove/undo/redo.
        self._all_annotations_cache: Dict[tuple, Dict[int, List[tuple]]] = {}
        self._all_boxes_cache: Dict[tuple, Dict[int, List[tuple]]] = {}
        # Tracks which instance keys have been modified since the last project load so that
        # save_points_for_batch can correctly overwrite (including clearing deleted points).
        self._dirty_keys: set = set()
        # Concepts the user marked for full re-detection (Reset & Re-detect button).
        # Disk cleanup + sentinel write is deferred to save_points_for_batch().
        self._concepts_pending_reset: set = set()

        # Vocabulary: concept_name -> [instance name, ...]  (for rename dropdown)
        self.vocabulary: Dict[str, List[str]] = {}
        self.vocabulary_path: Optional[str] = None

        # Display
        self.display_image = None
        self.photo = None
        self._canvas_image_id = None   # persistent canvas item; avoids delete+create each frame
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_masks_var = tk.BooleanVar(value=True)
        self.focus_mode_var = tk.BooleanVar(value=True)
        self.skip_delete_confirm_var = tk.BooleanVar(value=False)
        self.auto_jump_var = tk.BooleanVar(value=True)
        self.show_gaze_var = tk.BooleanVar(value=True)
        self.mask_alpha_var = tk.DoubleVar(value=0.5)

        # Rate-limit perf diagnostics: print at most once every 2 seconds
        self._last_perf_print: float = 0.0

        # App-level settings (last opened project dir, etc.)
        _app = _load_app_settings()
        self._last_project_dir: Optional[str] = _app.get("last_project_dir")

        # Suppress redundant display_frame() calls triggered by selection_set() inside
        # update_concept_tree() — the caller always calls display_frame() explicitly after.
        self._updating_tree = False

        # Flash animation state
        self.flash_mask_in_progress = False
        self.flash_mask_on = False
        self.flash_overlap_in_progress = False
        self.flash_overlap_on = False
        self.flash_overlap_computed = None
        self.flash_points_in_progress = False
        self.flash_points_on = False
        self.flash_custom_in_progress = False
        self.flash_custom_on = False
        self.flash_custom_mask = None

        # Setup UI
        self._setup_ui()

        # Status
        self.status_var.set("Ready. Load a video to start.")

        # Undo/redo (points and instance deletions)
        self.root.bind('<Control-z>', self.undo_action)
        self.root.bind('<Command-z>', self.undo_action)  # Mac support
        self.root.bind('<Control-y>', self.redo_action)
        self.root.bind('<Command-Shift-z>', self.redo_action)  # Mac primary
        self.root.bind('<Command-y>', self.redo_action)  # Mac alternative

        # Ctrl+S: save pending points/annotations to disk (no propagation)
        self.root.bind('<Control-s>', lambda e: self.save_points_for_batch())

        # Arrow-key navigation
        self.root.bind('<Left>', lambda e: self._handle_prev_frame_shortcut())
        self.root.bind('<Right>', lambda e: self._handle_next_frame_shortcut())
        self.root.bind('<Up>', lambda e: self._handle_prev_instance_shortcut())
        self.root.bind('<Down>', lambda e: self._handle_next_instance_shortcut())

        # Home/End: jump to first instance / last renamed (non-default-named) instance
        self.root.bind('<Home>', lambda e: self._handle_home_shortcut())
        self.root.bind('<End>', lambda e: self._handle_end_shortcut())

        # Page Up/Down: jump to previous/next frame with saved refinement points/boxes
        # for the selected instance (mirrors the SAM2 UI's annotated-frame jump)
        self.root.bind('<Prior>', lambda e: self._handle_annotated_frame_shortcut('prev'))
        self.root.bind('<Next>', lambda e: self._handle_annotated_frame_shortcut('next'))

        # Space bar: play / pause
        self.root.bind('<space>', lambda e: self._handle_play_shortcut())

        # Flash shortcuts
        self.root.bind('f', lambda e: self._handle_flash_shortcut())
        self.root.bind('o', lambda e: self._handle_overlap_shortcut())
        self.root.bind('r', lambda e: self._handle_removal_shortcut())

        # 'b': toggle box/point annotation mode (same as the Box Mode button)
        self.root.bind('b', lambda e: self._toggle_box_mode())

        # Clean up live sessions on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """Setup the 3-panel layout"""

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_command(label="Save Project (Metadata Only)", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Load Gaze CSV...", command=self.load_gaze_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Extract Frames to Project...",
                              command=self.extract_frames_to_project)
        file_menu.add_command(label="Re-encode Video (MJPEG)...",
                              command=self.reencode_video_dialog)
        file_menu.add_command(label="Delete Frame Cache...",
                              command=self.delete_frame_cache_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Concept List...",
                              command=self.export_concept_list)
        file_menu.add_command(label="Import Concept List...",
                              command=self.import_concept_list)
        file_menu.add_separator()
        file_menu.add_command(label="Manage Video Paths...",
                              command=self.manage_paths_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project",
                              command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        vocab_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Vocabulary", menu=vocab_menu)
        vocab_menu.add_command(label="Load Vocabulary File...",
                               command=self.load_vocabulary)
        vocab_menu.add_command(label="Save Vocabulary File",
                               command=self.save_vocabulary)
        vocab_menu.add_command(label="Save Vocabulary As...",
                               command=self.save_vocabulary_as)
        vocab_menu.add_separator()
        vocab_menu.add_command(label="Edit Vocabulary...",
                               command=self.edit_vocabulary_dialog)

        # Main container — three resizable panes
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                    sashrelief=tk.RAISED, sashwidth=5,
                                    handlesize=8)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Concepts tree
        left_panel = tk.Frame(main_paned, relief=tk.RAISED, borderwidth=1)
        main_paned.add(left_panel, minsize=150, width=300)

        self._setup_concepts_panel(left_panel)

        # Center panel: Video display
        center_panel = tk.Frame(main_paned, relief=tk.RAISED, borderwidth=1)
        main_paned.add(center_panel, minsize=200, stretch="always")

        self._setup_video_panel(center_panel)

        # Right panel: Instance controls
        right_panel = tk.Frame(main_paned, relief=tk.RAISED, borderwidth=1)
        main_paned.add(right_panel, minsize=150, width=300)

        self._setup_controls_panel(right_panel)

        # Bottom status bar
        status_bar = tk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        status_label = tk.Label(status_bar, textvariable=self.status_var,
                               anchor=tk.W, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, padx=5, pady=2)

    def _setup_concepts_panel(self, parent):
        """Setup left panel with concept tree"""

        tk.Label(parent, text="Concepts", font=("Arial", 12, "bold")).pack(pady=5)

        # Scrollable tree
        tree_frame = tk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tree_scroll = tk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.concept_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set,
                                         selectmode='extended')
        self.concept_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.concept_tree.yview)

        # Configure tree columns
        self.concept_tree['columns'] = ('status', 'coverage')
        self.concept_tree.column('#0', width=200, minwidth=150)
        self.concept_tree.column('status', width=60, minwidth=50)
        self.concept_tree.column('coverage', width=60, minwidth=50)

        self.concept_tree.heading('#0', text='Name', anchor=tk.W)
        self.concept_tree.heading('status', text='Status', anchor=tk.W)
        self.concept_tree.heading('coverage', text='Cov %', anchor=tk.W)

        # Bind selection and double-click
        self.concept_tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        self.concept_tree.bind('<Double-Button-1>', self.on_tree_double_click)

        # Left/right keys navigate frames even when the tree has focus; return "break"
        # to prevent the Treeview's native collapse/expand behavior from firing.
        self.concept_tree.bind('<Left>', lambda e: (self.jump_frames(-1), "break")[1])
        self.concept_tree.bind('<Right>', lambda e: (self.jump_frames(1), "break")[1])
        # Delete key removes selected instance(s) without reaching for the mouse.
        self.concept_tree.bind('<Delete>', lambda e: self.delete_instance())

        # Buttons
        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(btn_frame, text="Add Concept",
                 command=self.add_concept_dialog).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Remove Concept",
                 command=self.remove_concept_dialog).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Reset & Re-detect...",
                 command=self.reset_concept_redetect).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Edit Prompt...",
                 command=self.edit_concept_prompt_dialog).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Reload Project",
                 command=self.reload_project).pack(fill=tk.X, pady=2)
        self._export_btn = tk.Button(btn_frame, text="Export Video",
                                     command=self.export_video)
        self._export_btn.pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Export SAM2 Format...",
                 command=self.export_sam2_format_dialog).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Propagate",
                 command=self.apply_refinement).pack(fill=tk.X, pady=2)

    def _setup_video_panel(self, parent):
        """Setup center panel with video display"""

        # Project folder name bar (updated whenever a project is loaded)
        self._project_folder_label = tk.Label(
            parent, text="No project loaded",
            font=("Arial", 8), fg='#888888', bg='#1e1e1e', anchor=tk.W,
            padx=6, pady=1,
        )
        self._project_folder_label.pack(fill=tk.X, side=tk.TOP)

        # Canvas for video display
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind click events for refinement (left=positive point or box drag, right=negative)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<B1-Motion>', self._on_box_drag_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_box_drag_end)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)

        # Redraw when canvas is resized so video fills the new size
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Mouse-wheel scroll → frame navigation (Linux: Button-4/5; others: MouseWheel)
        self.canvas.bind('<Button-4>',    lambda e: self.jump_frames(-1))
        self.canvas.bind('<Button-5>',    lambda e: self.jump_frames(1))
        self.canvas.bind('<MouseWheel>',
                         lambda e: self.jump_frames(-1 if e.delta > 0 else 1))

        # ── Absorb wizard banner (hidden until wizard is active) ──────────────
        self._wizard_banner = tk.Frame(parent, bg='#1a3a1a', relief=tk.RIDGE, bd=1)
        # (not packed here — shown only during absorb wizard via _wizard_enter_phase)

        # Pack the button cluster FIRST so it claims its natural (right-aligned)
        # width before the text label competes for space — otherwise, on a
        # narrow window, the text (packed/measured first) would eat the cavity
        # and leave the buttons clipped or shoved off the right edge instead of
        # sitting flush against the banner's right boundary.
        wiz_right = tk.Frame(self._wizard_banner, bg='#1a3a1a')
        wiz_right.pack(side=tk.RIGHT, padx=4, pady=3)

        wiz_left = tk.Frame(self._wizard_banner, bg='#1a3a1a')
        wiz_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=3)
        self._wizard_title_lbl = tk.Label(wiz_left, text="", bg='#1a3a1a', fg='white',
                                          font=("Arial", 9, "bold"), anchor=tk.W)
        self._wizard_title_lbl.pack(fill=tk.X)
        self._wizard_instr_lbl = tk.Label(wiz_left, text="", bg='#1a3a1a', fg='#aaffaa',
                                          font=("Arial", 8), anchor=tk.W, wraplength=200)
        self._wizard_instr_lbl.pack(fill=tk.X)
        # Keep the instruction text wrapping to whatever width wiz_left actually
        # has, so long instructions never push the (right-packed) buttons off
        # the banner instead of wrapping onto a second line.
        wiz_left.bind(
            '<Configure>',
            lambda e: self._wizard_instr_lbl.config(wraplength=max(1, e.width))
        )

        wiz_row1 = tk.Frame(wiz_right, bg='#1a3a1a')
        wiz_row1.pack(fill=tk.X, pady=(0, 2))
        wiz_row2 = tk.Frame(wiz_right, bg='#1a3a1a')
        wiz_row2.pack(fill=tk.X)

        self._wiz_confirm_btn = tk.Button(wiz_row1, text="Use this mask",
                                          bg='#2a5a2a', fg='white', font=("Arial", 8),
                                          command=self._wizard_confirm_mask)
        self._wiz_confirm_btn.pack(side=tk.LEFT, padx=2)

        self._wiz_point_btn = tk.Button(wiz_row1, text="Add point instead",
                                        bg='#2a2a5a', fg='white', font=("Arial", 8),
                                        command=self._wizard_switch_to_point_mode)
        self._wiz_point_btn.pack(side=tk.LEFT, padx=2)

        self._wiz_confirm_pts_btn = tk.Button(wiz_row2, text="Confirm points",
                                              bg='#2a5a2a', fg='white', font=("Arial", 8),
                                              state=tk.DISABLED,
                                              command=self._wizard_confirm_points)
        self._wiz_confirm_pts_btn.pack(side=tk.LEFT, padx=2)

        self._wiz_back_btn = tk.Button(wiz_row2, text="Back to mask",
                                       bg='#404040', fg='white', font=("Arial", 8),
                                       state=tk.DISABLED,
                                       command=self._wizard_back_to_mask_mode)
        self._wiz_back_btn.pack(side=tk.LEFT, padx=2)

        self._wiz_cancel_btn = tk.Button(wiz_row2, text="Cancel",
                                         bg='#5a1a1a', fg='white', font=("Arial", 8),
                                         command=self._wizard_cancel)
        self._wiz_cancel_btn.pack(side=tk.LEFT, padx=2)

        # ── Row 1: play/pause + frame nav + counter ───────────────────────────
        self._nav_row1 = row1 = tk.Frame(parent)
        row1.pack(fill=tk.X, padx=5, pady=(3, 0))

        self.play_button = tk.Button(row1, text="Play", width=5,
                                     font=("Arial", 8), command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=1)

        for txt, d in [("<<", -10), ("<", -1), (">", 1), (">>", 10)]:
            tk.Button(row1, text=txt, width=3, font=("Arial", 8),
                     command=lambda delta=d: self.jump_frames(delta)).pack(side=tk.LEFT, padx=1)

        self.frame_label = tk.Label(row1, text="0 / 0", font=("Arial", 8))
        self.frame_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # Hidden slider — kept as a zoom-window state container so all existing
        # zoom/scrub logic that reads .cget('from')/('to') continues to work.
        self.frame_slider = tk.Scale(row1, from_=0, to=1, orient=tk.HORIZONTAL,
                                     command=self.on_slider_change, showvalue=False,
                                     sliderlength=10, width=10)
        # intentionally not packed — not visible

        # ── Row 2: combined timeline scrubber + instance presence bar ─────────
        self._nav_row2 = row2 = tk.Frame(parent)
        row2.pack(fill=tk.X, padx=5, pady=(1, 0))

        self.presence_canvas = tk.Canvas(row2, height=22, bg='#2a2a2a',
                                         relief=tk.SUNKEN, bd=1)
        self.presence_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._presence_canvas_w = 0
        self._presence_canvas_h = 0
        self.presence_canvas.bind('<Configure>', self._on_presence_canvas_configure)
        self.presence_canvas.bind('<Button-1>', self._on_presence_bar_click)
        self.presence_canvas.bind('<B1-Motion>', self._on_presence_bar_click)

        # ── Row 3 (compact): timeline zoom + playback speed ────────────────────
        self._nav_row3 = row3 = tk.Frame(parent)
        row3.pack(fill=tk.X, padx=5, pady=(1, 3))

        tk.Label(row3, text="Zoom:", font=("Arial", 7)).pack(side=tk.LEFT)
        for zoom, label in [(1, "Full"), (5, "5x"), (20, "20x"), (100, "100x")]:
            btn = tk.Button(row3, text=label, width=4, font=("Arial", 7), relief=tk.RAISED,
                           pady=0, command=lambda z=zoom: self.set_slider_zoom(z))
            btn.pack(side=tk.LEFT, padx=1)
            self.zoom_buttons[zoom] = btn

        self.zoom_info_label = tk.Label(row3, text="(Full)", fg="gray",
                                        font=("Arial", 7), width=12)
        self.zoom_info_label.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(row3, text="Speed:", font=("Arial", 7)).pack(side=tk.LEFT)
        for speed, label in [(0.25, ".25x"), (0.5, ".5x"), (1.0, "1x"),
                             (2.0, "2x"), (4.0, "4x")]:
            btn = tk.Button(row3, text=label, width=4, font=("Arial", 7), relief=tk.RAISED,
                           pady=0, command=lambda s=speed: self.set_playback_speed(s))
            btn.pack(side=tk.LEFT, padx=1)
            self.speed_buttons[speed] = btn

        self.speed_info_label = tk.Label(row3, text="(Norm)", fg="gray",
                                         font=("Arial", 7), width=7)
        self.speed_info_label.pack(side=tk.LEFT, padx=2)

        # Highlight the default zoom/speed buttons
        self._update_zoom_button_highlight()
        self._update_speed_button_highlight()

        # ── Row 4 (compact): quality colorbars (overlap + background ratio) ────
        # Uses grid (not pack) so the two canvases split the row's width exactly
        # 50/50 regardless of window size — pack's expand allocates space
        # sequentially, so the label sitting between them would otherwise steal
        # from the second canvas's share when the window is narrow.
        self._nav_row4 = row4 = tk.Frame(parent)
        row4.pack(fill=tk.X, padx=5, pady=(1, 3))
        row4.columnconfigure(1, weight=1, uniform="qbar")
        row4.columnconfigure(3, weight=1, uniform="qbar")

        tk.Label(row4, text="Overlap:", font=("Arial", 7), fg="gray", width=7,
                 anchor=tk.W).grid(row=0, column=0, sticky="w")
        self.quality_overlap_canvas = tk.Canvas(row4, height=10, bg='#2a2a2a',
                                                relief=tk.FLAT, bd=0)
        self.quality_overlap_canvas.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.quality_overlap_canvas.bind('<Button-1>', self._on_quality_bar_click)

        tk.Label(row4, text="BG:", font=("Arial", 7), fg="gray", width=3,
                 anchor=tk.W).grid(row=0, column=2, sticky="w")
        self.quality_bg_canvas = tk.Canvas(row4, height=10, bg='#2a2a2a',
                                           relief=tk.FLAT, bd=0)
        self.quality_bg_canvas.grid(row=0, column=3, sticky="ew")
        self.quality_bg_canvas.bind('<Button-1>', self._on_quality_bar_click)

        # ── Row 5 (compact): inter-frame change, spans full width ──────────
        self._nav_row5 = row5 = tk.Frame(parent)
        row5.pack(fill=tk.X, padx=5, pady=(0, 3))

        tk.Label(row5, text="Change:", font=("Arial", 7), fg="gray", width=7,
                 anchor=tk.W).pack(side=tk.LEFT)
        self.quality_inter_canvas = tk.Canvas(row5, height=8, bg='#2a2a2a',
                                              relief=tk.FLAT, bd=0)
        self.quality_inter_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.quality_inter_canvas.bind('<Button-1>', self._on_quality_bar_click)

    def _setup_controls_panel(self, parent):
        """Setup right panel with instance controls"""

        # Scrollable container so all controls are reachable on low-resolution screens
        _sc = tk.Canvas(parent, highlightthickness=0)
        _sc_sb = tk.Scrollbar(parent, orient="vertical", command=_sc.yview)
        _sc.configure(yscrollcommand=_sc_sb.set)
        _sc_sb.pack(side=tk.RIGHT, fill=tk.Y)
        _sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(_sc)
        _win_id = _sc.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(e):
            _sc.configure(scrollregion=_sc.bbox("all"))
        def _on_sc_resize(e):
            _sc.itemconfig(_win_id, width=e.width)
        inner.bind("<Configure>", _on_inner_configure)
        _sc.bind("<Configure>", _on_sc_resize)

        # Bind mousewheel only while the pointer is over the right panel
        def _mw(e): _sc.yview_scroll(int(-1 * (e.delta / 120)), "units")
        def _mw_lin(e): _sc.yview_scroll(-1 if e.num == 4 else 1, "units")
        def _on_enter(e):
            _sc.bind_all("<MouseWheel>", _mw)
            _sc.bind_all("<Button-4>", _mw_lin)
            _sc.bind_all("<Button-5>", _mw_lin)
        def _on_leave(e):
            _sc.unbind_all("<MouseWheel>")
            _sc.unbind_all("<Button-4>")
            _sc.unbind_all("<Button-5>")
        _sc.bind("<Enter>", _on_enter)
        _sc.bind("<Leave>", _on_leave)
        inner.bind("<Enter>", _on_enter)
        inner.bind("<Leave>", _on_leave)

        # Device selector
        device_frame = tk.LabelFrame(inner, text="Device", padx=8, pady=5)
        device_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        device_labels = [label for _, label in self.available_devices]
        device_menu = ttk.Combobox(device_frame, textvariable=self.device_var,
                                   values=device_labels, state="readonly")
        device_menu.pack(fill=tk.X)
        device_menu.bind('<<ComboboxSelected>>', self._on_device_change)

        self.model_status_label = tk.Label(device_frame, text="Model: not loaded",
                                           fg="gray", font=("Arial", 8))
        self.model_status_label.pack(anchor=tk.W, pady=(3, 0))

        tk.Label(inner, text="Instance Controls",
                font=("Arial", 12, "bold")).pack(pady=5)

        # Selected instance info
        info_frame = tk.LabelFrame(inner, text="Selected Instance", padx=10, pady=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.selected_label = tk.Label(info_frame, text="None",
                                       font=("Arial", 10), anchor=tk.W)
        self.selected_label.pack(fill=tk.X)

        # Rename controls
        rename_frame = tk.Frame(info_frame)
        rename_frame.pack(fill=tk.X, pady=5)

        tk.Label(rename_frame, text="Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Combobox(rename_frame, state='normal')
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.name_entry.bind('<Return>', lambda e: self.rename_instance())
        self.name_entry.bind('<FocusOut>', lambda e: self._on_name_entry_focus_out())
        self.name_entry.bind('<<ComboboxSelected>>', lambda e: self.rename_instance())
        tk.Button(rename_frame, text="Rename",
                 command=self.rename_instance).pack(side=tk.LEFT)

        # Frame navigation for selected instance
        nav_frame = tk.LabelFrame(inner, text="Navigate Instance", padx=8, pady=5)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        period_row = tk.Frame(nav_frame)
        period_row.pack(fill=tk.X, pady=(0, 3))
        tk.Button(period_row, text="|< First",
                  command=self.jump_to_first_period).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(period_row, text="< Prev",
                  command=self.jump_to_prev_period).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(period_row, text="Next >",
                  command=self.jump_to_next_period).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Checkbutton(nav_frame, text="Auto-jump to first period on select",
                       variable=self.auto_jump_var).pack(anchor=tk.W)

        ann_row = tk.Frame(nav_frame)
        ann_row.pack(fill=tk.X)
        tk.Button(ann_row, text="< Prev Ann",
                  command=self.jump_to_prev_annotation).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(ann_row, text="Next Ann >",
                  command=self.jump_to_next_annotation).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Instance operations
        ops_frame = tk.LabelFrame(inner, text="Operations", padx=10, pady=10)
        ops_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(ops_frame, text="New Instance (Manual Points)",
                 command=self.add_new_instance).pack(fill=tk.X, pady=2)
        tk.Button(ops_frame, text="Delete Instance",
                 command=self.delete_instance).pack(fill=tk.X, pady=2)
        tk.Checkbutton(ops_frame, text="Skip delete confirmation (this session)",
                       variable=self.skip_delete_confirm_var).pack(anchor=tk.W)
        tk.Button(ops_frame, text="Absorb Selected Into Target...",
                 command=self.absorb_instance_dialog).pack(fill=tk.X, pady=2)
        tk.Button(ops_frame, text="Save Displayed Mask As Anchor",
                 command=self.save_current_mask_as_anchor).pack(fill=tk.X, pady=2)

        # Annotation controls (point/box placement and per-instance correction)
        refine_frame = tk.LabelFrame(inner, text="Annotation", padx=10, pady=10)
        refine_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(refine_frame, text="Left click: positive  |  Right click: negative",
                 font=("Arial", 8), fg="gray").pack(anchor=tk.W, pady=(2, 0))
        self.box_mode_btn = tk.Button(
            refine_frame, text="Box Mode: OFF",
            command=self._toggle_box_mode,
            bg='#404040', fg='white', activebackground='#505050',
        )
        self.box_mode_btn.pack(fill=tk.X, pady=(2, 4))

        tk.Button(refine_frame, text="Clear Frame Annotations",
                 command=self.clear_frame_annotations).pack(fill=tk.X, pady=2)

        self.remove_mode_button = tk.Button(
            refine_frame, text="Removal Mode: OFF",
            command=self.toggle_point_removal_mode,
            bg='#404040', fg='white', activebackground='#505050'
        )
        self.remove_mode_button.pack(fill=tk.X, pady=2)

        tk.Button(refine_frame, text="Clear ALL Annotations (all frames)",
                 fg='darkred', command=self.clear_all_instance_annotations).pack(fill=tk.X, pady=2)
        tk.Button(refine_frame, text="Save Changes for Refinement",
                 command=self.save_points_for_batch).pack(fill=tk.X, pady=2)

        self.points_label = tk.Label(refine_frame, text="Points: 0  Boxes: 0", anchor=tk.W)
        self.points_label.pack(fill=tk.X, pady=2)

        self.session_label = tk.Label(refine_frame, text="Session: none",
                                      anchor=tk.W, fg="gray")
        self.session_label.pack(fill=tk.X)
        tk.Button(refine_frame, text="Release Session",
                 command=self._release_selected_session).pack(fill=tk.X, pady=2)

        # Display options (including flash inspection tools)
        display_frame = tk.LabelFrame(inner, text="Display", padx=10, pady=5)
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Checkbutton(display_frame, text="Show instance labels",
                       variable=self.show_labels_var,
                       command=self.display_frame).pack(anchor=tk.W)
        tk.Checkbutton(display_frame, text="Show masks",
                       variable=self.show_masks_var,
                       command=self.display_frame).pack(anchor=tk.W)
        tk.Checkbutton(display_frame, text="Focus: selected concept only",
                       variable=self.focus_mode_var,
                       command=self.display_frame).pack(anchor=tk.W)
        self.show_gaze_checkbutton = tk.Checkbutton(
            display_frame, text="Show gaze point overlay",
            variable=self.show_gaze_var,
            command=self.display_frame,
            state=tk.NORMAL if self.gaze_points_by_frame else tk.DISABLED)
        self.show_gaze_checkbutton.pack(anchor=tk.W)

        # Flash inspection buttons (shortcut: f / o)
        flash_row = tk.Frame(display_frame)
        flash_row.pack(fill=tk.X, pady=(4, 1))
        tk.Button(flash_row, text="Flash Mask (f)",
                 command=self.flash_selected_instance_mask).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(flash_row, text="Flash Points",
                 command=self.flash_frame_points).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(display_frame, text="Flash Overlap (o)",
                 command=self.flash_overlap_regions).pack(fill=tk.X, pady=(1, 4))

        alpha_row = tk.Frame(display_frame)
        alpha_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(alpha_row, text="Mask alpha:").pack(side=tk.LEFT)
        tk.Scale(alpha_row, variable=self.mask_alpha_var, from_=0.0, to=1.0,
                 resolution=0.05, orient=tk.HORIZONTAL, width=18,
                 sliderlength=20, troughcolor='#444444', activebackground='#ffd700',
                 command=lambda _: self.display_frame()).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ============================================================
    # Device Management
    # ============================================================

    @staticmethod
    def _detect_devices() -> List[tuple]:
        """Return list of (device_str, display_label) for all available devices."""
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total_mb = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
                devices.append((f"cuda:{i}", f"cuda:{i}  {name}  ({total_mb} MB)"))
        devices.append(("cpu", "cpu  (no GPU)"))
        return devices

    def _on_device_change(self, event=None):
        """Handle GPU selector change: unload model and update self.device."""
        label = self.device_var.get()
        new_device = next((d for d, l in self.available_devices if l == label), "cpu")
        if new_device == self.device:
            return
        self._release_all_sessions()
        self.sam3_model = None
        self.device = new_device
        self.model_status_label.config(text="Model: not loaded", fg="gray")
        self.status_var.set(f"Device changed to {new_device}. Model will reload on next use.")

    def _update_model_status(self):
        """Refresh the model-loaded indicator in the toolbar."""
        if self.sam3_model is not None:
            self.model_status_label.config(text=f"Model: loaded ({self.device})", fg="green")
        else:
            self.model_status_label.config(text="Model: not loaded", fg="gray")

    # ============================================================
    # Session Management
    # ============================================================

    def _release_session(self, concept_name: str):
        """Close a live SAM3 session and remove it from active_sessions."""
        session_id = self.active_sessions.pop(concept_name, None)
        if session_id and self.sam3_model:
            try:
                self.sam3_model.close_session(session_id=session_id)
                print(f"Released session for concept '{concept_name}'.")
            except Exception:
                pass
        self._update_session_status()

    def _release_all_sessions(self):
        """Close all live sessions."""
        for name in list(self.active_sessions.keys()):
            self._release_session(name)

    def _release_selected_session(self):
        """Release the session for the currently selected concept."""
        if not self.selected_concept:
            return
        name = self.selected_concept.name
        if name in self.active_sessions:
            self._release_session(name)
            self.status_var.set(f"Session released for '{name}'.")
        else:
            self.status_var.set(f"No active session for '{name}'.")

    def _update_session_status(self):
        """Refresh the session status label for the currently selected concept.

        Thread-safe: callers like _release_session run on background worker
        threads (e.g. process_thread releasing another concept's session), and
        Tk widget calls off the main thread crash or corrupt Tk state — so
        marshal onto the main loop when needed.
        """
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self._update_session_status)
            return
        if not self.selected_concept:
            self.session_label.config(text="Session: none", fg="gray")
            return
        name = self.selected_concept.name
        if name in self.active_sessions:
            self.session_label.config(text="Session: active (fast path)", fg="green")
        else:
            self.session_label.config(text="Session: none (will re-detect)", fg="gray")

    def _on_closing(self):
        """Handle window close: warn about unsaved annotation changes, then destroy."""
        if self._dirty_keys or self._concepts_pending_reset or self._metadata_dirty:
            parts = []
            if self._dirty_keys:
                parts.append(f"{len(self._dirty_keys)} instance(s) with unsaved points")
            if self._concepts_pending_reset:
                parts.append(f"{len(self._concepts_pending_reset)} concept reset(s) pending")
            if self._metadata_dirty:
                parts.append("unsaved renames/deletions")
            answer = messagebox.askyesnocancel(
                "Unsaved changes",
                f"You have: {', '.join(parts)}.\n\n"
                "Save now before closing?",
                default=messagebox.YES,
            )
            if answer is None:   # Cancel — don't close
                return
            if answer:           # Yes — save then close
                self.save_points_for_batch()
        self._release_all_sessions()
        self.root.destroy()

    # ============================================================
    # Instance Frame Navigation
    # ============================================================

    def _get_instance_periods(self) -> List[tuple]:
        """Return continuous_periods for the selected instance, computing lazily if empty."""
        if not self.selected_instance or not self.selected_concept or not self.project:
            return []
        inst = self.selected_instance
        if not inst.continuous_periods:
            mask_dir = os.path.join(
                self.project.project_dir, "concepts", self.selected_concept.name,
                "instances", str(inst.sam3_obj_id), "masks"
            )
            inst.continuous_periods = compute_continuous_periods(mask_dir)
        return inst.continuous_periods

    def _go_to_frame(self, frame_idx: int):
        """Navigate to a specific frame, loading any saved annotations for that frame."""
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        self.current_frame_idx = frame_idx
        self.frame_slider.set(frame_idx)
        self._load_annotations_for_current_frame()
        self.display_frame()

    def _load_annotations_for_current_frame(self):
        """Load ALL cached annotations (including already-propagated) for the current
        frame/instance into self.refinement_points/boxes. Does NOT touch the undo/redo stack."""
        self.refinement_points.clear()
        self.refinement_boxes.clear()
        key = self._instance_key()
        if key:
            cached_pts = self._all_annotations_cache.get(key, {}).get(self.current_frame_idx, [])
            self.refinement_points.extend(cached_pts)
            cached_boxes = self._all_boxes_cache.get(key, {}).get(self.current_frame_idx, [])
            self.refinement_boxes.extend(cached_boxes)
        self._update_ann_label()

    def jump_to_first_period(self):
        """Jump to the start of the first continuous detected period for this instance."""
        periods = self._get_instance_periods()
        if not periods:
            self.status_var.set("No detected periods for this instance.")
            return
        first = periods[0][0]
        self.status_var.set(f"Jumped to first period (frame {first}).")
        self._go_to_frame(first)

    def jump_to_prev_period(self):
        """Jump to the start of the previous continuous detected period (wraps around)."""
        periods = self._get_instance_periods()
        if not periods:
            self.status_var.set("No detected periods for this instance.")
            return
        cur = self.current_frame_idx
        # Last period whose start is before the current frame
        prev = next((s for s, e in reversed(periods) if s < cur), None)
        if prev is None:
            prev = periods[-1][0]  # wrap to last
            self.status_var.set(f"Wrapped to last period (frame {prev}).")
        else:
            self.status_var.set(f"Jumped to period start at frame {prev}.")
        self._go_to_frame(prev)

    def jump_to_next_period(self):
        """Jump to the start of the next continuous detected period (wraps around)."""
        periods = self._get_instance_periods()
        if not periods:
            self.status_var.set("No detected periods for this instance.")
            return
        cur = self.current_frame_idx
        nxt = next((s for s, e in periods if s > cur), None)
        if nxt is None:
            nxt = periods[0][0]  # wrap to first
            self.status_var.set(f"Wrapped to first period (frame {nxt}).")
        else:
            self.status_var.set(f"Jumped to period start at frame {nxt}.")
        self._go_to_frame(nxt)

    def _get_annotation_frames(self) -> List[int]:
        """Return sorted list of frame indices that have annotation points, boxes, or mask anchors."""
        if not self.selected_instance or not self.selected_concept:
            return []
        key = (self.selected_concept.name, self.selected_instance.sam3_obj_id)
        point_frames = set(self._all_annotations_cache.get(key, {}).keys())
        box_frames = set(self._all_boxes_cache.get(key, {}).keys())
        anchor_frames = set(self.selected_instance.mask_anchor_frames)
        return sorted(point_frames | box_frames | anchor_frames)

    def jump_to_prev_annotation(self):
        """Jump to the previous frame with saved annotations (wraps around)."""
        frames = self._get_annotation_frames()
        if not frames:
            self.status_var.set("No saved annotations for this instance.")
            return
        cur = self.current_frame_idx
        prev = next((f for f in reversed(frames) if f < cur), None)
        if prev is None:
            prev = frames[-1]
            self.status_var.set(f"Wrapped to last annotation (frame {prev}).")
        else:
            self.status_var.set(f"Jumped to annotation at frame {prev}.")
        self._go_to_frame(prev)

    def jump_to_next_annotation(self):
        """Jump to the next frame with saved annotations (wraps around)."""
        frames = self._get_annotation_frames()
        if not frames:
            self.status_var.set("No saved annotations for this instance.")
            return
        cur = self.current_frame_idx
        nxt = next((f for f in frames if f > cur), None)
        if nxt is None:
            nxt = frames[0]
            self.status_var.set(f"Wrapped to first annotation (frame {nxt}).")
        else:
            self.status_var.set(f"Jumped to annotation at frame {nxt}.")
        self._go_to_frame(nxt)

    # ============================================================
    # Keyboard Shortcuts
    # ============================================================

    def _should_ignore_keyboard_shortcut(self) -> bool:
        """Return True when an Entry widget has keyboard focus (user is typing)."""
        return isinstance(self.root.focus_get(), tk.Entry)

    def _handle_prev_frame_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self.jump_frames(-1)

    def _handle_next_frame_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self.jump_frames(1)

    def _handle_prev_instance_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self._navigate_instances(-1)

    def _handle_next_instance_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self._navigate_instances(1)

    def _handle_play_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self.toggle_play()

    def _current_concept_instances(self):
        """Navigable (concept, instance) list scoped to the selected concept only,
        in the same order the concept tree lists them. Falls back to the first
        concept in the project if none is selected."""
        concept = self.selected_concept
        if concept is None and self.project and self.project.concepts:
            concept = self.project.concepts[0]
        if concept is None:
            return []
        return [(concept, inst) for inst in concept.instances if not inst.deleted]

    def _handle_home_shortcut(self):
        """Jump to the first instance within the current concept."""
        if self._should_ignore_keyboard_shortcut():
            return
        all_items = self._current_concept_instances()
        if all_items:
            self._select_instance_at_index(all_items, 0)

    def _handle_end_shortcut(self):
        """Jump to the last instance within the current concept that has been
        renamed away from its default "<concept>_<id>" name (mirrors the SAM2
        UI's End shortcut, which skips auto-generated "Object_N" names). No-op
        if no instance in the concept has been renamed."""
        if self._should_ignore_keyboard_shortcut():
            return
        all_items = self._current_concept_instances()
        target_idx = None
        for i, (concept, inst) in enumerate(all_items):
            if inst.user_name != f"{concept.name}_{inst.sam3_obj_id}":
                target_idx = i
        if target_idx is not None:
            self._select_instance_at_index(all_items, target_idx)

    def _handle_annotated_frame_shortcut(self, direction: str):
        """Handle Page Up/Down: jump to the previous/next frame with saved
        annotations (points, boxes, or mask anchors) for the selected instance.
        Reuses the same jump_to_prev/next_annotation methods as the toolbar
        buttons, so behavior (including wraparound) stays identical."""
        if self._should_ignore_keyboard_shortcut():
            return
        if direction == 'prev':
            self.jump_to_prev_annotation()
        else:
            self.jump_to_next_annotation()

    def _all_navigable_instances(self):
        """Flattened (concept, instance) list of all non-deleted instances, in the
        same order used by Up/Down/Home/End navigation."""
        if not self.project:
            return []
        return [
            (concept, inst)
            for concept in self.project.concepts
            for inst in concept.instances
            if not inst.deleted
        ]

    def _navigate_instances(self, delta: int):
        """Move to the previous (delta=-1) or next (delta=+1) non-deleted instance."""
        all_items = self._all_navigable_instances()
        if not all_items:
            return
        current_idx = None
        if self.selected_instance and self.selected_concept:
            for i, (c, inst) in enumerate(all_items):
                if (c.name == self.selected_concept.name and
                        inst.sam3_obj_id == self.selected_instance.sam3_obj_id):
                    current_idx = i
                    break
        new_idx = 0 if current_idx is None else (current_idx + delta) % len(all_items)
        self._select_instance_at_index(all_items, new_idx)

    def _select_instance_at_index(self, all_items, new_idx: int):
        """Select all_items[new_idx], updating tree/labels and auto-jumping to its
        first detected period if enabled. Shared by _navigate_instances and the
        Home/End shortcuts."""
        prev_key = ((self.selected_concept.name, self.selected_instance.sam3_obj_id)
                    if self.selected_concept and self.selected_instance else None)
        new_concept, new_inst = all_items[new_idx]
        self.selected_concept = new_concept
        self.selected_instance = new_inst
        self.selected_instances = [(new_concept, new_inst)]
        self.selected_label.config(text=f"{new_inst.user_name} (ID: {new_inst.sam3_obj_id})")
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, new_inst.user_name)
        # Highlight in tree (suppress on_tree_select so it doesn't double-fire)
        self._updating_tree = True
        try:
            for concept_item in self.concept_tree.get_children():
                for inst_item in self.concept_tree.get_children(concept_item):
                    tags = self.concept_tree.item(inst_item, 'tags')
                    if (tags and len(tags) >= 3 and tags[0] == 'instance' and
                            tags[1] == new_concept.name and int(tags[2]) == new_inst.sam3_obj_id):
                        self.concept_tree.selection_set(inst_item)
                        self.concept_tree.see(inst_item)
                        break
        finally:
            self._updating_tree = False
        self._update_session_status()
        self._mark_presence_dirty()
        self._load_annotations_for_current_frame()
        new_key = (new_concept.name, new_inst.sam3_obj_id)
        jumped = False
        if new_key != prev_key and self.auto_jump_var.get():
            periods = self._get_instance_periods()
            if periods:
                self._go_to_frame(periods[0][0])
                jumped = True
        if not jumped:
            self.display_frame()
        self.status_var.set(f"Instance: {new_inst.user_name}")

    # ============================================================
    # Playback
    # ============================================================

    def toggle_play(self):
        """Toggle video playback forward."""
        if not self.num_frames:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Pause")
            threading.Thread(target=self._play_video_thread, daemon=True).start()
        else:
            self.play_button.config(text="Play")

    def _play_video_thread(self):
        """Background thread: advance frame at the configured speed.

        Uses an absolute deadline per frame (instead of a fixed sleep per
        iteration) so slow rendering doesn't compound into drift: if we're
        already at/past the deadline we move on immediately rather than
        waiting. We also shave off half a display refresh interval (assumed
        60 Hz if unknown) since the next repaint can't happen any sooner
        than that anyway.
        """
        import time
        speed = self.playback_speed.get()
        base_delay = 1.0 / max(self.fps, 1.0)
        frame_step = max(1, round(speed)) if speed >= 1.0 else 1
        adjusted_delay = base_delay if speed >= 1.0 else base_delay / speed
        refresh_interval = 1.0 / 60.0

        next_deadline = time.perf_counter()
        while self.playing and self.num_frames:
            if self.current_frame_idx < self.num_frames - 1:
                self.current_frame_idx = min(
                    self.current_frame_idx + frame_step, self.num_frames - 1)
                # Only queue a render if the previous one has been consumed; otherwise
                # the main thread would accumulate a backlog of stale frame renders.
                if not self._render_pending:
                    self._render_pending = True
                    self.root.after(0, self._update_frame_from_play)
                next_deadline += adjusted_delay
                wait = next_deadline - time.perf_counter() - refresh_interval / 2.0
                if wait > 0:
                    time.sleep(wait)
            else:
                self.playing = False
                self.root.after(0, lambda: self.play_button.config(text="Play"))
                break

    def _update_frame_from_play(self):
        """Called on main thread to update slider + display during playback."""
        self._render_pending = False  # allow background thread to queue the next render
        zoom = self.slider_zoom_level.get()
        if zoom > 1:
            slider_min = int(self.frame_slider.cget('from'))
            slider_max = int(self.frame_slider.cget('to'))
            window_size = max(1, slider_max - slider_min)
            near_edge = (self.current_frame_idx - slider_min < window_size // 10 or
                         slider_max - self.current_frame_idx < window_size // 10)
            if near_edge:
                self._execute_zoom_jump(self.current_frame_idx, window_size, self.num_frames)
        self.frame_slider.set(self.current_frame_idx)
        self.display_frame()

    # ============================================================
    # Timeline Zoom
    # ============================================================

    def set_slider_zoom(self, zoom_level: int):
        """Set timeline zoom level; recenters the slider window around the current frame."""
        if not self.num_frames:
            return
        self.slider_zoom_level.set(zoom_level)
        if zoom_level == 1:
            self.frame_slider.config(from_=0, to=self.num_frames - 1)
            self.frame_slider.set(self.current_frame_idx)
            self.zoom_info_label.config(text="(Full)")
        else:
            window_size = max(100, self.num_frames // zoom_level)
            half = window_size // 2
            ws = max(0, self.current_frame_idx - half)
            we = min(self.num_frames - 1, ws + window_size)
            if we == self.num_frames - 1:
                ws = max(0, self.num_frames - window_size)
            self.slider_window_center = self.current_frame_idx
            self.frame_slider.config(from_=ws, to=we)
            self.frame_slider.set(self.current_frame_idx)
            self.zoom_info_label.config(text=f"({ws}-{we})")
        self._update_zoom_button_highlight()
        self._mark_presence_dirty()
        self._update_presence_bar()

    def _execute_zoom_jump(self, frame_idx: int, window_size: int, total_frames: int):
        """Re-center the zoom window around frame_idx (called after edge approach)."""
        if self._absorb_wizard:
            return  # wizard controls the slider range; don't auto-recenter
        half = window_size // 2
        ws = max(0, frame_idx - half)
        we = min(total_frames - 1, ws + window_size)
        if we == total_frames - 1:
            ws = max(0, total_frames - window_size)
        self.slider_window_center = frame_idx
        self.frame_slider.config(from_=ws, to=we)
        self.zoom_info_label.config(text=f"({ws}-{we})")
        self.zoom_jump_scheduled = None
        self._mark_presence_dirty()
        self._update_presence_bar()

    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier."""
        self.playback_speed.set(speed)
        labels = {0.25: "Slow-", 0.5: "Slow", 1.0: "Norm", 2.0: "Fast", 4.0: "Fast+"}
        self.speed_info_label.config(text=f"({labels.get(speed, f'{speed}x')})")
        self._update_speed_button_highlight()

    def _update_zoom_button_highlight(self):
        """Sunken relief on the active zoom button."""
        cur = self.slider_zoom_level.get()
        for level, btn in self.zoom_buttons.items():
            btn.config(relief=tk.SUNKEN if level == cur else tk.RAISED)

    def _update_speed_button_highlight(self):
        """Sunken relief on the active speed button."""
        cur = self.playback_speed.get()
        for speed, btn in self.speed_buttons.items():
            btn.config(relief=tk.SUNKEN if speed == cur else tk.RAISED)

    # ============================================================
    # Presence Bar
    # ============================================================

    def _get_presence_frames(self) -> set:
        """Return cached set of frame indices where the selected instance has a mask."""
        if not self.selected_instance or not self.selected_concept or not self.project:
            return set()
        key = (self.selected_concept.name, self.selected_instance.sam3_obj_id)
        if key not in self._presence_cache:
            mask_dir = os.path.join(
                self.project.project_dir, "concepts", self.selected_concept.name,
                "instances", str(self.selected_instance.sam3_obj_id), "masks"
            )
            frames: set = set()
            if os.path.exists(mask_dir):
                for fname in os.listdir(mask_dir):
                    if fname.endswith('.png') or fname.endswith('.npz'):
                        try:
                            frames.add(int(os.path.splitext(fname)[0]))
                        except ValueError:
                            pass
            self._presence_cache[key] = frames
        return self._presence_cache[key]

    def _invalidate_presence_cache(self):
        """Clear presence cache (call after masks are written or instance changes)."""
        self._presence_cache.clear()
        self._presence_bar_dirty = True

    def _mark_presence_dirty(self):
        """Mark the presence bar and quality colorbars for a full redraw on next update."""
        self._presence_bar_dirty = True
        self._quality_bars_dirty = True

    # ============================================================
    # In-memory annotation cache
    # ============================================================

    def _instance_key(self):
        """Return (concept_name, obj_id) cache key for the selected instance, or None."""
        if self.selected_instance and self.selected_concept:
            return (self.selected_concept.name, self.selected_instance.sam3_obj_id)
        return None

    def _select_instance_for_undo(self, concept_name: str, obj_id: int):
        """Select a concept+instance in the tree and update internal state.
        Used by undo/redo to navigate to the context of a past action.
        Does NOT trigger a full display_frame cascade (caller does that)."""
        if not self.project:
            return
        concept = self.project.get_concept_by_name(concept_name)
        if not concept:
            return
        inst = next((i for i in concept.instances
                     if i.sam3_obj_id == obj_id and not i.deleted), None)
        if not inst:
            return
        self.selected_concept = concept
        self.selected_instance = inst
        # Update tree selection silently
        self._updating_tree = True
        try:
            for concept_item in self.concept_tree.get_children():
                for inst_item in self.concept_tree.get_children(concept_item):
                    tags = self.concept_tree.item(inst_item, 'tags')
                    if (len(tags) >= 3 and tags[0] == 'instance'
                            and tags[1] == concept_name
                            and int(tags[2]) == obj_id):
                        self.concept_tree.selection_set(inst_item)
                        self.concept_tree.see(inst_item)
                        return
        finally:
            self._updating_tree = False

    def _select_instance_in_tree(self, concept_name: str, obj_id: int) -> bool:
        """Select concept+instance in the tree and fully sync UI state.

        Like _select_instance_for_undo() but also updates selected_instances,
        the selection label and the per-concept quality bars.  Used after
        operations that replace the selected instance (merge, absorb) so the
        result stays highlighted.  Caller is responsible for display_frame().
        Returns True if the instance was found and selected.
        """
        if not self.project:
            return False
        concept = self.project.get_concept_by_name(concept_name)
        if not concept:
            return False
        inst = next((i for i in concept.instances
                     if i.sam3_obj_id == obj_id and not i.deleted), None)
        if not inst:
            return False
        self.selected_concept = concept
        self.selected_instance = inst
        self.selected_instances = [(concept, inst)]
        self.selected_label.config(text=f"{inst.user_name} (ID: {inst.sam3_obj_id})")
        self._updating_tree = True
        try:
            for concept_item in self.concept_tree.get_children():
                for inst_item in self.concept_tree.get_children(concept_item):
                    tags = self.concept_tree.item(inst_item, 'tags')
                    if (len(tags) >= 3 and tags[0] == 'instance'
                            and tags[1] == concept_name
                            and int(tags[2]) == obj_id):
                        self.concept_tree.selection_set(inst_item)
                        self.concept_tree.see(inst_item)
                        break
        finally:
            self._updating_tree = False
        self._refresh_concept_quality_bars(redraw=False)
        return True

    def _update_points_cache(self):
        """Sync refinement_points + refinement_boxes → caches for the current frame/instance."""
        key = self._instance_key()
        if key is None:
            return
        pts = list(self.refinement_points)
        boxes = list(self.refinement_boxes)
        # Points cache
        frame_dict = self._points_cache.setdefault(key, {})
        frame_dict[self.current_frame_idx] = pts  # empty = tombstone
        # Boxes cache
        box_dict = self._boxes_cache.setdefault(key, {})
        box_dict[self.current_frame_idx] = boxes  # empty = tombstone
        # All-annotations caches (used for navigation; don't carry tombstones)
        all_dict = self._all_annotations_cache.setdefault(key, {})
        if pts:
            all_dict[self.current_frame_idx] = pts
        else:
            all_dict.pop(self.current_frame_idx, None)
        all_box_dict = self._all_boxes_cache.setdefault(key, {})
        if boxes:
            all_box_dict[self.current_frame_idx] = boxes
        else:
            all_box_dict.pop(self.current_frame_idx, None)
        self._dirty_keys.add(key)

    def _load_points_cache_from_project(self):
        """Populate point/box caches (pending only) and all-annotation caches from refinements.json."""
        self._points_cache.clear()
        self._boxes_cache.clear()
        self._all_annotations_cache.clear()
        self._all_boxes_cache.clear()
        self._dirty_keys.clear()
        if not self.project:
            return
        for concept in self.project.concepts:
            for inst in concept.instances:
                if inst.deleted:
                    continue
                rpath = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(inst.sam3_obj_id), "refinements.json"
                )
                if not os.path.exists(rpath):
                    continue
                with open(rpath) as f:
                    entries = json.load(f).get("refinements", [])
                pending_pts: Dict[int, List[tuple]] = {}
                pending_box: Dict[int, List[tuple]] = {}
                all_pts: Dict[int, List[tuple]] = {}
                all_box: Dict[int, List[tuple]] = {}
                for entry in entries:
                    frame_idx = entry.get("frame_idx")
                    if frame_idx is None:
                        continue
                    pts = [(p["x"], p["y"], p["is_positive"])
                           for p in entry.get("points", [])]
                    boxes = [(b["x1"], b["y1"], b["x2"], b["y2"])
                             for b in entry.get("boxes", [])]
                    if pts:
                        all_pts[frame_idx] = pts
                    else:
                        all_pts.pop(frame_idx, None)
                    if boxes:
                        all_box[frame_idx] = boxes
                    else:
                        all_box.pop(frame_idx, None)
                    if not entry.get("propagated", False):
                        if pts:
                            pending_pts[frame_idx] = pts
                        else:
                            pending_pts.pop(frame_idx, None)
                        if boxes:
                            pending_box[frame_idx] = boxes
                        else:
                            pending_box.pop(frame_idx, None)
                key = (concept.name, inst.sam3_obj_id)
                if pending_pts:
                    self._points_cache[key] = pending_pts
                if pending_box:
                    self._boxes_cache[key] = pending_box
                if all_pts:
                    self._all_annotations_cache[key] = all_pts
                if all_box:
                    self._all_boxes_cache[key] = all_box

    def _on_canvas_configure(self, event):
        """Redraw video when the canvas is resized (debounced 80 ms to avoid per-pixel floods)."""
        if not self.compositor:
            return
        if self._canvas_resize_pending:
            self.root.after_cancel(self._canvas_resize_pending)
        self._canvas_resize_pending = self.root.after(80, self._canvas_resize_redraw)

    def _canvas_resize_redraw(self):
        self._canvas_resize_pending = None
        if self.compositor:
            self.display_frame()

    def _on_presence_canvas_configure(self, event):
        """Cache canvas size on resize and redraw."""
        self._presence_canvas_w = event.width
        self._presence_canvas_h = event.height
        self._mark_presence_dirty()
        self._update_presence_bar()

    def _update_presence_bar(self):
        """Update the instance presence bar.

        Background blocks are rendered to a numpy pixel array and cached as a
        PhotoImage — no per-block canvas items.  On plain frame navigation only
        the 3-item playhead is redrawn (O(1) Tk ops regardless of frame count).
        Full background regeneration happens only when dirty (instance/zoom changed).
        """
        if not hasattr(self, 'presence_canvas'):
            return
        canvas = self.presence_canvas
        w = self._presence_canvas_w
        h = self._presence_canvas_h
        if w <= 1 or h <= 1:
            return

        if not self.num_frames:
            return

        # Compute visible frame range (needed for both background and playhead)
        zoom = self.slider_zoom_level.get()
        if zoom == 1:
            f_start, f_end = 0, self.num_frames - 1
        else:
            f_start = int(self.frame_slider.cget('from'))
            f_end = int(self.frame_slider.cget('to'))
        visible = max(1, f_end - f_start + 1)

        if self._presence_bar_dirty:
            # Render background into a numpy pixel array (no canvas items, no X11 per-rect flush)
            import numpy as _np
            bg_img = _np.full((h, w, 3), [42, 42, 42], dtype=_np.uint8)

            if self.selected_instance and self.selected_concept:
                inst = self.selected_instance
                concept = self.selected_concept
                base = inst.color_rgb or concept.color_rgb or (100, 180, 255)
                r, g, b = base
                hv, sv, vv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                r2, g2, b2 = colorsys.hsv_to_rgb(hv, 1.0, 1.0)
                fill_rgb = [int(r2 * 255), int(g2 * 255), int(b2 * 255)]

                period_peaks = inst.period_peaks if inst.period_peaks else []
                # Build a list of valid (numeric) avg_pixel_ratio values — older
                # projects stored None when pixel counts weren't tracked yet.
                valid_ratios = [
                    p.get('avg_pixel_ratio') for p in period_peaks
                    if p.get('avg_pixel_ratio') is not None
                ]
                if period_peaks and valid_ratios:
                    # Use saved 99th-percentile norm so typical frames read near full
                    # height; fall back to max avg_pixel_ratio for old projects.
                    saved_norm = getattr(inst, 'presence_norm', 0.0)
                    if saved_norm > 0:
                        max_ratio = saved_norm
                    else:
                        max_ratio = max(valid_ratios)
                    if max_ratio <= 0:
                        max_ratio = 1.0
                    inner_h = h - 2
                    for period in period_peaks:
                        draw_start = max(period['start'], f_start)
                        draw_end = min(period['end'], f_end)
                        if draw_start > draw_end:
                            continue
                        avg = period.get('avg_pixel_ratio')
                        if avg is None:
                            avg = 0.0
                        rel = min(1.0, (avg / max_ratio) ** 0.5)
                        bar_h = max(2, int(rel * inner_h))
                        y0 = h - 1 - bar_h
                        x0 = max(0, int((draw_start - f_start) / visible * w))
                        x1 = min(w, max(x0 + 1, int((draw_end - f_start + 1) / visible * w)))
                        bg_img[y0:h - 1, x0:x1] = fill_rgb
                else:
                    # Fallback: binary presence from mask file scan (old projects / no peak data)
                    presence = self._get_presence_frames()
                    if presence:
                        sorted_f = sorted(f for f in presence if f_start <= f <= f_end)
                        if sorted_f:
                            run_s = run_e = sorted_f[0]
                            for fr in sorted_f[1:]:
                                if fr == run_e + 1:
                                    run_e = fr
                                else:
                                    x0 = max(0, int((run_s - f_start) / visible * w))
                                    x1 = min(w, max(x0 + 1, int((run_e - f_start + 1) / visible * w)))
                                    bg_img[1:h - 1, x0:x1] = fill_rgb
                                    run_s = run_e = fr
                            x0 = max(0, int((run_s - f_start) / visible * w))
                            x1 = min(w, max(x0 + 1, int((run_e - f_start + 1) / visible * w)))
                            bg_img[1:h - 1, x0:x1] = fill_rgb

            self._presence_bar_photo = ImageTk.PhotoImage(
                Image.fromarray(bg_img, mode='RGB'))
            if self._presence_bar_img_id is None:
                canvas.delete("all")
                self._presence_bar_img_id = canvas.create_image(
                    0, 0, image=self._presence_bar_photo, anchor=tk.NW)
            else:
                canvas.itemconfig(self._presence_bar_img_id,
                                  image=self._presence_bar_photo)
            self._presence_bar_dirty = False
        else:
            pass  # background image unchanged

        # Always remove old playhead before redrawing
        canvas.delete("playhead")

        # Draw playhead (always): black shadow + bright line + triangle, tagged "playhead"
        if f_start <= self.current_frame_idx <= f_end:
            cx = (self.current_frame_idx - f_start) / visible * w
            canvas.create_line(cx + 1, 0, cx + 1, h, fill='black', width=4, tags="playhead")
            canvas.create_line(cx, 0, cx, h, fill='#ffd700', width=3, tags="playhead")
            ts = 7
            canvas.create_polygon(cx - ts, 0, cx + ts, 0, cx, ts + 3,
                                   fill='#ffd700', outline='black', width=1, tags="playhead")

        # Update quality colorbars (only the playhead position needs to move)
        self._draw_quality_colorbars()

    def _on_presence_bar_click(self, event):
        """Click on the presence bar to jump to that frame."""
        if not self.num_frames:
            return
        w = self.presence_canvas.winfo_width()
        if w <= 1:
            return
        zoom = self.slider_zoom_level.get()
        if zoom == 1:
            f_start, f_end = 0, self.num_frames - 1
        else:
            f_start = int(self.frame_slider.cget('from'))
            f_end = int(self.frame_slider.cget('to'))
        visible = max(1, f_end - f_start + 1)
        frame_idx = int(f_start + event.x / w * visible)
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        self._go_to_frame(frame_idx)

    # ============================================================
    # Quality Metrics Colorbars
    # ============================================================

    def _load_quality_metrics(self, project_dir: str):
        """Load quality_metrics.npz from project_dir into self.quality_bg/overlap.

        The Overlap and Change bars are per-concept: they follow the currently
        selected concept (see _refresh_concept_quality_bars()).  The Background
        bar is always the single global track.  Older metrics files without
        per-concept data fall back to the global all-concepts-combined track.
        """
        try:
            from utils import load_quality_metrics_grouped
            data = load_quality_metrics_grouped(project_dir)
            if data is None:
                self.quality_bg = None
                self.quality_overlap_by_group = None
                self.quality_inter_by_group = None
                self._quality_overlap_global = None
                self._quality_inter_global = None
            else:
                self.quality_bg = data['background_ratios']
                self.quality_overlap_by_group = data['overlap_ratios_by_group']
                self.quality_inter_by_group = data['inter_frame_changes_by_group']
                # Fallback track for older files / concepts missing from the file.
                self._quality_overlap_global = data['overlap_ratios']
                self._quality_inter_global = data['inter_frame_changes']
        except Exception as e:
            print(f"[quality] Could not load metrics: {e}")
            self.quality_bg = None
            self.quality_overlap_by_group = None
            self.quality_inter_by_group = None
            self._quality_overlap_global = None
            self._quality_inter_global = None
        self._refresh_concept_quality_bars(redraw=True)

    def _refresh_concept_quality_bars(self, redraw: bool = True):
        """Point the Overlap/Change bars at the currently selected concept.

        When per-concept metrics exist and a concept is selected, show that
        concept's tracks; otherwise fall back to the global combined tracks.
        The Background bar is unaffected (always global).
        """
        name = self.selected_concept.name if self.selected_concept else None
        overlap_by_group = self.quality_overlap_by_group or {}
        key = name if name in overlap_by_group else None

        # No-op when the displayed concept hasn't changed — avoids marking the
        # bars dirty (and forcing a full background regen) on every tree click,
        # frame step, or playback tick that re-runs selection logic.
        if key == getattr(self, "_quality_bars_concept", "__unset__") and not redraw:
            return
        self._quality_bars_concept = key

        if key is not None:
            self.quality_overlap = overlap_by_group[key]
            self.quality_inter = (self.quality_inter_by_group or {}).get(key)
        else:
            self.quality_overlap = self._quality_overlap_global
            self.quality_inter = self._quality_inter_global

        self._quality_bars_dirty = True
        if redraw:
            self._draw_quality_colorbars()

    def _draw_quality_colorbars(self):
        """Render the overlap, background-ratio, and inter-frame change colorbars with a playhead.

        Background is rendered to a numpy pixel array and cached as a PhotoImage —
        O(canvas_width) numpy ops, zero per-pixel canvas items.  On plain frame
        navigation only the 3-item playhead is redrawn (O(1) Tk ops).
        Full background regeneration only when _quality_bars_dirty is set.
        """
        if not hasattr(self, 'quality_overlap_canvas'):
            return

        zoom = self.slider_zoom_level.get()
        if zoom == 1:
            f_start, f_end = 0, self.num_frames - 1
        else:
            f_start = int(self.frame_slider.cget('from'))
            f_end = int(self.frame_slider.cget('to'))
        visible = max(1, f_end - f_start + 1)

        def _make_bar_image(canvas, values, low_color, high_color):
            import numpy as _np
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w <= 1 or h <= 1 or not self.num_frames:
                return None
            lr, lg, lb = low_color
            hr, hg, hb = high_color
            img = _np.empty((h, w, 3), dtype=_np.uint8)
            img[:, :] = low_color
            if values:
                n = len(values)
                # Map pixel columns to frames — O(w) numpy, zero Tk calls
                px_arr = _np.arange(w)
                f_arr = (f_start + (px_arr * visible / w).astype(_np.int32))
                valid = f_arr < n
                t_arr = _np.clip(_np.array(values)[f_arr[valid]], 0.0, 1.0)
                img[:, px_arr[valid], 0] = (lr + t_arr * (hr - lr)).astype(_np.uint8)
                img[:, px_arr[valid], 1] = (lg + t_arr * (hg - lg)).astype(_np.uint8)
                img[:, px_arr[valid], 2] = (lb + t_arr * (hb - lb)).astype(_np.uint8)
            return ImageTk.PhotoImage(Image.fromarray(img, mode='RGB'))

        def _put_image(canvas, photo, img_id_attr, photo_attr):
            if photo is None:
                return
            setattr(self, photo_attr, photo)
            img_id = getattr(self, img_id_attr)
            if img_id is None:
                canvas.delete("all")
                setattr(self, img_id_attr,
                        canvas.create_image(0, 0, image=photo, anchor=tk.NW))
            else:
                canvas.itemconfig(img_id, image=photo)

        def _draw_playhead(canvas):
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w <= 1 or h <= 1 or not self.num_frames:
                return
            canvas.delete("qplayhead")
            if f_start <= self.current_frame_idx <= f_end:
                cx = (self.current_frame_idx - f_start) / visible * w
                canvas.create_line(cx + 1, 0, cx + 1, h, fill='black', width=4, tags="qplayhead")
                canvas.create_line(cx, 0, cx, h, fill='#ffd700', width=3, tags="qplayhead")
                ts = 6
                canvas.create_polygon(cx - ts, 0, cx + ts, 0, cx, ts + 3,
                                      fill='#ffd700', outline='black', width=1, tags="qplayhead")

        if self._quality_bars_dirty:
            _put_image(self.quality_overlap_canvas,
                       _make_bar_image(self.quality_overlap_canvas,
                                       self.quality_overlap, (42, 42, 42), (220, 60, 30)),
                       '_quality_overlap_img_id', '_quality_overlap_photo')
            _put_image(self.quality_bg_canvas,
                       _make_bar_image(self.quality_bg_canvas,
                                       self.quality_bg, (42, 42, 42), (30, 120, 200)),
                       '_quality_bg_img_id', '_quality_bg_photo')
            _put_image(self.quality_inter_canvas,
                       _make_bar_image(self.quality_inter_canvas,
                                       self.quality_inter, (42, 42, 42), (60, 200, 90)),
                       '_quality_inter_img_id', '_quality_inter_photo')
            self._quality_bars_dirty = False

        _draw_playhead(self.quality_overlap_canvas)
        _draw_playhead(self.quality_bg_canvas)
        _draw_playhead(self.quality_inter_canvas)

    def _on_quality_bar_click(self, event):
        """Click on a quality colorbar to jump to that frame."""
        if not self.num_frames:
            return
        canvas = event.widget
        w = canvas.winfo_width()
        if w <= 1:
            return
        zoom = self.slider_zoom_level.get()
        if zoom == 1:
            f_start, f_end = 0, self.num_frames - 1
        else:
            f_start = int(self.frame_slider.cget('from'))
            f_end = int(self.frame_slider.cget('to'))
        visible = max(1, f_end - f_start + 1)
        frame_idx = int(f_start + event.x / w * visible)
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        self._go_to_frame(frame_idx)

    # ============================================================
    # Video Loading
    # ============================================================

    def _create_compositor(self):
        """Create DynamicFrameCompositor for self.project.

        Search order when the stored paths are unreachable:
          1. Stored video_path / mjpeg_video_path / frames_dir (handled inside compositor)
          2. alt_video_paths list (tried automatically, verified by frame count)
          3. alt_frames_dirs list (tried automatically, verified by frame count)
          4. User prompt (chosen path added to alt list so it is found automatically next time)

        The compositor is created with a temporarily patched project attribute when using
        an alternative path; the patch is always reverted so save() never writes the
        machine-local override back to project.json.

        Returns the compositor, or None if the user cancelled.
        """
        try:
            return DynamicFrameCompositor(self.project)
        except ValueError as exc:
            if "Failed to open video" not in str(exc):
                raise

        # Try stored alternative paths before prompting the user
        comp = self._try_alt_paths_compositor()
        if comp is not None:
            return comp

        # All automatic options exhausted — ask the user
        stored = self.project.video_path or "(unknown)"
        messagebox.showinfo(
            "Video Not Found",
            f"The video file could not be opened:\n\n{stored}\n\n"
            "Please locate the video file or a frames directory to continue.",
        )
        return self._prompt_user_for_video()

    # ── Alternative-path helpers ─────────────────────────────────────────────

    def _try_alt_paths_compositor(self) -> Optional[DynamicFrameCompositor]:
        """Try alt_video_paths then alt_frames_dirs; return compositor on first hit."""
        for alt in list(getattr(self.project, 'alt_video_paths', [])):
            comp = self._try_video_path(alt)
            if comp is not None:
                print(f"[VideoPath] Auto-found video at alt path: {alt}")
                return comp
        for alt in list(getattr(self.project, 'alt_frames_dirs', [])):
            comp = self._try_frames_dir(alt)
            if comp is not None:
                print(f"[VideoPath] Auto-found frames at alt dir: {alt}")
                return comp
        return None

    def _try_video_path(self, path: str) -> Optional[DynamicFrameCompositor]:
        """Try to create a compositor using *path* as the video file.

        Returns compositor on success, None on any failure.
        Cross-platform: silently skips paths that do not exist on this OS.
        """
        try:
            p = Path(path)
            if not p.is_file():
                return None
            if not self._verify_video_frame_count(str(p)):
                return None
            _orig = self.project.video_path
            self.project.video_path = str(p)
            try:
                return DynamicFrameCompositor(self.project)
            except Exception:
                return None
            finally:
                self.project.video_path = _orig
        except Exception:
            return None

    def _try_frames_dir(self, path: str) -> Optional[DynamicFrameCompositor]:
        """Try to create a compositor using *path* as the frames directory.

        Returns compositor on success, None on any failure.
        """
        try:
            p = Path(path)
            if not p.is_dir():
                return None
            if not self._verify_frames_dir_count(str(p)):
                return None
            _orig = self.project.frames_dir
            self.project.frames_dir = str(p)
            try:
                return DynamicFrameCompositor(self.project)
            except Exception:
                return None
            finally:
                self.project.frames_dir = _orig
        except Exception:
            return None

    def _verify_video_frame_count(self, video_path: str) -> bool:
        """Return True if the video has roughly the expected number of frames (≤1% off)."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            expected = self.project.num_frames
            return abs(count - expected) <= max(2, int(expected * 0.01))
        except Exception:
            return False

    def _verify_frames_dir_count(self, frames_dir: str) -> bool:
        """Return True if the directory contains roughly the expected number of image files."""
        try:
            count = sum(
                1 for f in os.listdir(frames_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            expected = self.project.num_frames
            return abs(count - expected) <= max(2, int(expected * 0.01))
        except Exception:
            return False

    def _prompt_user_for_video(self) -> Optional[DynamicFrameCompositor]:
        """Ask user to locate video file or frames directory.

        On success, saves the chosen path to the project's alt list so future
        loads find it automatically without prompting.
        """
        # Offer two options
        choice = messagebox.askyesno(
            "Locate Video",
            "Would you like to locate a video file?\n\n"
            "Click Yes to browse for a video file.\n"
            "Click No to browse for a frames directory instead.",
        )
        if choice:  # Yes → video file
            chosen = filedialog.askopenfilename(
                title="Locate source video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                    ("All files", "*.*"),
                ],
            )
            if not chosen:
                return None
            if not self._verify_video_frame_count(chosen):
                if not messagebox.askyesno(
                    "Frame Count Mismatch",
                    f"This video's frame count does not match the project "
                    f"({self.project.num_frames} frames expected).\n\n"
                    "Use it anyway?",
                ):
                    return None
            _orig = self.project.video_path
            self.project.video_path = chosen
            try:
                comp = DynamicFrameCompositor(self.project)
                self._add_alt_video_path(chosen)
                return comp
            except Exception as e:
                messagebox.showerror("Error", f"Could not open video:\n{e}")
                return None
            finally:
                self.project.video_path = _orig
        else:  # No → frames directory
            chosen = filedialog.askdirectory(title="Locate frames directory")
            if not chosen:
                return None
            if not self._verify_frames_dir_count(chosen):
                if not messagebox.askyesno(
                    "Frame Count Mismatch",
                    f"This directory's frame count does not match the project "
                    f"({self.project.num_frames} frames expected).\n\n"
                    "Use it anyway?",
                ):
                    return None
            _orig = self.project.frames_dir
            self.project.frames_dir = chosen
            try:
                comp = DynamicFrameCompositor(self.project)
                self._add_alt_frames_dir(chosen)
                return comp
            except Exception as e:
                messagebox.showerror("Error", f"Could not use frames directory:\n{e}")
                return None
            finally:
                self.project.frames_dir = _orig

    def _add_alt_video_path(self, path: str) -> None:
        """Add *path* to project.alt_video_paths (dedup) and save."""
        path = str(Path(path))  # normalise separators for this OS
        if path not in self.project.alt_video_paths:
            self.project.alt_video_paths.append(path)
            self._safe_project_save()

    def _add_alt_frames_dir(self, path: str) -> None:
        """Add *path* to project.alt_frames_dirs (dedup) and save."""
        path = str(Path(path))
        if path not in self.project.alt_frames_dirs:
            self.project.alt_frames_dirs.append(path)
            self._safe_project_save()

    # ── Last-project-dir helpers ─────────────────────────────────────────────

    def _get_last_project_parent(self) -> Optional[str]:
        """Return the parent directory of the last opened project, for initialdir."""
        if self._last_project_dir:
            try:
                parent = str(Path(self._last_project_dir).parent)
                if os.path.isdir(parent):
                    return parent
            except Exception:
                pass
        return None

    def _remember_project_dir(self, project_dir: str) -> None:
        """Persist *project_dir* as the last-opened project dir."""
        try:
            self._last_project_dir = str(Path(project_dir))
            _app = _load_app_settings()
            _app["last_project_dir"] = self._last_project_dir
            _save_app_settings(_app)
        except Exception as e:
            print(f"Warning: could not persist last project dir: {e}")

    # ── Project label helper ──────────────────────────────────────────────────

    def _update_project_label(self) -> None:
        """Refresh the project folder name shown above the canvas."""
        if not hasattr(self, '_project_folder_label'):
            return
        if not self.project:
            self._project_folder_label.config(text="No project loaded", fg='#888888')
            return
        path = self.project.project_dir
        max_len = 70
        if len(path) > max_len:
            display = f"...{path[-(max_len - 3):]}"
        else:
            display = path
        self._project_folder_label.config(
            text=f"Project: {display}", fg='#cccccc'
        )

    # ── Manage paths dialog ───────────────────────────────────────────────────

    def manage_paths_dialog(self) -> None:
        """Show a dialog listing all video/frame paths and allowing deletions."""
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        top = tk.Toplevel(self.root)
        top.title("Manage Video Paths")
        top.geometry("700x480")
        top.transient(self.root)
        top.grab_set()

        tk.Label(top, text="Video / Frame paths for this project",
                 font=("Arial", 10, "bold")).pack(pady=(10, 4))
        tk.Label(top,
                 text="Alternative paths are tried automatically when the primary path is missing.",
                 font=("Arial", 8), fg='#666666').pack()

        frame = tk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Scrollable content
        canvas = tk.Canvas(frame, bg='#f5f5f5')
        sb = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg='#f5f5f5')
        inner_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        def _on_inner_resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind('<Configure>', _on_inner_resize)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(inner_id, width=e.width))

        def _section(label):
            tk.Label(inner, text=label, font=("Arial", 9, "bold"),
                     bg='#f5f5f5', anchor=tk.W).pack(fill=tk.X, padx=4, pady=(10, 2))

        def _ro_row(text, tag="(primary)"):
            row = tk.Frame(inner, bg='#f5f5f5')
            row.pack(fill=tk.X, padx=4, pady=1)
            tk.Label(row, text=tag, font=("Arial", 8), fg='#999999',
                     bg='#f5f5f5', width=12, anchor=tk.E).pack(side=tk.LEFT)
            tk.Label(row, text=text or "(none)", font=("Arial", 8),
                     fg='#444444', bg='#e8e8e8', anchor=tk.W,
                     relief=tk.SUNKEN, padx=4).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        def _alt_list(path_list: list, remove_fn):
            """Render a deletable list of alternative paths."""
            if not path_list:
                tk.Label(inner, text="  (none)", font=("Arial", 8),
                         fg='#aaaaaa', bg='#f5f5f5').pack(anchor=tk.W, padx=16)
                return
            for i, p in enumerate(list(path_list)):
                row = tk.Frame(inner, bg='#f5f5f5')
                row.pack(fill=tk.X, padx=4, pady=1)
                reachable = False
                try:
                    reachable = Path(p).exists()
                except Exception:
                    pass
                color = '#335522' if reachable else '#aa3322'
                status = "[found]" if reachable else "[missing]"
                tk.Label(row, text=status, font=("Arial", 8), fg=color,
                         bg='#f5f5f5', width=10, anchor=tk.E).pack(side=tk.LEFT)
                tk.Label(row, text=p, font=("Arial", 8), fg='#444444',
                         bg='#e8e8e8', anchor=tk.W, relief=tk.SUNKEN,
                         padx=4).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
                idx_capture = i

                def _del(idx=idx_capture, pl=path_list, fn=remove_fn):
                    fn(idx)
                    top.destroy()
                    self.manage_paths_dialog()  # refresh

                tk.Button(row, text="Remove", font=("Arial", 8),
                          command=_del).pack(side=tk.RIGHT, padx=4)

        _section("Primary video path")
        _ro_row(self.project.video_path)

        _section("MJPEG video path")
        _ro_row(self.project.mjpeg_video_path)

        _section("Primary frames directory")
        _ro_row(self.project.frames_dir)

        _section("Alternative video paths (tried in order)")
        def _rm_video(idx):
            self.project.alt_video_paths.pop(idx)
            self._safe_project_save()

        def _rm_frames(idx):
            self.project.alt_frames_dirs.pop(idx)
            self._safe_project_save()

        _alt_list(self.project.alt_video_paths, _rm_video)

        _section("Alternative frames directories (tried in order)")
        _alt_list(self.project.alt_frames_dirs, _rm_frames)

        tk.Button(top, text="Close", command=top.destroy).pack(pady=8)

    # ── Image-mode helpers ────────────────────────────────────────────────────

    _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.gif'}

    @property
    def is_image_mode(self) -> bool:
        return self.project is not None and getattr(self.project, 'is_image', False)

    def _get_image_info(self, image_path: str):
        """Return (num_frames=1, (width, height), fps=1.0) from an image file."""
        with Image.open(image_path) as img:
            width, height = img.size
        return 1, (width, height), 1.0

    def _prepare_image_frames_dir(self, image_path: str, project_dir: str) -> str:
        """Convert image to 000000.jpg in <project_dir>/frames/. Returns frames_dir."""
        frames_dir = os.path.join(project_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        dest = os.path.join(frames_dir, "000000.jpg")
        with Image.open(image_path) as img:
            img.convert("RGB").save(dest, "JPEG", quality=95)
        return frames_dir

    def _update_nav_visibility(self):
        """Show or hide video-only nav rows based on whether current project is an image."""
        rows_and_packs = [
            (self._nav_row1, dict(fill=tk.X, padx=5, pady=(3, 0))),
            (self._nav_row2, dict(fill=tk.X, padx=5, pady=(1, 0))),
            (self._nav_row3, dict(fill=tk.X, padx=5, pady=(1, 3))),
            (self._nav_row4, dict(fill=tk.X, padx=5, pady=(1, 3))),
            (self._nav_row5, dict(fill=tk.X, padx=5, pady=(0, 3))),
        ]
        if self.is_image_mode:
            for row, _ in rows_and_packs:
                if row:
                    row.pack_forget()
            if self._export_btn:
                self._export_btn.config(text="Export Image")
        else:
            for row, pack_kw in rows_and_packs:
                if row:
                    row.pack(**pack_kw)
            if self._export_btn:
                self._export_btn.config(text="Export Video")

    def load_gaze_csv(self, gaze_path: Optional[str] = None, world_path: Optional[str] = None,
                       link_to_project: bool = True, silent: bool = False):
        """Load a gaze CSV + world-timestamps CSV and overlay the gaze position.

        When link_to_project is True and a project is open, the pair of paths is stored in
        project.json so it is auto-loaded (silently) next time the project is opened.
        """
        if gaze_path is None or world_path is None:
            if self.gaze_points_by_frame is not None:
                choice = messagebox.askyesnocancel(
                    "Gaze Data Loaded",
                    f"Gaze data is already loaded ({len(self.gaze_points_by_frame)} frames).\n\n"
                    "Yes = load a different pair of CSVs\nNo = clear the current gaze overlay",
                )
                if choice is None:
                    return
                if choice is False:
                    self.gaze_points_by_frame = None
                    self.show_gaze_checkbutton.config(state=tk.DISABLED)
                    self.display_frame()
                    return

            initial_dir = os.path.dirname(self.gaze_csv_path) if self.gaze_csv_path else None
            gaze_path = filedialog.askopenfilename(
                title="Select gaze CSV (timestamp [ns], gaze x [px], gaze y [px])",
                initialdir=initial_dir,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if not gaze_path:
                return
            initial_dir = os.path.dirname(self.gaze_world_path) if self.gaze_world_path else os.path.dirname(gaze_path)
            world_path = filedialog.askopenfilename(
                title="Select world-timestamps CSV (one row per video frame, timestamp [ns])",
                initialdir=initial_dir,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if not world_path:
                return

        try:
            mapping = gaze_overlay.load_gaze_csv(gaze_path, world_path)
        except Exception as e:
            if silent:
                self.status_var.set(f"Linked gaze CSV failed to load: {e}")
                return
            messagebox.showerror("Load Gaze CSV Failed", str(e))
            return

        self.gaze_points_by_frame = mapping
        self.gaze_csv_path = gaze_path
        self.gaze_world_path = world_path
        self.show_gaze_var.set(True)
        self.show_gaze_checkbutton.config(state=tk.NORMAL)

        if link_to_project and self.project is not None:
            if (getattr(self.project, "gaze_csv_path", None) != gaze_path
                    or getattr(self.project, "gaze_world_path", None) != world_path):
                self.project.gaze_csv_path = gaze_path
                self.project.gaze_world_path = world_path
                self._metadata_dirty = True

        if silent:
            self.status_var.set(f"Gaze CSV auto-loaded: {os.path.basename(gaze_path)}")
        else:
            messagebox.showinfo("Gaze CSV Loaded", gaze_overlay.summarize(mapping))
        self.display_frame()

    def load_video(self):
        """Load a video file or image and create new project"""

        video_path = filedialog.askopenfilename(
            title="Select Video or Image File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp"),
                ("All files", "*.*"),
            ],
            initialdir=self._get_last_project_parent(),
        )

        if not video_path:
            return

        # Release any live sessions before loading a new video
        self._release_all_sessions()

        is_image = Path(video_path).suffix.lower() in self._IMAGE_EXTS

        try:
            self.status_var.set("Loading image..." if is_image else "Loading video...")
            self.root.update()

            # Get media info
            if is_image:
                num_frames, dims, fps = self._get_image_info(video_path)
            else:
                num_frames, dims, fps = get_video_info(video_path)

            # Ask for project directory
            project_dir = filedialog.askdirectory(
                title="Select Project Directory",
                initialdir=self._get_last_project_parent(),
            )
            if not project_dir:
                self.status_var.set("Load cancelled.")
                return

            # For images: copy/convert into a persistent frames dir so SAM3 can use it.
            frames_dir = None
            if is_image:
                self.status_var.set("Preparing image frames...")
                self.root.update()
                frames_dir = self._prepare_image_frames_dir(video_path, project_dir)

            # Create project
            self.project = SAM3Project.create_new(
                project_dir=project_dir,
                video_path=video_path,
                num_frames=num_frames,
                frame_dimensions=dims,
                fps=fps,
                device=self.device,
                is_image=is_image,
                frames_dir=frames_dir,
            )

            # Carry over any vocabulary already loaded in this session
            if self.vocabulary_path:
                self.project.vocabulary_path = self.vocabulary_path

            # Save project
            self._safe_project_save()

            # Persist last project dir for next file dialog
            self._remember_project_dir(project_dir)

            # Setup video state
            self.video_path = video_path
            self.num_frames = num_frames
            self.frame_dimensions = dims
            self.fps = fps
            self.current_frame_idx = 0

            # Initialize compositor
            self.compositor = DynamicFrameCompositor(self.project)

            # Populate in-memory annotation cache from project files
            self._load_points_cache_from_project()

            # Update UI — hide/show nav controls based on image vs video
            self._update_nav_visibility()
            self._update_project_label()
            self.frame_slider.configure(to=num_frames - 1)
            self.selected_concept = None
            self.selected_instance = None
            self.selected_instances = []
            self.update_concept_tree()
            self._auto_select_first_instance()
            self.display_frame()

            if is_image:
                self.status_var.set(f"Loaded image: {Path(video_path).name} ({dims[0]}x{dims[1]})")
            else:
                self.status_var.set(f"Loaded: {Path(video_path).name} ({num_frames} frames, {dims[0]}x{dims[1]}, {fps:.2f} fps)")

        except Exception as e:
            import traceback
            print(f"\n=== ERROR: Failed to load {'image' if is_image else 'video'} ===")
            traceback.print_exc()
            print(f"=====================================\n")
            messagebox.showerror("Error", f"Failed to load {'image' if is_image else 'video'}:\n{e}")
            self.status_var.set("Error loading file.")

    def load_project(self):
        """Load existing SAM3 project"""

        project_dir = filedialog.askdirectory(
            title="Select Project Directory",
            initialdir=self._get_last_project_parent(),
        )
        if not project_dir:
            return

        self._load_project_from_dir(project_dir)

    def _load_project_from_dir(self, project_dir: str):
        """Load a SAM3 project given its directory path (shared by load_project() and --project CLI arg)"""

        # Release any live sessions before loading a different project
        self._release_all_sessions()

        try:
            self.status_var.set("Loading project...")
            self.root.update()

            _tp0 = _time.perf_counter()
            # Load project
            self.project = SAM3Project.load(project_dir)
            _tp1 = _time.perf_counter()

            # Persist last project dir for next file dialog
            self._remember_project_dir(project_dir)

            # Setup video state
            self.video_path = self.project.video_path
            self.num_frames = self.project.num_frames
            self.frame_dimensions = self.project.frame_dimensions
            self.fps = self.project.fps
            self.current_frame_idx = 0

            # Initialize compositor (prompts for video if stored path is inaccessible)
            self.compositor = self._create_compositor()
            if self.compositor is None:
                self.status_var.set("Load cancelled: video not found.")
                return
            _tp2 = _time.perf_counter()

            # Presence cache must be cleared so it gets rebuilt from actual mask files
            # (not stale pre-reload values, which may be wrong after external batch refinement)
            self._invalidate_presence_cache()

            # Populate in-memory annotation cache from project files
            self._load_points_cache_from_project()
            _tp3 = _time.perf_counter()

            # Load quality metrics if available
            self._load_quality_metrics(project_dir)

            # Update UI — hide/show nav controls based on image vs video
            self._update_nav_visibility()
            self._update_project_label()
            self.frame_slider.configure(to=self.num_frames - 1)
            self.selected_concept = None
            self.selected_instance = None
            self.selected_instances = []
            self.update_concept_tree()
            self._auto_select_first_instance()

            # Auto-load the vocabulary file linked to this project, if any.
            # load_vocabulary() re-reads the file, so external edits are picked up.
            self.vocabulary = {}
            self.vocabulary_path = None
            linked_vocab = getattr(self.project, "vocabulary_path", None)
            if linked_vocab:
                if os.path.isfile(linked_vocab):
                    self.load_vocabulary(linked_vocab, link_to_project=False)
                else:
                    self.status_var.set(
                        f"Linked vocabulary not found: {os.path.basename(linked_vocab)}"
                    )

            # Auto-load the gaze CSV pair linked to this project, if any.
            self.gaze_points_by_frame = None
            self.gaze_csv_path = None
            self.gaze_world_path = None
            self.show_gaze_var.set(True)
            self.show_gaze_checkbutton.config(state=tk.DISABLED)
            linked_gaze_csv = getattr(self.project, "gaze_csv_path", None)
            linked_gaze_world = getattr(self.project, "gaze_world_path", None)
            if linked_gaze_csv and linked_gaze_world:
                if os.path.isfile(linked_gaze_csv) and os.path.isfile(linked_gaze_world):
                    self.load_gaze_csv(linked_gaze_csv, linked_gaze_world,
                                        link_to_project=False, silent=True)
                else:
                    self.status_var.set(
                        f"Linked gaze CSV not found: {os.path.basename(linked_gaze_csv)}"
                    )

            _tp4 = _time.perf_counter()
            self.display_frame()
            _tp5 = _time.perf_counter()

            print(f"[LOAD] total={(_tp5-_tp0)*1000:.1f}ms | "
                  f"json={(_tp1-_tp0)*1000:.1f}ms | "
                  f"compositor={(_tp2-_tp1)*1000:.1f}ms | "
                  f"points_cache={(_tp3-_tp2)*1000:.1f}ms | "
                  f"update_tree={(_tp4-_tp3)*1000:.1f}ms | "
                  f"display_frame={(_tp5-_tp4)*1000:.1f}ms")

            self.status_var.set(f"Loaded project: {Path(project_dir).name}")

        except Exception as e:
            import traceback
            print(f"\n=== ERROR: Failed to load project ===")
            traceback.print_exc()
            print(f"=====================================\n")
            messagebox.showerror("Error", f"Failed to load project:\n{e}")
            self.status_var.set("Error loading project.")

    def save_project(self):
        """Save current project"""

        if not self.project:
            messagebox.showwarning("Warning", "No project to save.")
            return

        try:
            self._safe_project_save()
            self._metadata_dirty = False
            self.status_var.set("Project saved.")
            messagebox.showinfo("Success", "Project saved successfully.")
        except Exception as e:
            import traceback
            print(f"\n=== ERROR: Failed to save project ===")
            traceback.print_exc()
            print(f"=====================================\n")
            messagebox.showerror("Error", f"Failed to save project:\n{e}")

    def _safe_project_save(self, write_concept_metadata: bool = True) -> bool:
        """Save self.project, but refuse if a `--refine`/`--redetect` job is running on it,
        or if its files changed on disk since this copy was loaded (a job finished in the
        background). Either case means an unconditional save would clobber that job's
        results. Returns True if the save happened, False if it was refused.

        Thread-safe: worker threads (e.g. merge_thread) call this too, and Tk
        dialogs must not be created off the main thread — marshal the warning
        via root.after in that case and just return False.
        """
        if not self.project:
            return False

        import threading
        _on_main = threading.current_thread() is threading.main_thread()

        def _warn(title, msg):
            if _on_main:
                messagebox.showwarning(title, msg)
            else:
                self.root.after(0, lambda: messagebox.showwarning(title, msg))

        lock = check_refine_lock(self.project.project_dir)
        if lock is not None:
            _warn(
                "Project Locked",
                "A batch refinement/re-detection job (sam3_process.py --refine) is "
                "currently running for this project. Saving now would risk overwriting "
                "its results.\n\nWait for the job to finish, then try again."
            )
            return False

        stale = self.project.check_staleness()
        if stale:
            _warn(
                "Project Changed On Disk",
                "This project was modified on disk since it was loaded here - likely by "
                "a batch refinement job that finished while this window was open. Saving "
                "now would overwrite those results.\n\n"
                "Click 'Reload Project', then redo your change."
            )
            return False

        self.project.save(write_concept_metadata=write_concept_metadata)
        return True

    def reload_project(self):
        """Reload current project from disk (picks up external batch processing results)"""

        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        project_dir = self.project.project_dir
        try:
            self.status_var.set("Reloading project...")
            self.root.update()

            self.project = SAM3Project.load(project_dir)

            self.video_path = self.project.video_path
            self.num_frames = self.project.num_frames
            self.frame_dimensions = self.project.frame_dimensions
            self.fps = self.project.fps

            self.compositor = self._create_compositor()
            if self.compositor is None:
                self.status_var.set("Reload cancelled: video not found.")
                return
            self._invalidate_presence_cache()
            self._load_points_cache_from_project()
            self._load_quality_metrics(project_dir)

            self.frame_slider.configure(to=self.num_frames - 1)
            self.update_concept_tree()
            self.display_frame()

            self.status_var.set(f"Project reloaded from {Path(project_dir).name}.")

        except Exception as e:
            import traceback
            print(f"\n=== ERROR: Failed to reload project ===")
            traceback.print_exc()
            print(f"=======================================\n")
            messagebox.showerror("Error", f"Failed to reload project:\n{e}")
            self.status_var.set("Error reloading project.")

    # ============================================================
    # Frame Display
    # ============================================================

    def _highlight_selected_instance(self, frame_rgb: np.ndarray,
                                     masks_cache: Optional[dict] = None) -> np.ndarray:
        """Draw a white+black contour around every selected instance's mask.

        With multi-select (Shift/Ctrl), all selected instances are outlined.
        Masks are served from masks_cache when available (no extra disk reads).
        """
        targets = self.selected_instances if self.selected_instances else (
            [(self.selected_concept, self.selected_instance)]
            if self.selected_concept and self.selected_instance else []
        )
        if not targets or not self.project:
            return frame_rgb

        frame = None
        for concept, inst in targets:
            if not concept or not inst:
                continue
            cache_key = (concept.name, inst.sam3_obj_id)
            mask = masks_cache.get(cache_key) if masks_cache else None
            if mask is None:
                _inst_mask_dir = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(inst.sam3_obj_id), "masks",
                )
                mask = load_sam3_mask(_inst_mask_dir, self.current_frame_idx)
            if mask is None:
                continue
            if frame is None:
                frame = frame_rgb.copy()
            binary = (mask > 127).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 3)
            cv2.drawContours(frame, contours, -1, (0, 0, 0), 1)
        return frame if frame is not None else frame_rgb

    def _draw_instance_labels(self, frame_rgb: np.ndarray,
                              masks_cache: Optional[dict] = None) -> np.ndarray:
        """Draw each visible instance's name at its mask centroid.

        Text is rendered in a fully-saturated version of the instance's color
        (the blended mask color is too pastel to read; boosting saturation to 1.0
        gives a vivid label that still matches the overlay hue).
        A black outline behind the text keeps it legible on any background.

        masks_cache: optional {(concept_name, obj_id) -> mask} from get_composited_frame
        to avoid re-reading mask files that were already loaded for compositing.
        """
        if not self.project:
            return frame_rgb

        # Defer the frame copy until the first label is actually drawn — skips
        # the ~2 MB clone on frames where no instance mask is present.
        frame = None

        focus_concept = (
            self.selected_concept
            if self.focus_mode_var.get() and self.selected_concept
            else None
        )

        for concept in self.project.concepts:
            if not concept.visible:
                continue
            if focus_concept is not None and concept is not focus_concept:
                continue
            for inst in concept.instances:
                if inst.deleted or not inst.visible:
                    continue

                cache_key = (concept.name, inst.sam3_obj_id)
                mask = masks_cache.get(cache_key) if masks_cache else None
                if mask is None:
                    _inst_mask_dir = os.path.join(
                        self.project.project_dir, "concepts", concept.name,
                        "instances", str(inst.sam3_obj_id), "masks",
                    )
                    mask = load_sam3_mask(_inst_mask_dir, self.current_frame_idx)
                if mask is None:
                    continue

                binary = (mask > 127).astype(np.uint8)
                if binary.sum() < 200:
                    continue

                M = cv2.moments(binary)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Full-saturation color so text is vivid against the pastel overlay
                base_color = inst.color_rgb or concept.color_rgb or (200, 200, 200)
                r, g, b = base_color
                h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                r2, g2, b2 = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                text_color = (int(r2 * 255), int(g2 * 255), int(b2 * 255))

                label = inst.user_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 1

                if frame is None:
                    frame = frame_rgb.copy()
                cv2.putText(frame, label, (cx, cy), font, font_scale,
                            (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, label, (cx, cy), font, font_scale,
                            text_color, thickness, cv2.LINE_AA)

        return frame if frame is not None else frame_rgb

    def display_frame(self):
        """Display current frame with composited masks"""

        if not self.compositor:
            return

        try:
            _perf_reset()
            _t_total = _time.perf_counter()

            # During playback: focus on the selected concept only (reduces mask I/O & blend
            # work from all concepts down to one), and skip label drawing (expensive centroid
            # computation not useful while video is moving).
            _is_playing = self.playing
            if _is_playing:
                _focus_concept = self.selected_concept.name if self.selected_concept else None
            elif self.focus_mode_var.get() and self.selected_concept:
                _focus_concept = self.selected_concept.name
            else:
                _focus_concept = None

            # During playback: downscale to canvas resolution before compositing so all
            # numpy work (blend, cvtcolor, etc.) runs at ~6× fewer pixels.
            # During scrubbing: composite at native resolution for full visual quality.
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            target_hw = None
            _scale_x = _scale_y = 1.0
            if _is_playing and canvas_width > 1 and canvas_height > 1:
                _nw = self.compositor.native_w
                _nh = self.compositor.native_h
                if _nw > 1 and _nh > 1:
                    if _nw / _nh > canvas_width / canvas_height:
                        _disp_w = canvas_width
                        _disp_h = int(canvas_width * _nh / _nw)
                    else:
                        _disp_h = canvas_height
                        _disp_w = int(canvas_height * _nw / _nh)
                    if _disp_w > 1 and _disp_h > 1:
                        target_hw = (_disp_h, _disp_w)
                        _scale_x = _disp_w / _nw
                        _scale_y = _disp_h / _nh

            # Get composited frame; masks loaded during compositing are cached for reuse.
            _alpha = self.mask_alpha_var.get() if self.show_masks_var.get() else 0.0
            with _T("compositor"):
                frame_rgb = self.compositor.get_composited_frame(
                    self.current_frame_idx,
                    alpha_multiplier=_alpha,
                    focus_concept_name=_focus_concept,
                    target_hw=target_hw,
                    render_absorbed_for=(self.selected_concept.name
                                        if self.selected_concept else None),
                )
            masks_cache = self.compositor.get_last_masks()

            # White+black outline on the selected instance (uses cached mask, no extra disk read)
            with _T("highlight"):
                frame_rgb = self._highlight_selected_instance(frame_rgb, masks_cache)

            # Wizard mode: draw cyan contour of current wizard instance's mask so the
            # user can judge quality before confirming.  Also draw pending wizard points.
            if self._absorb_wizard and not _is_playing:
                wiz = self._absorb_wizard
                wiz_inst = wiz['source'] if wiz['phase'] == 'source' else wiz['target']
                wiz_mask_dir = os.path.join(
                    self.project.project_dir, "concepts", wiz['concept'].name,
                    "instances", str(wiz_inst.sam3_obj_id), "masks",
                )
                wiz_mask = load_sam3_mask(wiz_mask_dir, self.current_frame_idx)
                if wiz_mask is not None:
                    fh, fw = frame_rgb.shape[:2]
                    if wiz_mask.shape[0] != fh or wiz_mask.shape[1] != fw:
                        wiz_mask = cv2.resize(wiz_mask, (fw, fh),
                                              interpolation=cv2.INTER_NEAREST)
                    binary = (wiz_mask > 127).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_rgb, contours, -1, (0, 220, 220), 2)

                if wiz['mode'] == 'point':
                    for wx, wy, wpos in self._wizard_pending_points:
                        wpx, wpy = int(wx * _scale_x), int(wy * _scale_y)
                        cv2.circle(frame_rgb, (wpx, wpy), 7, (255, 255, 255), 2)
                        if wpos:
                            cv2.circle(frame_rgb, (wpx, wpy), 5, (60, 220, 90), -1)
                        else:
                            cv2.circle(frame_rgb, (wpx, wpy), 5, (150, 20, 20), 2)
                            cv2.circle(frame_rgb, (wpx, wpy), 2, (150, 20, 20), -1)
                    for bx1, by1, bx2, by2 in self._wizard_pending_boxes:
                        cv2.rectangle(frame_rgb,
                                      (int(bx1 * _scale_x), int(by1 * _scale_y)),
                                      (int(bx2 * _scale_x), int(by2 * _scale_y)),
                                      (0, 220, 220), 2)

            # Normal mode: draw orange anchor boundary if current frame is a mask anchor.
            elif (not _is_playing and self.selected_instance and self.selected_concept
                  and self.current_frame_idx in self.selected_instance.mask_anchor_frames):
                anc_dir = os.path.join(
                    self.project.project_dir, "concepts", self.selected_concept.name,
                    "instances", str(self.selected_instance.sam3_obj_id), "mask_anchors",
                )
                anc_mask = load_sam3_mask(anc_dir, self.current_frame_idx)
                if anc_mask is not None:
                    fh, fw = frame_rgb.shape[:2]
                    if anc_mask.shape[0] != fh or anc_mask.shape[1] != fw:
                        anc_mask = cv2.resize(anc_mask, (fw, fh),
                                              interpolation=cv2.INTER_NEAREST)
                    binary = (anc_mask > 127).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_rgb, contours, -1, (255, 140, 0), 3)
                    cv2.drawContours(frame_rgb, contours, -1, (180, 80, 0), 1)
                    cv2.putText(frame_rgb, "ANCHOR", (5, frame_rgb.shape[0] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1,
                                cv2.LINE_AA)

            # Text labels at mask centroids — skipped during playback (expensive, not useful)
            if self.show_labels_var.get() and not _is_playing:
                with _T("labels"):
                    frame_rgb = self._draw_instance_labels(frame_rgb, masks_cache)

            # Flash mask: white overlay on selected instance's mask region
            if self.flash_mask_in_progress and self.flash_mask_on and self.selected_instance and self.selected_concept:
                _fmask_key = (self.selected_concept.name, self.selected_instance.sam3_obj_id)
                m = masks_cache.get(_fmask_key)
                if m is None:
                    _fmask_dir = os.path.join(
                        self.project.project_dir, "concepts", self.selected_concept.name,
                        "instances", str(self.selected_instance.sam3_obj_id), "masks",
                    )
                    m = load_sam3_mask(_fmask_dir, self.current_frame_idx)
                if m is not None:
                    fh, fw = frame_rgb.shape[:2]
                    if m.shape[0] != fh or m.shape[1] != fw:
                        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    white = np.zeros_like(frame_rgb)
                    white[m > 127] = [255, 255, 255]
                    frame_rgb = cv2.addWeighted(frame_rgb, 0.3, white, 0.7, 0)

            # Flash overlap: orange-red overlay on pixels claimed by 2+ instances
            if self.flash_overlap_in_progress and self.flash_overlap_on and self.flash_overlap_computed is not None:
                overlap_mask = self.flash_overlap_computed
                fh, fw = frame_rgb.shape[:2]
                if overlap_mask.shape != (fh, fw):
                    overlap_mask = cv2.resize(
                        overlap_mask.astype(np.uint8), (fw, fh),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                red_overlay = np.zeros_like(frame_rgb)
                red_overlay[overlap_mask] = [255, 60, 0]

                frame_rgb = cv2.addWeighted(frame_rgb, 0.4, red_overlay, 0.6, 0)

            # Flash custom mask: cyan overlay for an arbitrary mask array (e.g. a
            # single union-dialog component the user wants to visually identify)
            if self.flash_custom_in_progress and self.flash_custom_on and self.flash_custom_mask is not None:
                cm = self.flash_custom_mask
                fh, fw = frame_rgb.shape[:2]
                if cm.shape[0] != fh or cm.shape[1] != fw:
                    cm = cv2.resize(cm.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
                cyan = np.zeros_like(frame_rgb)
                cyan[cm > 127] = [0, 255, 255]
                frame_rgb = cv2.addWeighted(frame_rgb, 0.3, cyan, 0.7, 0)

            # Draw saved/pending refinement points and boxes — skipped during playback since
            # they're tied to the frame the user was working on, not to whatever frame is
            # currently playing.
            if not _is_playing and (self.refinement_points or self.refinement_boxes):
                frame_rgb = frame_rgb.copy()
                flash_pts = self.flash_points_in_progress and self.flash_points_on
                radius = 10 if flash_pts else 5
                halo = radius + 2
                for x, y, is_pos in self.refinement_points:
                    px, py = int(x * _scale_x), int(y * _scale_y)
                    # Colour-blind safe: positive and negative differ in luminance AND
                    # shape (bright filled dot vs. dark hollow ring), not hue alone.
                    if flash_pts:
                        cv2.circle(frame_rgb, (px, py), halo + 3, (255, 255, 255), 3)
                    if is_pos:
                        cv2.circle(frame_rgb, (px, py), radius, (60, 220, 90), -1)   # bright green, filled
                        cv2.circle(frame_rgb, (px, py), halo, (255, 255, 255), 2)
                    else:
                        cv2.circle(frame_rgb, (px, py), halo, (255, 255, 255), 2)   # white halo ring
                        cv2.circle(frame_rgb, (px, py), radius, (150, 20, 20), 2)   # dark red ring (hollow)
                        cv2.circle(frame_rgb, (px, py), max(1, radius // 3), (150, 20, 20), -1)
                for x1, y1, x2, y2 in self.refinement_boxes:
                    cv2.rectangle(frame_rgb,
                                  (int(x1 * _scale_x), int(y1 * _scale_y)),
                                  (int(x2 * _scale_x), int(y2 * _scale_y)),
                                  (0, 255, 100), 2)
                    cv2.rectangle(frame_rgb,
                                  (int(x1 * _scale_x) - 1, int(y1 * _scale_y) - 1),
                                  (int(x2 * _scale_x) + 1, int(y2 * _scale_y) + 1),
                                  (255, 255, 255), 1)

            # Gaze position overlay (gaze x/y are in native video coords -> scale to display)
            if self.gaze_points_by_frame and self.show_gaze_var.get():
                gaze_pts = self.gaze_points_by_frame.get(self.current_frame_idx)
                if gaze_pts:
                    frame_rgb = np.ascontiguousarray(frame_rgb).copy()
                    gaze_overlay.draw_gaze_marker(frame_rgb, gaze_pts, _scale_x, _scale_y)

            # Wrap in PIL — compositor already returned at canvas size, no resize needed.
            # Fallback PIL resize only when canvas wasn't ready (target_hw is None).
            with _T("pil_resize"):
                pil_image = Image.fromarray(frame_rgb)
                if target_hw is None and canvas_width > 1 and canvas_height > 1:
                    img_aspect = pil_image.width / pil_image.height
                    if img_aspect > canvas_width / canvas_height:
                        new_width = canvas_width
                        new_height = int(canvas_width / img_aspect)
                    else:
                        new_height = canvas_height
                        new_width = int(canvas_height * img_aspect)
                    pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)

            # Display
            with _T("canvas"):
                self.photo = ImageTk.PhotoImage(pil_image)
                cx = canvas_width // 2
                cy = canvas_height // 2
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.canvas.create_image(
                        cx, cy, image=self.photo, anchor=tk.CENTER)
                else:
                    self.canvas.itemconfig(self._canvas_image_id, image=self.photo)
                    self.canvas.coords(self._canvas_image_id, cx, cy)

            # Update frame label
            self.frame_label.config(text=f"{self.current_frame_idx} / {self.num_frames - 1}")

            # Update presence bar cursor position
            with _T("presence_bar"):
                self._update_presence_bar()

            # Rate-limited perf report: print at most once every 2 seconds
            _now = _time.perf_counter()
            if _now - self._last_perf_print >= 2.0:
                _perf_report(f"display_frame[{self.current_frame_idx}]", _now - _t_total)
                self._last_perf_print = _now

        except Exception as e:
            print(f"Error displaying frame: {e}")
            import traceback
            traceback.print_exc()

    def on_slider_change(self, value):
        """Handle frame slider change; re-center zoom window when near edge."""
        # During playback the background thread updates current_frame_idx directly and
        # calls display_frame() via _update_frame_from_play; the slider.set() there
        # would fire this callback a second time, doubling all rendering work.
        if self.playing:
            return
        new_idx = int(float(value))
        zoom = self.slider_zoom_level.get()
        if zoom > 1:
            slider_min = int(self.frame_slider.cget('from'))
            slider_max = int(self.frame_slider.cget('to'))
            window_size = max(1, slider_max - slider_min)
            near_edge = (new_idx - slider_min < window_size // 10 or
                         slider_max - new_idx < window_size // 10)
            if near_edge:
                if self.zoom_jump_scheduled:
                    self.root.after_cancel(self.zoom_jump_scheduled)
                self.zoom_jump_scheduled = self.root.after(
                    30, lambda: self._execute_zoom_jump(new_idx, window_size, self.num_frames))
        self.current_frame_idx = new_idx
        self._load_annotations_for_current_frame()
        self.display_frame()

    def jump_frames(self, delta):
        """Jump forward/backward by delta frames, keeping zoom window centered.

        The step is scaled by the current playback speed (rounded to a
        positive int) so that fast-forward/rewind and arrow-key stepping
        feel proportional to the selected speed (e.g. 4x -> 4 frames/step).
        """
        if not self.num_frames:
            return
        speed_factor = max(1, round(self.playback_speed.get()))
        delta *= speed_factor
        new_idx = max(0, min(self.current_frame_idx + delta, self.num_frames - 1))
        # Clamp to wizard period when active so frame navigation stays within the period
        if self._absorb_wizard:
            wiz = self._absorb_wizard
            period = (wiz['source_period'] if wiz['phase'] == 'source'
                      else wiz.get('target_period'))
            if period:
                new_idx = max(period[0], min(period[1], new_idx))
        self.current_frame_idx = new_idx
        zoom = self.slider_zoom_level.get()
        if zoom > 1:
            slider_min = int(self.frame_slider.cget('from'))
            slider_max = int(self.frame_slider.cget('to'))
            if new_idx < slider_min or new_idx > slider_max:
                window_size = max(1, slider_max - slider_min)
                self._execute_zoom_jump(new_idx, window_size, self.num_frames)
        self.frame_slider.set(new_idx)
        self._load_annotations_for_current_frame()
        self.display_frame()

    # ============================================================
    # Concept Tree Management
    # ============================================================

    def update_concept_tree(self):
        """Refresh the concept tree view, preserving expansion state and selection."""

        # Snapshot expanded concepts and current selection before clearing
        expanded_concepts: set = set()
        for item in self.concept_tree.get_children():
            if self.concept_tree.item(item, 'open'):
                tags = self.concept_tree.item(item, 'tags')
                if tags and tags[0] == 'concept':
                    expanded_concepts.add(tags[1])

        selected_tags = None
        selection = self.concept_tree.selection()
        if selection:
            selected_tags = self.concept_tree.item(selection[0], 'tags')

        # Clear tree
        for item in self.concept_tree.get_children():
            self.concept_tree.delete(item)

        if not self.project:
            return

        # Suppress display_frame() calls fired by selection_set() below;
        # the caller is responsible for calling display_frame() once afterward.
        self._updating_tree = True

        # Add concepts
        for concept in self.project.concepts:
            status_str = concept.status.value
            if concept.name in self._concepts_pending_reset:
                status_str = "reset*"

            # Add concept node; restore open state (default open if was open before, or
            # newly added concepts start collapsed — user can expand as desired)
            concept_id = self.concept_tree.insert(
                '', 'end', text=f"[x] {concept.name}" if concept.visible else f"[ ] {concept.name}",
                values=(status_str, ''),
                tags=('concept', concept.name),
                open=concept.name in expanded_concepts
            )

            # Restore concept-level selection
            if (selected_tags and len(selected_tags) >= 2 and
                    selected_tags[0] == 'concept' and selected_tags[1] == concept.name):
                self.concept_tree.selection_set(concept_id)

            # Add instances
            for instance in concept.instances:
                if instance.deleted:
                    continue

                coverage = (instance.num_frames_with_mask / self.num_frames * 100) if self.num_frames > 0 else 0
                checkbox = "[x]" if instance.visible else "[ ]"

                inst_item = self.concept_tree.insert(
                    concept_id, 'end',
                    text=f"  {checkbox} {instance.user_name}",
                    values=('', f"{coverage:.1f}%"),
                    tags=('instance', concept.name, str(instance.sam3_obj_id))
                )

                # Restore instance-level selection
                if (selected_tags and len(selected_tags) >= 3 and
                        selected_tags[0] == 'instance' and
                        selected_tags[1] == concept.name and
                        int(selected_tags[2]) == instance.sam3_obj_id):
                    self.concept_tree.selection_set(inst_item)
                    self.concept_tree.see(inst_item)

        self._updating_tree = False

    def _auto_select_first_instance(self):
        """Select the first concept+instance with detected masks after project load.

        Walks concepts in order; skips concepts with no live instances and concepts
        whose instances have no masks yet (status != completed).  Falls back to the
        first live instance regardless of status if none are completed.  Does nothing
        when no instances exist at all — safe to call on empty projects.
        """
        if not self.project:
            return

        # Two-pass: prefer a completed concept, fall back to any live instance.
        target_concept = None
        target_inst = None
        fallback_concept = None
        fallback_inst = None

        for concept in self.project.concepts:
            live = [i for i in concept.instances if not i.deleted]
            if not live:
                continue
            if fallback_concept is None:
                fallback_concept, fallback_inst = concept, live[0]
            if concept.status.value == "completed" and target_concept is None:
                target_concept, target_inst = concept, live[0]

        if target_concept is None:
            target_concept, target_inst = fallback_concept, fallback_inst

        if target_concept is None:
            # No instances at all — nothing to select.
            return

        self.selected_concept = target_concept
        self.selected_instance = target_inst
        self.selected_instances = [(target_concept, target_inst)]

        # Sync the tree widget selection without triggering display_frame recursion.
        self._updating_tree = True
        try:
            for concept_item in self.concept_tree.get_children():
                tags = self.concept_tree.item(concept_item, 'tags')
                if not (tags and tags[0] == 'concept' and tags[1] == target_concept.name):
                    continue
                self.concept_tree.item(concept_item, open=True)
                for inst_item in self.concept_tree.get_children(concept_item):
                    itags = self.concept_tree.item(inst_item, 'tags')
                    if (len(itags) >= 3 and itags[0] == 'instance'
                            and int(itags[2]) == target_inst.sam3_obj_id):
                        self.concept_tree.selection_set(inst_item)
                        self.concept_tree.see(inst_item)
                        break
                break
        finally:
            self._updating_tree = False

        self.selected_label.config(
            text=f"{target_inst.user_name} (ID: {target_inst.sam3_obj_id})")
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, target_inst.user_name)
        self._update_name_combobox_values()
        self._refresh_concept_quality_bars(redraw=False)
        self._mark_presence_dirty()
        self._load_annotations_for_current_frame()

    def on_tree_select(self, event):
        """Handle tree selection (single or multi via Shift/Ctrl)."""

        selection = self.concept_tree.selection()
        if not selection:
            return

        # Remember what was selected before, so a switch to a genuinely different
        # single instance can auto-jump to its first detected frame below.
        prev_key = ((self.selected_concept.name, self.selected_instance.sam3_obj_id)
                    if self.selected_concept and self.selected_instance else None)

        # Collect all selected instances across items.
        self.selected_instances = []
        last_concept = None
        last_instance = None
        for item in selection:
            tags = self.concept_tree.item(item, 'tags')
            if not tags:
                continue
            if tags[0] == 'instance':
                concept_name = tags[1]
                sam3_obj_id = int(tags[2])
                concept = self.project.get_concept_by_name(concept_name)
                if concept:
                    inst = concept.get_instance_by_sam3_id(sam3_obj_id)
                    if inst:
                        self.selected_instances.append((concept, inst))
                        last_concept = concept
                        last_instance = inst
            elif tags[0] == 'concept':
                last_concept = self.project.get_concept_by_name(tags[1])
                last_instance = None

        n_inst = len(self.selected_instances)
        if n_inst > 1:
            # Multiple instances selected — expose primary as selected_concept/instance
            # so existing single-instance operations still have a target.
            self.selected_concept, self.selected_instance = self.selected_instances[0]
            self.selected_label.config(text=f"{n_inst} instances selected")
            self.name_entry.delete(0, tk.END)
        elif n_inst == 1:
            self.selected_concept, self.selected_instance = self.selected_instances[0]
            inst = self.selected_instance
            self.selected_label.config(
                text=f"{inst.user_name} (ID: {inst.sam3_obj_id})")
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, inst.user_name)
        else:
            # Concept row selected (no instance)
            self.selected_concept = last_concept
            self.selected_instance = None
            if last_concept:
                self.selected_label.config(text=f"Concept: {last_concept.name}")
            self.name_entry.delete(0, tk.END)

        self._update_name_combobox_values()
        self._update_session_status()
        # Overlap/Change colorbars track the selected concept; redraw happens in
        # the display_frame() / _go_to_frame() call below.
        self._refresh_concept_quality_bars(redraw=False)
        self._mark_presence_dirty()
        self._load_annotations_for_current_frame()
        if not self._updating_tree:
            new_key = ((self.selected_concept.name, self.selected_instance.sam3_obj_id)
                       if n_inst == 1 else None)
            jumped = False
            if new_key is not None and new_key != prev_key and self.auto_jump_var.get():
                # Switched to a different single instance — jump to where it's first
                # detected so the user doesn't land on a frame where it's absent.
                # Gated behind a checkbox: when comparing masks across instances on
                # the same frame, auto-jumping away is unwanted (see SAM3_ANNOTATION_GUIDE.md).
                periods = self._get_instance_periods()
                if periods:
                    self._go_to_frame(periods[0][0])  # calls display_frame() internally
                    jumped = True
            if not jumped:
                self.display_frame()

    def on_tree_double_click(self, event):
        """Handle tree double-click to toggle visibility"""

        selection = self.concept_tree.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.concept_tree.item(item, 'tags')

        if not tags:
            return

        if tags[0] == 'concept':
            # Toggle concept visibility
            concept_name = tags[1]
            concept = self.project.get_concept_by_name(concept_name)
            if concept:
                concept.visible = not concept.visible
                self._safe_project_save()
                self.update_concept_tree()
                self.compositor = DynamicFrameCompositor(self.project)  # Refresh compositor
                self.display_frame()
                self.status_var.set(f"Concept '{concept_name}' {'shown' if concept.visible else 'hidden'}.")

        elif tags[0] == 'instance':
            # Toggle instance visibility
            concept_name = tags[1]
            sam3_obj_id = int(tags[2])
            concept = self.project.get_concept_by_name(concept_name)
            if concept:
                instance = concept.get_instance_by_sam3_id(sam3_obj_id)
                if instance:
                    instance.visible = not instance.visible
                    self._safe_project_save()
                    self.update_concept_tree()
                    self.compositor = DynamicFrameCompositor(self.project)  # Refresh compositor
                    self.display_frame()
                    self.status_var.set(f"Instance '{instance.user_name}' {'shown' if instance.visible else 'hidden'}.")

        return "break"  # prevent default Treeview expand/collapse

    # ============================================================
    # Concept Operations
    # ============================================================

    def add_concept_dialog(self):
        """Show dialog to add new concept"""

        if not self.project:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        dialog = AddConceptDialog(self.root, self)
        self.root.wait_window(dialog.top)

    def add_concept(self, concept_name: str, text_prompt: str, detection_frame: int,
                    max_instances: int = -1):
        """Add and process a new concept (runs in background thread)"""

        # Validate
        if not validate_text_prompt(text_prompt):
            messagebox.showerror("Error", "Invalid text prompt. Must be 1-200 characters.")
            return

        # Check for duplicate name
        if self.project.get_concept_by_name(concept_name):
            messagebox.showerror("Error", f"Concept '{concept_name}' already exists.")
            return

        # Create concept
        concept = SAM3Concept(
            name=concept_name,
            text_prompt=text_prompt,
            color_rgb=generate_concept_color(len(self.project.concepts)),
            detection_frame=detection_frame,
            max_instances=max_instances
        )

        self.project.add_concept(concept)
        # Persist the new concept (status=PENDING, no instances) to disk immediately —
        # not just after detection succeeds. Without this, a concept that errors out
        # (or whose detection is interrupted, e.g. the app is closed mid-run) exists
        # only in this session's in-memory self.project.concepts and was never written
        # to project.json/concept_metadata.json, so a separate `sam3_process.py --refine`
        # run (which loads project.json fresh) has no way to know it exists and can
        # never pick it up for redetection.
        self._safe_project_save()
        self.update_concept_tree()

        # Process in background with progress dialog
        self.status_var.set(f"Processing concept '{concept_name}'...")

        progress_dialog = ProgressDialog(
            self.root,
            "Processing Concept",
            f"Detecting instances for '{concept_name}'..."
        )

        def progress_callback(frame_idx, num_frames):
            """Update progress from background thread"""
            self.root.after(0, progress_dialog.update_progress, frame_idx, num_frames)

        def process_thread():
            try:
                # SAM3's process-wide bf16 autocast is thread-local; re-enter it
                # on this worker thread or the model hits bf16/float32 mismatches.
                ensure_bf16_autocast()
                # Release any other live sessions before starting (GPU memory)
                for other_name in list(self.active_sessions.keys()):
                    if other_name != concept_name:
                        self._release_session(other_name)

                # Load model if needed
                if not self.sam3_model:
                    self.root.after(0, progress_dialog.update_progress, 0, 1, "Loading SAM3 model...")
                    self.sam3_model = load_sam3_model(device=self.device)
                    self.root.after(0, self._update_model_status)

                # Process detection — keep session alive for fast online refinement
                self.root.after(0, progress_dialog.update_progress, 0, self.num_frames, "Propagating masks...")
                updated_concept = process_concept_detection(
                    sam3_model=self.sam3_model,
                    project=self.project,
                    concept=concept,
                    device=self.device,
                    progress_callback=progress_callback,
                    keep_session_alive=True,
                )

                # Store live session_id for online refinement
                live_session_id = getattr(updated_concept, "_live_session_id", None)
                if live_session_id:
                    self.active_sessions[concept_name] = live_session_id

                # Update project
                for i, c in enumerate(self.project.concepts):
                    if c.name == concept_name:
                        self.project.concepts[i] = updated_concept
                        break

                # Update UI (must be on main thread); save happens there too
                self.root.after(0, progress_dialog.close)
                self.root.after(0, self._concept_processing_complete, updated_concept)

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.root.after(0, progress_dialog.close)
                self.root.after(0, self._concept_processing_error, concept_name, error_msg)

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def _concept_processing_complete(self, concept: SAM3Concept):
        """Called when concept processing completes"""

        self._safe_project_save()
        self._invalidate_presence_cache()
        self.update_concept_tree()
        self.compositor = DynamicFrameCompositor(self.project)  # Refresh compositor
        self.display_frame()
        self._update_session_status()

        num_instances = len(concept.instances)
        session_note = " (session kept alive for fast refinement)" if concept.name in self.active_sessions else ""
        self.status_var.set(f"Concept '{concept.name}' complete: {num_instances} instances detected.{session_note}")
        messagebox.showinfo("Success",
                           f"Concept '{concept.name}' processed successfully.\n"
                           f"Detected {num_instances} instances.")

    def add_concept_pending(self, concept_name: str, text_prompt: str,
                            detection_frame: int, max_instances: int = -1):
        """Register a new concept as pending without running detection.

        The concept is saved to project.json with status=PENDING so it can be
        processed later by:  python sam3_process.py --project <dir> --device cuda:0
        """
        if not validate_text_prompt(text_prompt):
            messagebox.showerror("Error", "Invalid text prompt. Must be 1-200 characters.")
            return

        if self.project.get_concept_by_name(concept_name):
            messagebox.showerror("Error", f"Concept '{concept_name}' already exists.")
            return

        concept = SAM3Concept(
            name=concept_name,
            text_prompt=text_prompt,
            color_rgb=generate_concept_color(len(self.project.concepts)),
            detection_frame=detection_frame,
            max_instances=max_instances,
        )

        self.project.add_concept(concept)
        self.project.save()
        self.update_concept_tree()
        self.status_var.set(f"Concept '{concept_name}' saved (pending - run sam3_process.py to detect).")
        messagebox.showinfo(
            "Saved",
            f"Concept '{concept_name}' saved as pending.\n\n"
            f"Run detection later with:\n"
            f"  python sam3_process.py --project <project_dir> --device {self.device}"
        )

    def _concept_processing_error(self, concept_name: str, error: str):
        """Called when concept processing fails"""

        print(f"\n=== ERROR: Failed to process concept '{concept_name}' ===")
        print(f"{error}")
        print(f"==========================================================\n")
        self.status_var.set(f"Error processing concept '{concept_name}'.")
        # process_concept_detection already set concept.status = ERROR in-memory before
        # raising; persist that so the on-disk record matches (and so a later
        # `sam3_process.py --refine` sees status=error, not a stale pending/completed
        # state from before this attempt).
        self._safe_project_save()
        messagebox.showerror("Error", f"Failed to process concept '{concept_name}':\n{error}")

    def remove_concept_dialog(self):
        """Remove the currently selected concept (and all its mask files) from the project."""
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        if not self.selected_concept:
            messagebox.showwarning("Warning", "Please select a concept first.")
            return

        concept = self.selected_concept
        confirm = messagebox.askyesno(
            "Confirm Remove Concept",
            f"Remove concept '{concept.name}'?\n\n"
            f"This will permanently delete all mask files for every instance "
            f"of this concept from disk.\n\nThis cannot be undone."
        )
        if not confirm:
            return

        import shutil

        # Release live session if one exists
        self._release_session(concept.name)

        # Delete concept directory (contains all instance mask files)
        concept_dir = self.project.get_concept_dir(concept.name)
        if os.path.isdir(concept_dir):
            shutil.rmtree(concept_dir)

        # Remove from project metadata (mirrors sam3_process.py delete_concept)
        self.project.remove_concept(concept.name)
        self.selected_concept = None
        self.selected_instance = None
        self.selected_label.config(text="None")
        self.name_entry.delete(0, tk.END)

        # Invalidate caches
        self._invalidate_presence_cache()
        keys_to_drop = [k for k in list(self._points_cache) if k[0] == concept.name]
        for k in keys_to_drop:
            self._points_cache.pop(k, None)
            self._all_annotations_cache.pop(k, None)
            self._dirty_keys.discard(k)

        self._safe_project_save()
        self.update_concept_tree()
        self.compositor = DynamicFrameCompositor(self.project)
        self.display_frame()
        self._update_session_status()
        self.status_var.set(f"Concept '{concept.name}' removed.")

    def reset_concept_redetect(self):
        """Mark the selected concept for full re-detection.

        Takes effect on disk only when 'Save Changes for Refinement' is clicked — no files
        are touched here.  The action is undoable/redoable within the session.
        """
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return
        if not self.selected_concept:
            messagebox.showwarning("Warning", "Please select a concept first.")
            return

        concept = self.selected_concept

        if concept.name in self._concepts_pending_reset:
            messagebox.showinfo("Already pending",
                                f"Concept '{concept.name}' is already marked for "
                                f"re-detection. Save Changes for Refinement to apply.")
            return

        n_inst = len([i for i in concept.instances if not i.deleted])
        confirm = messagebox.askyesno(
            "Reset & Re-detect",
            f"Mark concept '{concept.name}' for re-detection?\n\n"
            f"This will (when you click 'Save Changes for Refinement'):\n"
            f"  - Delete all {n_inst} instance(s) and their masks from disk\n"
            f"  - Clear all annotation points for this concept\n"
            f"  - Write a sentinel so --refine re-runs detection ({concept.prompt_label()})\n\n"
            f"Nothing is deleted yet. You can undo this with Ctrl+Z."
        )
        if not confirm:
            return

        import copy

        self._release_session(concept.name)

        # Snapshot current state for undo
        snapshot_instances = copy.deepcopy(concept.instances)
        snapshot_status = concept.status
        snapshot_highest_obj_id = concept.highest_obj_id
        snapshot_completed_at = concept.completed_at

        # Clear in memory only — disk untouched until save
        concept.instances = []
        concept.status = ConceptStatus.PENDING
        concept.highest_obj_id = -1
        concept.completed_at = None
        self._concepts_pending_reset.add(concept.name)

        # Discard any unsaved annotation changes for this concept's instances
        keys_to_drop = [k for k in list(self._points_cache) if k[0] == concept.name]
        for k in keys_to_drop:
            self._points_cache.pop(k, None)
            self._all_annotations_cache.pop(k, None)
            self._dirty_keys.discard(k)
        self._invalidate_presence_cache()

        self.selected_instance = None
        self.selected_label.config(text="None")
        self.name_entry.delete(0, tk.END)
        self.refinement_points.clear()
        self.points_label.config(text="Points: 0  Boxes: 0")

        self.undo_stack.append({
            'type': 'reset_concept',
            'concept_name': concept.name,
            'snapshot_instances': snapshot_instances,
            'snapshot_status': snapshot_status,
            'snapshot_highest_obj_id': snapshot_highest_obj_id,
            'snapshot_completed_at': snapshot_completed_at,
        })
        self.redo_stack.clear()

        self.update_concept_tree()
        self.compositor = DynamicFrameCompositor(self.project)
        self.display_frame()
        self._update_session_status()
        self.status_var.set(
            f"Concept '{concept.name}' marked for re-detection. "
            f"Click 'Save Changes for Refinement' to apply, or Ctrl+Z to undo.")

    def edit_concept_prompt_dialog(self):
        """Open a dialog to change the text prompt of the selected concept.

        Changing the prompt marks the concept for full re-detection (same as
        Reset & Re-detect) so stale masks are cleared and fresh detection runs
        with the new prompt.  The change is undoable/redoable within the session.
        """
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return
        if not self.selected_concept:
            messagebox.showwarning("Warning", "Please select a concept first.")
            return

        concept = self.selected_concept
        dialog = EditPromptDialog(self.root, self, concept)
        self.root.wait_window(dialog.top)

    def _change_concept_prompt(self, concept, new_prompt: str):
        """Update the concept's text prompt in memory and save to disk."""
        old_prompt = concept.text_prompt
        if old_prompt == new_prompt:
            self.status_var.set("Prompt unchanged.")
            return

        concept.text_prompt = new_prompt
        self.project.save()

        self.undo_stack.append({
            'type': 'change_prompt',
            'concept_name': concept.name,
            'old_prompt': old_prompt,
            'new_prompt': new_prompt,
        })
        self.redo_stack.clear()

        self.update_concept_tree()
        self.status_var.set(f"Prompt for '{concept.name}' updated to '{new_prompt}'.")

    # ============================================================
    # Instance Operations
    # ============================================================

    def _on_name_entry_focus_out(self):
        """Silently commit a pending rename when the entry loses focus (e.g. user
        clicks elsewhere), so keyboard focus doesn't stay trapped in the box and
        single-letter shortcuts (like 'f' for flash) keep working."""
        if not self.selected_instance:
            return
        if self._absorb_wizard:
            # The wizard's own focus_set() (moving focus off the entry to the
            # canvas) fires this handler as a side effect — not a real user
            # edit. Ignore it so it doesn't re-trigger rename_instance() and
            # collide with the wizard already in progress.
            return
        new_name = self.name_entry.get().strip()
        if not new_name or new_name == self.selected_instance.user_name:
            return
        self.rename_instance()

    def rename_instance(self):
        """Rename selected instance, via a mask/point confirmation wizard (same UX
        as absorb's source-confirmation phase) so the rename also locks in a
        self-anchor for the instance. Falls back to a plain rename if the
        instance has no masks yet to confirm (e.g. never propagated)."""

        if not self.selected_instance:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return
        if not self.selected_concept:
            messagebox.showwarning("Warning", "No concept selected.")
            return

        new_name = self.name_entry.get().strip()
        if not new_name:
            messagebox.showwarning("Warning", "Name cannot be empty.")
            return
        if new_name == self.selected_instance.user_name:
            return
        if self._absorb_wizard:
            messagebox.showwarning("Warning", "Finish or cancel the current wizard first.")
            return

        self._start_rename_wizard(self.selected_concept, self.selected_instance, new_name)

    def _apply_rename_directly(self, concept, instance, new_name):
        """Plain rename with no mask/point confirmation (used when there's no
        mask to confirm yet)."""
        old_name = instance.user_name
        instance.user_name = new_name
        self.undo_stack.append({
            'type': 'rename_instance',
            'instance': instance,
            'concept_name': concept.name,
            'obj_id': instance.sam3_obj_id,
            'old_name': old_name,
            'new_name': new_name,
        })
        self.redo_stack.clear()
        self._metadata_dirty = True
        self.update_concept_tree()
        self.status_var.set(f"Renamed instance to '{new_name}'. Ctrl+Z to undo.")
        self.canvas.focus_set()

    def _start_rename_wizard(self, concept, instance, new_name):
        """Launch the single-phase confirmation wizard for a rename.

        Reuses the absorb wizard's source-confirmation phase (confirm a mask,
        or fall back to placing a point) but there is no second (target) phase
        here since only one instance is involved — the confirmed anchor is
        written back onto the same instance being renamed, not copied to a
        separate target.

        If the instance already has a confirmed anchor (mask anchor, absorb
        anchor, or placed refinement points) — e.g. from a previous rename or
        absorb — there's nothing new to confirm, so skip straight to a plain
        rename instead of re-prompting the wizard.
        """
        if self._target_already_has_anchor(concept, instance):
            self._apply_rename_directly(concept, instance, new_name)
            return
        info = self._get_first_period_info(concept, instance)
        if info is None:
            self._apply_rename_directly(concept, instance, new_name)
            return
        best, start, end = info

        self._absorb_wizard = {
            'kind': 'rename',
            'concept': concept,
            'target': instance,
            'source': instance,
            'phase': 'source',
            'source_period': (start, end),
            'target_period': None,
            'target_peak_frame': None,
            'source_result': None,
            'target_result': None,
            'mode': 'confirm',
            'saved_slider_from': int(self.frame_slider.cget('from')),
            'saved_slider_to': int(self.frame_slider.cget('to')),
            'saved_zoom': self.slider_zoom_level.get(),
            'original_concept': self.selected_concept,
            'original_instance': self.selected_instance,
            'new_name': new_name,
            'old_name': instance.user_name,
        }
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        self._wizard_enter_phase('source', best)

    def _complete_rename_wizard(self):
        """Stage the confirmed self-anchor (mask or points), apply the rename,
        and push an undo entry."""
        from datetime import datetime as _dt
        wiz = self._absorb_wizard
        if wiz is None:
            return

        concept = wiz['concept']
        instance = wiz['target']  # target == source for a rename
        new_name = wiz['new_name']
        old_name = wiz['old_name']
        result = wiz['source_result']
        copied_paths: List[str] = []
        staged_point_entries: List[dict] = []

        if result['type'] == 'mask':
            frame_idx = result['frame_idx']
            mask_dir = os.path.join(
                self.project.project_dir, "concepts", concept.name,
                "instances", str(instance.sam3_obj_id), "masks"
            )
            mask_np = load_sam3_mask(mask_dir, frame_idx)
            if mask_np is None:
                messagebox.showwarning(
                    "Warning",
                    f"Mask not found at frame {frame_idx} for "
                    f"'{instance.user_name}'. Renaming without an anchor."
                )
            else:
                dst_dir = Path(self.project.project_dir) / "concepts" / concept.name / \
                          "instances" / str(instance.sam3_obj_id) / "mask_anchors"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / f"{frame_idx:06d}.png"
                if dst_path.exists():
                    existing = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
                    if existing is not None and existing.shape == mask_np.shape:
                        import numpy as np
                        mask_np = np.maximum(mask_np.astype("uint8"), existing.astype("uint8"))
                cv2.imwrite(str(dst_path), mask_np)
                if frame_idx not in instance.mask_anchor_frames:
                    instance.mask_anchor_frames.append(frame_idx)
                copied_paths.append(str(dst_path))
                rel_path = os.path.join(
                    "concepts", concept.name, "instances", str(instance.sam3_obj_id),
                    "mask_anchors", f"{frame_idx:06d}.png"
                )
                trigger_entry = {
                    "timestamp": _dt.now().isoformat(),
                    "type": "mask_anchor",
                    "frame_idx": frame_idx,
                    "mask_path": rel_path,
                    "propagated": False,
                    "auto_rename_self_anchor": True,
                }
                self._stage_anchors_to_cache(concept, instance, [trigger_entry])
                staged_point_entries.append(trigger_entry)
        elif result['type'] == 'points':
            frame_idx = result['frame_idx']
            pts = result.get('points', [])
            boxes = result.get('boxes', [])
            entry = {
                "timestamp": _dt.now().isoformat(),
                "frame_idx": frame_idx,
                "points": [{"x": x, "y": y, "is_positive": p} for x, y, p in pts],
                "propagated": False,
                "auto_rename_self_anchor": True,
            }
            if boxes:
                entry["boxes"] = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                  for x1, y1, x2, y2 in boxes]
            self._stage_anchors_to_cache(concept, instance, [entry])
            staged_point_entries.append(entry)

        instance.user_name = new_name

        self.undo_stack.append({
            'type': 'rename_wizard',
            'instance': instance,
            'concept': concept,
            'concept_name': concept.name,
            'obj_id': instance.sam3_obj_id,
            'old_name': old_name,
            'new_name': new_name,
            'copied_mask_paths': copied_paths,
            'staged_point_entries': staged_point_entries,
        })
        self.redo_stack.clear()

        self._wizard_finish()

        self._metadata_dirty = True
        self.update_concept_tree()
        self.compositor = DynamicFrameCompositor(self.project)
        self.display_frame()

        desc = (f"mask at frame {result['frame_idx']}" if result['type'] == 'mask'
                else f"point(s) at frame {result['frame_idx']}")
        self.status_var.set(
            f"Renamed to '{new_name}' (anchor: {desc}). "
            "Save Changes for Refinement then --refine to apply. Ctrl+Z to undo."
        )

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _update_name_combobox_values(self):
        """Refresh the rename combobox dropdown values for the currently selected concept."""
        vocab = []
        if self.selected_concept:
            vocab = self.vocabulary.get(self.selected_concept.name, [])
        self.name_entry['values'] = vocab

    def load_vocabulary(self, path: Optional[str] = None, link_to_project: bool = True):
        """Load a vocabulary JSON file (concept_name -> [instance names]).

        When link_to_project is True and a project is open, the path is stored in
        project.json so it is auto-loaded next time the project is opened.
        """
        if path is None:
            path = filedialog.askopenfilename(
                title="Load Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object at top level")
            self.vocabulary = {str(k): list(v) for k, v in data.items()}
            self.vocabulary_path = path
            self._update_name_combobox_values()
            self.status_var.set(f"Vocabulary loaded: {os.path.basename(path)}")
            if link_to_project and self.project is not None:
                if getattr(self.project, "vocabulary_path", None) != path:
                    self.project.vocabulary_path = path
                    self._metadata_dirty = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load vocabulary:\n{e}")

    def _link_vocabulary_to_project(self):
        """Record self.vocabulary_path in the open project so it auto-loads next time."""
        if self.project is None or not self.vocabulary_path:
            return
        if getattr(self.project, "vocabulary_path", None) != self.vocabulary_path:
            self.project.vocabulary_path = self.vocabulary_path
            self._metadata_dirty = True

    def save_vocabulary(self):
        """Save vocabulary to the current path (prompts for path if none set)."""
        if not self.vocabulary_path:
            self.save_vocabulary_as()
            return
        try:
            with open(self.vocabulary_path, 'w') as f:
                json.dump(self.vocabulary, f, indent=2)
            self.status_var.set(f"Vocabulary saved: {os.path.basename(self.vocabulary_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save vocabulary:\n{e}")

    def save_vocabulary_as(self):
        """Prompt for a path and save the vocabulary."""
        path = filedialog.asksaveasfilename(
            title="Save Vocabulary As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        self.vocabulary_path = path
        self.save_vocabulary()
        self._link_vocabulary_to_project()

    def edit_vocabulary_dialog(self):
        """Open the vocabulary editor dialog."""
        dialog = VocabularyEditorDialog(self.root, self)
        self.root.wait_window(dialog.top)
        self._update_name_combobox_values()

    def _compute_per_period_anchors(self, concept, instance, is_positive: bool,
                                    tag: Optional[str] = None,
                                    first_period_only: bool = False) -> List[dict]:
        """For each continuous detection period of instance, find the frame with the
        largest mask area and return a refinements.json entry (propagated=False) whose
        point is the centroid of that mask.

        is_positive: polarity of the generated points.
        tag: optional key added to each entry for identification (e.g. 'auto_absorb_positive').
        first_period_only: if True, only the instance's first detected period is used.
            Later periods can have drifted segmentation (tracking error), so callers that
            treat the instance's identity as established only at its first appearance
            (e.g. absorb) should set this so the user can judge correctness from that
            single occurrence rather than propagating potential drift everywhere.

        Samples at most 20 evenly-spaced frames per period so long videos don't block
        the UI, then reads only the single best-candidate PNG to compute the centroid.
        """
        if not self.project:
            return []
        mask_dir = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "masks"
        )
        if not os.path.exists(mask_dir):
            return []

        # Use stored period_peaks when available — avoids scanning any PNGs to find the
        # best frame; only the single peak-frame PNG per period needs to be read for centroid.
        # Fall back to sampling when period_peaks is absent (old projects, post-edit state).
        if instance.period_peaks:
            peak_iter = [(p["best_frame"], None) for p in instance.period_peaks]
            if first_period_only:
                peak_iter = peak_iter[:1]
        else:
            periods = instance.continuous_periods or compute_continuous_periods(mask_dir)
            if not periods:
                return []
            if first_period_only:
                periods = periods[:1]
            peak_iter = []
            for start, end in periods:
                period_frames = list(range(start, end + 1))
                if len(period_frames) > 20:
                    step = len(period_frames) / 20
                    period_frames = [period_frames[int(i * step)] for i in range(20)]
                best_frame = None
                best_area = 0
                for frame_idx in period_frames:
                    m = load_sam3_mask(mask_dir, frame_idx)
                    if m is None:
                        continue
                    area = int((m > 127).sum())
                    if area > best_area:
                        best_area = area
                        best_frame = frame_idx
                if best_frame is not None and best_area > 0:
                    peak_iter.append((best_frame, None))

        from datetime import datetime
        entries = []
        for best_frame, _ in peak_iter:
            if best_frame is None:
                continue

            mask = load_sam3_mask(mask_dir, best_frame)
            if mask is None:
                continue
            binary = (mask > 127).astype(np.uint8)
            M = cv2.moments(binary)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # The moments-based centroid is just the mean pixel position, which can land
            # outside the mask for non-convex shapes (e.g. a "C"/crescent/ring, or two
            # blobs joined by a thin neck). A guidance point outside the mask is
            # meaningless to SAM3, so fall back to the point most distant from the mask
            # boundary (distance-transform peak) — guaranteed to lie inside the mask, and
            # generally a more representative "interior" point than the centroid anyway.
            if not binary[int(round(cy)), int(round(cx))]:
                dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
                iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
                cx, cy = float(ix), float(iy)

            entry: dict = {
                "timestamp": datetime.now().isoformat(),
                "frame_idx": best_frame,
                "points": [{"x": float(cx), "y": float(cy), "is_positive": is_positive}],
                "propagated": False,
            }
            if tag:
                entry[tag] = True
            entries.append(entry)

        return entries

    def _stage_anchors_to_cache(self, concept, instance, entries: List[dict]):
        """Merge anchor entries into both annotation caches (in-memory only) and mark dirty.

        mask_anchor type entries have no points and cannot go through _points_cache (which
        only stores point coordinates).  They are written directly to refinements.json via
        _write_mask_anchor_triggers so that _collect_pending_refinements can find them.
        """
        if not entries:
            return
        key = (concept.name, instance.sam3_obj_id)

        # mask_anchor entries carry no points — write them directly to disk.
        mask_anchor_entries = [e for e in entries if e.get("type") == "mask_anchor"]
        point_entries = [e for e in entries if e.get("type") != "mask_anchor"]

        if mask_anchor_entries and self.project:
            self._write_mask_anchor_triggers(concept.name, instance.sam3_obj_id, mask_anchor_entries)

        if point_entries:
            for cache in (self._points_cache, self._all_annotations_cache):
                frame_dict = cache.setdefault(key, {})
                for entry in point_entries:
                    frame_idx = entry["frame_idx"]
                    if frame_idx not in frame_dict and cache is self._points_cache:
                        # Seed with the frame's full existing point set: the flush treats
                        # _points_cache as the frame's canonical set and supersedes the
                        # historical entry, so starting empty would drop the instance's
                        # prior clicks on this frame.  (Reads _all_annotations_cache
                        # before anchors are appended to it in the next loop pass.)
                        frame_dict[frame_idx] = list(
                            self._all_annotations_cache.get(key, {}).get(frame_idx, []))
                    pts = frame_dict.setdefault(frame_idx, [])
                    for p in entry.get("points", []):
                        pts.append((p["x"], p["y"], p["is_positive"]))
        self._dirty_keys.add(key)

    def _write_mask_anchor_triggers(self, concept_name: str, obj_id: int,
                                    entries: List[dict]):
        """Append mask_anchor trigger entries directly to refinements.json.

        mask_anchor entries have no point prompts; they only signal that
        _inject_mask_anchors should run for this instance during propagation.
        Skips duplicates (same frame_idx already pending on disk).
        """
        if not self.project:
            return
        rpath = os.path.join(
            self.project.project_dir, "concepts", concept_name,
            "instances", str(obj_id), "refinements.json"
        )
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        existing: List[dict] = []
        if os.path.exists(rpath):
            with open(rpath) as f:
                existing = json.load(f).get("refinements", [])
        # Avoid duplicate pending triggers for the same frame
        pending_frames = {r.get("frame_idx") for r in existing
                         if not r.get("propagated", False) and r.get("type") == "mask_anchor"}
        new_entries = [e for e in entries if e.get("frame_idx") not in pending_frames]
        if new_entries:
            with open(rpath, "w") as f:
                json.dump({"refinements": existing + new_entries}, f, indent=2)

    def _remove_mask_anchor_triggers(self, concept_name: str, obj_id: int,
                                     frame_indices: List[int]):
        """Remove pending (propagated=False) mask_anchor entries for given frames.
        Called on undo of absorb_wizard so the trigger no longer appears pending."""
        if not self.project or not frame_indices:
            return
        rpath = os.path.join(
            self.project.project_dir, "concepts", concept_name,
            "instances", str(obj_id), "refinements.json"
        )
        if not os.path.exists(rpath):
            return
        with open(rpath) as f:
            existing = json.load(f).get("refinements", [])
        frame_set = set(frame_indices)
        updated = [r for r in existing
                   if not (r.get("type") == "mask_anchor"
                           and not r.get("propagated", False)
                           and r.get("frame_idx") in frame_set)]
        if len(updated) != len(existing):
            with open(rpath, "w") as f:
                json.dump({"refinements": updated}, f, indent=2)

    def _unstage_anchors_from_cache(self, concept, instance, entries: List[dict]):
        """Remove previously staged anchor entries from both annotation caches."""
        if not entries:
            return
        key = (concept.name, instance.sam3_obj_id)
        for cache in (self._points_cache, self._all_annotations_cache):
            frame_dict = cache.get(key, {})
            for entry in entries:
                frame_idx = entry["frame_idx"]
                if frame_idx not in frame_dict:
                    continue
                to_remove = {(p["x"], p["y"], p["is_positive"]) for p in entry["points"]}
                frame_dict[frame_idx] = [pt for pt in frame_dict[frame_idx]
                                         if pt not in to_remove]
                if not frame_dict[frame_idx]:
                    del frame_dict[frame_idx]
        self._dirty_keys.add(key)

    def add_new_instance(self):
        """Create a new manually-added instance in the selected concept.

        The instance starts with no masks.  The user places positive points on it
        and runs 'Save Changes for Refinement', then `sam3_process.py --refine` initializes it
        via those points and propagates it alongside all other concept instances.

        Object IDs are assigned as concept.highest_obj_id + 1 rather than
        max(current instance ids) + 1: the latter would shrink back down after the user
        deletes/consolidates instances and purges them from the list, risking collision
        with whatever small integer SAM3's own auto-detector assigns to a genuinely new
        object in a later refine session. highest_obj_id only grows (or is reset to -1 on
        a full concept reset/re-detect) — see SAM3Concept.highest_obj_id.
        """
        if not self.selected_concept or not self.project:
            messagebox.showwarning("Warning", "Please select a concept first.")
            return

        vocab_names = self.vocabulary.get(self.selected_concept.name, [])
        dlg = NewInstanceNameDialog(self.root, self.selected_concept.name, vocab_names)
        self.root.wait_window(dlg.top)
        if not dlg.result:
            return
        name = dlg.result

        new_id = self.selected_concept.highest_obj_id + 1
        self.selected_concept.highest_obj_id = new_id

        from sam3_project import SAM3Instance
        new_inst = SAM3Instance(
            sam3_obj_id=new_id,
            user_name=name,
            concept_name=self.selected_concept.name,
            color_rgb=None,
            manually_added=True,
        )
        self.selected_concept.instances.append(new_inst)
        self._metadata_dirty = True

        # Rebuild tree, then select the new instance (suppressing auto-jump so
        # we stay on the current frame instead of jumping to first detected frame).
        self.update_concept_tree()
        self._updating_tree = True
        try:
            for concept_item in self.concept_tree.get_children():
                for inst_item in self.concept_tree.get_children(concept_item):
                    tags = self.concept_tree.item(inst_item, 'tags')
                    if (tags and len(tags) >= 3 and tags[0] == 'instance' and
                            tags[1] == self.selected_concept.name and
                            int(tags[2]) == new_id):
                        self.concept_tree.selection_set(inst_item)
                        self.concept_tree.see(inst_item)
                        break
        finally:
            self._updating_tree = False
        # Ensure Python state reflects the new instance regardless of tree events.
        self.selected_instance = new_inst
        self._load_annotations_for_current_frame()
        self._update_presence_bar()
        self.display_frame()
        self.status_var.set(
            f"Created manual instance '{name}' (ID {new_id}) in '{self.selected_concept.name}'. "
            f"Place positive points, then 'Save Changes for Refinement' and run --refine."
        )

    def delete_instance(self):
        """Delete selected instance(s). Supports multi-selection via Shift/Ctrl."""

        targets = self.selected_instances if len(self.selected_instances) > 1 else (
            [(self.selected_concept, self.selected_instance)]
            if self.selected_concept and self.selected_instance else []
        )
        if not targets:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return

        if not self.skip_delete_confirm_var.get():
            if len(targets) == 1:
                _, inst = targets[0]
                msg = (
                    f"Delete instance '{inst.user_name}'?\n"
                    f"The instance will be hidden immediately. Mask files stay on disk until\n"
                    f"'Save Changes for Refinement' or 'Apply Changes & Propagate', so Ctrl+Z is a real undo."
                )
            else:
                names = ", ".join(f"'{i.user_name}'" for _, i in targets)
                msg = (
                    f"Delete {len(targets)} instances: {names}?\n"
                    f"All will be hidden immediately. Mask files stay on disk until\n"
                    f"'Save Changes for Refinement' or 'Apply Changes & Propagate', so Ctrl+Z undoes all."
                )
            if not messagebox.askyesno("Confirm Delete", msg):
                return

        deleted_names = []
        for concept, inst in targets:
            cache_key = (concept.name, inst.sam3_obj_id)
            self._presence_cache.pop(cache_key, None)
            self.undo_stack.append({
                'type': 'delete_instance',
                'instance': inst,
                'concept': concept,
            })
            inst.deleted = True
            inst.visible = False
            deleted_names.append(inst.user_name)

        self.redo_stack.clear()
        self.selected_instances = []
        self.selected_instance = None
        self.selected_label.config(text="None")
        self.name_entry.delete(0, tk.END)

        self._mark_presence_dirty()
        self._metadata_dirty = True
        self.update_concept_tree()
        self.compositor = DynamicFrameCompositor(self.project)
        self.display_frame()

        if len(deleted_names) == 1:
            self.status_var.set(
                f"'{deleted_names[0]}' marked for deletion. "
                f"Masks removed on Save/Apply. Ctrl+Z to undo."
            )
        else:
            self.status_var.set(
                f"{len(deleted_names)} instances marked for deletion. Ctrl+Z undoes each."
            )

    def absorb_instance_dialog(self):
        """Show dialog to absorb the currently selected instance into a target the
        user picks from the rest of the concept's instances.

        This is the recommended approach when instances are well-segmented but
        a real-world object was split across two detections. It places a positive
        point at the centroid of the (selected/current) source instance's largest
        mask frame, then runs full re-propagation on the target, and offers to
        delete the source.
        """
        if not self.selected_instance or not self.selected_concept:
            messagebox.showwarning("Warning", "Please select the instance to absorb first.")
            return

        targets = [inst for inst in self.selected_concept.instances
                   if not inst.deleted and inst.sam3_obj_id != self.selected_instance.sam3_obj_id]
        if not targets:
            messagebox.showwarning("Warning", "No other instances to absorb into.")
            return

        dialog = AbsorbInstanceDialog(self.root, self, self.selected_concept,
                                      self.selected_instance, targets)
        self.root.wait_window(dialog.top)

    # ============================================================
    # Absorb Wizard — mask-based conditioning anchor selection
    # ============================================================

    def _get_first_period_info(self, concept, instance):
        """Return (first_frame, period_start, period_end) for first detected period or None."""
        mask_dir = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "masks"
        )
        if not os.path.exists(mask_dir):
            return None
        periods = instance.continuous_periods or compute_continuous_periods(mask_dir)
        if not periods:
            return None
        start, end = periods[0]
        # Walk forward from the period start to find the first frame with a non-empty mask.
        for f in range(start, end + 1):
            m = load_sam3_mask(mask_dir, f)
            if m is not None and int((m > 127).sum()) > 0:
                return f, start, end
        return None

    def absorb_instance(self, target: 'SAM3Instance', source: 'SAM3Instance'):
        """Launch the guided mask-confirmation wizard for an absorb.

        The wizard lets the user confirm (or navigate to) the best conditioning frame
        for the source instance, then optionally for the target (if first absorb into
        it).  The confirmed frame's mask is copied to the target's mask_anchors/ folder
        and registered in target.mask_anchor_frames; the pipeline injects it via
        Sam3TrackerPredictor.add_new_mask() during replay.

        If the user cannot find a clean mask, they can fall back to placing a point
        (same path as the old centroid-based absorb).
        """
        concept = self.selected_concept
        self._start_absorb_wizard(concept, target, source)

    def _target_already_has_anchor(self, concept, target) -> bool:
        """True if the target instance already has a confirmed mask anchor or
        user-placed annotation (points/boxes) — e.g. from a prior absorb into
        it, or from confirming a self-anchor via the rename wizard. In that
        case there's no need to also ask the user to confirm the target's own
        mask during a later absorb.
        """
        if target.received_absorb_anchor or target.mask_anchor_frames:
            return True
        if not self.project:
            return False
        rpath = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(target.sam3_obj_id), "refinements.json"
        )
        if os.path.exists(rpath):
            try:
                with open(rpath) as f:
                    if json.load(f).get("refinements", []):
                        return True
            except Exception:
                pass
        return False

    def _start_absorb_wizard(self, concept, target, source):
        """Initialise and enter the absorb wizard (source confirmation phase)."""
        src_info = self._get_first_period_info(concept, source)
        if src_info is None:
            messagebox.showerror(
                "Error",
                f"No masks found for source instance '{source.user_name}'.\n"
                "Cannot start absorb wizard."
            )
            return
        src_best, src_start, src_end = src_info

        tgt_info = None
        if not self._target_already_has_anchor(concept, target):
            tgt_info = self._get_first_period_info(concept, target)

        self._absorb_wizard = {
            'concept': concept,
            'target': target,
            'source': source,
            'phase': 'source',
            'source_period': (src_start, src_end),
            'target_period': (tgt_info[1], tgt_info[2]) if tgt_info else None,
            'target_peak_frame': tgt_info[0] if tgt_info else None,
            'source_result': None,
            'target_result': None,
            'mode': 'confirm',
            'saved_slider_from': int(self.frame_slider.cget('from')),
            'saved_slider_to': int(self.frame_slider.cget('to')),
            'saved_zoom': self.slider_zoom_level.get(),
            'original_concept': self.selected_concept,
            'original_instance': self.selected_instance,
        }
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        self._wizard_enter_phase('source', src_best)

    def _wizard_enter_phase(self, phase: str, jump_to: int):
        """Set up slider zoom, select wizard instance, and update the banner."""
        wiz = self._absorb_wizard
        wiz['phase'] = phase
        wiz['mode'] = 'confirm'
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        self._wizard_pending_boxes = []

        if phase == 'source':
            inst = wiz['source']
            period_start, period_end = wiz['source_period']
            if wiz.get('kind') == 'rename':
                title = f"Confirm mask to rename '{inst.user_name}' -> '{wiz['new_name']}'"
            else:
                n_steps = 1 if wiz['target_period'] is None else 2
                title = f"Wizard (1/{n_steps}): confirm mask for SOURCE '{inst.user_name}'"
        else:
            inst = wiz['target']
            period_start, period_end = wiz['target_period']
            title = f"Wizard (2/2): confirm mask for TARGET '{inst.user_name}'"

        instr = (f"Frames {period_start}-{period_end}. Confirm the frame "
                 "where the mask cleanly covers the object (no bleed, not blurry). "
                 "Or 'Add point instead' to place a point manually.")

        # Zoom presence bar to this period
        self.frame_slider.config(from_=period_start, to=period_end)
        self._mark_presence_dirty()
        self._update_presence_bar()

        # Select wizard instance so highlight + mask loading are consistent
        self.selected_concept = wiz['concept']
        self.selected_instance = inst
        self._load_annotations_for_current_frame()
        self._go_to_frame(max(period_start, min(period_end, jump_to)))

        # Show / update wizard banner
        self._wizard_title_lbl.config(text=title)
        self._wizard_instr_lbl.config(text=instr)
        self._wizard_update_action_buttons()
        # Move focus off the name entry so Left/Right arrow keys navigate
        # frames instead of moving the text cursor within the entry.
        self.canvas.focus_set()
        if not self._wizard_banner.winfo_ismapped():
            self._wizard_banner.pack(fill=tk.X, padx=5, pady=2, after=self.canvas)

        self.status_var.set(
            f"Wizard: navigate within {period_start}-{period_end}, "
            f"then 'Use this mask' or 'Add point instead'."
        )

    def _wizard_update_action_buttons(self):
        """Enable/disable wizard buttons for the current interaction mode."""
        if not self._absorb_wizard:
            return
        mode = self._absorb_wizard['mode']
        if mode == 'confirm':
            self._wiz_confirm_btn.config(state=tk.NORMAL)
            self._wiz_point_btn.config(state=tk.NORMAL)
            self._wiz_confirm_pts_btn.config(state=tk.DISABLED)
            self._wiz_back_btn.config(state=tk.DISABLED)
        else:
            self._wiz_confirm_btn.config(state=tk.DISABLED)
            self._wiz_point_btn.config(state=tk.DISABLED)
            has_ann = bool(self._wizard_pending_points) or bool(self._wizard_pending_boxes)
            self._wiz_confirm_pts_btn.config(state=tk.NORMAL if has_ann else tk.DISABLED)
            self._wiz_back_btn.config(state=tk.NORMAL)

    def _wizard_confirm_mask(self):
        """User accepted the current frame's mask as conditioning anchor.

        For the source phase: immediately check whether the target or any
        instance previously absorbed into it has a mask at this same frame.
        If so, show a per-component union dialog before advancing.  The user
        can choose to union any subset, switch to point+box mode (which also
        discards the mask just accepted), or skip the union and keep the mask
        as-is.
        """
        wiz = self._absorb_wizard
        if wiz is None:
            return
        frame_idx = self.current_frame_idx

        if wiz['phase'] == 'source':
            if wiz.get('kind') == 'rename':
                # Rename has no separate target to union against — accept directly.
                wiz['source_result'] = {'type': 'mask', 'frame_idx': frame_idx}
                wiz['source_extra_union_mask'] = None
            else:
                extra_mask, use_annot = self._wizard_check_union_at_frame(wiz, frame_idx)
                if use_annot:
                    # User wants to annotate instead — discard the accepted mask and
                    # enter point+box mode so they can place their own prompts.
                    self._wizard_switch_to_point_mode()
                    return
                wiz['source_result'] = {'type': 'mask', 'frame_idx': frame_idx}
                wiz['source_extra_union_mask'] = extra_mask
        else:
            wiz[f"{wiz['phase']}_result"] = {'type': 'mask', 'frame_idx': frame_idx}
        self._wizard_advance()

    def _wizard_switch_to_point_mode(self):
        """Switch to manual point-placement fallback mode."""
        wiz = self._absorb_wizard
        if wiz is None:
            return
        wiz['mode'] = 'point'
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        self._wizard_update_action_buttons()
        self.status_var.set(
            "Point mode: left-click positive, right-click negative. "
            "Click 'Confirm points' when done."
        )
        self.display_frame()

    def _wizard_back_to_mask_mode(self):
        """Return from point mode back to mask-confirm mode."""
        wiz = self._absorb_wizard
        if wiz is None:
            return
        wiz['mode'] = 'confirm'
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        self._wizard_update_action_buttons()
        self.display_frame()

    def _wizard_confirm_points(self):
        """User confirmed their manually-placed points/boxes for this phase."""
        wiz = self._absorb_wizard
        if wiz is None:
            return
        if not self._wizard_pending_points and not self._wizard_pending_boxes:
            messagebox.showwarning("No annotation", "Place at least one point or draw a box first.")
            return
        wiz[f"{wiz['phase']}_result"] = {
            'type': 'points',
            'frame_idx': self.current_frame_idx,
            'points': list(self._wizard_pending_points),
            'boxes': list(self._wizard_pending_boxes),
        }
        self._wizard_advance()

    def _wizard_advance(self):
        """Advance to target phase if needed, otherwise complete the wizard."""
        wiz = self._absorb_wizard
        if wiz is None:
            return
        if wiz['phase'] == 'source' and wiz['target_period'] is not None:
            # Restore slider before entering target phase (so save/restore round-trips cleanly)
            self._wizard_exit_zoom()
            self._wizard_enter_phase('target', wiz['target_peak_frame'])
        elif wiz.get('kind') == 'rename':
            self._complete_rename_wizard()
        else:
            self._complete_absorb_wizard()

    def _wizard_exit_zoom(self):
        """Restore slider zoom to the pre-wizard state."""
        wiz = self._absorb_wizard
        if wiz is None:
            return
        self.frame_slider.config(from_=wiz['saved_slider_from'], to=wiz['saved_slider_to'])
        self.slider_zoom_level.set(wiz['saved_zoom'])
        self._mark_presence_dirty()
        self._update_presence_bar()

    def _wizard_check_union_at_frame(self, wiz, frame_idx):
        """Check if the target (or any instance it absorbed) has masks at frame_idx.

        Components checked:
          1. Target's own detection masks (from its masks/ dir).
          2. Each instance previously absorbed into the target (absorbed_source_ids),
             using their original masks/ dir — so each prior absorb is shown separately.

        If any components are found, shows a per-component dialog (with the current
        frame already visible behind it) offering three choices:
          - "Use as union": pixel-wise OR of source mask + checked components.
          - "Point/box mode": discard the accepted mask, enter point+box annotation
            mode so the user can place their own prompts from scratch.
          - "Skip union": keep only the source mask, ignore all components.

        Returns:
            (extra_union_mask, use_annotation_mode)
            extra_union_mask : numpy uint8 array or None
            use_annotation_mode : bool — if True, caller should switch to point/box mode
        """
        import numpy as np
        concept = wiz['concept']
        target = wiz['target']
        source = wiz['source']

        components = []   # list of (label: str, mask: np.ndarray)

        # Target's own original detection mask at frame_idx
        tgt_mask_dir = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(target.sam3_obj_id), "masks"
        )
        m = load_sam3_mask(tgt_mask_dir, frame_idx)
        if m is not None and int((m > 127).sum()) > 0:
            components.append(
                (f"'{target.user_name}' own detection (obj {target.sam3_obj_id})", m)
            )

        # Each instance previously absorbed into the target — original masks
        for src_id in target.absorbed_source_ids:
            if src_id == source.sam3_obj_id:
                continue
            src_inst = concept.get_instance_by_sam3_id(src_id)
            label = (f"'{src_inst.user_name}'" if src_inst else f"obj {src_id}")
            src_mask_dir = os.path.join(
                self.project.project_dir, "concepts", concept.name,
                "instances", str(src_id), "masks"
            )
            m = load_sam3_mask(src_mask_dir, frame_idx)
            if m is not None and int((m > 127).sum()) > 0:
                components.append(
                    (f"{label} original mask (prev. absorbed, obj {src_id})", m)
                )

        if not components:
            return None, False

        result = {'extra': None, 'use_annot': False}
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Target overlap at frame {frame_idx}")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        tk.Label(
            dlg,
            text=(f"At frame {frame_idx}, '{target.user_name}' (or instances\n"
                  f"previously absorbed into it) also have masks.\n\n"
                  f"'{source.user_name}' is being absorbed into '{target.user_name}'.\n"
                  f"Select which component masks to include in the anchor\n"
                  f"(pixel-wise union with '{source.user_name}' mask):"),
            justify=tk.LEFT,
        ).pack(padx=12, pady=(12, 4), anchor=tk.W)

        check_vars = []
        for label, m in components:
            v = tk.BooleanVar(value=True)
            check_vars.append((v, m))
            row = tk.Frame(dlg)
            row.pack(anchor=tk.W, padx=24, fill=tk.X)
            tk.Checkbutton(row, text=label, variable=v).pack(side=tk.LEFT)
            tk.Button(
                row, text="Flash", width=6,
                command=lambda m=m: self.flash_mask_array(m),
            ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(
            dlg,
            text=("Tip: if the masks are not clean enough to union properly,\n"
                  "use 'Point/box mode' to place your own prompts and discard\n"
                  "the mask you just accepted."),
            fg="#aaaaaa", justify=tk.LEFT, font=("Arial", 8),
        ).pack(padx=12, pady=(8, 2), anchor=tk.W)

        def on_union():
            extra = None
            for v, m in check_vars:
                if v.get():
                    binary = (m > 127).astype(np.uint8) * 255
                    extra = binary if extra is None else np.maximum(extra, binary)
            result['extra'] = extra
            dlg.destroy()

        def on_annot():
            result['use_annot'] = True
            dlg.destroy()

        def on_skip():
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=10)
        tk.Button(btn_row, text="Use as union", command=on_union,
                  width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Point/box mode", command=on_annot,
                  width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Skip union", command=on_skip,
                  width=12).pack(side=tk.LEFT, padx=4)

        self.root.wait_window(dlg)
        return result['extra'], result['use_annot']

    def _complete_absorb_wizard(self):
        """Stage all confirmed anchors, mark source deleted, push undo entry."""
        import shutil as _shutil
        import numpy as np
        wiz = self._absorb_wizard
        if wiz is None:
            return

        concept = wiz['concept']
        target = wiz['target']
        source = wiz['source']
        copied_paths: List[str] = []
        staged_point_entries: List[dict] = []
        target_anchor_was_new = wiz['target_period'] is not None

        # Union mask collected during source-phase confirmation (already decided by user).
        extra_union_mask = wiz.get('source_extra_union_mask')
        union_frame = (wiz['source_result']['frame_idx']
                       if wiz['source_result'] and wiz['source_result']['type'] == 'mask'
                       else None)

        def _apply_result(result, src_inst, label_tag: str):
            """Copy mask or stage points for the TARGET instance."""
            if result['type'] == 'mask':
                from datetime import datetime as _dt
                frame_idx = result['frame_idx']
                src_mask_dir = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(src_inst.sam3_obj_id), "masks"
                )
                mask_np = load_sam3_mask(src_mask_dir, frame_idx)
                if mask_np is None:
                    messagebox.showwarning(
                        "Warning",
                        f"Mask not found at frame {frame_idx} for "
                        f"'{src_inst.user_name}'. Skipping mask anchor."
                    )
                    return
                # Union with user-selected extra masks (only for the source anchor at the
                # confirmed frame).  For the target phase writing at the same frame, union
                # with whatever was already written so neither overwrites the other.
                if label_tag == "auto_absorb_positive" and extra_union_mask is not None and frame_idx == union_frame:
                    mask_np = np.maximum(mask_np.astype(np.uint8), extra_union_mask)
                dst_dir = Path(self.project.project_dir) / "concepts" / concept.name / \
                          "instances" / str(target.sam3_obj_id) / "mask_anchors"
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / f"{frame_idx:06d}.png"
                # If anchor file already exists at this frame (e.g. source phase already
                # wrote a union mask here), union rather than overwrite.
                if dst_path.exists():
                    existing = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
                    if existing is not None and existing.shape == mask_np.shape:
                        mask_np = np.maximum(mask_np.astype(np.uint8), existing.astype(np.uint8))
                cv2.imwrite(str(dst_path), mask_np)
                if frame_idx not in target.mask_anchor_frames:
                    target.mask_anchor_frames.append(frame_idx)
                copied_paths.append(str(dst_path))
                # Stage a mask-anchor entry so _collect_pending_refinements sees
                # this instance as pending and "Apply Changes & Propagate" proceeds
                # to call _inject_mask_anchors.  "type": "mask_anchor" identifies
                # this entry explicitly; the pipeline skips it during point-prompt
                # construction (no SAM3 point prompts added) but still runs
                # propagation and injects the mask via _inject_mask_anchors.
                rel_path = os.path.join(
                    "concepts", concept.name,
                    "instances", str(target.sam3_obj_id),
                    "mask_anchors", f"{frame_idx:06d}.png"
                )
                trigger_entry = {
                    "timestamp": _dt.now().isoformat(),
                    "type": "mask_anchor",
                    "frame_idx": frame_idx,
                    "mask_path": rel_path,
                    "propagated": False,
                    label_tag: True,
                }
                self._stage_anchors_to_cache(concept, target, [trigger_entry])
                staged_point_entries.append(trigger_entry)
            elif result['type'] == 'points':
                from datetime import datetime as _dt
                frame_idx = result['frame_idx']
                pts = result.get('points', [])
                boxes = result.get('boxes', [])
                entry = {
                    "timestamp": _dt.now().isoformat(),
                    "frame_idx": frame_idx,
                    "points": [{"x": x, "y": y, "is_positive": p} for x, y, p in pts],
                    "propagated": False,
                    label_tag: True,
                }
                if boxes:
                    entry["boxes"] = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                      for x1, y1, x2, y2 in boxes]
                self._stage_anchors_to_cache(concept, target, [entry])
                staged_point_entries.append(entry)

        _apply_result(wiz['source_result'], source, "auto_absorb_positive")
        if wiz['target_result'] is not None:
            _apply_result(wiz['target_result'], target, "auto_absorb_self_anchor")

        source.deleted = True
        source.absorbed_source = True
        source.visible = False
        target.received_absorb_anchor = True
        # Record the source's ID on the target so future absorbs into the same
        # target can surface each historical component's original masks for
        # individual union decisions.
        if source.sam3_obj_id not in target.absorbed_source_ids:
            target.absorbed_source_ids.append(source.sam3_obj_id)

        self.undo_stack.append({
            'type': 'absorb_wizard',
            'source': source,
            'target': target,
            'concept': concept,
            'source_result': wiz['source_result'],
            'target_result': wiz['target_result'],
            'target_anchor_was_new': target_anchor_was_new,
            'copied_mask_paths': copied_paths,
            'staged_point_entries': staged_point_entries,
        })
        self.redo_stack.clear()

        self._wizard_finish()

        self._metadata_dirty = True
        self.update_concept_tree()
        # The source is now deleted; keep the surviving target highlighted.
        self._select_instance_in_tree(target.concept_name, target.sam3_obj_id)
        self.compositor = DynamicFrameCompositor(self.project)
        self.display_frame()

        sr = wiz['source_result']
        desc = (f"mask at frame {sr['frame_idx']}" if sr['type'] == 'mask'
                else f"point(s) at frame {sr['frame_idx']}")
        self.status_var.set(
            f"Absorb: '{source.user_name}' → '{target.user_name}' "
            f"(source anchor: {desc}). "
            "Save Changes for Refinement then --refine to apply. Ctrl+Z to undo."
        )

    def _wizard_cancel(self):
        """Cancel wizard with no changes."""
        self._wizard_finish()
        self.status_var.set("Absorb wizard cancelled.")

    def _wizard_finish(self):
        """Common teardown: hide banner, restore slider zoom, restore selection."""
        if self._absorb_wizard:
            self._wizard_exit_zoom()
            orig_c = self._absorb_wizard.get('original_concept')
            orig_i = self._absorb_wizard.get('original_instance')
            if orig_c is not None:
                self.selected_concept = orig_c
            if orig_i is not None:
                self.selected_instance = orig_i
            self._absorb_wizard = None
        self._wizard_pending_points = []
        self._wizard_pending_boxes = []
        if self._wizard_banner.winfo_ismapped():
            self._wizard_banner.pack_forget()
        self._load_annotations_for_current_frame()
        self.display_frame()

    def remove_mask_anchor_at_location(self, x: float, y: float) -> bool:
        """In removal mode: if (x, y) is inside the current frame's mask anchor, remove it."""
        if not self.selected_instance or not self.selected_concept:
            return False
        inst = self.selected_instance
        frame_idx = self.current_frame_idx
        if frame_idx not in inst.mask_anchor_frames:
            return False
        anchor_dir = os.path.join(
            self.project.project_dir, "concepts", self.selected_concept.name,
            "instances", str(inst.sam3_obj_id), "mask_anchors"
        )
        mask_np = load_sam3_mask(anchor_dir, frame_idx)
        if mask_np is None:
            return False
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= iy < mask_np.shape[0] and 0 <= ix < mask_np.shape[1]):
            return False
        if mask_np[iy, ix] < 128:
            return False
        mask_data = mask_np.copy()
        inst.mask_anchor_frames.remove(frame_idx)
        # Delete the file (PNG or NPZ)
        for ext in ('.png', '.npz'):
            p = Path(anchor_dir) / f"{frame_idx:06d}{ext}"
            if p.exists():
                p.unlink()
        self.undo_stack.append({
            'type': 'remove_mask_anchor',
            'instance': inst,
            'concept': self.selected_concept,
            'frame_idx': frame_idx,
            'mask_data': mask_data,
            'anchor_dir': anchor_dir,
        })
        self.redo_stack.clear()
        self._metadata_dirty = True
        self.display_frame()
        self.status_var.set(f"Removed mask anchor at frame {frame_idx}. Ctrl+Z to undo.")
        return True

    def save_current_mask_as_anchor(self):
        """Save the currently displayed mask of the selected instance, at the
        current frame, as a mask anchor for that instance.

        Stored exactly the way rename/absorb self-anchors are stored (PNG under
        instances/<id>/mask_anchors/, registered in instance.mask_anchor_frames,
        plus a pending 'mask_anchor' trigger in refinements.json) so it is picked
        up by _inject_mask_anchors on the next --refine/propagation, and so it
        carries over as a historical component if this instance is later absorbed
        into another one.
        """
        from datetime import datetime as _dt

        if not self.selected_instance or not self.selected_concept:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return
        concept = self.selected_concept
        instance = self.selected_instance
        frame_idx = self.current_frame_idx

        mask_dir = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "masks"
        )
        mask_np = load_sam3_mask(mask_dir, frame_idx)
        if mask_np is None:
            messagebox.showwarning(
                "Warning", f"No mask displayed for '{instance.user_name}' at frame "
                           f"{frame_idx}; nothing to save as an anchor.")
            return

        anchor_dir = os.path.join(
            self.project.project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "mask_anchors"
        )
        os.makedirs(anchor_dir, exist_ok=True)
        dst_path = Path(anchor_dir) / f"{frame_idx:06d}.png"

        # Preserve prior anchor bytes (if any) so undo can restore them exactly.
        prev_mask_data = None
        if dst_path.exists():
            prev_mask_data = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
            if prev_mask_data is not None and prev_mask_data.shape == mask_np.shape:
                mask_np = np.maximum(mask_np.astype("uint8"), prev_mask_data.astype("uint8"))

        cv2.imwrite(str(dst_path), mask_np)
        was_new_frame = frame_idx not in instance.mask_anchor_frames
        if was_new_frame:
            instance.mask_anchor_frames.append(frame_idx)
            instance.mask_anchor_frames.sort()

        rel_path = os.path.join(
            "concepts", concept.name, "instances", str(instance.sam3_obj_id),
            "mask_anchors", f"{frame_idx:06d}.png"
        )
        trigger_entry = {
            "timestamp": _dt.now().isoformat(),
            "type": "mask_anchor",
            "frame_idx": frame_idx,
            "mask_path": rel_path,
            "propagated": False,
            "manual_save_as_anchor": True,
        }
        self._stage_anchors_to_cache(concept, instance, [trigger_entry])

        self.undo_stack.append({
            'type': 'save_mask_anchor',
            'instance': instance,
            'concept': concept,
            'frame_idx': frame_idx,
            'mask_data': mask_np,
            'prev_mask_data': prev_mask_data,
            'was_new_frame': was_new_frame,
            'anchor_dir': anchor_dir,
            'trigger_entry': trigger_entry,
        })
        self.redo_stack.clear()
        self._metadata_dirty = True
        self.display_frame()
        self.status_var.set(
            f"Saved mask anchor for '{instance.user_name}' at frame {frame_idx}. Ctrl+Z to undo.")

    def merge_instances_dialog(self):
        """Show dialog to merge multiple instances by pixel-union of their masks.

        NOTE: This is a purely mask-level operation — it unions the PNG masks
        frame-by-frame.  Use this only when every instance is already correctly
        segmented AND every occurrence of the concept has been detected.
        If masks are imperfect, use 'Absorb into Selected' (refinement-based)
        instead, which lets SAM3 re-propagate with guidance.
        """
        if not self.selected_concept:
            messagebox.showwarning("Warning", "Please select a concept first.")
            return

        # Get non-deleted instances
        instances = [inst for inst in self.selected_concept.instances if not inst.deleted]

        if len(instances) < 2:
            messagebox.showwarning("Warning", "Need at least 2 instances to merge.")
            return

        if not messagebox.askokcancel(
            "Merge Warning",
            "Merge (union) combines instances by OR-ing their saved masks frame-by-frame.\n\n"
            "Requirements for a good result:\n"
            "  - Each instance must already be well-segmented\n"
            "  - All occurrences of the concept must have been detected\n\n"
            "This operation CANNOT be re-prompted through SAM3 afterward.\n"
            "If masks are imperfect, use 'Absorb into Selected' instead.\n\n"
            "Proceed?"
        ):
            return

        # Show merge dialog
        dialog = MergeInstancesDialog(self.root, self, self.selected_concept, instances)
        self.root.wait_window(dialog.top)

    def merge_instances(self, concept: SAM3Concept, instance_ids: List[int], new_name: str):
        """Merge instances (runs in background thread)"""

        # Show progress dialog
        progress_dialog = ProgressDialog(
            self.root,
            "Merging Instances",
            f"Merging {len(instance_ids)} instances into '{new_name}'..."
        )

        def merge_thread():
            try:
                from sam3_utils import merge_instances

                # Get frame count for progress
                # Estimate total frames to process
                all_frames = set()
                for inst_id in instance_ids:
                    instance = concept.get_instance_by_sam3_id(inst_id)
                    if instance:
                        all_frames.update(range(instance.first_detection_frame, instance.last_detection_frame + 1))

                total_frames = len(all_frames)
                processed_frames = 0

                # Merge (this processes frame-by-frame)
                self.root.after(0, progress_dialog.update_progress, 0, total_frames, "Merging masks...")

                merged_instance = merge_instances(
                    concept=concept,
                    instance_ids=instance_ids,
                    new_name=new_name,
                    project_dir=self.project.project_dir
                )

                # Save project
                self._safe_project_save()

                # Close dialog and refresh
                self.root.after(0, progress_dialog.close)
                self.root.after(0, self._merge_complete, new_name,
                                concept.name, merged_instance.sam3_obj_id)

            except Exception as e:
                import traceback
                print(f"\n=== ERROR: Failed to merge instances ===")
                traceback.print_exc()
                print(f"=========================================\n")
                self.root.after(0, progress_dialog.close)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to merge instances:\n{e}"
                ))
                self.root.after(0, lambda: self.status_var.set("Merge failed."))

        thread = threading.Thread(target=merge_thread, daemon=True)
        thread.start()

    def _merge_complete(self, new_name: str, concept_name: str = None, merged_id: int = None):
        """Called when merge completes"""
        self._invalidate_presence_cache()
        self.update_concept_tree()
        # Keep the merged instance highlighted (the source instances are gone).
        if concept_name is not None and merged_id is not None:
            self._select_instance_in_tree(concept_name, merged_id)
        self.compositor = DynamicFrameCompositor(self.project)  # Refresh compositor
        self.display_frame()
        self.status_var.set(f"Instances merged into '{new_name}'.")
        messagebox.showinfo("Success", f"Instances merged successfully into '{new_name}'.")

    # ============================================================
    # Refinement
    # ============================================================

    def toggle_point_removal_mode(self):
        """Toggle click-to-remove mode. When active, clicking near a point, box edge, or
        mask anchor removes it."""
        self.point_removal_mode = not self.point_removal_mode
        if self.point_removal_mode:
            self.remove_mode_button.config(text="Removal Mode: ON",
                                           bg='#DC143C', activebackground='#FF6347')
            self.status_var.set("Removal Mode: click near a point or box to delete it.")
        else:
            self.remove_mode_button.config(text="Removal Mode: OFF",
                                           bg='#404040', activebackground='#505050')
            self.status_var.set("Removal Mode off.")

    def remove_point_at_location(self, x: float, y: float) -> bool:
        """Remove the closest refinement point within 20px of (x, y). Returns True if removed."""
        if not self.refinement_points:
            return False
        min_dist = float('inf')
        closest_idx = None
        for i, (px, py, _) in enumerate(self.refinement_points):
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx is None or min_dist > 20:
            return False
        removed = self.refinement_points.pop(closest_idx)
        self.undo_stack.append({'type': 'remove_point', 'point': removed, 'index': closest_idx,
                                'concept_name': self.selected_concept.name,
                                'obj_id': self.selected_instance.sam3_obj_id,
                                'frame_idx': self.current_frame_idx})
        self.redo_stack.clear()
        self._update_points_cache()
        self._update_ann_label()
        self.display_frame()
        self.status_var.set(f"Removed point at ({removed[0]:.0f}, {removed[1]:.0f}). Ctrl+Z to undo.")
        return True

    def remove_box_at_location(self, x: float, y: float) -> bool:
        """Remove the smallest refinement box containing (x, y). Returns True if removed."""
        if not self.refinement_boxes:
            return False
        best_idx = None
        best_area = float('inf')
        for i, (x1, y1, x2, y2) in enumerate(self.refinement_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_idx = i
        if best_idx is None:
            return False
        removed = self.refinement_boxes.pop(best_idx)
        self.undo_stack.append({'type': 'remove_box', 'box': removed, 'index': best_idx,
                                'concept_name': self.selected_concept.name,
                                'obj_id': self.selected_instance.sam3_obj_id,
                                'frame_idx': self.current_frame_idx})
        self.redo_stack.clear()
        self._update_points_cache()
        self._update_ann_label()
        self.display_frame()
        self.status_var.set(f"Removed box at ({removed[0]:.0f}, {removed[1]:.0f})-"
                            f"({removed[2]:.0f}, {removed[3]:.0f}). Ctrl+Z to undo.")
        return True

    def _wizard_remove_pending_point(self, x: float, y: float):
        """Remove the closest pending wizard point within 20px of (x, y). Ctrl+Z also works."""
        if not self._wizard_pending_points:
            self.status_var.set("No wizard points to remove.")
            return
        min_dist = float('inf')
        closest_idx = None
        for i, (px, py, _) in enumerate(self._wizard_pending_points):
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx is None or min_dist > 20:
            self.status_var.set("No wizard point nearby to remove.")
            return
        removed = self._wizard_pending_points.pop(closest_idx)
        self._wizard_update_action_buttons()
        self.display_frame()
        kind = "positive" if removed[2] else "negative"
        self.status_var.set(f"Removed {kind} wizard point at ({removed[0]:.0f}, {removed[1]:.0f}). Ctrl+Z to undo.")

    def _canvas_to_image_coords(self, event_x: int, event_y: int):
        """Convert canvas pixel coordinates to image pixel coordinates. Returns (x, y) or None."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return None
        img_width, img_height = self.frame_dimensions
        img_aspect = img_width / img_height
        canvas_aspect = canvas_width / canvas_height
        if img_aspect > canvas_aspect:
            display_width = canvas_width
            display_height = int(canvas_width / img_aspect)
        else:
            display_height = canvas_height
            display_width = int(canvas_height * img_aspect)
        offset_x = (canvas_width - display_width) / 2
        offset_y = (canvas_height - display_height) / 2
        x = (event_x - offset_x) / display_width * img_width
        y = (event_y - offset_y) / display_height * img_height
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        return x, y

    def _toggle_box_mode(self):
        self.box_draw_mode = not self.box_draw_mode
        if self.box_draw_mode:
            self.box_mode_btn.config(text="Box Mode: ON", bg='#2a6a2a')
        else:
            self.box_mode_btn.config(text="Box Mode: OFF", bg='#404040')
            self._box_draw_start = None
            if self._box_draw_canvas_id:
                self.canvas.delete(self._box_draw_canvas_id)
                self._box_draw_canvas_id = None

    def _on_box_drag_motion(self, event):
        """Update live-drag rectangle preview when in box mode."""
        if not self._box_draw_start:
            return
        x0c, y0c = self._box_draw_start
        if self._box_draw_canvas_id:
            self.canvas.delete(self._box_draw_canvas_id)
        preview_color = 'yellow'
        self._box_draw_canvas_id = self.canvas.create_rectangle(
            x0c, y0c, event.x, event.y,
            outline=preview_color, width=2, dash=(4, 2),
        )

    def _on_box_drag_end(self, event):
        """Finish a box drag: convert to image coords and store."""
        if not self._box_draw_start:
            return
        x0c, y0c = self._box_draw_start
        self._box_draw_start = None
        if self._box_draw_canvas_id:
            self.canvas.delete(self._box_draw_canvas_id)
            self._box_draw_canvas_id = None

        # Require a minimum drag distance to distinguish from a plain click
        if abs(event.x - x0c) < 5 or abs(event.y - y0c) < 5:
            return

        c0 = self._canvas_to_image_coords(x0c, y0c)
        c1 = self._canvas_to_image_coords(event.x, event.y)
        if c0 is None or c1 is None:
            return
        x1, y1 = min(c0[0], c1[0]), min(c0[1], c1[1])
        x2, y2 = max(c0[0], c1[0]), max(c0[1], c1[1])
        box = (x1, y1, x2, y2)

        in_wizard = self._absorb_wizard and self._absorb_wizard['mode'] == 'point'
        if in_wizard:
            self._wizard_pending_boxes.append(box)
            self._wizard_update_action_buttons()
            self.display_frame()
            return

        self.refinement_boxes.append(box)
        self.undo_stack.append({'type': 'box', 'box': box,
                                'concept_name': self.selected_concept.name,
                                'obj_id': self.selected_instance.sam3_obj_id,
                                'frame_idx': self.current_frame_idx})
        self.redo_stack.clear()
        self._update_points_cache()
        self._update_ann_label()
        self.display_frame()

    def on_canvas_click(self, event):
        """Handle canvas click for refinement (add positive point, or start box drag)."""

        if not self.selected_instance:
            return

        # In box mode: record drag start; the actual box is committed on ButtonRelease.
        if self.box_draw_mode or (self._absorb_wizard and self._absorb_wizard['mode'] == 'point' and self.box_draw_mode):
            if not self.point_removal_mode:
                self._box_draw_start = (event.x, event.y)
                return

        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        x, y = coords

        if self._absorb_wizard and self._absorb_wizard['mode'] == 'point':
            if self.point_removal_mode:
                self._wizard_remove_pending_point(x, y)
            else:
                self._wizard_pending_points.append((x, y, True))
                self._wizard_update_action_buttons()
                self.display_frame()
            return

        if self.point_removal_mode:
            if not self.remove_point_at_location(x, y):
                if not self.remove_box_at_location(x, y):
                    self.remove_mask_anchor_at_location(x, y)
            return

        # Add positive point (left click in point mode)
        pt = (x, y, True)
        self.refinement_points.append(pt)
        self.undo_stack.append({'type': 'point', 'point': pt,
                                'concept_name': self.selected_concept.name,
                                'obj_id': self.selected_instance.sam3_obj_id,
                                'frame_idx': self.current_frame_idx})
        self.redo_stack.clear()
        self._update_points_cache()
        self._update_ann_label()
        self.display_frame()

    def on_canvas_right_click(self, event):
        """Right-click adds a negative refinement point (or removes in removal mode)."""
        if not self.selected_instance:
            return

        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        x, y = coords

        if self._absorb_wizard and self._absorb_wizard['mode'] == 'point':
            if self.point_removal_mode:
                self._wizard_remove_pending_point(x, y)
            else:
                self._wizard_pending_points.append((x, y, False))
                self._wizard_update_action_buttons()
                self.display_frame()
            return

        if self.point_removal_mode:
            if not self.remove_point_at_location(x, y):
                if not self.remove_box_at_location(x, y):
                    self.remove_mask_anchor_at_location(x, y)
            return

        pt = (x, y, False)
        self.refinement_points.append(pt)
        self.undo_stack.append({'type': 'point', 'point': pt,
                                'concept_name': self.selected_concept.name,
                                'obj_id': self.selected_instance.sam3_obj_id,
                                'frame_idx': self.current_frame_idx})
        self.redo_stack.clear()
        self._update_points_cache()
        self._update_ann_label()
        self.display_frame()

    def _update_ann_label(self):
        """Refresh the points/boxes count label."""
        n_pts = len(self.refinement_points)
        n_box = len(self.refinement_boxes)
        self.points_label.config(text=f"Points: {n_pts}  Boxes: {n_box}")

    # clear_refinement_points removed — duplicate of clear_frame_annotations.
    # The "Clear Unsaved Points" button has been removed; use Ctrl+Z to undo
    # individual point additions, or "Clear Frame Annotations" to wipe all.

    def _apply_point_action_context(self, action):
        """Navigate to the frame/instance stored in a point action, then reload annotations.
        Returns False if the target instance no longer exists (deleted)."""
        cn = action.get('concept_name')
        oid = action.get('obj_id')
        fi = action.get('frame_idx')
        if cn is None or oid is None or fi is None:
            return True  # old-format action without context — apply in-place
        self._select_instance_for_undo(cn, oid)
        if self.selected_instance is None or self.selected_instance.sam3_obj_id != oid:
            return False  # instance gone
        if fi != self.current_frame_idx:
            self.current_frame_idx = fi
            self.frame_slider.set(fi)
        self._load_annotations_for_current_frame()
        return True

    def undo_action(self, event=None):
        """Undo the last undoable action (Ctrl+Z). Navigates to the action's frame."""
        if self._absorb_wizard and self._absorb_wizard['mode'] == 'point':
            if self._wizard_pending_points:
                removed = self._wizard_pending_points.pop()
                self._wizard_update_action_buttons()
                self.display_frame()
                kind = "positive" if removed[2] else "negative"
                self.status_var.set(f"Undid {kind} wizard point at ({removed[0]:.0f}, {removed[1]:.0f}).")
            else:
                self.status_var.set("No wizard points to undo.")
            return
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        self.redo_stack.append(action)

        t = action['type']

        if t == 'point':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot undo: instance no longer exists.")
                return
            # Remove the specific point that was added (last occurrence for LIFO safety)
            pt = action['point']
            try:
                idx = max(i for i, p in enumerate(self.refinement_points) if p == pt)
                self.refinement_points.pop(idx)
            except (ValueError, TypeError):
                pass
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            kind = "positive" if pt[2] else "negative"
            self.status_var.set(f"Undid {kind} point at frame {action.get('frame_idx', '?')}.")

        elif t == 'box':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot undo: instance no longer exists.")
                return
            box = action['box']
            try:
                idx = max(i for i, b in enumerate(self.refinement_boxes) if b == box)
                self.refinement_boxes.pop(idx)
            except (ValueError, TypeError):
                pass
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Undid box at frame {action.get('frame_idx', '?')}.")

        elif t == 'remove_point':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot undo: instance no longer exists.")
                return
            idx = min(action['index'], len(self.refinement_points))
            self.refinement_points.insert(idx, action['point'])
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Restored removed point at frame {action.get('frame_idx', '?')}.")

        elif t == 'remove_box':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot undo: instance no longer exists.")
                return
            idx = min(action['index'], len(self.refinement_boxes))
            self.refinement_boxes.insert(idx, action['box'])
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Restored removed box at frame {action.get('frame_idx', '?')}.")

        elif t == 'clear_frame':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot undo: instance no longer exists.")
                return
            self.refinement_points.extend(action['points'])
            self.refinement_boxes.extend(action.get('boxes', []))
            self._update_points_cache()
            # Restore mask anchor if one was removed.
            if action.get('anchor_data') is not None:
                inst = self.selected_instance
                frame_idx = action['frame_idx']
                anchor_dir = action['anchor_dir']
                Path(anchor_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(anchor_dir) / f"{frame_idx:06d}.png"), action['anchor_data'])
                if frame_idx not in inst.mask_anchor_frames:
                    inst.mask_anchor_frames.append(frame_idx)
                    inst.mask_anchor_frames.sort()
                self._metadata_dirty = True
            self._update_ann_label()
            self.display_frame()
            n_pts = len(action['points'])
            n_box = len(action.get('boxes', []))
            self.status_var.set(f"Restored {n_pts} point(s) and {n_box} box(es) at frame "
                                f"{action.get('frame_idx', '?')}.")

        elif t == 'clear_all_instance':
            key = action['key']
            # Restore _points_cache to its pre-clear state.
            cache_entry = self._points_cache.setdefault(key, {})
            # Remove tombstones we added, then put back whatever was pending before.
            for f in action['tombstoned_frames']:
                cache_entry.pop(f, None)
            cache_entry.update(action['pending_snap'])
            # Restore _boxes_cache to its pre-clear state.
            box_cache_entry = self._boxes_cache.setdefault(key, {})
            for f in action['tombstoned_frames']:
                box_cache_entry.pop(f, None)
            box_cache_entry.update(action.get('box_pending_snap', {}))
            # Restore all-annotations caches.
            all_dict = self._all_annotations_cache.setdefault(key, {})
            all_dict.clear()
            all_dict.update(action['all_snap'])
            all_box_dict = self._all_boxes_cache.setdefault(key, {})
            all_box_dict.clear()
            all_box_dict.update(action.get('box_all_snap', {}))
            self._dirty_keys.add(key)
            # Restore current-frame live points/boxes if we're still on the same instance.
            if (self.selected_concept and self.selected_instance
                    and self.selected_concept.name == action['concept_name']
                    and self.selected_instance.sam3_obj_id == action['obj_id']):
                self.refinement_points.clear()
                self.refinement_points.extend(action['current_pts_snap'])
                self.refinement_boxes.clear()
                self.refinement_boxes.extend(action.get('current_boxes_snap', []))
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(
                f"Undid clear-all: restored annotations for instance (obj {action['obj_id']}).")

        elif t == 'delete_instance':
            action['instance'].deleted = False
            action['instance'].visible = True
            self._metadata_dirty = True
            self._mark_presence_dirty()
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Restored '{action['instance'].user_name}'.")

        elif t == 'absorb_instance':
            action['source'].deleted = False
            action['source'].absorbed_source = False
            action['source'].visible = True
            self._unstage_anchors_from_cache(
                action['concept'], action['target'], action['pos_entries'])
            if action.get('target_anchor_was_new'):
                self._unstage_anchors_from_cache(
                    action['concept'], action['target'], action['target_self_entries'])
                action['target'].received_absorb_anchor = False
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Absorb undone: '{action['source'].user_name}' restored.")

        elif t == 'absorb_wizard':
            action['source'].deleted = False
            action['source'].absorbed_source = False
            action['source'].visible = True
            # Remove from target's absorbed history
            src_id = action['source'].sam3_obj_id
            if src_id in action['target'].absorbed_source_ids:
                action['target'].absorbed_source_ids.remove(src_id)
            # Remove copied mask files from target's mask_anchors/
            for p in action.get('copied_mask_paths', []):
                try:
                    Path(p).unlink(missing_ok=True)
                    frame_idx = int(Path(p).stem)
                    if frame_idx in action['target'].mask_anchor_frames:
                        action['target'].mask_anchor_frames.remove(frame_idx)
                except Exception:
                    pass
            # Unstage any point anchors
            if action.get('staged_point_entries'):
                self._unstage_anchors_from_cache(
                    action['concept'], action['target'], action['staged_point_entries'])
                # Remove any pending mask_anchor trigger entries from disk
                mask_anchor_entries = [e for e in action['staged_point_entries']
                                       if e.get('type') == 'mask_anchor']
                if mask_anchor_entries:
                    self._remove_mask_anchor_triggers(
                        action['concept'].name, action['target'].sam3_obj_id,
                        [e['frame_idx'] for e in mask_anchor_entries])
            if action.get('target_anchor_was_new'):
                action['target'].received_absorb_anchor = False
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Absorb undone: '{action['source'].user_name}' restored.")

        elif t == 'remove_mask_anchor':
            inst = action['instance']
            frame_idx = action['frame_idx']
            anchor_dir = action['anchor_dir']
            dst = Path(anchor_dir) / f"{frame_idx:06d}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), action['mask_data'])
            if frame_idx not in inst.mask_anchor_frames:
                inst.mask_anchor_frames.append(frame_idx)
                inst.mask_anchor_frames.sort()
            self._metadata_dirty = True
            self.display_frame()
            self.status_var.set(f"Restored mask anchor at frame {frame_idx}.")

        elif t == 'save_mask_anchor':
            inst = action['instance']
            concept = action['concept']
            frame_idx = action['frame_idx']
            anchor_dir = action['anchor_dir']
            dst = Path(anchor_dir) / f"{frame_idx:06d}.png"
            if action['prev_mask_data'] is not None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), action['prev_mask_data'])
            else:
                dst.unlink(missing_ok=True)
            if action['was_new_frame'] and frame_idx in inst.mask_anchor_frames:
                inst.mask_anchor_frames.remove(frame_idx)
            self._remove_mask_anchor_triggers(
                concept.name, inst.sam3_obj_id, [frame_idx])
            self._metadata_dirty = True
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Undid saved mask anchor at frame {frame_idx}.")

        elif t == 'rename_instance':
            action['instance'].user_name = action['old_name']
            self._metadata_dirty = True
            self.update_concept_tree()
            self.status_var.set(f"Renamed back to '{action['old_name']}'.")

        elif t == 'rename_wizard':
            action['instance'].user_name = action['old_name']
            # Remove copied mask anchor file(s)
            for p in action.get('copied_mask_paths', []):
                try:
                    Path(p).unlink(missing_ok=True)
                    frame_idx = int(Path(p).stem)
                    if frame_idx in action['instance'].mask_anchor_frames:
                        action['instance'].mask_anchor_frames.remove(frame_idx)
                except Exception:
                    pass
            if action.get('staged_point_entries'):
                self._unstage_anchors_from_cache(
                    action['concept'], action['instance'], action['staged_point_entries'])
                mask_anchor_entries = [e for e in action['staged_point_entries']
                                       if e.get('type') == 'mask_anchor']
                if mask_anchor_entries:
                    self._remove_mask_anchor_triggers(
                        action['concept'].name, action['instance'].sam3_obj_id,
                        [e['frame_idx'] for e in mask_anchor_entries])
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Renamed back to '{action['old_name']}'.")

        elif t == 'reset_concept':
            cname = action['concept_name']
            concept = self.project.get_concept_by_name(cname)
            if concept:
                concept.instances = action['snapshot_instances']
                concept.status = action['snapshot_status']
                concept.highest_obj_id = action['snapshot_highest_obj_id']
                concept.completed_at = action.get('snapshot_completed_at')
                self._concepts_pending_reset.discard(cname)
                self.update_concept_tree()
                self.compositor = DynamicFrameCompositor(self.project)
                self.display_frame()
                self.status_var.set(f"Undid reset of concept '{cname}'.")

        elif t == 'change_prompt':
            cname = action['concept_name']
            concept = self.project.get_concept_by_name(cname)
            if concept:
                concept.text_prompt = action['old_prompt']
                self.project.save()
                self.update_concept_tree()
                self.status_var.set(
                    f"Undid prompt change for '{cname}' "
                    f"(restored: '{action['old_prompt']}').")

    def redo_action(self, event=None):
        """Redo the last undone action (Ctrl+Y). Navigates to the action's frame."""
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        self.undo_stack.append(action)

        t = action['type']

        if t == 'point':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot redo: instance no longer exists.")
                return
            self.refinement_points.append(action['point'])
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            kind = "positive" if action['point'][2] else "negative"
            self.status_var.set(f"Redid {kind} point at frame {action.get('frame_idx', '?')}.")

        elif t == 'box':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot redo: instance no longer exists.")
                return
            self.refinement_boxes.append(action['box'])
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Redid box at frame {action.get('frame_idx', '?')}.")

        elif t == 'remove_point':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot redo: instance no longer exists.")
                return
            idx = min(action['index'], len(self.refinement_points))
            if 0 <= idx < len(self.refinement_points):
                self.refinement_points.pop(idx)
            elif self.refinement_points:
                self.refinement_points.pop()
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Re-removed point at frame {action.get('frame_idx', '?')}.")

        elif t == 'remove_box':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot redo: instance no longer exists.")
                return
            idx = min(action['index'], len(self.refinement_boxes))
            if 0 <= idx < len(self.refinement_boxes):
                self.refinement_boxes.pop(idx)
            elif self.refinement_boxes:
                self.refinement_boxes.pop()
            self._update_points_cache()
            self._update_ann_label()
            self.display_frame()
            self.status_var.set(f"Re-removed box at frame {action.get('frame_idx', '?')}.")

        elif t == 'clear_frame':
            if not self._apply_point_action_context(action):
                self.status_var.set("Cannot redo: instance no longer exists.")
                return
            self.refinement_points.clear()
            self.refinement_boxes.clear()
            self._update_points_cache()
            if action.get('anchor_data') is not None:
                inst = self.selected_instance
                frame_idx = action['frame_idx']
                anchor_dir = action['anchor_dir']
                if frame_idx in inst.mask_anchor_frames:
                    inst.mask_anchor_frames.remove(frame_idx)
                self._remove_mask_anchor_triggers(
                    action['concept_name'], inst.sam3_obj_id, [frame_idx])
                for ext in ('.png', '.npz'):
                    p = Path(anchor_dir) / f"{frame_idx:06d}{ext}"
                    if p.exists():
                        p.unlink()
                self._metadata_dirty = True
            self.points_label.config(text="Points: 0  Boxes: 0")
            self.display_frame()
            self.status_var.set(f"Re-cleared frame {action.get('frame_idx', '?')}.")

        elif t == 'clear_all_instance':
            key = action['key']
            cache_entry = self._points_cache.setdefault(key, {})
            box_cache_entry = self._boxes_cache.setdefault(key, {})
            for f in action['tombstoned_frames']:
                cache_entry[f] = []
                box_cache_entry[f] = []
            all_dict = self._all_annotations_cache.setdefault(key, {})
            all_dict.clear()
            all_box_dict = self._all_boxes_cache.setdefault(key, {})
            all_box_dict.clear()
            self._dirty_keys.add(key)
            if (self.selected_concept and self.selected_instance
                    and self.selected_concept.name == action['concept_name']
                    and self.selected_instance.sam3_obj_id == action['obj_id']):
                self.refinement_points.clear()
                self.refinement_boxes.clear()
            self.points_label.config(text="Points: 0  Boxes: 0")
            self.display_frame()
            self.status_var.set(
                f"Re-cleared all annotations for instance (obj {action['obj_id']}).")

        elif t == 'delete_instance':
            action['instance'].deleted = True
            action['instance'].visible = False
            self._metadata_dirty = True
            self._mark_presence_dirty()
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Re-deleted '{action['instance'].user_name}'.")

        elif t == 'absorb_instance':
            action['source'].deleted = True
            action['source'].visible = False
            self._stage_anchors_to_cache(action['concept'], action['target'], action['pos_entries'])
            if action.get('target_anchor_was_new'):
                self._stage_anchors_to_cache(
                    action['concept'], action['target'], action['target_self_entries'])
                action['target'].received_absorb_anchor = True
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Re-absorbed '{action['source'].user_name}'.")

        elif t == 'absorb_wizard':
            # Re-copy masks + re-stage points + re-delete source
            import shutil as _shutil
            action['source'].deleted = True
            action['source'].absorbed_source = True
            action['source'].visible = False
            action['target'].received_absorb_anchor = True
            src_id = action['source'].sam3_obj_id
            if src_id not in action['target'].absorbed_source_ids:
                action['target'].absorbed_source_ids.append(src_id)
            for p in action.get('copied_mask_paths', []):
                try:
                    frame_idx = int(Path(p).stem)
                    # The source mask might still exist; copy it back
                    # (we stored the path; if missing, skip — impact is just a missing anchor)
                    if not Path(p).exists():
                        pass  # can't redo mask copy without source
                    if frame_idx not in action['target'].mask_anchor_frames:
                        action['target'].mask_anchor_frames.append(frame_idx)
                except Exception:
                    pass
            if action.get('staged_point_entries'):
                self._stage_anchors_to_cache(
                    action['concept'], action['target'], action['staged_point_entries'])
                # Re-write any mask_anchor trigger entries to disk
                mask_anchor_entries = [e for e in action['staged_point_entries']
                                       if e.get('type') == 'mask_anchor']
                if mask_anchor_entries:
                    self._write_mask_anchor_triggers(
                        action['concept'].name, action['target'].sam3_obj_id,
                        mask_anchor_entries)
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Re-absorbed '{action['source'].user_name}'.")

        elif t == 'remove_mask_anchor':
            inst = action['instance']
            frame_idx = action['frame_idx']
            anchor_dir = action['anchor_dir']
            dst = Path(anchor_dir) / f"{frame_idx:06d}.png"
            if dst.exists():
                dst.unlink()
            if frame_idx in inst.mask_anchor_frames:
                inst.mask_anchor_frames.remove(frame_idx)
            self._metadata_dirty = True
            self.display_frame()
            self.status_var.set(f"Re-removed mask anchor at frame {frame_idx}.")

        elif t == 'save_mask_anchor':
            inst = action['instance']
            concept = action['concept']
            frame_idx = action['frame_idx']
            anchor_dir = action['anchor_dir']
            dst = Path(anchor_dir) / f"{frame_idx:06d}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), action['mask_data'])
            if frame_idx not in inst.mask_anchor_frames:
                inst.mask_anchor_frames.append(frame_idx)
                inst.mask_anchor_frames.sort()
            self._stage_anchors_to_cache(concept, inst, [action['trigger_entry']])
            self._metadata_dirty = True
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Redid mask anchor save at frame {frame_idx}.")

        elif t == 'rename_instance':
            action['instance'].user_name = action['new_name']
            self._metadata_dirty = True
            self.update_concept_tree()
            self.status_var.set(f"Renamed to '{action['new_name']}'.")

        elif t == 'rename_wizard':
            action['instance'].user_name = action['new_name']
            for p in action.get('copied_mask_paths', []):
                frame_idx = int(Path(p).stem)
                if frame_idx not in action['instance'].mask_anchor_frames:
                    action['instance'].mask_anchor_frames.append(frame_idx)
            if action.get('staged_point_entries'):
                self._stage_anchors_to_cache(
                    action['concept'], action['instance'], action['staged_point_entries'])
                mask_anchor_entries = [e for e in action['staged_point_entries']
                                       if e.get('type') == 'mask_anchor']
                if mask_anchor_entries:
                    self._write_mask_anchor_triggers(
                        action['concept'].name, action['instance'].sam3_obj_id,
                        mask_anchor_entries)
            self._metadata_dirty = True
            self.update_concept_tree()
            self.compositor = DynamicFrameCompositor(self.project)
            self.display_frame()
            self.status_var.set(f"Renamed to '{action['new_name']}'.")

        elif t == 'reset_concept':
            cname = action['concept_name']
            concept = self.project.get_concept_by_name(cname)
            if concept:
                concept.instances = []
                concept.status = ConceptStatus.PENDING
                concept.highest_obj_id = -1
                concept.completed_at = None
                self._concepts_pending_reset.add(cname)
                keys_to_drop = [k for k in list(self._points_cache) if k[0] == cname]
                for k in keys_to_drop:
                    self._points_cache.pop(k, None)
                    self._all_annotations_cache.pop(k, None)
                    self._dirty_keys.discard(k)
                self._invalidate_presence_cache()
                self.update_concept_tree()
                self.compositor = DynamicFrameCompositor(self.project)
                self.display_frame()
                self.status_var.set(f"Re-applied reset of concept '{cname}'.")

        elif t == 'change_prompt':
            cname = action['concept_name']
            concept = self.project.get_concept_by_name(cname)
            if concept:
                concept.text_prompt = action['new_prompt']
                self.project.save()
                self.update_concept_tree()
                self.status_var.set(
                    f"Re-applied prompt change for '{cname}' "
                    f"(now: '{action['new_prompt']}').")

    def _load_frame_annotations(self, frame_idx: int) -> List[tuple]:
        """Load saved points for the selected instance at frame_idx (any propagation state)."""
        if not self.selected_instance or not self.selected_concept or not self.project:
            return []
        rpath = os.path.join(
            self.project.project_dir, "concepts", self.selected_concept.name,
            "instances", str(self.selected_instance.sam3_obj_id), "refinements.json"
        )
        if not os.path.exists(rpath):
            return []
        with open(rpath) as f:
            entries = json.load(f).get("refinements", [])
        for entry in reversed(entries):
            if entry.get("frame_idx") == frame_idx:
                return [(p["x"], p["y"], p["is_positive"]) for p in entry.get("points", [])]
        return []

    def _save_current_points_to_file(self):
        """Persist self.refinement_points to refinements.json, replacing any existing entry for the current frame."""
        if not self.selected_instance or not self.selected_concept or not self.project:
            return
        rpath = os.path.join(
            self.project.project_dir, "concepts", self.selected_concept.name,
            "instances", str(self.selected_instance.sam3_obj_id), "refinements.json"
        )
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        existing = []
        if os.path.exists(rpath):
            with open(rpath) as f:
                existing = json.load(f).get("refinements", [])
        # Replace any existing entry for this frame
        existing = [r for r in existing if r.get("frame_idx") != self.current_frame_idx]
        if self.refinement_points:
            from datetime import datetime
            existing.append({
                "timestamp": datetime.now().isoformat(),
                "frame_idx": self.current_frame_idx,
                "points": [{"x": x, "y": y, "is_positive": is_pos}
                           for x, y, is_pos in self.refinement_points],
                "propagated": False,
            })
        with open(rpath, "w") as f:
            json.dump({"refinements": existing}, f, indent=2)

    def _flush_instance_cache_to_file(self, concept_name: str, obj_id: int) -> int:
        """Overwrite all propagated=False entries in refinements.json with the current cache
        state for (concept_name, obj_id).  propagated=True (historical) entries are preserved
        EXCEPT for frames the user edited this session: the UI loads a frame's historical
        points into the editable list, so the cache holds that frame's complete point set
        and the old entry is superseded (keeping it would duplicate points — or resurrect
        deleted ones — when --refine replays all saved annotations).
        Calling this when the cache is empty correctly removes any stale pending entries.
        Returns the number of pending frames written."""
        if not self.project:
            return 0
        key = (concept_name, obj_id)
        cached_pts = self._points_cache.get(key, {})
        cached_boxes = self._boxes_cache.get(key, {})
        # Frames touched by either points or boxes
        all_touched_frames = set(cached_pts) | set(cached_boxes)
        rpath = os.path.join(
            self.project.project_dir, "concepts", concept_name,
            "instances", str(obj_id), "refinements.json"
        )
        if not all_touched_frames and not os.path.exists(rpath):
            return 0
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        applied = []
        if os.path.exists(rpath):
            with open(rpath) as f:
                applied = [r for r in json.load(f).get("refinements", [])
                           if r.get("frame_idx") not in all_touched_frames
                           or (not r.get("propagated", False) and r.get("type") == "mask_anchor")]
        from datetime import datetime
        pending = []
        for frame_idx in sorted(all_touched_frames):
            pts = cached_pts.get(frame_idx, [])
            boxes = cached_boxes.get(frame_idx, [])
            if pts or boxes:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_idx": frame_idx,
                    "propagated": False,
                }
                if pts:
                    entry["points"] = [{"x": x, "y": y, "is_positive": is_pos}
                                       for x, y, is_pos in pts]
                if boxes:
                    entry["boxes"] = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                      for x1, y1, x2, y2 in boxes]
                pending.append(entry)
        with open(rpath, "w") as f:
            json.dump({"refinements": applied + pending}, f, indent=2)
        return len(pending)

    def clear_frame_annotations(self):
        """Clear ALL annotation points (historical + pending) for the selected instance
        at the current frame. Also removes any mask anchor at this frame.
        Undoable within the session; deferred to disk until save."""
        if not self.selected_instance or not self.selected_concept:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return
        inst = self.selected_instance
        frame_idx = self.current_frame_idx

        snapshot = list(self.refinement_points)
        box_snapshot = list(self.refinement_boxes)
        self.refinement_points.clear()
        self.refinement_boxes.clear()
        # Tombstone in cache — disk write deferred to explicit save.
        self._update_points_cache()

        # Also remove any mask anchor at this frame.
        anchor_data = None
        if frame_idx in inst.mask_anchor_frames:
            anchor_dir = os.path.join(
                self.project.project_dir, "concepts", self.selected_concept.name,
                "instances", str(inst.sam3_obj_id), "mask_anchors"
            )
            mask_np = load_sam3_mask(anchor_dir, frame_idx)
            if mask_np is not None:
                anchor_data = mask_np.copy()
            inst.mask_anchor_frames.remove(frame_idx)
            self._remove_mask_anchor_triggers(
                self.selected_concept.name, inst.sam3_obj_id, [frame_idx])
            for ext in ('.png', '.npz'):
                p = Path(anchor_dir) / f"{frame_idx:06d}{ext}"
                if p.exists():
                    p.unlink()
            self._metadata_dirty = True

        if snapshot or box_snapshot or anchor_data is not None:
            self.undo_stack.append({
                'type': 'clear_frame',
                'concept_name': self.selected_concept.name,
                'obj_id': inst.sam3_obj_id,
                'frame_idx': frame_idx,
                'points': snapshot,
                'boxes': box_snapshot,
                'anchor_data': anchor_data,
                'anchor_dir': anchor_dir if anchor_data is not None else None,
            })
            self.redo_stack.clear()
        self.points_label.config(text="Points: 0  Boxes: 0")
        self.display_frame()
        self.status_var.set(f"Cleared all annotations at frame {frame_idx}. "
                            "Ctrl+Z to undo. Save Changes for Refinement or Apply to persist.")

    def clear_all_instance_annotations(self):
        """Clear ALL annotation points for the selected instance across every frame.
        Tombstones every frame so --refine won't replay any of them.
        Undoable within the session; deferred to disk until save."""
        if not self.selected_instance or not self.selected_concept:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return
        inst = self.selected_instance
        concept = self.selected_concept
        key = self._instance_key()

        # Gather what exists: pending cache + all-annotations cache (covers historical too)
        pending_snap = dict(self._points_cache.get(key, {}))
        all_snap = dict(self._all_annotations_cache.get(key, {}))
        box_pending_snap = dict(self._boxes_cache.get(key, {}))
        box_all_snap = dict(self._all_boxes_cache.get(key, {}))
        current_pts_snap = list(self.refinement_points)
        current_boxes_snap = list(self.refinement_boxes)

        all_frames = sorted(set(pending_snap) | set(all_snap) | set(box_pending_snap) | set(box_all_snap))
        total = len(all_frames)

        if not messagebox.askyesno(
            "Clear ALL Annotations",
            f"This will delete all annotation points and boxes for '{inst.user_name}' "
            f"across {total} frame(s).\n\n"
            "The instance itself is kept; only the click-point/box annotations are removed. "
            "You can Ctrl+Z to undo within this session.\n\n"
            "Proceed?"
        ):
            return

        # Tombstone every frame in _points_cache/_boxes_cache so _flush writes empty entries
        # (which supersede historical propagated=True entries on --refine).
        cache_entry = self._points_cache.setdefault(key, {})
        box_cache_entry = self._boxes_cache.setdefault(key, {})
        for f in all_frames:
            cache_entry[f] = []
            box_cache_entry[f] = []
        # Also wipe the current-frame live lists and their all-annotations entries.
        self.refinement_points.clear()
        self.refinement_boxes.clear()
        all_dict = self._all_annotations_cache.setdefault(key, {})
        all_dict.clear()
        all_box_dict = self._all_boxes_cache.setdefault(key, {})
        all_box_dict.clear()
        self._dirty_keys.add(key)

        self.undo_stack.append({
            'type': 'clear_all_instance',
            'concept_name': concept.name,
            'obj_id': inst.sam3_obj_id,
            'key': key,
            'pending_snap': pending_snap,
            'all_snap': all_snap,
            'box_pending_snap': box_pending_snap,
            'box_all_snap': box_all_snap,
            'current_pts_snap': current_pts_snap,
            'current_boxes_snap': current_boxes_snap,
            'tombstoned_frames': all_frames,
        })
        self.redo_stack.clear()

        self.points_label.config(text="Points: 0  Boxes: 0")
        self.display_frame()
        self.status_var.set(
            f"Cleared all annotations for '{inst.user_name}' ({total} frame(s)). "
            "Ctrl+Z to undo. Save Changes for Refinement or Apply to persist."
        )

    def _purge_deleted_instance_masks(self, concepts=None):
        """Remove mask dirs for user-deleted instances.

        Absorbed sources (deleted=True, absorbed_source=True) are intentionally
        skipped here — their original mask files are preserved until the next
        --refine run processes the absorb and deletes them at that point.

        Args:
            concepts: iterable of SAM3Concept to check, or None to check all.
        """
        import shutil
        if concepts is None:
            concepts = [self.project.get_concept_by_name(n)
                        for n in self.project.concept_order]
        for concept in concepts:
            if concept is None:
                continue
            for inst in concept.instances:
                if not inst.deleted:
                    continue
                if inst.absorbed_source:
                    continue  # preserved until --refine completes
                masks_dir = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(inst.sam3_obj_id), "masks"
                )
                if os.path.isdir(masks_dir):
                    shutil.rmtree(masks_dir)
                mask_anchors_dir = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(inst.sam3_obj_id), "mask_anchors"
                )
                if os.path.isdir(mask_anchors_dir):
                    shutil.rmtree(mask_anchors_dir)
                cache_key = (concept.name, inst.sam3_obj_id)
                self._presence_cache.pop(cache_key, None)

    def save_points_for_batch(self):
        """Save current points to refinements.json and flush any pending concept resets.

        Marks annotation entries as propagated=False so sam3_process.py --refine
        will replay them.  For concepts pending reset: deletes stale instance dirs /
        cond states / inference state, then writes the redetect sentinel.
        """
        def _any_pending_deletes():
            for n in self.project.concept_order:
                c = self.project.get_concept_by_name(n)
                if c and any(inst.deleted for inst in c.instances):
                    return True
            return False

        if (not self._dirty_keys and not self._concepts_pending_reset
                and not _any_pending_deletes() and not self._metadata_dirty):
            self.status_var.set("No unsaved changes to write.")
            return

        # Flush annotation points
        total_frames = 0
        total_points = 0
        for key in list(self._dirty_keys):
            concept_name, obj_id = key
            n = self._flush_instance_cache_to_file(concept_name, obj_id)
            total_frames += n
            total_points += sum(len(pts) for pts in self._points_cache.get(key, {}).values())
        self._dirty_keys.clear()

        # Flush pending concept resets — write sentinel only.
        # Actual folder deletion (instances/, cond_states/, inference_state.pkl)
        # is left to sam3_process.py --refine, which runs on the machine where
        # the data lives. The in-memory state was already cleared in
        # reset_concept_redetect(), so the compositor shows nothing for these concepts.
        for cname in list(self._concepts_pending_reset):
            concept_dir = self.project.get_concept_dir(cname)
            os.makedirs(concept_dir, exist_ok=True)
            open(os.path.join(concept_dir, "redetect"), "w").close()
        n_resets = len(self._concepts_pending_reset)
        self._concepts_pending_reset.clear()

        # Warn that pending user-initiated deletions are permanent once committed.
        # Absorbed sources (deleted=True, absorbed_source=True) are NOT shown here —
        # their mask files are intentionally kept as historical data for future union checks.
        pending_deletes = [
            (c.name, inst.user_name)
            for n in self.project.concept_order
            for c in [self.project.get_concept_by_name(n)] if c
            for inst in c.instances if inst.deleted and not inst.absorbed_source
        ]
        if pending_deletes:
            names_preview = ", ".join(f"'{nm}'" for _, nm in pending_deletes[:5])
            if len(pending_deletes) > 5:
                names_preview += f" and {len(pending_deletes) - 5} more"
            if not messagebox.askyesno(
                "Permanent Deletion",
                f"The following instance(s) will have their mask files permanently deleted "
                f"from disk (cannot be undone):\n\n{names_preview}\n\n"
                f"Continue?"
            ):
                return

        # Commit pending user-initiated deletions: remove mask dirs from disk now,
        # then remove them from concept.instances.
        # Absorbed sources are kept in concept.instances (with deleted=True, absorbed_source=True)
        # so future absorbs into the same target can still find their historical masks.
        self._purge_deleted_instance_masks()
        for n in self.project.concept_order:
            c = self.project.get_concept_by_name(n)
            if c:
                c.instances = [inst for inst in c.instances
                               if not inst.deleted or inst.absorbed_source]

        # After purging, stale delete_instance undo entries would restore ghost instances
        # with no mask files — scrub them so Ctrl+Z can't create ghosts.
        self.undo_stack = [a for a in self.undo_stack if a.get('type') != 'delete_instance']

        self._safe_project_save()
        self._metadata_dirty = False

        parts = []
        if total_frames:
            parts.append(f"{total_points} point(s) across {total_frames} frame(s)")
        if n_resets:
            parts.append(f"{n_resets} concept reset(s) written")
        self.status_var.set(
            f"Saved: {', '.join(parts)}. "
            f"Run: python sam3_process.py --project <dir> --refine"
        )

    def _collect_pending_refinements(self) -> List[tuple]:
        """Return list of (instance, pending_entries) for every non-deleted instance in the
        selected concept that has at least one refinement entry with propagated=False."""
        if not self.selected_concept or not self.project:
            return []
        result = []
        for inst in self.selected_concept.instances:
            if inst.deleted:
                continue
            rpath = os.path.join(
                self.project.project_dir, "concepts", self.selected_concept.name,
                "instances", str(inst.sam3_obj_id), "refinements.json"
            )
            if not os.path.exists(rpath):
                continue
            with open(rpath) as f:
                entries = json.load(f).get("refinements", [])
            pending = [r for r in entries if not r.get("propagated", True)]
            if pending:
                result.append((inst, pending))
        return result

    def apply_refinement(self):
        """Apply refinement points and re-propagate"""

        if not self.selected_instance or not self.selected_concept:
            messagebox.showwarning("Warning", "Please select an instance first.")
            return

        # Flush all cached points for every concept instance to file so that
        # _collect_pending_refinements sees the full picture (including points added on
        # frames other than the currently displayed one).
        for inst in self.selected_concept.instances:
            if not inst.deleted:
                self._flush_instance_cache_to_file(
                    self.selected_concept.name, inst.sam3_obj_id)

        # Check whether there is anything to apply (in memory or newly flushed to file)
        any_in_memory = bool(self.refinement_points)
        pending_check = self._collect_pending_refinements()
        if not any_in_memory and not pending_check:
            messagebox.showwarning("Warning", "No refinement points to apply.")
            return

        # Count total pending points for the confirmation dialog
        total_pts = len(self.refinement_points)

        # Confirm
        confirm = messagebox.askyesno(
            "Confirm Refinement",
            f"Apply refinement points and re-propagate?\n"
            f"This will update masks for instance '{self.selected_instance.user_name}'.\n\n"
            f"Any saved pending points for other instances in this concept will also be\n"
            f"applied in the same pass so non-overlapping constraints are respected."
        )

        if not confirm:
            return

        # Collect ALL pending (propagated=False) corrections across the concept — this
        # ensures that corrections for multiple instances (e.g. +A and -B) are applied
        # jointly in a single propagation pass where non-overlapping is enforced.
        all_pending_snap = self._collect_pending_refinements()
        n_pending_instances = len(all_pending_snap)

        # Capture concept ref on the main thread before any dialogs.
        concept_snap = self.selected_concept

        # Pre-check: if we know we're going offline and frames are missing, resolve the
        # video path now on the main thread so the background thread doesn't have to.
        # This preserves project.frames_dir (which may point to another host) in metadata.
        video_for_extraction = self.project.video_path
        if not self.active_sessions.get(concept_snap.name):
            from sam3_utils import frames_dir_is_complete
            if (not frames_dir_is_complete(self.project.get_frames_dir(),
                                           self.project.num_frames)
                    and not os.path.exists(self.project.video_path)):
                from tkinter import filedialog
                video_for_extraction = filedialog.askopenfilename(
                    title="Video not found - locate source video for frame extraction",
                    filetypes=[
                        ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                        ("All files", "*.*"),
                    ]
                )
                if not video_for_extraction:
                    return  # user cancelled

        # Show progress dialog
        progress_dialog = ProgressDialog(
            self.root,
            "Refining Masks",
            f"Re-propagating corrections for {n_pending_instances} instance(s)..."
        )

        def progress_callback(frame_idx, num_frames):
            """Update progress from background thread"""
            self.root.after(0, progress_dialog.update_progress, frame_idx, num_frames)

        def refine_thread():
            try:
                # SAM3's process-wide bf16 autocast is thread-local; re-enter it
                # on this worker thread or the model hits bf16/float32 mismatches.
                ensure_bf16_autocast()
                from sam3_utils import extract_frames_from_video

                # Load model if needed
                if not self.sam3_model:
                    self.root.after(0, progress_dialog.update_progress, 0, 1, "Loading SAM3 model...")
                    self.sam3_model = load_sam3_model(device=self.device)
                    self.root.after(0, self._update_model_status)

                orig_width, orig_height = self.frame_dimensions
                live_session_id = self.active_sessions.get(concept_snap.name)

                # Commit pending instance deletions for this concept before propagating.
                # Also scrub stale undo entries so Ctrl+Z can't restore ghost instances.
                self._purge_deleted_instance_masks(concepts=[concept_snap])
                self.undo_stack = [a for a in self.undo_stack if a.get('type') != 'delete_instance']

                if live_session_id:
                    # Online fast path: add all correction anchors then propagate ONCE.
                    # Non-overlapping is enforced jointly across all instances.
                    print(f"Online refinement path — {n_pending_instances} instance(s), "
                          f"session {live_session_id[:8]}...")
                    self.root.after(0, progress_dialog.update_progress, 0, self.num_frames,
                                    f"Propagating {n_pending_instances} instance correction(s) jointly...")
                    try:
                        online_replay_concept_refinements(
                            concept=concept_snap,
                            instances_with_pending=all_pending_snap,
                            orig_width=orig_width,
                            orig_height=orig_height,
                            num_frames=self.num_frames,
                            sam3_model=self.sam3_model,
                            session_id=live_session_id,
                            project_dir=self.project.project_dir,
                            progress_callback=progress_callback,
                            mask_format=self.project.mask_format,
                            new_instance_policy="allow",
                        )
                    except Exception as online_err:
                        # Session may have gone stale — fall back to offline path
                        import traceback
                        print(f"Online path failed ({online_err}); falling back to offline path.")
                        traceback.print_exc()
                        self.active_sessions.pop(concept_snap.name, None)
                        self.root.after(0, self._update_session_status)
                        live_session_id = None  # trigger offline path below

                if not live_session_id:
                    # Offline path: re-detect + restore prior cond states, apply all instances jointly
                    print(f"Offline refinement path — {n_pending_instances} instance(s), re-detecting...")
                    from sam3_utils import frames_dir_is_complete
                    resource_path = self.project.get_frames_dir()
                    if not frames_dir_is_complete(resource_path, self.project.num_frames):
                        self.root.after(0, progress_dialog.update_progress, 0, 1, "Extracting frames...")
                        # video_for_extraction may differ from project.video_path when the user
                        # located a video on this machine for cross-machine use; we intentionally
                        # do NOT write it back to project.video_path or project.frames_dir so
                        # that the originating host's paths are preserved in project.json.
                        # Resume-safe: keeps frames from a previously interrupted extraction.
                        extract_frames_from_video(video_for_extraction, resource_path)

                    self.root.after(0, progress_dialog.update_progress, 0, self.num_frames,
                                    f"Propagating {n_pending_instances} instance correction(s) jointly (re-detecting)...")
                    replay_concept_refinements(
                        concept=concept_snap,
                        instances_with_pending=all_pending_snap,
                        resource_path=resource_path,
                        orig_width=orig_width,
                        orig_height=orig_height,
                        num_frames=self.num_frames,
                        sam3_model=self.sam3_model,
                        project_dir=self.project.project_dir,
                        progress_callback=progress_callback,
                        device=self.device,
                        mask_format=self.project.mask_format,
                        preload_original_masks=False,
                        new_instance_policy="allow",
                    )

                # Close dialog and refresh display; save + metadata reset happen there
                self.root.after(0, progress_dialog.close)
                self.root.after(0, self._refinement_complete)

            except Exception as e:
                import traceback
                print(f"\n=== ERROR: Failed to apply refinement ===")
                traceback.print_exc()
                print(f"==========================================\n")
                self.root.after(0, progress_dialog.close)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to apply refinement:\n{e}"
                ))
                self.root.after(0, lambda: self.status_var.set("Refinement failed."))

        thread = threading.Thread(target=refine_thread, daemon=True)
        thread.start()

    def _refinement_complete(self):
        """Called when refinement completes"""
        self._safe_project_save()
        self._metadata_dirty = False
        self.refinement_points.clear()

        # All pending points have been applied — clear them from the in-memory cache
        # so the concept starts fresh.  File entries are already marked propagated=True.
        if self.selected_concept:
            for inst in self.selected_concept.instances:
                key = (self.selected_concept.name, inst.sam3_obj_id)
                self._points_cache.pop(key, None)
                self._dirty_keys.discard(key)
        self._invalidate_presence_cache()
        self.compositor = DynamicFrameCompositor(self.project)  # Refresh compositor
        self.update_concept_tree()  # Refresh coverage % from updated metadata
        self.display_frame()
        self.points_label.config(text="Points: 0  Boxes: 0")
        self.status_var.set("Refinement applied successfully.")
        messagebox.showinfo("Success", "Refinement applied and masks updated.")

    # ============================================================
    # Flash Animations
    # ============================================================

    def _reset_flash_state(self):
        self.flash_mask_in_progress = False
        self.flash_mask_on = False
        self.flash_overlap_in_progress = False
        self.flash_overlap_on = False
        self.flash_overlap_computed = None
        self.flash_points_in_progress = False
        self.flash_points_on = False
        self.flash_custom_in_progress = False
        self.flash_custom_on = False
        self.flash_custom_mask = None

    def _handle_flash_shortcut(self):
        if self._should_ignore_keyboard_shortcut():
            return
        if self.selected_instance and self.selected_concept and self.project:
            self.flash_selected_instance_mask()
        elif self.refinement_points:
            self.flash_frame_points()

    def _handle_overlap_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self.flash_overlap_regions()

    def _handle_removal_shortcut(self):
        if not self._should_ignore_keyboard_shortcut():
            self.toggle_point_removal_mode()

    def flash_selected_instance_mask(self):
        """Flash the selected instance's mask white (3×, 300 ms each)."""
        if not self.selected_instance or not self.selected_concept or not self.project:
            messagebox.showinfo("Info", "Please select an instance first.")
            return
        _flash_mask_dir = os.path.join(
            self.project.project_dir, "concepts", self.selected_concept.name,
            "instances", str(self.selected_instance.sam3_obj_id), "masks",
        )
        if load_sam3_mask(_flash_mask_dir, self.current_frame_idx) is None:
            messagebox.showinfo("Info", "No mask for this instance on the current frame.")
            return

        self.flash_mask_in_progress = True
        self.flash_mask_on = False
        flash_count = [0]

        def flash_step():
            if flash_count[0] >= 3:
                self._reset_flash_state()
                self.display_frame()
                return
            self.flash_mask_on = not self.flash_mask_on
            self.display_frame()
            if not self.flash_mask_on:
                flash_count[0] += 1
            self.root.after(300, flash_step)

        flash_step()

    def flash_mask_array(self, mask_np):
        """Flash an arbitrary mask array (cyan overlay, 3x, 300ms each).

        Used by dialogs that let the user choose among several candidate masks
        (e.g. the absorb/rename union dialog) so they can visually identify
        which mask a given checkbox/label refers to before deciding.
        """
        if mask_np is None:
            return
        self.flash_custom_mask = mask_np
        self.flash_custom_in_progress = True
        self.flash_custom_on = False
        flash_count = [0]

        def flash_step():
            if flash_count[0] >= 3:
                self._reset_flash_state()
                self.display_frame()
                return
            self.flash_custom_on = not self.flash_custom_on
            self.display_frame()
            if not self.flash_custom_on:
                flash_count[0] += 1
            self.root.after(300, flash_step)

        flash_step()

    def flash_frame_points(self):
        """Flash the current refinement points enlarged (3×, 300 ms each)."""
        if not self.refinement_points:
            return
        self.flash_points_in_progress = True
        self.flash_points_on = True
        flash_count = [0]

        def flash_step():
            if flash_count[0] >= 3:
                self.flash_points_in_progress = False
                self.flash_points_on = False
                self.display_frame()
                return
            self.flash_points_on = not self.flash_points_on
            self.display_frame()
            if not self.flash_points_on:
                flash_count[0] += 1
            self.root.after(300, flash_step)

        flash_step()

    def _compute_overlap_mask_for_frame(self, frame_idx: int):
        """Return boolean H×W mask of pixels claimed by 2+ visible instances, or None."""
        if not self.project:
            return None
        masks = []
        for concept in self.project.concepts:
            if not concept.visible:
                continue
            for inst in concept.instances:
                if inst.deleted or not inst.visible:
                    continue
                _ov_mask_dir = os.path.join(
                    self.project.project_dir, "concepts", concept.name,
                    "instances", str(inst.sam3_obj_id), "masks",
                )
                m = load_sam3_mask(_ov_mask_dir, frame_idx)
                if m is not None:
                    masks.append(m > 127)
        if len(masks) < 2:
            return None
        h, w = masks[0].shape
        count = np.zeros((h, w), dtype=np.int32)
        for m in masks:
            if m.shape != (h, w):
                m_pil = Image.fromarray(m.astype(np.uint8) * 255)
                m_pil = m_pil.resize((w, h), Image.NEAREST)
                m = np.array(m_pil) > 0
            count[m] += 1
        overlap = count >= 2
        return overlap if np.any(overlap) else None

    def flash_overlap_regions(self):
        """Flash pixels claimed by 2+ instances with an orange-red overlay (3×, 300 ms)."""
        if not self.project:
            messagebox.showinfo("Info", "No project loaded.")
            return
        overlap_mask = self._compute_overlap_mask_for_frame(self.current_frame_idx)
        if overlap_mask is None:
            messagebox.showinfo("No Overlap",
                "No pixels are claimed by multiple visible instances on this frame.\n\n"
                "Either fewer than 2 instances have masks here, or there is no overlap.")
            return

        self.flash_overlap_computed = overlap_mask
        self.flash_overlap_in_progress = True
        self.flash_overlap_on = False
        flash_count = [0]

        def flash_step():
            if flash_count[0] >= 3:
                self._reset_flash_state()
                self.display_frame()
                return
            self.flash_overlap_on = not self.flash_overlap_on
            self.display_frame()
            if not self.flash_overlap_on:
                flash_count[0] += 1
            self.root.after(300, flash_step)

        flash_step()

    # ============================================================
    # Concept List Import / Export
    # ============================================================

    def export_concept_list(self):
        """Export concept names and text prompts to a JSON file for reuse across projects."""
        if not self.project or not self.project.concepts:
            messagebox.showwarning("Warning", "No concepts to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Concept List",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        concept_list = [
            {"name": c.name, "text_prompt": c.text_prompt}
            for c in self.project.concepts
        ]
        with open(path, "w") as f:
            json.dump({"concepts": concept_list}, f, indent=2)
        self.status_var.set(f"Exported {len(concept_list)} concept(s) to {Path(path).name}.")

    def import_concept_list(self):
        """Import concept names and prompts from a JSON file.

        Adds concepts that don't already exist in the project (matched by name).
        Does NOT trigger detection — the user runs detection per-concept afterward.
        """
        if not self.project:
            messagebox.showwarning("Warning", "Please load a project first.")
            return

        path = filedialog.askopenfilename(
            title="Import Concept List",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path) as f:
                data = json.load(f)
            concepts = data.get("concepts", [])
            if not concepts:
                messagebox.showwarning("Warning", "No concepts found in the file.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read concept list:\n{e}")
            return

        existing_names = {c.name for c in self.project.concepts}
        added = []
        skipped = []
        for entry in concepts:
            name = entry.get("name", "").strip()
            prompt = entry.get("text_prompt", "").strip()
            if not name or not prompt:
                continue
            if name in existing_names:
                skipped.append(name)
                continue
            concept = SAM3Concept(
                name=name,
                text_prompt=prompt,
                color_rgb=generate_concept_color(len(self.project.concepts)),
            )
            self.project.add_concept(concept)
            existing_names.add(name)
            added.append(name)

        if not added:
            messagebox.showinfo("Import", f"No new concepts to add.\nSkipped (already exist): {', '.join(skipped)}")
            return

        self._safe_project_save()
        self.update_concept_tree()

        msg = f"Added {len(added)} concept(s): {', '.join(added)}."
        if skipped:
            msg += f"\nSkipped (already exist): {', '.join(skipped)}."
        msg += "\n\nSelect each new concept and click 'Add Concept' to run detection."
        messagebox.showinfo("Import Complete", msg)
        self.status_var.set(f"Imported {len(added)} concept(s) from {Path(path).name}.")

    # ============================================================
    # Frame Cache Management
    # ============================================================

    def extract_frames_to_project(self):
        """Extract video frames as JPEGs into the project directory for fast random access."""
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        frames_dir = os.path.join(self.project.project_dir, "frames")
        if os.path.isdir(frames_dir):
            n = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            if not messagebox.askyesno(
                "Re-extract Frames",
                f"Frame cache already exists at:\n{frames_dir}\n({n} frames)\n\n"
                "Re-extract and overwrite?"
            ):
                return

        self.status_var.set("Extracting frames...")
        self.root.update()

        def extract_thread():
            try:
                from sam3_utils import extract_frames_from_video
                # Explicit user-requested re-extract: overwrite existing files
                extract_frames_from_video(self.project.video_path, frames_dir,
                                          overwrite=True)
                self.project.frames_dir = frames_dir
                self._safe_project_save()
                self.compositor = DynamicFrameCompositor(self.project)
                self.root.after(0, lambda: self.status_var.set(
                    f"Frame cache ready: {frames_dir}"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Done", f"Frames extracted to:\n{frames_dir}\n\n"
                    "Frame loading will now use this cache for faster navigation."))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Frame extraction failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Frame extraction failed."))

        threading.Thread(target=extract_thread, daemon=True).start()

    def reencode_video_dialog(self):
        """Re-encode the project video to MJPEG for fast random-access playback."""
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        mjpeg_path = os.path.join(self.project.project_dir, "video_mjpeg.avi")
        msg = (
            f"Re-encode video to MJPEG (all-keyframe) format?\n\n"
            f"Source: {self.project.video_path}\n"
            f"Output: {mjpeg_path}\n\n"
            "This takes a few minutes but makes frame seeking instant. "
            "If a frame cache (extracted JPEGs) already exists, it is faster than MJPEG."
        )
        if not messagebox.askyesno("Re-encode Video", msg):
            return

        self.status_var.set("Re-encoding video to MJPEG...")
        self.root.update()

        def encode_thread():
            try:
                from sam3_utils import force_reencode_video_mjpeg
                result = force_reencode_video_mjpeg(
                    self.project.video_path, self.project.project_dir)
                self.project.mjpeg_video_path = result
                self._safe_project_save()
                self.compositor = DynamicFrameCompositor(self.project)
                self.root.after(0, lambda: self.status_var.set(
                    f"MJPEG video ready: {result}"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Done", f"MJPEG video saved to:\n{result}"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Re-encoding failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Re-encoding failed."))

        threading.Thread(target=encode_thread, daemon=True).start()

    def delete_frame_cache_dialog(self):
        """Delete the persistent frame cache directory after user confirmation."""
        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        frames_dir = self.project.frames_dir
        if not frames_dir or not os.path.isdir(frames_dir):
            messagebox.showinfo("No Frame Cache",
                                "No persistent frame cache is registered for this project.")
            return

        try:
            n_files = sum(1 for f in os.listdir(frames_dir) if f.endswith('.jpg'))
            size_mb = sum(
                os.path.getsize(os.path.join(frames_dir, f))
                for f in os.listdir(frames_dir) if f.endswith('.jpg')
            ) / (1024 ** 2)
        except OSError:
            n_files, size_mb = 0, 0.0

        confirm = messagebox.askyesno(
            "Delete Frame Cache",
            f"Delete the frame cache directory?\n\n"
            f"Path:  {frames_dir}\n"
            f"Files: {n_files} JPEGs  ({size_mb:.1f} MB)\n\n"
            "The project will fall back to reading directly from the video file."
        )
        if not confirm:
            return

        import shutil
        shutil.rmtree(frames_dir)
        self.project.frames_dir = None
        self._safe_project_save()
        self.compositor = DynamicFrameCompositor(self.project)
        self.status_var.set(f"Frame cache deleted: {frames_dir}")

    # ============================================================
    # Export
    # ============================================================

    def export_image(self):
        """Export composited image for image-mode projects."""

        if not self.compositor:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("TIFF Image", "*.tiff"),
                ("JPEG Image", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if not output_path:
            return

        try:
            alpha = self.mask_alpha_var.get() if self.show_masks_var.get() else 0.0
            frame_rgb = self.compositor.get_composited_frame(0, alpha_multiplier=alpha)
            Image.fromarray(frame_rgb).save(output_path)
            self.status_var.set(f"Image exported to {Path(output_path).name}")
            messagebox.showinfo("Success", f"Image exported to:\n{output_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to export image:\n{e}")

    def export_video(self):
        """Export composited video (or image if project is in image mode)."""

        if not self.compositor:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        if self.is_image_mode:
            self.export_image()
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")]
        )

        if not output_path:
            return

        # Show progress dialog
        progress_dialog = ProgressDialog(
            self.root,
            "Exporting Video",
            "Compositing frames and encoding video..."
        )

        def progress_callback(frame_idx, num_frames):
            """Update progress from background thread"""
            self.root.after(0, progress_dialog.update_progress, frame_idx, num_frames)

        def export_thread():
            try:
                self.compositor.export_video(output_path, progress_callback=progress_callback)

                # Close dialog and show success
                self.root.after(0, progress_dialog.close)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", f"Video exported successfully to:\n{output_path}"
                ))
                self.root.after(0, lambda: self.status_var.set(f"Video exported to {Path(output_path).name}"))

            except Exception as e:
                import traceback
                print(f"\n=== ERROR: Failed to export video ===")
                traceback.print_exc()
                print(f"=====================================\n")
                self.root.after(0, progress_dialog.close)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to export video:\n{e}"
                ))
                self.root.after(0, lambda: self.status_var.set("Video export failed."))

        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()

    def export_sam2_format_dialog(self):
        """Open dialog to assign SAM2 object IDs and export SAM2-compatible masks"""

        if not self.project:
            messagebox.showwarning("Warning", "No project loaded.")
            return

        instances = []
        for concept in self.project.concepts:
            for inst in concept.instances:
                if not inst.deleted:
                    instances.append((concept, inst))

        if not instances:
            messagebox.showwarning("Warning", "No instances to export.")
            return

        ExportSAM2Dialog(self.root, self, instances)


def _load_sam2_object_list_csv(csv_path: str) -> dict:
    """Load a SAM2 object-list CSV and return {name: {"id": int, "color": [R,G,B]}}."""
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


class ExportSAM2Dialog:
    """Dialog for assigning SAM2 object IDs and writing sam2_handoff.json.

    Supports optional SAM2 object list CSV for name-based matching.  When a CSV
    is provided, matched instances are shown read-only; unmatched ones have an
    inline action selector (New ID / Merge into existing / Discard).  A second
    section handles CSV entries that had no SAM3 instance assigned, letting the
    user mark them as "covered by SAM3" and specify sub-ID unions.
    """

    def __init__(self, parent, ui: 'SAM3VideoUI', instances: List[tuple]):
        self.ui = ui
        self.instances = instances  # List of (concept, instance)
        self.csv_name_map: dict = {}   # {name: {id, color}} from CSV
        self.existing_handoff: dict = {}
        self.existing_mapping: dict = {}
        self.existing_covered: dict = {}
        self.existing_retired: dict = {}  # handoff "retired_ids": deleted instances' ids
        self.reverse_lookup: dict = {}  # (concept_name, sam3_obj_id) → sam2_id

        self._load_existing_handoff()

        self.top = tk.Toplevel(parent)
        self.top.title("Export SAM2 Format")
        self.top.geometry("820x640")
        self.top.transient(parent)
        self.top.grab_set()

        # Per-row widget state, filled by _rebuild_instance_table
        self.row_states: List[dict] = []
        # Per-uncovered-CSV-entry widgets, filled by _rebuild_covered_section
        self.covered_states: List[dict] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Handoff loading
    # ------------------------------------------------------------------

    def _load_existing_handoff(self):
        handoff_path = os.path.join(self.ui.project.project_dir, "sam2_handoff.json")
        if os.path.exists(handoff_path):
            try:
                with open(handoff_path) as f:
                    self.existing_handoff = json.load(f)
            except Exception:
                pass
        self.existing_mapping = self.existing_handoff.get("object_mapping", {})
        self.existing_covered = self.existing_handoff.get("sam2_covered_ids", {})
        self.existing_retired = self.existing_handoff.get("retired_ids", {})
        # Retired entries first so a live mapping wins; a restored instance gets its old id back.
        for sam2_id_str, entry in (list(self.existing_retired.items())
                                   + list(self.existing_mapping.items())):
            key = (entry.get("concept"), entry.get("instance_id"))
            self.reverse_lookup[key] = int(sam2_id_str)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = self.top

        tk.Label(top, text="Export SAM2 Format",
                 font=("Arial", 12, "bold")).pack(pady=(8, 2))

        # Output directory row
        out_frame = tk.Frame(top)
        out_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(out_frame, text="Output dir:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.out_var = tk.StringVar(
            value=str(Path(self.ui.project.project_dir) / "sam2_export"))
        tk.Entry(out_frame, textvariable=self.out_var, width=48).pack(side=tk.LEFT, padx=5)
        tk.Button(out_frame, text="Browse...", command=self._browse_output).pack(side=tk.LEFT)

        # CSV picker row
        csv_frame = tk.Frame(top)
        csv_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(csv_frame, text="SAM2 object list:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        self.csv_path_var = tk.StringVar()
        tk.Entry(csv_frame, textvariable=self.csv_path_var, width=40,
                 state="readonly").pack(side=tk.LEFT, padx=5)
        tk.Button(csv_frame, text="Load CSV...", command=self._pick_csv).pack(side=tk.LEFT, padx=2)
        tk.Button(csv_frame, text="Clear", command=self._clear_csv).pack(side=tk.LEFT, padx=2)
        tk.Label(csv_frame, text="(optional - enables name matching)",
                 fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=6)

        # Instance table (LabelFrame so we can retitle it)
        self.table_lf = tk.LabelFrame(top, text="Instance Assignments")
        self.table_lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._rebuild_instance_table()

        # Uncovered CSV section — packed/unpacked dynamically
        self.covered_lf = tk.LabelFrame(
            top, text="Uncovered SAM2 Objects (from CSV)")
        # Not packed yet; _rebuild_covered_section handles packing

        # Buttons
        self.btn_frame = tk.Frame(top)
        self.btn_frame.pack(pady=8)
        tk.Button(self.btn_frame, text="Export", command=self._do_export,
                  width=14).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Cancel", command=self.top.destroy,
                  width=14).pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # CSV management
    # ------------------------------------------------------------------

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="Load SAM2 object list CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.csv_name_map = _load_sam2_object_list_csv(path)
        except Exception as e:
            messagebox.showerror("CSV Error", f"Could not load CSV:\n{e}")
            return
        self.csv_path_var.set(path)
        print(f"Loaded {len(self.csv_name_map)} entries from {path}")
        self._rebuild_instance_table()
        self._rebuild_covered_section()

    def _clear_csv(self):
        self.csv_name_map = {}
        self.csv_path_var.set("")
        self._rebuild_instance_table()
        self._rebuild_covered_section()

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.out_var.set(d)

    # ------------------------------------------------------------------
    # Instance table
    # ------------------------------------------------------------------

    def _next_free_id(self, reserved: set) -> int:
        i = max(reserved, default=0) + 1
        while i in reserved:
            i += 1
        return i

    def _rebuild_instance_table(self):
        for w in self.table_lf.winfo_children():
            w.destroy()
        self.row_states = []

        has_csv = bool(self.csv_name_map)

        # --- Header ---
        hdr = tk.Frame(self.table_lf)
        hdr.pack(fill=tk.X)
        _H = lambda t, w: tk.Label(hdr, text=t, width=w, anchor=tk.W,
                                   font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=3)
        _H("Concept", 13)
        _H("Instance", 18)
        _H("Frames", 6)
        if has_csv:
            _H("Status", 14)
            _H("Action / Assignment", 30)
        else:
            _H("SAM2 ID", 7)
            _H("Include", 6)
            tk.Label(hdr, text="(same ID → union)",
                     font=("Arial", 8), fg="gray").pack(side=tk.LEFT, padx=2)

        # --- Scrollable body ---
        body_frame = tk.Frame(self.table_lf)
        body_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(body_frame, height=230)
        sb = tk.Scrollbar(body_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Compute ID assignments (retired ids stay reserved, never proposed for new rows)
        reserved: set = {int(k) for k in self.existing_mapping} | {int(k) for k in self.existing_retired}
        if self.csv_name_map:
            reserved.update(e["id"] for e in self.csv_name_map.values())
        next_id = self._next_free_id(reserved)

        for concept, inst in self.instances:
            key = (concept.name, inst.sam3_obj_id)
            state: dict = {"concept": concept, "inst": inst}

            if key in self.reverse_lookup:
                state["kind"] = "existing"
                state["sam2_id"] = self.reverse_lookup[key]
            elif has_csv and inst.user_name in self.csv_name_map:
                state["kind"] = "csv_match"
                state["sam2_id"] = self.csv_name_map[inst.user_name]["id"]
            elif has_csv:
                state["kind"] = "unmatched"
                state["auto_id"] = next_id
                reserved.add(next_id)
                next_id = self._next_free_id(reserved)
            else:
                state["kind"] = "auto"
                state["id_var"] = tk.IntVar(value=next_id)
                state["inc_var"] = tk.BooleanVar(value=True)
                reserved.add(next_id)
                next_id = self._next_free_id(reserved)

            row = tk.Frame(inner, relief=tk.FLAT)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=concept.name, width=13, anchor=tk.W).pack(
                side=tk.LEFT, padx=3)
            tk.Label(row, text=inst.user_name, width=18, anchor=tk.W).pack(
                side=tk.LEFT, padx=3)
            tk.Label(row, text=str(inst.num_frames_with_mask), width=6,
                     anchor=tk.W).pack(side=tk.LEFT, padx=3)

            kind = state["kind"]
            if kind == "existing":
                tk.Label(row, text="existing", width=14, fg="#2255cc",
                         anchor=tk.W).pack(side=tk.LEFT, padx=3)
                tk.Label(row, text=f"ID {state['sam2_id']} (preserved)",
                         fg="#2255cc", anchor=tk.W).pack(side=tk.LEFT, padx=3)
            elif kind == "csv_match":
                tk.Label(row, text="CSV match", width=14, fg="#228822",
                         anchor=tk.W).pack(side=tk.LEFT, padx=3)
                tk.Label(row, text=f"ID {state['sam2_id']} (from CSV)",
                         fg="#228822", anchor=tk.W).pack(side=tk.LEFT, padx=3)
            elif kind == "unmatched":
                tk.Label(row, text="unmatched", width=14, fg="#cc6600",
                         anchor=tk.W).pack(side=tk.LEFT, padx=3)
                # Action selector
                act_var = tk.StringVar(value="new")
                state["action_var"] = act_var
                merge_id_var = tk.IntVar(value=1)
                state["merge_id_var"] = merge_id_var

                action_frame = tk.Frame(row)
                action_frame.pack(side=tk.LEFT, padx=3)

                rb_new = tk.Radiobutton(
                    action_frame, text=f"New ID {state['auto_id']}",
                    variable=act_var, value="new")
                rb_new.pack(side=tk.LEFT)

                rb_merge = tk.Radiobutton(
                    action_frame, text="Union with ID:",
                    variable=act_var, value="merge")
                rb_merge.pack(side=tk.LEFT, padx=(8, 0))

                merge_spin = tk.Spinbox(
                    action_frame, from_=1, to=9999,
                    textvariable=merge_id_var, width=5)
                merge_spin.pack(side=tk.LEFT)

                rb_discard = tk.Radiobutton(
                    action_frame, text="Discard",
                    variable=act_var, value="discard")
                rb_discard.pack(side=tk.LEFT, padx=8)
            else:  # auto (no CSV)
                tk.Spinbox(row, from_=1, to=9999,
                           textvariable=state["id_var"], width=6).pack(
                    side=tk.LEFT, padx=3)
                tk.Checkbutton(row, variable=state["inc_var"]).pack(
                    side=tk.LEFT, padx=3)

            self.row_states.append(state)

    # ------------------------------------------------------------------
    # Uncovered CSV section
    # ------------------------------------------------------------------

    def _rebuild_covered_section(self):
        for w in self.covered_lf.winfo_children():
            w.destroy()
        self.covered_states = []

        if not self.csv_name_map:
            self.covered_lf.pack_forget()
            return

        # Determine which CSV IDs are already consumed (by name-match or existing covered)
        matched_ids: set = set()
        for concept, inst in self.instances:
            key = (concept.name, inst.sam3_obj_id)
            if key in self.reverse_lookup:
                matched_ids.add(self.reverse_lookup[key])
            elif inst.user_name in self.csv_name_map:
                matched_ids.add(self.csv_name_map[inst.user_name]["id"])

        uncovered: list = []
        for name, entry in self.csv_name_map.items():
            cid = entry["id"]
            if cid not in matched_ids and str(cid) not in self.existing_covered:
                uncovered.append((cid, name, entry["color"]))
        uncovered.sort(key=lambda x: x[0])

        if not uncovered:
            self.covered_lf.pack_forget()
            return

        # Collect all auto-assigned SAM3-derived IDs for the hint label
        csv_all_ids = {e["id"] for e in self.csv_name_map.values()}

        # Pack the LabelFrame just before the buttons frame
        self.covered_lf.pack(fill=tk.X, padx=10, pady=2,
                             before=self.btn_frame)

        tk.Label(self.covered_lf,
                 text="These SAM2 objects from your CSV were not matched to any SAM3 instance.\n"
                      "You can mark them as 'covered by SAM3' (read-only in sam2_ui, "
                      "excluded from re-segmentation).",
                 wraplength=760, justify=tk.LEFT,
                 font=("Arial", 8), fg="#555555").pack(anchor=tk.W, padx=6, pady=2)

        for cid, cname, ccolor in uncovered:
            state: dict = {"csv_id": cid, "csv_name": cname, "csv_color": ccolor}

            row = tk.Frame(self.covered_lf)
            row.pack(fill=tk.X, padx=6, pady=2)

            covered_var = tk.BooleanVar(value=False)
            state["covered_var"] = covered_var
            tk.Checkbutton(row, text="Mark as covered",
                           variable=covered_var).pack(side=tk.LEFT)

            tk.Label(row, text=f"  '{cname}' (id={cid})",
                     fg="#333333").pack(side=tk.LEFT)

            tk.Label(row, text="  Sub-IDs (comma-sep, optional):",
                     fg="#555555").pack(side=tk.LEFT, padx=(12, 2))
            sub_var = tk.StringVar()
            state["sub_ids_var"] = sub_var
            tk.Entry(row, textvariable=sub_var, width=20).pack(side=tk.LEFT)

            self.covered_states.append(state)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _do_export(self):
        import time as _time_mod

        project = self.ui.project
        handoff_path = os.path.join(project.project_dir, "sam2_handoff.json")
        mask_format = getattr(project, "mask_format", "png")

        new_mapping: dict = dict(self.existing_mapping)
        new_colors: dict = dict(self.existing_handoff.get("object_colors", {}))
        new_covered: dict = dict(self.existing_covered)

        # --- Compute reserved IDs and next_free helper ---
        # Retired ids (deleted SAM3 instances) stay reserved so they are never reused.
        reserved: set = {int(k) for k in new_mapping} | {int(k) for k in self.existing_retired}
        if self.csv_name_map:
            reserved.update(e["id"] for e in self.csv_name_map.values())
        next_id = self._next_free_id(reserved)

        # --- Collect all instance assignments into pending dict first ---
        # pending: {sam2_id: {"name": str, "color": list, "subs": [sub_entry, ...]}}
        # Collecting before writing prevents a later row from silently overwriting
        # an earlier one that mapped to the same SAM2 ID.
        pending: dict = {}
        errors: list = []

        for state in self.row_states:
            concept = state["concept"]
            inst = state["inst"]
            kind = state["kind"]

            mask_dir_rel = os.path.join(
                "concepts", concept.name, "instances",
                str(inst.sam3_obj_id), "masks")
            abs_mask_dir = os.path.join(project.project_dir, mask_dir_rel)
            if not os.path.isdir(abs_mask_dir):
                continue

            if kind == "existing":
                sam2_id = state["sam2_id"]
                color = list(inst.get_effective_color(concept.color_rgb or (200, 200, 200)))
            elif kind == "csv_match":
                sam2_id = state["sam2_id"]
                color = list(self.csv_name_map[inst.user_name]["color"])
            elif kind == "unmatched":
                action = state["action_var"].get()
                if action == "discard":
                    continue
                elif action == "merge":
                    sam2_id = state["merge_id_var"].get()
                    # Target must be either already pending (assigned in this pass)
                    # or present in the existing handoff.
                    if sam2_id not in pending and str(sam2_id) not in new_mapping:
                        errors.append(
                            f"Instance '{inst.user_name}': union target ID {sam2_id} "
                            f"does not exist in the current handoff or this export batch. "
                            f"Choose an existing ID or assign this instance a new ID.")
                        continue
                else:  # new
                    sam2_id = state["auto_id"]
                    reserved.add(sam2_id)
                    next_id = self._next_free_id(reserved)
                color = list(inst.get_effective_color(concept.color_rgb or (200, 200, 200)))
            else:  # auto (no CSV)
                if not state["inc_var"].get():
                    continue
                sam2_id = state["id_var"].get()
                if str(sam2_id) in self.existing_retired:
                    errors.append(
                        f"Instance '{inst.user_name}': ID {sam2_id} is reserved - it belonged to "
                        f"a deleted SAM3 instance. Choose a different ID.")
                    continue
                color = list(inst.get_effective_color(concept.color_rgb or (200, 200, 200)))

            sub_entry = {
                "concept": concept.name,
                "instance_id": inst.sam3_obj_id,
                "name": inst.user_name,
                "mask_dir_rel": mask_dir_rel,
                "mask_filename_pattern": "{frame:06d}." + mask_format,
            }
            if sam2_id not in pending:
                pending[sam2_id] = {"name": inst.user_name, "color": color, "subs": []}
            pending[sam2_id]["subs"].append(sub_entry)

        if errors:
            messagebox.showwarning("Assignment Errors", "\n\n".join(errors))
            return

        # --- Confirm intentional multi-instance unions ---
        unions = [(sid, info) for sid, info in pending.items() if len(info["subs"]) > 1]
        if unions:
            union_lines = "\n".join(
                f"  SAM2 ID {sid}: "
                + " + ".join(f"'{s['name']}' ({s['concept']})" for s in info["subs"])
                for sid, info in unions
            )
            if not messagebox.askyesno(
                "Confirm Multi-Instance Union",
                f"The following SAM2 IDs will have their masks pixel-unioned "
                f"from multiple SAM3 instances:\n\n{union_lines}\n\nContinue?"
            ):
                return

        # --- Write pending assignments to new_mapping ---
        for sam2_id, info in pending.items():
            subs = info["subs"]
            if len(subs) == 1:
                # Single instance — flat format (backward compatible)
                sub = subs[0]
                new_mapping[str(sam2_id)] = {
                    "concept": sub["concept"],
                    "instance_id": sub["instance_id"],
                    "name": info["name"],
                    "mask_dir_rel": sub["mask_dir_rel"],
                    "mask_filename_pattern": sub["mask_filename_pattern"],
                }
            else:
                # Multiple instances → pixel-union at export time
                new_mapping[str(sam2_id)] = {
                    "name": info["name"],
                    "mask_sub_instances": subs,
                }
            new_colors[str(sam2_id)] = info["color"]

        # Retire ids of instances deleted since the last export (and trim deleted parts
        # out of unions); otherwise sam2_ui.py / sam2_process.py keep using them.
        new_mapping, new_retired, retire_messages = retire_deleted_handoff_ids(
            project, new_mapping, self.existing_retired, new_covered)
        for message in retire_messages:
            print(message)

        if not new_mapping:
            messagebox.showwarning("Warning", "No instances selected for export.")
            return

        # --- Process covered section ---
        for state in self.covered_states:
            if not state["covered_var"].get():
                continue
            cid = state["csv_id"]
            sub_ids: list = []
            for tok in state["sub_ids_var"].get().split(","):
                tok = tok.strip()
                if tok.isdigit():
                    sub_ids.append(int(tok))
            new_covered[str(cid)] = {
                "name": state["csv_name"],
                "color": state["csv_color"],
                "sam3_sub_ids": sub_ids,
            }

        # --- Write handoff ---
        now = _time_mod.strftime("%Y-%m-%d %H:%M:%S")
        handoff = {
            "version": 1,
            "created": self.existing_handoff.get("created", now),
            "updated": now,
            "sam3_project_dir": os.path.abspath(project.project_dir),
            "original_video_path": project.video_path,
            "num_frames": project.num_frames,
            "object_mapping": new_mapping,
            "object_colors": new_colors,
            "sam2_covered_ids": new_covered,
            "retired_ids": new_retired,
            "sam2_results_subdir": "sam2_results",
        }

        try:
            with open(handoff_path, "w") as f:
                json.dump(handoff, f, indent=2)
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Export Error", f"Could not write handoff:\n{e}")
            return

        project.save()
        self.top.destroy()

        n = len(new_mapping)
        n_covered = len(new_covered)
        msg = (f"sam2_handoff.json written.\n\n"
               f"{n} SAM2 object(s) assigned.\n"
               + (f"{n_covered} object(s) marked as covered by SAM3.\n" if n_covered else "")
               + (f"{len(retire_messages)} change(s) for deleted instances (see console).\n"
                  if retire_messages else "")
               + "\n"
               + f"Open in sam2_ui.py: File → Import Masks → {project.project_dir}")
        messagebox.showinfo("Export Complete", msg)


class NewInstanceNameDialog:
    """Small dialog for naming a new instance.

    Shows a combobox pre-populated with vocabulary names for the concept when a
    vocabulary is loaded; otherwise behaves like a plain text entry.
    """

    def __init__(self, parent, concept_name: str, vocab_names: List[str]):
        self.result: Optional[str] = None

        self.top = tk.Toplevel(parent)
        self.top.title("New Instance")
        self.top.geometry("380x160")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)

        tk.Label(self.top,
                 text=f"Name for the new instance in concept '{concept_name}':",
                 font=("Arial", 10), wraplength=350, justify=tk.LEFT
                 ).pack(pady=(14, 6), padx=14, anchor=tk.W)

        self.combo = ttk.Combobox(self.top, values=vocab_names, width=38)
        self.combo.pack(padx=14, anchor=tk.W)
        self.combo.focus_set()

        if vocab_names:
            tk.Label(self.top,
                     text="(Choose from vocabulary or type a custom name)",
                     font=("Arial", 8), fg="gray").pack(padx=14, anchor=tk.W, pady=(2, 0))

        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="OK", width=10, command=self._ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10, command=self.top.destroy).pack(side=tk.LEFT, padx=5)

        self.top.bind("<Return>", lambda e: self._ok())
        self.top.bind("<Escape>", lambda e: self.top.destroy())

    def _ok(self):
        name = self.combo.get().strip()
        if name:
            self.result = name
        self.top.destroy()


class AbsorbInstanceDialog:
    """Dialog for choosing which instance the current (selected) instance should be
    absorbed into.

    The absorb workflow places a positive SAM3 guidance point at the centroid of
    the source instance's (the current/selected one) most-visible frame within its
    FIRST detected period (later periods are ignored, since their segmentation may
    have drifted), then re-propagates the target. This is appropriate when instances
    are well-segmented but a real-world object was split across two detections.
    """

    def __init__(self, parent, ui: 'SAM3VideoUI', concept: 'SAM3Concept',
                 source: 'SAM3Instance', targets: List['SAM3Instance']):
        self.ui = ui
        self.concept = concept
        self.source = source
        self.targets = targets

        self.top = tk.Toplevel(parent)
        self.top.title("Absorb Current Instance Into...")
        self.top.geometry("500x350")
        self.top.transient(parent)
        self.top.grab_set()

        tk.Label(self.top,
                 text=f"Source (will be absorbed & deleted): {source.user_name}",
                 font=("Arial", 10, "bold")).pack(pady=(12, 2), padx=10, anchor=tk.W)

        tk.Label(self.top,
                 text="SAM3 will be guided by a positive point at the centroid of\n"
                      "this instance's most-visible frame in its FIRST period\n"
                      "(later periods are ignored - they may have drifted), then\n"
                      "the chosen target below will be re-propagated. You will be\n"
                      "asked whether to delete this instance afterward. Make sure\n"
                      "the first appearance is really the same object as the target.",
                 font=("Arial", 9), fg="gray", justify=tk.LEFT).pack(padx=10, anchor=tk.W)

        tk.Label(self.top, text="Target instance to absorb into (kept):",
                 font=("Arial", 10)).pack(pady=(12, 2), padx=10, anchor=tk.W)

        list_frame = tk.Frame(self.top)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        sb = tk.Scrollbar(list_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE,
                                  yscrollcommand=sb.set, height=6)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.listbox.yview)

        has_higher_id = any(inst.sam3_obj_id > source.sam3_obj_id for inst in targets)
        for i, inst in enumerate(targets):
            coverage = (inst.num_frames_with_mask / ui.num_frames * 100) if ui.num_frames > 0 else 0
            label = f"{inst.user_name}  (ID {inst.sam3_obj_id}, {coverage:.1f}% coverage)"
            self.listbox.insert(tk.END, label)
            if inst.sam3_obj_id > source.sam3_obj_id:
                self.listbox.itemconfig(i, foreground="#aaaaaa")

        if has_higher_id:
            tk.Label(self.top,
                     text="Grayed entries have a higher ID than the source - they were\n"
                          "detected later and absorbing into them is less reliable.",
                     font=("Arial", 8), fg="#aaaaaa", justify=tk.LEFT).pack(padx=10, anchor=tk.W)

        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="Absorb", width=10, command=self.on_absorb).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10, command=self.top.destroy).pack(side=tk.LEFT, padx=5)

        self.top.bind("<Return>", lambda e: self.on_absorb())

    def on_absorb(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "Please select a target instance.")
            return
        target = self.targets[sel[0]]
        self.top.destroy()
        self.ui.absorb_instance(target, self.source)


class MergeInstancesDialog:
    """Dialog for merging multiple instances"""

    def __init__(self, parent, ui: SAM3VideoUI, concept: SAM3Concept, instances: List[SAM3Instance]):
        self.ui = ui
        self.concept = concept
        self.instances = instances
        self.selected_instances = []

        self.top = tk.Toplevel(parent)
        self.top.title("Merge Instances")
        self.top.geometry("500x400")
        self.top.transient(parent)
        self.top.grab_set()

        # Instructions
        tk.Label(self.top, text="Select instances to merge:",
                font=("Arial", 10, "bold")).pack(pady=10, padx=10, anchor=tk.W)

        # Scrollable list
        list_frame = tk.Frame(self.top)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE,
                                  yscrollcommand=scrollbar.set, height=10)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Populate list
        for inst in instances:
            coverage = (inst.num_frames_with_mask / ui.num_frames * 100) if ui.num_frames > 0 else 0
            self.listbox.insert(tk.END, f"{inst.user_name} (ID: {inst.sam3_obj_id}, {coverage:.1f}% coverage)")

        # Merged instance name
        tk.Label(self.top, text="Name for merged instance:",
                font=("Arial", 10)).pack(pady=(10, 0), padx=10, anchor=tk.W)

        self.name_entry = tk.Entry(self.top, width=50)
        self.name_entry.pack(pady=5, padx=10, fill=tk.X)
        self.name_entry.insert(0, f"{concept.name}_merged")

        # Buttons
        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Merge", width=10,
                 command=self.on_merge).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10,
                 command=self.top.destroy).pack(side=tk.LEFT, padx=5)

    def on_merge(self):
        """Merge selected instances"""

        selection_indices = self.listbox.curselection()

        if len(selection_indices) < 2:
            messagebox.showwarning("Warning", "Please select at least 2 instances to merge.")
            return

        new_name = self.name_entry.get().strip()
        if not new_name:
            messagebox.showwarning("Warning", "Merged instance name cannot be empty.")
            return

        # Get selected instance IDs
        selected_ids = [self.instances[i].sam3_obj_id for i in selection_indices]

        self.top.destroy()

        # Perform merge in background
        self.ui.merge_instances(self.concept, selected_ids, new_name)


class ProgressDialog:
    """Progress dialog for long-running operations"""

    def __init__(self, parent, title: str, message: str):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("400x150")
        self.top.transient(parent)
        self.top.grab_set()

        # Message
        self.message_var = tk.StringVar(value=message)
        tk.Label(self.top, textvariable=self.message_var,
                font=("Arial", 10)).pack(pady=(20, 10), padx=20)

        # Progress bar
        self.progress = ttk.Progressbar(self.top, mode='determinate', length=350)
        self.progress.pack(pady=10, padx=20)

        # Status label
        self.status_var = tk.StringVar(value="0 / 0")
        tk.Label(self.top, textvariable=self.status_var,
                font=("Arial", 9), fg="gray").pack(pady=5)

        # Cancel button
        self.cancelled = False
        tk.Button(self.top, text="Cancel", width=10,
                 command=self.cancel).pack(pady=10)

        # Center window
        self.top.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.top.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.top.winfo_height()) // 2
        self.top.geometry(f"+{x}+{y}")

    def update_progress(self, current: int, total: int, message: str = None):
        """Update progress (call from main thread via after())"""
        if total > 0:
            self.progress['value'] = (current / total) * 100
            self.status_var.set(f"{current} / {total}")
        if message:
            self.message_var.set(message)
        self.top.update()

    def cancel(self):
        """Mark as cancelled"""
        self.cancelled = True

    def close(self):
        """Close dialog"""
        try:
            self.top.destroy()
        except:
            pass


class AddConceptDialog:
    """Dialog for adding a new concept"""

    def __init__(self, parent, ui: SAM3VideoUI):
        self.ui = ui
        self.top = tk.Toplevel(parent)
        self.top.title("Add Concept")
        self.top.geometry("420x380")
        self.top.minsize(400, 380)
        self.top.transient(parent)
        self.top.grab_set()

        # Concept name
        tk.Label(self.top, text="Concept Name:").pack(pady=(10, 0), padx=10, anchor=tk.W)
        self.name_entry = tk.Entry(self.top, width=40)
        self.name_entry.pack(pady=5, padx=10, fill=tk.X)

        # Text prompt
        tk.Label(self.top, text="Text Prompt:").pack(pady=(10, 0), padx=10, anchor=tk.W)
        self.prompt_entry = tk.Entry(self.top, width=40)
        self.prompt_entry.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(self.top, text="Examples: 'person', 'car', 'toy', 'furniture'",
                font=("Arial", 8), fg="gray").pack(padx=10, anchor=tk.W)

        # Detection frame
        tk.Label(self.top, text="Initial Preview Frame:").pack(pady=(10, 0), padx=10, anchor=tk.W)
        self.frame_entry = tk.Entry(self.top, width=40)
        self.frame_entry.insert(0, str(ui.current_frame_idx))
        self.frame_entry.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(self.top, text="(Detection will scan the entire video)",
                font=("Arial", 8), fg="gray").pack(padx=10, anchor=tk.W)

        # Max instances
        tk.Label(self.top, text="Max Instances:").pack(pady=(10, 0), padx=10, anchor=tk.W)
        self.max_instances_entry = tk.Entry(self.top, width=40)
        self.max_instances_entry.insert(0, "-1")
        self.max_instances_entry.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(self.top, text="(-1 = no limit; caps how many instances SAM3 may create)",
                font=("Arial", 8), fg="gray").pack(padx=10, anchor=tk.W)

        # Buttons
        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Process Now", width=12,
                 command=self.on_process).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save for Later", width=12,
                 command=self.on_save_pending).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10,
                 command=self.top.destroy).pack(side=tk.LEFT, padx=5)

    def _validate_inputs(self):
        """Validate and return (name, prompt, frame_idx, max_instances) or None on error."""
        name = self.name_entry.get().strip()
        prompt = self.prompt_entry.get().strip()

        if not name:
            messagebox.showwarning("Warning", "Concept name is required.")
            return None

        if not prompt:
            messagebox.showwarning("Warning", "Text prompt is required.")
            return None

        try:
            frame_idx = int(self.frame_entry.get())
            if frame_idx < 0 or frame_idx >= self.ui.num_frames:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Warning", "Invalid frame index.")
            return None

        try:
            max_instances = int(self.max_instances_entry.get())
        except ValueError:
            messagebox.showwarning("Warning", "Max Instances must be an integer (-1 for no limit).")
            return None

        return name, prompt, frame_idx, max_instances

    def on_process(self):
        """Process the new concept immediately."""
        inputs = self._validate_inputs()
        if inputs is None:
            return
        name, prompt, frame_idx, max_instances = inputs
        self.top.destroy()
        self.ui.add_concept(name, prompt, frame_idx, max_instances)

    def on_save_pending(self):
        """Save concept as pending (to be processed later via sam3_process.py)."""
        inputs = self._validate_inputs()
        if inputs is None:
            return
        name, prompt, frame_idx, max_instances = inputs
        self.top.destroy()
        self.ui.add_concept_pending(name, prompt, frame_idx, max_instances)


class EditPromptDialog:
    """Dialog for editing the text prompt of an existing concept."""

    def __init__(self, parent, ui: SAM3VideoUI, concept):
        self.ui = ui
        self.concept = concept
        self.top = tk.Toplevel(parent)
        self.top.title(f"Edit Prompt - {concept.name}")
        self.top.geometry("420x180")
        self.top.transient(parent)
        self.top.grab_set()

        tk.Label(self.top, text=f"Concept: {concept.name}",
                 font=("Arial", 10, "bold")).pack(pady=(10, 2), padx=10, anchor=tk.W)

        tk.Label(self.top, text="Text Prompt:").pack(pady=(8, 0), padx=10, anchor=tk.W)
        self.prompt_entry = tk.Entry(self.top, width=50)
        self.prompt_entry.insert(0, concept.text_prompt)
        self.prompt_entry.pack(pady=5, padx=10, fill=tk.X)
        self.prompt_entry.select_range(0, tk.END)
        self.prompt_entry.focus_set()

        tk.Label(self.top,
                 text="Updates the prompt text. Existing instances and masks are kept.",
                 font=("Arial", 8), fg="gray").pack(padx=10, anchor=tk.W)

        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="Apply", width=10,
                  command=self.on_apply).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=10,
                  command=self.top.destroy).pack(side=tk.LEFT, padx=5)

        self.top.bind('<Return>', lambda e: self.on_apply())
        self.top.bind('<Escape>', lambda e: self.top.destroy())

        self.top.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.top.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.top.winfo_height()) // 2
        self.top.geometry(f"+{x}+{y}")

    def on_apply(self):
        new_prompt = self.prompt_entry.get().strip()
        if not new_prompt:
            messagebox.showwarning("Warning", "Text prompt cannot be empty.")
            return
        if not validate_text_prompt(new_prompt):
            messagebox.showerror("Error", "Invalid text prompt. Must be 1-200 characters.")
            return
        self.top.destroy()
        self.ui._change_concept_prompt(self.concept, new_prompt)


class VocabularyEditorDialog:
    """Two-pane editor for the concept → instance-name vocabulary.

    Left pane: list of concepts in the vocabulary.
    Right pane: ordered list of instance names for the selected concept.
    Changes are applied in-memory on Save & Close; the caller is responsible
    for persisting to disk via ui.save_vocabulary() if desired.
    """

    def __init__(self, parent, ui: SAM3VideoUI):
        self.ui = ui
        # Deep-copy so Cancel can discard changes
        self._vocab: Dict[str, List[str]] = copy.deepcopy(ui.vocabulary)

        self.top = tk.Toplevel(parent)
        self.top.title("Edit Vocabulary")
        self.top.geometry("700x500")
        self.top.transient(parent)
        self.top.grab_set()

        self._build_ui()
        self._refresh_concept_list()

        self.top.bind('<Escape>', lambda e: self.top.destroy())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top instruction label
        tk.Label(
            self.top,
            text="Map concept names to allowed instance names. "
                 "Users can still type freely - the list is a suggestion menu.",
            font=("Arial", 9), fg="gray", wraplength=680, justify=tk.LEFT
        ).pack(fill=tk.X, padx=10, pady=(8, 4))

        # Two-pane area
        paned = tk.PanedWindow(self.top, orient=tk.HORIZONTAL,
                               sashrelief=tk.RAISED, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- Left: concept list ---
        left = tk.Frame(paned)
        paned.add(left, minsize=160, width=200)

        tk.Label(left, text="Concepts", font=("Arial", 9, "bold")).pack(anchor=tk.W)

        self.concept_lb = tk.Listbox(left, selectmode=tk.SINGLE, exportselection=False)
        scrollL = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.concept_lb.yview)
        self.concept_lb.config(yscrollcommand=scrollL.set)
        self.concept_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollL.pack(side=tk.LEFT, fill=tk.Y)
        self.concept_lb.bind('<<ListboxSelect>>', self._on_concept_select)

        btn_left = tk.Frame(left)
        btn_left.pack(fill=tk.X, pady=(4, 0))
        tk.Button(btn_left, text="+ Add", command=self._add_concept).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_left, text="- Delete", command=self._delete_concept).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Right: instance name list ---
        right = tk.Frame(paned)
        paned.add(right, minsize=200)

        self.right_label = tk.Label(right, text="Names", font=("Arial", 9, "bold"))
        self.right_label.pack(anchor=tk.W)

        name_frame = tk.Frame(right)
        name_frame.pack(fill=tk.BOTH, expand=True)

        self.name_lb = tk.Listbox(name_frame, selectmode=tk.SINGLE, exportselection=False)
        scrollR = tk.Scrollbar(name_frame, orient=tk.VERTICAL, command=self.name_lb.yview)
        self.name_lb.config(yscrollcommand=scrollR.set)
        self.name_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollR.pack(side=tk.LEFT, fill=tk.Y)
        self.name_lb.bind('<Double-Button-1>', self._edit_name)

        btn_right = tk.Frame(right)
        btn_right.pack(fill=tk.X, pady=(4, 0))
        tk.Button(btn_right, text="+ Add", command=self._add_name).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_right, text="Edit", command=self._edit_name).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_right, text="- Delete", command=self._delete_name).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_right, text="^ Up", command=self._move_name_up).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_right, text="v Down", command=self._move_name_down).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Bottom bar
        bottom = tk.Frame(self.top)
        bottom.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(bottom, text="Load File...", command=self._load_file).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bottom, text="Save File", command=self._save_file).pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(bottom, text="Save & Close", width=14,
                  command=self._save_and_close).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Button(bottom, text="Cancel", width=10,
                  command=self.top.destroy).pack(side=tk.RIGHT, padx=(4, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_concept_name(self) -> Optional[str]:
        sel = self.concept_lb.curselection()
        if not sel:
            return None
        return self.concept_lb.get(sel[0])

    def _refresh_concept_list(self, select_name: Optional[str] = None):
        self.concept_lb.delete(0, tk.END)
        for name in sorted(self._vocab.keys()):
            self.concept_lb.insert(tk.END, name)
        # Re-select
        if select_name:
            for i in range(self.concept_lb.size()):
                if self.concept_lb.get(i) == select_name:
                    self.concept_lb.selection_set(i)
                    self.concept_lb.see(i)
                    break
        self._refresh_name_list()

    def _refresh_name_list(self, select_idx: Optional[int] = None):
        concept = self._selected_concept_name()
        self.name_lb.delete(0, tk.END)
        if concept is None:
            self.right_label.config(text="Names")
            return
        self.right_label.config(text=f"Names for \"{concept}\"")
        for name in self._vocab.get(concept, []):
            self.name_lb.insert(tk.END, name)
        if select_idx is not None:
            idx = max(0, min(select_idx, self.name_lb.size() - 1))
            if self.name_lb.size() > 0:
                self.name_lb.selection_set(idx)
                self.name_lb.see(idx)

    # ------------------------------------------------------------------
    # Concept actions
    # ------------------------------------------------------------------

    def _on_concept_select(self, _event=None):
        self._refresh_name_list()

    def _add_concept(self):
        name = simpledialog.askstring("Add Concept", "Concept name:", parent=self.top)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._vocab:
            messagebox.showwarning("Warning", f"Concept '{name}' already exists.", parent=self.top)
            return
        self._vocab[name] = []
        self._refresh_concept_list(select_name=name)

    def _delete_concept(self):
        name = self._selected_concept_name()
        if name is None:
            return
        if not messagebox.askyesno("Delete Concept",
                                   f"Delete concept '{name}' and all its names?",
                                   parent=self.top):
            return
        del self._vocab[name]
        self._refresh_concept_list()

    # ------------------------------------------------------------------
    # Name actions
    # ------------------------------------------------------------------

    def _add_name(self):
        concept = self._selected_concept_name()
        if concept is None:
            messagebox.showwarning("Warning", "Select a concept first.", parent=self.top)
            return
        name = simpledialog.askstring("Add Name", "Instance name:", parent=self.top)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        names = self._vocab.setdefault(concept, [])
        if name in names:
            messagebox.showwarning("Warning", f"'{name}' already in list.", parent=self.top)
            return
        names.append(name)
        self._refresh_name_list(select_idx=len(names) - 1)

    def _edit_name(self, _event=None):
        concept = self._selected_concept_name()
        if concept is None:
            return
        sel = self.name_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        old = self.name_lb.get(idx)
        new = simpledialog.askstring("Edit Name", "Instance name:", initialvalue=old, parent=self.top)
        if not new:
            return
        new = new.strip()
        if not new or new == old:
            return
        names = self._vocab[concept]
        names[idx] = new
        self._refresh_name_list(select_idx=idx)

    def _delete_name(self):
        concept = self._selected_concept_name()
        if concept is None:
            return
        sel = self.name_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        del self._vocab[concept][idx]
        self._refresh_name_list(select_idx=idx)

    def _move_name_up(self):
        concept = self._selected_concept_name()
        if concept is None:
            return
        sel = self.name_lb.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        names = self._vocab[concept]
        names[idx - 1], names[idx] = names[idx], names[idx - 1]
        self._refresh_name_list(select_idx=idx - 1)

    def _move_name_down(self):
        concept = self._selected_concept_name()
        if concept is None:
            return
        sel = self.name_lb.curselection()
        names = self._vocab.get(concept, [])
        if not sel or sel[0] >= len(names) - 1:
            return
        idx = sel[0]
        names[idx], names[idx + 1] = names[idx + 1], names[idx]
        self._refresh_name_list(select_idx=idx + 1)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Load Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.top
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object at top level")
            self._vocab = {str(k): list(v) for k, v in data.items()}
            self.ui.vocabulary_path = path
            self.ui._link_vocabulary_to_project()
            self._refresh_concept_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}", parent=self.top)

    def _save_file(self):
        path = self.ui.vocabulary_path
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save Vocabulary File",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                parent=self.top
            )
        if not path:
            return
        try:
            with open(path, 'w') as f:
                json.dump(self._vocab, f, indent=2)
            self.ui.vocabulary_path = path
            self.ui._link_vocabulary_to_project()
            messagebox.showinfo("Saved", f"Vocabulary saved to:\n{path}", parent=self.top)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}", parent=self.top)

    def _save_and_close(self):
        self.ui.vocabulary = self._vocab
        self.top.destroy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 text-prompt annotation UI")
    parser.add_argument(
        "project", nargs="?", default=None,
        help="Path to a SAM3 project directory to load on startup",
    )
    parser.add_argument(
        "--project", dest="project_opt", default=None,
        help="Path to a SAM3 project directory to load on startup (alternative to positional arg)",
    )
    args = parser.parse_args()
    project_dir = args.project_opt or args.project

    os.umask(0o002)  # ensure group-writable output for shared results dirs
    root = tk.Tk()
    app = SAM3VideoUI(root)
    if project_dir:
        root.update()  # ensure window is realized before loading (status bar, etc.)
        app._load_project_from_dir(project_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
