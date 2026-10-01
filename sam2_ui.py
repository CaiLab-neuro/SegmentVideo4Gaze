import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import tempfile
import shutil
import hashlib
import glob
from pathlib import Path
import traceback
from omegaconf import OmegaConf, DictConfig
import csv
import time
import threading
import re
import argparse
import colorsys
from collections import deque
import gaze_overlay

# Import utility functions
from utils import (
    load_mask as load_mask_from_disk,
    load_all_masks_for_frame,
    export_mask_to_disk as export_mask_to_disk_util,
    calculate_quality_metrics,
    save_quality_metrics,
    load_quality_metrics,
    IncrementalQualityMetricsCalculator,
    DisableCUDADuringInit,
    should_disable_cuda_for_device,
    patch_sam3_for_device,
    compress_video_with_ffmpeg,
    export_segmented_video,
    get_contrasting_text_color,
    HybridFrameSource,
)

# Import lazy loader to prevent SAM2 from loading all frames into RAM at inference time
from sam_lazy_loader import enable_lazy_loading

# Import shared segmentation module
from segment import (
    PointAnnotation,
    SegmentationConfig,
    SegmentationResult,
    VideoSegmenter,
    ProgressCallback,
)

# Add SAM2 path to Python path - dynamically detect project root
def get_project_root():
    """Dynamically detect the project root directory (where sam2_ui.py lives)"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # The directory containing sam2_ui.py is the Sam2UI root
    if os.path.basename(current_file) == 'sam2_ui.py':
        return current_dir

    # Fallback: search upwards for indicators
    indicators = ['sam2', 'setup.py']
    search_dir = current_dir
    while search_dir != os.path.dirname(search_dir):  # Not at filesystem root
        if all(os.path.exists(os.path.join(search_dir, indicator)) for indicator in indicators):
            return search_dir
        search_dir = os.path.dirname(search_dir)

    # Final fallback to current directory
    return current_dir

SAM2_PATH = get_project_root()

# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None

# Check SAM3 availability
def _check_sam3_available():
    """
    Check if SAM3 is installed and usable.

    NOTE: We intentionally avoid importing sam3.model_builder here because
    it triggers the full import chain including modules with hardcoded "cuda"
    allocations. Instead, we check for the package existence and do a
    lightweight import check.
    """
    try:
        sam3_path = os.path.join(SAM2_PATH, "sam_models", "sam3")
        if not os.path.exists(sam3_path):
            return False
        # Lightweight check - just verify the package is importable
        # Don't import model_builder as it triggers cuda allocations
        import importlib.util
        spec = importlib.util.find_spec("sam3")
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False

SAM3_AVAILABLE = _check_sam3_available()


def _probe_fa3() -> bool:
    """Return True if flash_attn_interface is installed AND the current GPU is Hopper (sm_90+)."""
    try:
        import flash_attn_interface  # noqa: F401
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            return True
    except ImportError:
        pass
    return False


class _AutoNameDict(dict):
    """Dict that auto-generates 'Object_N' names for missing keys."""
    def __missing__(self, key):
        self[key] = f"Object_{key}"
        return self[key]


class _AutoColorDict(dict):
    """Dict that auto-generates distinct HSV colors for missing keys."""
    def __missing__(self, key):
        hue = (key * 137.5) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.8 + (key % 3) * 0.1, 0.9 - (key % 2) * 0.2)
        self[key] = [int(r * 255), int(g * 255), int(b * 255)]
        return self[key]


class TkProgressCallback:
    """
    Tkinter-based progress callback for VideoSegmenter.

    Updates a Tkinter progress bar and status label during segmentation.
    """

    def __init__(self, root: tk.Tk, progress_var: tk.DoubleVar, status_label: tk.Label):
        """
        Initialize the Tkinter progress callback.

        Args:
            root: Tkinter root window for update_idletasks()
            progress_var: Tkinter DoubleVar bound to a progress bar
            status_label: Tkinter Label for status messages
        """
        self.root = root
        self.progress_var = progress_var
        self.status_label = status_label
        self._phase_progress_base = 0.0
        self._phase_progress_range = 0.0

    def on_phase_start(self, phase: str, total_steps: int) -> None:
        """Called when a new phase begins."""
        # Map phases to progress bar ranges
        phase_ranges = {
            "extracting": (0, 30),
            "initializing": (30, 35),
            "adding_points": (35, 40),
            "forward": (40, 70),
            "backward": (70, 100),
        }
        start, end = phase_ranges.get(phase, (0, 100))
        self._phase_progress_base = start
        self._phase_progress_range = end - start

        phase_labels = {
            "extracting": "Extracting frames...",
            "initializing": "Initializing SAM inference...",
            "adding_points": "Adding annotation prompts...",
            "forward": "Propagating masks forward...",
            "backward": "Propagating masks backward...",
        }
        self.status_label.config(text=phase_labels.get(phase, f"Processing {phase}..."))
        self.progress_var.set(start)
        self.root.update_idletasks()

    def on_progress(self, phase: str, current: int, total: int, message: str) -> None:
        """Report progress within a phase."""
        if total > 0:
            progress = self._phase_progress_base + (current / total) * self._phase_progress_range
            self.progress_var.set(min(progress, 100))

        self.status_label.config(text=message)

        # Only update UI every few iterations to avoid lag
        if current % 10 == 0:
            self.root.update_idletasks()

    def on_phase_complete(self, phase: str) -> None:
        """Called when a phase completes."""
        # Set progress to end of phase range
        phase_ends = {
            "extracting": 30,
            "initializing": 35,
            "adding_points": 40,
            "forward": 70,
            "backward": 100,
        }
        self.progress_var.set(phase_ends.get(phase, 100))
        self.root.update_idletasks()


class SAM2VideoUI:
    def __init__(self, root, working_dir=None):
        self.root = root
        self.working_dir = working_dir if working_dir else os.getcwd()
        self.root.title("SAM Video Segmentation Tool")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        
        # Dynamic paths - automatically detect project root
        self.sam2_base_path = SAM2_PATH  # Sam2UI root
        self.sam2_repo_path = os.path.join(SAM2_PATH, "sam_models", "sam2")  # SAM2 repository location
        self.checkpoint_dir = os.path.join(self.sam2_repo_path, "checkpoints")
        self.config_dir = os.path.join(self.sam2_repo_path, "sam2", "configs")  # Configs are in sam2/sam2/configs/
        
        # Variables
        self.video_path = None
        self.video_cap = None
        self.frames = []
        self.current_frame_idx = 0
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0

        # Display zoom (separate from fit-to-canvas scale)
        self.display_zoom_level = 1.0  # 1.0 = fit to canvas, > 1.0 = zoomed in
        self.zoom_center_img_x = 0.5  # Zoom center as fraction of image width (0.0-1.0)
        self.zoom_center_img_y = 0.5  # Zoom center as fraction of image height (0.0-1.0)

        self.click_points = []  # Store click coordinates with object IDs
        self.gaze_points_by_frame = None  # {frame_idx: [(x, y), ...]} from a loaded gaze CSV
        self.masks = {}  # Store masks for each frame {frame_idx: {obj_id: mask}}
        self.has_segmentation = False  # Track if segmentation has been completed or loaded
        self.playing = False
        self.playing_backward = False
        self.inference_state = None
        self.current_object_id = 1  # Currently selected object ID
        self.max_object_id = 1  # Track highest object ID used

        # Enhanced object management
        self.object_names = {}  # Maps obj_id to custom name
        self.object_colors = {}  # Dynamic color assignment
        self.point_removal_mode = False

        # SAM3 integration state
        self.sam3_object_ids: set = set()
        self.sam3_mask_dirs: dict = {}     # obj_id → (abs_mask_dir, filename_pattern)
        self.sam3_project_dir: str | None = None
        # SAM2 objects declared "covered by SAM3" — read-only, masks are union of sub-ids
        # {int_id: {"name": str, "color": list, "sam3_sub_ids": list[int]}}
        self.sam2_covered_ids: dict = {}

        # Multi-frame annotation mode (always enabled)
        self.multi_frame_annotation_mode = True
        self.annotated_frames = set()  # Track which frames have been annotated
        self.annotations_dirty = False  # True when annotations changed since last export/import
        self.updated_objects: set = set()  # Object IDs with changed annotations since last segmentation

        # Undo/Redo functionality for annotation points
        self.undo_stack = deque(maxlen=10)  # Store removed points for undo (max 10)
        self.redo_stack = deque(maxlen=10)  # Store points for redo (max 10)

        # Mask flash animation state
        self.flash_in_progress = False
        self.flash_obj_id = None
        self.flash_white_on = False

        # Point flash animation state
        self.flash_points_in_progress = False
        self.flash_points_on = False

        # Overlap flash animation state
        self.flash_overlap_in_progress = False
        self.flash_overlap_on = False
        self.flash_overlap_computed = None  # Cached per-pixel overlap bool mask (H,W) for current flash

        # Background task tracking
        self.active_exports = []
        self.active_segmentation = []

        # GPU selection
        self.available_gpus = self._detect_available_gpus()
        self.selected_gpu = tk.StringVar(value="auto")  # Default to auto selection
        self.gpu_device = None  # Will be set when model loads

        # Model selection
        self.available_models = self._detect_available_models()
        self.selected_model = tk.StringVar(value="auto")  # Auto-select best model
        self.current_model_info = None  # Store loaded model info

        # Model type (SAM2/SAM3)
        self.sam3_available = SAM3_AVAILABLE
        self.model_type_var = tk.StringVar(value="SAM2")  # Default to SAM2
        self.using_sam3 = False

        # Segmentation options
        self.exclusive_masks_var = tk.BooleanVar(value=False)  # Winner-takes-all per pixel

        # Lazy loading option for large videos
        self.lazy_load_var = tk.BooleanVar(value=True)  # Load frames on demand
        self.video_cap_lazy = None  # Keep video capture open for lazy loading

        # Button references for highlighting selected options
        self.zoom_buttons = {}  # Map zoom level -> button widget
        self.speed_buttons = {}  # Map speed -> button widget
        self.step_buttons = {}  # Map step size -> button widget

        # Export folder memory (persists across sessions)
        self.last_export_dir = self._load_last_export_dir()

        # Per-dialog-type last-used directories (first use falls back to working_dir)
        self.last_dir_video = None
        self.last_dir_annotations = None
        self.last_dir_object_list = None
        self.last_dir_results = None
        self.last_dir_video_export = None

        # Slider zoom functionality for precise navigation
        self.slider_zoom_level = tk.IntVar(value=1)  # 1 = full range, 10/100/1000 = zoomed
        self.slider_window_center = 0  # Center frame for zoomed slider window
        self.zoom_jump_scheduled = None  # Timer ID for delayed jump
        self.last_jump_notification = 0  # Timestamp of last notification
        self.slider_manual_change = False  # Track if slider change is from user dragging

        # Navigation step size (frames per left/right/scroll/play tick, independent of slider zoom)
        self.nav_step_size = tk.IntVar(value=1)

        # Playback speed control
        self.playback_speed = tk.DoubleVar(value=1.0)  # 1.0 = normal speed
        self.video_fps = 30  # Will be set from actual video FPS

        # Video dimensions stored in video_props dict

        # State for loading segmentation results
        self.loaded_from_results = False  # Flag if loaded from results
        self.original_video_path_for_resegment = None  # Path for re-segmentation
        self.segmented_video_displayed = False  # Track if displaying segmented vs original
        self.has_prerendered_masks = False  # Track if frames have masks baked in (performance optimization)
        self.results_output_dir = None  # Store output directory
        self.saved_opacity = 0.4  # Persisted overlay opacity (read from metadata on import)
        self.original_frame_source = None  # Frame source for original video (used when segmented video is displayed)

        # Display text for canvas overlays (top left corner)
        self.video_display_text = ""  # Video filename or results path to display on canvas

        # Session-based frame cache (cleaned up when app closes)
        self.session_cache_dir = None  # Current temp directory for extracted frames
        self.session_video_hash = None  # Hash of video being processed (to detect reuse)

        # Mask loading cache (for flash mask feature)
        self.mask_cache = {}  # Cache: {(frame_idx, obj_id): mask_array}
        self.mask_cache_size = 50  # LRU cache limit

        # Segmentation quality visualization
        self.inter_frame_changes = []  # Ratio of pixels changing category between frames
        self.background_ratios = []     # Ratio of background pixels per frame
        self.overlap_ratios = []        # Ratio of pixels with multiple category assignments
        self.change_viz_canvas = None   # Canvas for inter-frame change visualization
        self.quality_bg_rendered = False  # Track if background colorbars are cached
        self.change_viz_image = None    # Cached PhotoImage for change visualization
        self.bg_viz_image = None        # Cached PhotoImage for background visualization
        self.overlap_viz_image = None   # Cached PhotoImage for overlap visualization
        self.bg_viz_canvas = None       # Canvas for background ratio visualization
        self.overlap_viz_canvas = None  # Canvas for overlap ratio visualization

        # Quality visualization zoom tracking
        self.quality_viz_range_start = 0    # Start frame of rendered range
        self.quality_viz_range_end = 0      # End frame of rendered range
        self.quality_viz_zoom_level = 1     # Zoom level when last rendered

        # Refine segmentation controls
        self.refine_frame_start_var = tk.StringVar(value="")
        self.refine_frame_end_var = tk.StringVar(value="")
        self.refine_button = None

        # Initialize default colors and names
        self._initialize_objects()
        
        # Temporary single-frame segmentation (added for quick frame preview)
        self.temp_masks = {}  # {frame_idx: {obj_id: mask_array}} - memory-only masks
        self.temp_mask_frames = set()  # Track which frames have temp masks
        self.sam2_image_predictor = None  # Lazy-loaded SAM2ImagePredictor (SAM2 only)
        self.sam3_image_model = None  # Lazy-loaded SAM3 image model (SAM3 only)
        self.sam3_processor = None  # Lazy-loaded Sam3Processor (SAM3 only)

        # Responsive sizing state
        self._resize_job = None  # Debounce job for window resize
        self._left_panel_frac = 0.18  # Fraction of window width for left panel (updated on sash drag)
        self._initial_left_w = 280  # Computed in setup_ui(); used by setup_left_panel()
        self._video_panel_frac = 0.62  # Fraction of right panel height for video canvas (updated on sash drag)

        # SAM2 model
        self.sam2_model = None
        self.model_loaded = False
        self.device = None  # Track segmentation device for GPU overlay

        # UI styling
        self.setup_styles()
        self.setup_ui()
        
    def _initialize_objects(self):
        """Initialize auto-generating object name/color dicts."""
        self.object_names = _AutoNameDict()
        self.object_colors = _AutoColorDict()
        # Pre-populate entry for object 1 so it exists immediately
        _ = self.object_names[1]
        _ = self.object_colors[1]
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#404040', foreground='white')
        style.map('TButton',
                 background=[('active', '#505050'), ('pressed', '#303030')])

        # Highlighted button styles for selected zoom/speed
        style.configure('Selected.TButton', background='#00AA88', foreground='white',
                       font=('TkDefaultFont', 9, 'bold'))
        style.map('Selected.TButton',
                 background=[('active', '#00CC99'), ('pressed', '#008866')],
                 foreground=[('active', 'white'), ('pressed', 'white')])
        
    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Compute screen-relative initial left panel width
        # 1920 px → 300 | 1440 px → 259 | 1280 px → 230 | 1024 px → 200
        screen_w = self.root.winfo_screenwidth()
        initial_left_w = max(200, min(300, screen_w * 18 // 100))
        self._initial_left_w = initial_left_w
        self._left_panel_frac = initial_left_w / max(1, screen_w)

        # Create paned window - use tk.PanedWindow instead of ttk.PanedWindow
        self.paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL,
                                    sashwidth=5, bg='#2b2b2b')
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Create panels
        left_panel = ttk.Frame(self.paned)
        right_panel = ttk.Frame(self.paned)

        # Store panel references for focus management
        self.left_panel = left_panel
        self.right_panel = right_panel

        # Add panels
        self.paned.add(left_panel, width=initial_left_w)
        self.paned.add(right_panel)

        # Track user sash drags so we can restore the fraction on resize
        self.paned.bind('<ButtonRelease-1>', self._on_sash_moved)

        # Setup panels
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)

        # Bind clicks on panels to remove entry focus
        left_panel.bind('<Button-1>', self._on_panel_click)
        right_panel.bind('<Button-1>', self._on_panel_click)

        # Force sash position after window renders
        self.root.after(100, lambda w=initial_left_w: self.paned.sash_place(0, w, 1))
        
    def setup_left_panel(self, parent):
        canvas = tk.Canvas(parent, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas to expand scrollable_frame to canvas width
        def configure_scroll_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Make the scrollable_frame match the canvas width
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind("<Configure>", configure_scroll_frame)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar FIRST, then canvas - this ensures scrollbar stays visible
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Mouse wheel scrolling - simplified working version
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Bind when mouse enters/leaves the parent frame
        parent.bind("<Enter>", lambda e: parent.bind_all("<MouseWheel>", _on_mousewheel))
        parent.bind("<Leave>", lambda e: parent.unbind_all("<MouseWheel>"))
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="SAM UI", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # File operations
        file_frame = ttk.LabelFrame(scrollable_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        file_frame.bind('<Button-1>', self._on_panel_click)

        ttk.Button(file_frame, text="Load Video",
                  command=self.load_video, width=15).pack(fill=tk.X, pady=2)

        ttk.Button(file_frame, text="Import Object List",
                  command=self.import_object_list, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Object List",
                  command=self.export_object_list, width=15).pack(fill=tk.X, pady=2)

        ttk.Button(file_frame, text="Import Annotations",
                  command=self.import_annotations, width=15).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Annotations",
                  command=self.export_annotations, width=15).pack(fill=tk.X, pady=2)

        self.gaze_button = ttk.Button(file_frame, text="Load Gaze CSV",
                  command=self.load_gaze_csv, width=15)
        self.gaze_button.pack(fill=tk.X, pady=2)

        # Object Annotation
        obj_frame = ttk.LabelFrame(scrollable_frame, text="Object Annotation", padding=10)
        obj_frame.pack(fill=tk.X, pady=(0, 10))
        obj_frame.bind('<Button-1>', self._on_panel_click)

        # Current object selection with more space
        current_obj_frame = ttk.Frame(obj_frame)
        current_obj_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(current_obj_frame, text="Current:").pack(side=tk.LEFT)

        self.object_var = tk.IntVar(value=1)
        self.object_spinbox = tk.Spinbox(current_obj_frame, from_=1, to=9999,
                                        textvariable=self.object_var, width=5,
                                        command=self.on_object_change,
                                        bg='#404040', fg='white', insertbackground='white')
        self.object_spinbox.pack(side=tk.LEFT, padx=(5, 5))

        # Object color indicator (a real colored swatch - the unicode block char
        # renders as an empty box in the default Linux Tk font)
        self.object_color_label = tk.Label(current_obj_frame, text="  ",
                                           relief=tk.SOLID, borderwidth=1)
        self.object_color_label.pack(side=tk.LEFT, padx=(5, 0))

        # Object name entry
        name_frame = ttk.Frame(obj_frame)
        name_frame.pack(fill=tk.X, pady=5)

        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.object_name_var = tk.StringVar(value="Object_1")
        self.object_name_entry = tk.Entry(name_frame, textvariable=self.object_name_var,
                                         bg='#404040', fg='white', insertbackground='white')
        self.object_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.object_name_entry.bind('<Return>', self.update_object_name)
        self.object_name_entry.bind('<Escape>', self.cancel_object_name_edit)

        ttk.Button(name_frame, text="Save", command=self.update_object_name, width=5).pack(side=tk.RIGHT)

        # Object control buttons
        obj_buttons_frame = ttk.Frame(obj_frame)
        obj_buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(obj_buttons_frame, text="Add New",
                  command=self.add_new_object, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(obj_buttons_frame, text="Clear Obj",
                  command=self.clear_current_object, width=10).pack(side=tk.LEFT)

        # Object list with scrollbar
        list_frame = ttk.Frame(obj_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create Treeview for object list
        self.object_tree = ttk.Treeview(list_frame, columns=("name", "points"),
                                       show="tree headings", height=8)
        self.object_tree.heading("#0", text="ID")
        self.object_tree.heading("name", text="Name")
        self.object_tree.heading("points", text="Points")

        self.object_tree.column("#0", width=40)
        self.object_tree.column("name", width=120)
        self.object_tree.column("points", width=60)

        # Scrollbar for object list
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.object_tree.yview)
        self.object_tree.configure(yscrollcommand=tree_scroll.set)

        self.object_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # SAM3-covered objects are read-only in this UI - show them greyed out
        self.object_tree.tag_configure("covered", foreground="gray")

        self.object_tree.bind('<ButtonRelease-1>', self.on_object_tree_select)

        # ADDED: Mark this widget as having its own scrollbar
        # When mouse is over this widget, it should handle its own scrolling
        self.object_tree.bind('<Enter>', lambda e: setattr(self, '_mouse_over_scrollable', True))
        self.object_tree.bind('<Leave>', lambda e: setattr(self, '_mouse_over_scrollable', False))

        # Point management controls
        point_mgmt_frame = ttk.Frame(obj_frame)
        point_mgmt_frame.pack(fill=tk.X, pady=2)

        self.remove_point_button = tk.Button(point_mgmt_frame, text="Remove Point",
                  command=self.toggle_point_removal_mode, underline=0,
                  bg='#404040', fg='white', activebackground='#505050')
        self.remove_point_button.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ttk.Button(point_mgmt_frame, text="Clear Frame",
                  command=self.clear_current_frame_points).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Flash buttons row (mask and points side by side, both share 'f' shortcut)
        flash_row = ttk.Frame(obj_frame)
        flash_row.pack(fill=tk.X, pady=(2, 1))

        self.flash_mask_button = ttk.Button(flash_row, text="Flash Mask",
                                           command=self.flash_selected_object_mask,
                                           underline=0, state='disabled')
        self.flash_mask_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.flash_points_button = ttk.Button(flash_row, text="Flash Points",
                                             command=self.flash_frame_points,
                                             underline=0, state='disabled')
        self.flash_points_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Flash Overlap button on its own row - highlights pixels claimed by 2+ objects (shortcut: 'o')
        flash_overlap_row = ttk.Frame(obj_frame)
        flash_overlap_row.pack(fill=tk.X, pady=(1, 2))

        self.flash_overlap_button = ttk.Button(flash_overlap_row, text="Flash Overlap",
                                              command=self.flash_overlap_regions,
                                              state='disabled')
        self.flash_overlap_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Model Configuration (Model selection + GPU selection merged)
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        model_frame.bind('<Button-1>', self._on_panel_click)

        # Model Type Selection (SAM2/SAM3) - only show if SAM3 is available
        if self.sam3_available:
            ttk.Label(model_frame, text="Model Type:").pack(anchor=tk.W, pady=(0, 0))
            model_type_frame = ttk.Frame(model_frame)
            model_type_frame.pack(fill=tk.X, pady=(0, 5))

            ttk.Radiobutton(
                model_type_frame,
                text="SAM2",
                variable=self.model_type_var,
                value="SAM2",
                command=self.on_model_type_change
            ).pack(side=tk.LEFT, padx=(0, 10))

            ttk.Radiobutton(
                model_type_frame,
                text="SAM3",
                variable=self.model_type_var,
                value="SAM3",
                command=self.on_model_type_change
            ).pack(side=tk.LEFT)

            # Note about SAM3 capabilities
            note_label = ttk.Label(
                model_frame,
                text="Text prompts coming later",
                foreground="gray",
                font=("TkDefaultFont", 8, "italic")
            )
            note_label.pack(anchor=tk.W, pady=(0, 5))

        # Model Variant Selection
        ttk.Label(model_frame, text="Model Variant:").pack(anchor=tk.W, pady=(5, 0))
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=self._format_model_list(),
            state="readonly",
            width=30
        )
        self.model_combo.pack(fill=tk.X, pady=(0, 5))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selection_change)

        ttk.Button(model_frame, text="Load SAM Model",
                  command=self.load_sam2_model, width=15).pack(fill=tk.X, pady=2)

        # Model status
        self.model_status_label = ttk.Label(model_frame, text="Model Not Loaded",
                                           foreground='red')
        self.model_status_label.pack(pady=5)

        # GPU Device Selection
        ttk.Label(model_frame, text="GPU Device:").pack(anchor=tk.W, pady=(5, 0))
        self.gpu_combo = ttk.Combobox(model_frame, textvariable=self.selected_gpu,
                                     values=self.available_gpus, state="readonly", width=30)
        self.gpu_combo.pack(fill=tk.X, pady=(0, 5))
        self.gpu_combo.bind('<<ComboboxSelected>>', self.on_gpu_selection_change)

        # GPU info display
        self.gpu_info_label = ttk.Label(model_frame, text="", foreground='gray', font=('Arial', 8))
        self.gpu_info_label.pack(anchor=tk.W)

        # Update GPU info display
        self._update_gpu_info_display()

        # Segmentation controls
        seg_frame = ttk.LabelFrame(scrollable_frame, text="Segmentation", padding=10)
        seg_frame.pack(fill=tk.X, pady=(0, 10))
        seg_frame.bind('<Button-1>', self._on_panel_click)

        ttk.Button(seg_frame, text="Segment Video",
                  command=self.segment_video, width=15).pack(fill=tk.X, pady=2)

        # Exclusive masks option
        ttk.Checkbutton(seg_frame, text="Exclusive masks (winner-takes-all)",
                        variable=self.exclusive_masks_var).pack(anchor=tk.W, pady=(3, 0))
        ttk.Label(seg_frame,
                  text="Each pixel assigned to at most one object (highest logit wins).",
                  foreground='gray', font=('Arial', 8, 'italic'),
                  wraplength=200).pack(anchor=tk.W)

        # Mask opacity slider
        opacity_frame = ttk.Frame(seg_frame)
        opacity_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(opacity_frame, text="Mask Opacity:").pack(anchor=tk.W)
        self.mask_opacity_var = tk.DoubleVar(value=0.4)
        opacity_slider = ttk.Scale(opacity_frame, from_=0.0, to=1.0,
                                   variable=self.mask_opacity_var,
                                   orient=tk.HORIZONTAL,
                                   command=self.on_mask_opacity_change)
        opacity_slider.pack(fill=tk.X, padx=(0, 5))
        self.opacity_label = ttk.Label(opacity_frame, text="40%", foreground='gray')
        self.opacity_label.pack(anchor=tk.W)

        ttk.Button(seg_frame, text="Import Segmentation",
                  command=self.import_masks, width=15).pack(fill=tk.X, pady=2)

        # Refine Segmentation controls
        refine_frame = ttk.LabelFrame(seg_frame, text="Refine Segmentation", padding=5)
        refine_frame.pack(fill=tk.X, pady=(10, 5))

        # Frame range inputs
        range_frame = ttk.Frame(refine_frame)
        range_frame.pack(fill=tk.X, pady=2)

        ttk.Label(range_frame, text="Start:", width=5).pack(side=tk.LEFT)
        refine_start_entry = ttk.Entry(range_frame, textvariable=self.refine_frame_start_var, width=8)
        refine_start_entry.pack(side=tk.LEFT, padx=(0, 10))
        refine_start_entry.bind('<KeyRelease>', lambda e: self._update_refine_button_state())

        ttk.Label(range_frame, text="End:", width=4).pack(side=tk.LEFT)
        refine_end_entry = ttk.Entry(range_frame, textvariable=self.refine_frame_end_var, width=8)
        refine_end_entry.pack(side=tk.LEFT)
        refine_end_entry.bind('<KeyRelease>', lambda e: self._update_refine_button_state())

        # Help text
        ttk.Label(refine_frame, text="(1-indexed, inclusive)",
                 foreground='gray', font=('Arial', 8, 'italic')).pack(anchor=tk.W)


        # Refine button (initially disabled)
        self.refine_button = ttk.Button(refine_frame, text="Refine Range",
                                        command=self.refine_segmentation, width=12)
        self.refine_button.pack(fill=tk.X, pady=(5, 0))
        self.refine_button.configure(state='disabled')

        # Status info
        status_frame = ttk.LabelFrame(scrollable_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X)
        status_frame.bind('<Button-1>', self._on_panel_click)

        self.status_label = ttk.Label(status_frame, text="Ready",
                                      wraplength=max(150, self._initial_left_w - 30))
        self.status_label.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        
    def setup_right_panel(self, parent):
        """Setup the right video display panel"""
        # Vertical paned window: video canvas on top, controls below; user can drag sash to resize
        self.right_paned = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                          sashwidth=6, sashrelief=tk.RAISED, bg='#555555')
        self.right_paned.pack(fill=tk.BOTH, expand=True)

        # Video display area
        display_frame = ttk.Frame(self.right_paned)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg='#1a1a1a', highlightthickness=0)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Overlay labels (placed over canvas, always at canvas-widget corners)
        _overlay_font = ('TkDefaultFont', 10)
        _overlay_bg = '#1a1a1a'
        self._overlay_label_tl = tk.Label(self.canvas, text="", fg='#cccccc', bg=_overlay_bg,
                                           font=_overlay_font, padx=4, pady=2)
        self._overlay_label_tr = tk.Label(self.canvas, text="", fg='white', bg=_overlay_bg,
                                           font=_overlay_font, padx=4, pady=2)
        self._overlay_label_bl = tk.Label(self.canvas, text="", fg='#aaaaaa', bg=_overlay_bg,
                                           font=_overlay_font, padx=4, pady=2)
        # Place at canvas corners (relx/rely = relative to canvas widget, x/y = pixel offset from that anchor)
        self._overlay_label_tl.place(relx=0.0, rely=0.0, x=6, y=6, anchor=tk.NW)
        self._overlay_label_tr.place(relx=1.0, rely=0.0, x=-6, y=6, anchor=tk.NE)
        self._overlay_label_bl.place(relx=0.0, rely=1.0, x=6, y=-6, anchor=tk.SW)

        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Mouse wheel: frame navigation (no Ctrl) or zoom (with Ctrl)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)      # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)        # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)        # Linux scroll down

        # Keyboard shortcuts - cross-platform support
        self.root.bind('f', lambda e: self._handle_flash_shortcut())
        self.root.bind('o', lambda e: self._handle_overlap_shortcut())

        # Undo: Ctrl+Z (Windows/Linux) or Cmd+Z (Mac)
        self.root.bind('<Control-z>', self.undo_last_point)
        self.root.bind('<Command-z>', self.undo_last_point)  # Mac support

        # Redo: Ctrl+Y (Windows/Linux) or Cmd+Shift+Z (Mac primary) or Cmd+Y (Mac alternative)
        self.root.bind('<Control-y>', self.redo_last_point)
        self.root.bind('<Command-Shift-z>', self.redo_last_point)  # Mac primary
        self.root.bind('<Command-y>', self.redo_last_point)  # Mac alternative

        # Frame navigation: Left/Right arrows
        self.root.bind('<Left>', lambda e: self._handle_prev_frame_shortcut())
        self.root.bind('<Right>', lambda e: self._handle_next_frame_shortcut())

        # Object navigation: Up/Down arrows
        self.root.bind('<Up>', lambda e: self._handle_prev_object_shortcut())
        self.root.bind('<Down>', lambda e: self._handle_next_object_shortcut())

        # Toggle point removal mode: R key
        self.root.bind('r', lambda e: self._handle_toggle_removal_shortcut())

        # Play/pause: Space bar
        self.root.bind('<space>', lambda e: self._handle_play_shortcut())

        # Category list navigation: Home/End keys
        self.root.bind('<Home>', lambda e: self._handle_home_shortcut())
        self.root.bind('<End>', lambda e: self._handle_end_shortcut())

        # Jump to prev/next annotated frame for current object: PageUp/PageDown
        self.root.bind('<Prior>', lambda e: self._handle_annotated_frame_shortcut(e, 'prev'))
        self.root.bind('<Next>', lambda e: self._handle_annotated_frame_shortcut(e, 'next'))

        # Display zoom: Ctrl++ / Ctrl+= to zoom in, Ctrl+- to zoom out
        self.root.bind('<Control-plus>', lambda e: self._handle_keyboard_zoom(1))
        self.root.bind('<Control-equal>', lambda e: self._handle_keyboard_zoom(1))   # unshifted + key
        self.root.bind('<Control-minus>', lambda e: self._handle_keyboard_zoom(-1))
        self.root.bind('<Command-plus>', lambda e: self._handle_keyboard_zoom(1))    # Mac
        self.root.bind('<Command-equal>', lambda e: self._handle_keyboard_zoom(1))   # Mac unshifted
        self.root.bind('<Command-minus>', lambda e: self._handle_keyboard_zoom(-1))  # Mac

        # Add video display pane to the vertical paned window
        self.right_paned.add(display_frame, stretch="always", minsize=150)

        # Video controls (lower pane — user can drag sash above to give more space to video)
        controls_frame = ttk.Frame(self.right_paned)
        controls_frame.bind('<Button-1>', self._on_panel_click)

        # Scrollable wrapper: when the sash squeezes the controls pane below its
        # natural height a scrollbar appears so every control stays reachable.
        _ctrl_canvas = tk.Canvas(controls_frame, highlightthickness=0)
        _ctrl_sb = ttk.Scrollbar(controls_frame, orient=tk.VERTICAL, command=_ctrl_canvas.yview)
        _ctrl_canvas.configure(yscrollcommand=_ctrl_sb.set)
        _ctrl_sb.pack(side=tk.RIGHT, fill=tk.Y)
        _ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls_inner = ttk.Frame(_ctrl_canvas)
        _ctrl_win = _ctrl_canvas.create_window((0, 0), window=controls_inner, anchor="nw")

        def _ctrl_inner_configure(event):
            _ctrl_canvas.configure(scrollregion=_ctrl_canvas.bbox("all"))
            # Show/hide scrollbar depending on whether content fits
            needed = controls_inner.winfo_reqheight()
            available = _ctrl_canvas.winfo_height()
            if needed > available:
                _ctrl_sb.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                _ctrl_sb.pack_forget()

        def _ctrl_canvas_configure(event):
            # Keep inner frame width == canvas width so controls don't wrap oddly
            _ctrl_canvas.itemconfig(_ctrl_win, width=event.width)
            _ctrl_inner_configure(event)

        controls_inner.bind("<Configure>", _ctrl_inner_configure)
        _ctrl_canvas.bind("<Configure>", _ctrl_canvas_configure)

        # Mouse-wheel scrolling on the controls pane
        def _ctrl_scroll(event):
            if _ctrl_sb.winfo_ismapped():
                _ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _ctrl_scroll_linux(event):
            if _ctrl_sb.winfo_ismapped():
                _ctrl_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        for _w in (controls_frame, _ctrl_canvas, controls_inner):
            _w.bind("<MouseWheel>", _ctrl_scroll)
            _w.bind("<Button-4>", _ctrl_scroll_linux)
            _w.bind("<Button-5>", _ctrl_scroll_linux)

        # Playback controls
        playback_frame = ttk.Frame(controls_inner)
        playback_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        self.play_backward_button = ttk.Button(playback_frame, text="Play Rev", command=self.toggle_play_backward)
        self.play_backward_button.pack(side=tk.LEFT, padx=(0, 5))

        # Prev/Next buttons with hold-to-scroll support
        self.prev_button = tk.Button(playback_frame, text="Prev", bg='#404040', fg='white')
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        self.prev_button.bind("<ButtonPress-1>", lambda e: self._start_continuous_nav("prev"))
        self.prev_button.bind("<ButtonRelease-1>", lambda e: self._stop_continuous_nav())

        self.next_button = tk.Button(playback_frame, text="Next", bg='#404040', fg='white')
        self.next_button.pack(side=tk.LEFT, padx=(0, 5))
        self.next_button.bind("<ButtonPress-1>", lambda e: self._start_continuous_nav("next"))
        self.next_button.bind("<ButtonRelease-1>", lambda e: self._stop_continuous_nav())

        ttk.Button(playback_frame, text="Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Jump to annotated frames buttons
        ttk.Button(playback_frame, text="◄ Ann", command=self.jump_to_prev_annotated_frame).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(playback_frame, text="Ann ►", command=self.jump_to_next_annotated_frame).pack(side=tk.LEFT, padx=(0, 5))

        # Single-frame segmentation button
        ttk.Button(playback_frame, text="Segment Frame",
                   command=self.segment_current_frame_only).pack(side=tk.LEFT, padx=(10, 5))

        # Frame slider
        slider_frame = ttk.Frame(controls_inner)
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        
        self.frame_var = tk.IntVar()
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100,
                                     orient=tk.HORIZONTAL, variable=self.frame_var,
                                     command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))

        # Bind mouse events to track manual slider dragging
        self.frame_slider.bind('<ButtonPress-1>', lambda e: setattr(self, 'slider_manual_change', True))
        self.frame_slider.bind('<ButtonRelease-1>', lambda e: setattr(self, 'slider_manual_change', False))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)

        # Shared row for zoom + speed
        zoom_speed_row = ttk.Frame(controls_inner)
        zoom_speed_row.pack(fill=tk.X, pady=(0, 5))

        # ---- Slider Zoom ----
        zoom_frame = ttk.Frame(zoom_speed_row)
        zoom_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(zoom_frame, text="Slider Zoom:").pack(side=tk.LEFT)

        self.zoom_buttons[1] = ttk.Button(
            zoom_frame, text="Full",
            command=lambda: self.set_slider_zoom(1), width=6
        )
        self.zoom_buttons[1].pack(side=tk.LEFT, padx=2)

        self.zoom_buttons[5] = ttk.Button(
            zoom_frame, text="5x",
            command=lambda: self.set_slider_zoom(5), width=6
        )
        self.zoom_buttons[5].pack(side=tk.LEFT, padx=2)

        self.zoom_buttons[20] = ttk.Button(
            zoom_frame, text="20x",
            command=lambda: self.set_slider_zoom(20), width=6
        )
        self.zoom_buttons[20].pack(side=tk.LEFT, padx=2)

        self.zoom_buttons[100] = ttk.Button(
            zoom_frame, text="100x",
            command=lambda: self.set_slider_zoom(100), width=6
        )
        self.zoom_buttons[100].pack(side=tk.LEFT, padx=2)

        self.zoom_buttons[500] = ttk.Button(
            zoom_frame, text="500x",
            command=lambda: self.set_slider_zoom(500), width=6
        )
        self.zoom_buttons[500].pack(side=tk.LEFT, padx=2)

        self.zoom_info_label = ttk.Label(
            zoom_frame, text="(Full range)", foreground='gray'
        )
        self.zoom_info_label.pack(side=tk.LEFT, padx=(10, 0))


        # ---- Playback Speed ----
        speed_frame = ttk.Frame(zoom_speed_row)
        speed_frame.pack(side=tk.LEFT)

        ttk.Label(speed_frame, text="Playback Speed:").pack(side=tk.LEFT)

        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_buttons[speed] = ttk.Button(
                speed_frame, text=f"{speed}x",
                command=lambda s=speed: self.set_playback_speed(s),
                width=6
            )
            self.speed_buttons[speed].pack(side=tk.LEFT, padx=2)

        self.speed_info_label = ttk.Label(
            speed_frame, text="(Normal)", foreground='gray'
        )
        self.speed_info_label.pack(side=tk.LEFT, padx=(10, 0))

        # ---- Step Size ----
        step_row = ttk.Frame(controls_inner)
        step_row.pack(fill=tk.X, pady=(0, 5))

        step_frame = ttk.Frame(step_row)
        step_frame.pack(side=tk.LEFT)

        ttk.Label(step_frame, text="Step Size:").pack(side=tk.LEFT)

        for step in (1, 5, 20):
            self.step_buttons[step] = ttk.Button(
                step_frame, text=str(step),
                command=lambda s=step: self.set_nav_step_size(s),
                width=6
            )
            self.step_buttons[step].pack(side=tk.LEFT, padx=2)

        ttk.Label(step_frame, text="(frames per arrow/scroll/play tick)", foreground='gray').pack(side=tk.LEFT, padx=(10, 0))

        # Initialize button highlighting
        self._update_zoom_button_highlight()
        self._update_speed_button_highlight()
        self._update_step_button_highlight()

        # Segmentation quality indicators
        viz_frame = ttk.LabelFrame(controls_inner, text="Segmentation Quality Indicators", padding=5)
        viz_frame.pack(fill=tk.X, pady=(10, 0))

        # Inter-frame change visualization
        change_label = ttk.Label(viz_frame, text="Inter-frame Changes:")
        change_label.pack(anchor=tk.W)

        self.change_viz_canvas = tk.Canvas(viz_frame, height=15, bg='gray80')
        self.change_viz_canvas.pack(fill=tk.X, pady=(2, 5))
        self.change_viz_canvas.bind("<Button-1>", self._on_viz_canvas_click)
        self.change_viz_canvas.bind("<Configure>", self._on_quality_viz_resize)

        # Background ratio visualization
        bg_label = ttk.Label(viz_frame, text="Background Ratio:")
        bg_label.pack(anchor=tk.W)

        self.bg_viz_canvas = tk.Canvas(viz_frame, height=15, bg='gray80')
        self.bg_viz_canvas.pack(fill=tk.X, pady=(2, 5))
        self.bg_viz_canvas.bind("<Button-1>", self._on_viz_canvas_click)
        self.bg_viz_canvas.bind("<Configure>", self._on_quality_viz_resize)

        # Overlap ratio visualization
        overlap_label = ttk.Label(viz_frame, text="Overlap Ratio:")
        overlap_label.pack(anchor=tk.W)

        self.overlap_viz_canvas = tk.Canvas(viz_frame, height=15, bg='gray80')
        self.overlap_viz_canvas.pack(fill=tk.X, pady=(2, 0))
        self.overlap_viz_canvas.bind("<Button-1>", self._on_viz_canvas_click)
        self.overlap_viz_canvas.bind("<Configure>", self._on_quality_viz_resize)

        # Bind root Configure for responsive resizing (debounced)
        self.root.bind('<Configure>', self._on_root_resize)

        # Info panel
        info_frame = ttk.Frame(controls_inner)
        info_frame.pack(fill=tk.X)
        
        self.points_label = ttk.Label(info_frame, text="No points (Left: +, Right: -)")
        self.points_label.pack(fill=tk.X)

        # Add controls pane to the vertical paned window and bind its sash.
        # minsize=40 lets the sash slide almost all the way up; the scrollable
        # inner frame keeps all controls reachable via the scroll wheel.
        self.right_paned.add(controls_frame, stretch="never", minsize=40)
        self.right_paned.bind('<ButtonRelease-1>', self._on_right_sash_moved)
        self.root.after(250, self._set_initial_right_sash)

    def _on_right_sash_moved(self, event):
        """Store vertical sash fraction when user drags the video/controls divider."""
        try:
            rph = self.right_paned.winfo_height()
            if rph > 0:
                sash_y = self.right_paned.sash_coord(0)[1]
                self._video_panel_frac = sash_y / rph
        except Exception:
            pass

    def _set_initial_right_sash(self):
        """Place the vertical sash at the stored fraction once the pane is sized."""
        try:
            rph = self.right_paned.winfo_height()
            if rph > 1:
                y = max(150, min(rph - 120, int(rph * self._video_panel_frac)))
                self.right_paned.sash_place(0, 1, y)
        except Exception:
            pass

    def _on_sash_moved(self, event):
        """Update stored left-panel fraction when the user drags the sash."""
        try:
            sash_x = self.paned.sash_coord(0)[0]
            w = self.root.winfo_width()
            if w > 0:
                self._left_panel_frac = sash_x / w
        except Exception:
            pass

    def _on_root_resize(self, event):
        """Debounced handler for root window Configure events."""
        if event.widget is not self.root:
            return
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self._apply_responsive_sizes)

    def _apply_responsive_sizes(self):
        """Apply screen-relative sizes after a window resize."""
        self._resize_job = None
        h = self.root.winfo_height()
        w = self.root.winfo_width()

        # Quality canvas height: ~1.5 % of window height, bounded 8–20 px
        canvas_h = max(8, min(20, h // 65))
        for c in (self.change_viz_canvas, self.bg_viz_canvas, self.overlap_viz_canvas):
            if c:
                c.configure(height=canvas_h)

        # Re-apply horizontal sash position using stored fraction
        new_left_w = max(180, min(320, int(w * self._left_panel_frac)))
        try:
            self.paned.sash_place(0, new_left_w, 1)
        except Exception:
            pass

        # Re-apply vertical sash position using stored fraction
        try:
            rph = self.right_paned.winfo_height()
            if rph > 1:
                new_y = max(150, min(rph - 120, int(rph * self._video_panel_frac)))
                self.right_paned.sash_place(0, 1, new_y)
        except Exception:
            pass

    def _mark_object_dirty(self, obj_id: int):
        """Record that this object's annotations have changed since last segmentation"""
        self.updated_objects.add(obj_id)
        self.annotations_dirty = True

    def update_object_list(self):
        """Update the object list display"""
        # Clear current items
        for item in self.object_tree.get_children():
            self.object_tree.delete(item)
        
        # Add objects that should be shown
        used_objects = set()

        # Pattern for generic object names (e.g., "Object 1", "Object 99")
        generic_pattern = re.compile(r'^Object_\d+$')

        # Find objects with points
        for _, _, _, obj_id, _ in self.click_points:
            used_objects.add(obj_id)

        # Find objects with non-generic names
        for obj_id, name in self.object_names.items():
            if not generic_pattern.match(name):
                used_objects.add(obj_id)

        # Always show current object
        used_objects.add(self.current_object_id)

        for obj_id in sorted(used_objects):
            # Count points for this object
            point_count = sum(1 for _, _, _, oid, _ in self.click_points if oid == obj_id)

            # Insert into tree (show [SAM3] and/or * markers)
            display_name = self.object_names[obj_id]
            row_tags = ()
            if obj_id in self.sam2_covered_ids:
                display_name += " [SAM3-covered]"
                row_tags = ("covered",)
            elif obj_id in self.sam3_object_ids:
                display_name += " [SAM3]"
            if obj_id in self.updated_objects:
                display_name += " *"
            item = self.object_tree.insert("", "end", text=str(obj_id),
                                          values=(display_name, point_count),
                                          tags=row_tags)
            
            # Highlight current object
            if obj_id == self.current_object_id:
                self.object_tree.selection_set(item)
                
    def on_object_tree_select(self, event):
        """Handle object tree selection"""
        # Remove focus from any text entry widgets
        self.canvas.focus_set()

        selection = self.object_tree.selection()
        if selection:
            item = selection[0]
            obj_id = int(self.object_tree.item(item, "text"))
            self.current_object_id = obj_id
            self.object_var.set(obj_id)
            self.object_name_var.set(self.object_names[obj_id])
            self.update_object_color_display()
            if self.frames:
                self.display_current_frame()
                
    def import_object_list(self):
        """Import object names from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Import Object List",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.last_dir_object_list or self.working_dir
        )

        if not file_path:
            return

        self.last_dir_object_list = os.path.dirname(file_path)

        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                imported_count = 0
                
                for row in reader:
                    if 'id' in row and 'name' in row:
                        try:
                            obj_id = int(row['id'])
                            if obj_id >= 1:
                                self.object_names[obj_id] = row['name'].strip()
                                imported_count += 1
                        except ValueError:
                            continue
                            
                self.update_object_list()
                self.object_name_var.set(self.object_names[self.current_object_id])

                custom_count = self._count_custom_objects()

                # Clear undo/redo history when importing (new data, no history)
                self.undo_stack.clear()
                self.redo_stack.clear()

                self.status_label.config(text=f"Imported {custom_count} custom objects ({imported_count} available)")

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import object list: {str(e)}")
            
    def export_object_list(self):
        """Export object names to CSV file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Object List",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.last_dir_object_list or self.working_dir
        )

        if not file_path:
            return

        self.last_dir_object_list = os.path.dirname(file_path)
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name', 'color_r', 'color_g', 'color_b']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for obj_id in range(1, self.max_object_id + 1):
                    if obj_id in self.object_colors:
                        color = self.object_colors[obj_id]
                        writer.writerow({
                            'id': obj_id,
                            'name': self.object_names.get(obj_id, f"Object_{obj_id}"),
                            'color_r': color[0],
                            'color_g': color[1],
                            'color_b': color[2]
                        })
                    
            self.status_label.config(text=f"Object list exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export object list: {str(e)}")
    
    def load_gaze_csv(self):
        """Load a gaze CSV + world-timestamps CSV and overlay the gaze position."""
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
                self.gaze_button.config(text="Load Gaze CSV")
                self.display_current_frame()
                return

        gaze_path = filedialog.askopenfilename(
            title="Select gaze CSV (timestamp [ns], gaze x [px], gaze y [px])",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not gaze_path:
            return
        world_path = filedialog.askopenfilename(
            title="Select world-timestamps CSV (one row per video frame, timestamp [ns])",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not world_path:
            return

        try:
            mapping = gaze_overlay.load_gaze_csv(gaze_path, world_path)
        except Exception as e:
            messagebox.showerror("Load Gaze CSV Failed", str(e))
            return

        self.gaze_points_by_frame = mapping
        self.gaze_button.config(text="Gaze CSV: reload / clear")
        messagebox.showinfo("Gaze CSV Loaded", gaze_overlay.summarize(mapping))
        self.display_current_frame()

    def export_annotations(self):
        """Export click point annotations to JSON file"""
        if not self.click_points:
            messagebox.showwarning("No Annotations", "No annotations to export. Please add some click points first.")
            return
        
        try:
            # Get export file path
            file_path = self._get_export_file_path_with_creation(
                title="Export Annotations",
                default_name="sam2_annotations.json",
                file_types=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                default_ext=".json",
                last_dir_attr='last_dir_annotations'
            )
            
            if not file_path:
                return
            
            # Prepare annotation data
            annotation_data = {
                "video_path": self.video_path,
                "total_frames": len(self.frames),
                "total_annotations": len(self.click_points),
                "annotated_frames": sorted(self.annotated_frames),
                "object_names": self.object_names,
                "object_colors": {str(k): v for k, v in self.object_colors.items()},
                "updated_objects": sorted(self.updated_objects),
                "sam3_objects": sorted(self.sam3_object_ids),
                "sam3_mask_dirs": {
                    str(obj_id): {
                        "mask_dir_rel": os.path.relpath(mask_dir, self.sam3_project_dir)
                                        if self.sam3_project_dir else mask_dir,
                        "mask_filename_pattern": pattern,
                        "name": self.object_names.get(obj_id, f"Object_{obj_id}"),
                    }
                    for obj_id, (mask_dir, pattern) in self.sam3_mask_dirs.items()
                },
                "sam2_covered_ids": {str(k): v for k, v in self.sam2_covered_ids.items()},
                "annotations": []
            }

            # Convert click points to export format
            for point in self.click_points:
                img_x, img_y, is_positive, obj_id, frame_idx = point

                annotation = {
                    "frame_index": frame_idx,
                    "x": float(img_x),
                    "y": float(img_y),
                    "is_positive": is_positive,
                    "object_id": obj_id,
                    "object_name": self.object_names.get(obj_id, f"Object_{obj_id}")
                }
                annotation_data["annotations"].append(annotation)
            
            # Sort annotations by frame index
            annotation_data["annotations"].sort(key=lambda x: (x["frame_index"], x["object_id"]))
            
            # Add metadata
            annotation_data["export_info"] = {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": "SAM Video UI v1.0",
                "coordinate_system": "original",  # ADDED: Indicate coordinate system
                "multi_frame_mode": self.multi_frame_annotation_mode
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            # Show success message (auto-dismisses after 4s)
            self.annotations_dirty = False
            self.updated_objects.clear()
            self._show_auto_dismiss("Export Complete",
                              f"Annotations exported successfully!\n\n"
                              f"File: {file_path}\n"
                              f"Total annotations: {len(self.click_points)}\n"
                              f"Annotated frames: {len(self.annotated_frames)}\n"
                              f"Objects: {len(self.object_names)}", timeout_ms=4000)
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export annotations: {str(e)}")
    
    def import_annotations(self):
        """Import click point annotations from JSON file"""
        try:
            # Get import file path
            file_path = filedialog.askopenfilename(
                title="Import Annotations",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                initialdir=self.last_dir_annotations or self.working_dir
            )

            if not file_path:
                return

            self.last_dir_annotations = os.path.dirname(file_path)

            # Load annotation data
            with open(file_path, 'r') as f:
                annotation_data = json.load(f)
            
            # Validate the file format
            if "annotations" not in annotation_data:
                messagebox.showerror("Import Error", "Invalid annotation file format. Missing 'annotations' field.")
                return
            
            # Check if video is loaded
            if not self.frames:
                messagebox.showwarning("No Video", "Please load a video first before importing annotations.")
                return

            # Clear temporary masks when importing new annotations
            self._clear_temporary_masks()

            # ADDED: Check for coordinate system compatibility
            saved_metadata = annotation_data.get("video_metadata", {})
            # Check for frame count mismatch
            saved_total_frames = annotation_data.get("total_frames", 0)
            current_total_frames = len(self.frames)

            if saved_total_frames != current_total_frames:
                messagebox.showwarning(
                    "Frame Count Mismatch",
                    f"Warning: Annotation file has {saved_total_frames} frames,\n"
                    f"but loaded video has {current_total_frames} frames.\n\n"
                    f"Some annotations may be skipped if they reference out-of-bounds frames."
                )
            
            # Ask user if they want to clear existing annotations
            replace_mode = False  # Track whether to replace or add
            if self.click_points:
                result = self._show_three_button_dialog(
                    "Existing Annotations",
                    f"You have {len(self.click_points)} existing annotations.\n\n"
                    f"How would you like to import new annotations?",
                    "Replace All",
                    "Add to Existing",
                    "Cancel"
                )

                if result is None:  # Cancel
                    return
                elif result == 0:  # Replace All - clear existing
                    self.click_points.clear()
                    self.object_names.clear()
                    self.object_colors.clear()
                    self.annotated_frames.clear()
                    replace_mode = True
            else:
                # No existing annotations, treat as replace mode
                replace_mode = True

            # Load top-level object names and colors from annotation file
            if "object_names" in annotation_data:
                object_names_raw = annotation_data["object_names"]
                for obj_id_str, name in object_names_raw.items():
                    obj_id = int(obj_id_str)
                    # In replace mode, always use imported names
                    # In add mode, only add names for new object IDs
                    if replace_mode or obj_id not in self.object_names:
                        self.object_names[obj_id] = name

            if "object_colors" in annotation_data:
                object_colors_raw = annotation_data["object_colors"]
                for obj_id_str, color in object_colors_raw.items():
                    obj_id = int(obj_id_str)
                    # In replace mode, always use imported colors
                    # In add mode, only add colors for new object IDs
                    if replace_mode or obj_id not in self.object_colors:
                        self.object_colors[obj_id] = color

            # Import annotations with frame validation
            imported_count = 0
            skipped_count = 0

            for annotation in annotation_data["annotations"]:
                try:
                    frame_idx = annotation["frame_index"]
                    x = annotation["x"]
                    y = annotation["y"]
                    is_positive = annotation["is_positive"]
                    obj_id = annotation["object_id"]
                    obj_name = annotation.get("object_name", f"Object_{obj_id}")

                    # Simple bounds check
                    if frame_idx >= len(self.frames):
                        print(f"Warning: Frame {frame_idx} out of bounds, skipping")
                        skipped_count += 1
                        continue

                    # Add the annotation point
                    self.click_points.append((x, y, is_positive, obj_id, frame_idx))

                    # Update object names and colors
                    if obj_id not in self.object_names:
                        self.object_names[obj_id] = obj_name
                        # Assign a color if not already assigned
                        if obj_id not in self.object_colors:
                            # Try to load color from annotation data if available
                            if "object_colors" in annotation_data and str(obj_id) in annotation_data["object_colors"]:
                                self.object_colors[obj_id] = annotation_data["object_colors"][str(obj_id)]
                            else:
                                self.object_colors[obj_id] = self._get_next_color()

                    # Track annotated frames
                    self.annotated_frames.add(frame_idx)

                    imported_count += 1

                except KeyError as e:
                    print(f"Warning: Skipping invalid annotation: {e}")
                    skipped_count += 1
                    continue

            # Prune phantom entries: old files pre-populated all 100 object_names/colors
            # even for unused objects.  Keep only IDs that have actual click points.
            used_ids = {pt[3] for pt in self.click_points}
            phantom_ids = set(self.object_names.keys()) - used_ids
            if phantom_ids:
                print(f"Pruning {len(phantom_ids)} unused object entries from imported file "
                      f"(e.g. old pre-populated Object_1..100 names).")
                for pid in phantom_ids:
                    self.object_names.pop(pid, None)
                    self.object_colors.pop(pid, None)
            if used_ids:
                self.max_object_id = max(used_ids)

            # Update UI
            self.update_points_display()
            self.update_object_list()
            self.display_current_frame()

            # Show success message with warnings if applicable
            custom_count = self._count_custom_objects()
            total_objects = len(self.object_names)

            message = f"Annotations imported successfully!\n\n" \
                    f"File: {file_path}\n" \
                    f"Imported annotations: {imported_count}\n"

            if skipped_count > 0:
                message += f"Skipped annotations: {skipped_count} (invalid frame indices)\n"

            message += f"\nTotal annotations: {len(self.click_points)}\n" \
                    f"Objects: {custom_count} custom ({total_objects} total)"

            # Clear undo/redo history when importing (new data, no history)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.annotations_dirty = False
            self.updated_objects.clear()

            self._show_auto_dismiss("Import Complete", message, timeout_ms=4000)

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import annotations: {str(e)}")

    def import_masks(self):
        """Load segmentation results: segmented video, annotations, and mask directory"""
        # Select output directory
        output_dir = filedialog.askdirectory(
            title="Select Result Directory to Load",
            initialdir=self.last_dir_results or self.working_dir
        )
        if not output_dir:
            return

        self.last_dir_results = os.path.dirname(output_dir)
        self._load_results_from_dir(output_dir)

    def _load_results_from_dir(self, output_dir):
        """Load segmentation results (segmented video, annotations, masks) from a project/result
        directory given directly (shared by import_masks()'s file dialog and the --project CLI arg)."""
        try:
            # Detect SAM3 project handoff
            handoff_path = os.path.join(output_dir, "sam2_handoff.json")
            if os.path.exists(handoff_path):
                self._load_from_sam3_handoff(output_dir, handoff_path)
                return

            # Load metadata first
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            if not os.path.exists(metadata_path):
                messagebox.showerror("Error", "processing_metadata.json not found in directory")
                return

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Get file paths from metadata (with backward compatibility)
            file_paths = metadata.get("file_paths", {})
            segmented_video_filename = file_paths.get("segmented_video_filename", "segmented_video.avi")

            # Store original video path for re-segmentation
            self.original_video_path_for_resegment = file_paths.get("original_video_path")

            # Locate segmented video
            segmented_video_path = os.path.join(output_dir, segmented_video_filename)
            if not os.path.exists(segmented_video_path):
                # Prompt user to locate the video
                messagebox.showwarning("Video Not Found",
                                     f"Segmented video not found at:\n{segmented_video_path}\n\n"
                                     f"Please locate the segmented video file.")
                segmented_video_path = filedialog.askopenfilename(
                    title="Select Segmented Video",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
                    initialdir=self.last_dir_results or self.working_dir
                )
                if not segmented_video_path:
                    return
                self.last_dir_results = os.path.dirname(segmented_video_path)

            # Clean up any existing lazy loading state (like load_video does)
            if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
                self.video_cap_lazy.release()
                self.video_cap_lazy = None

            # Clear frame cache to prevent stale data from previous lazy loading session
            if hasattr(self, 'frame_cache'):
                self.frame_cache = {}

            # Clear existing state
            self.frames = []
            self.masks = {}
            self.mask_cache = {}  # Clear mask cache to prevent stale data from previous session
            self.click_points = []
            self.annotated_frames = set()

            # Load segmented video for display
            # Note: self.video_path is set to segmented video for UI display purposes
            # Original video path is preserved in self.original_video_path_for_resegment (set above on line 1112)
            print(f"Loading segmented video: {segmented_video_path}")
            self.video_path = segmented_video_path
            self.segmented_video_displayed = True
            self.has_prerendered_masks = True  # Frames have masks baked in
            self.results_output_dir = output_dir

            # Store result folder path for display on canvas
            self.video_display_text = f"Results: {self._truncate_path(output_dir)}"

            self.load_video_frames()

            # Initialize original frame source for segment_current_frame_only
            self._initialize_original_frame_source()

            # Set mask export directory for flash functionality
            masks_dir = os.path.join(output_dir, "masks")
            if not os.path.exists(masks_dir):
                messagebox.showerror("Error", "masks/ directory not found")
                return
            self.mask_export_dir = masks_dir

            # Reconstruct self.masks from processing_info (all objects have masks for all frames)
            # This enables flash functionality, mask lookup, and Refine Range button
            processing_info = metadata.get('processing_info', {})
            total_frames = processing_info.get('total_frames_processed', len(self.frames))
            object_ids = processing_info.get('objects_detected', [])

            # Restore persisted overlay opacity
            self.saved_opacity = processing_info.get('overlay_opacity', 0.4)
            self.mask_opacity_var.set(self.saved_opacity)
            self.opacity_label.config(text=f"{int(self.saved_opacity * 100)}%")

            for frame_idx in range(total_frames):
                self.masks[frame_idx] = {}
                for obj_id in object_ids:
                    # Store placeholder - actual mask loaded from disk on demand
                    self.masks[frame_idx][obj_id] = None

            # Load annotations from metadata
            self._load_annotations_from_metadata(metadata)

            # Validate masks exist on first frame (just check, don't load)
            first_frame_with_annotations = min(
                (pt[4] for pt in self.click_points),
                default=None
            )
            if first_frame_with_annotations is not None:
                # Try to find at least one mask file for the first frame (PNG or NPZ bundle)
                found_mask = False
                for filename in os.listdir(masks_dir):
                    if filename.startswith(f"mask_f{first_frame_with_annotations:06d}_"):
                        found_mask = True
                        break
                if not found_mask:
                    # Check NPZ bundle
                    npz_bundle = os.path.join(masks_dir, f"masks_f{first_frame_with_annotations:06d}.npz")
                    if os.path.exists(npz_bundle):
                        found_mask = True

                if not found_mask:
                    print(f"WARNING: No mask files found for first annotated frame {first_frame_with_annotations}")

            # Update UI
            self.loaded_from_results = True
            self.update_object_list()
            self.display_current_frame()

            # Load quality metrics if available
            self._load_quality_metrics(output_dir)

            num_objects = len(self.object_names)
            num_annotations = len(self.click_points)

            # Enable Flash Mask button now that segmentation is loaded
            self._enable_flash_mask_button()

            # Enable Refine button if model is loaded
            self._update_refine_button_state()

            # Clear undo/redo history when importing (new data, no history)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.annotations_dirty = False
            self.updated_objects.clear()

            self._show_auto_dismiss("Success",
                              f"Loaded segmentation results successfully!\n\n"
                              f"Video: {os.path.basename(segmented_video_path)}\n"
                              f"Frames: {len(self.frames)}\n"
                              f"Objects: {num_objects}\n"
                              f"Annotations: {num_annotations}", timeout_ms=4000)

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import masks: {str(e)}")

    def _load_from_sam3_handoff(self, project_dir, handoff_path):
        """Load a SAM3 project via its sam2_handoff.json, reading masks from native SAM3 paths."""
        try:
            with open(handoff_path, 'r') as f:
                handoff = json.load(f)

            mapping = handoff.get("object_mapping", {})
            if not mapping:
                messagebox.showerror("Error", "sam2_handoff.json has no object_mapping entries.")
                return

            num_frames = handoff.get("num_frames", 0)
            if num_frames <= 0:
                messagebox.showerror("Error", "sam2_handoff.json has invalid num_frames.")
                return

            # Build sam3_mask_dirs: {obj_id: (abs_mask_dir, filename_pattern)}
            self.sam3_mask_dirs = {}
            for k, v in mapping.items():
                obj_id = int(k)
                abs_mask_dir = os.path.normpath(os.path.join(project_dir, v["mask_dir_rel"]))
                self.sam3_mask_dirs[obj_id] = (abs_mask_dir, v["mask_filename_pattern"])

            # Resolve original video path; prompt relink if missing
            video_path = handoff.get("original_video_path", "")
            if not video_path or not os.path.exists(video_path):
                answer = messagebox.askokcancel(
                    "Video Not Found",
                    f"Original video not found:\n{video_path}\n\nLocate it now?")
                if not answer:
                    return
                new_path = filedialog.askopenfilename(
                    title="Locate Original Video",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                               ("All files", "*.*")],
                    initialdir=project_dir)
                if not new_path:
                    return
                video_path = new_path
                # Persist the relinked path
                handoff["original_video_path"] = video_path
                handoff["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(handoff_path, 'w') as f:
                    json.dump(handoff, f, indent=2)

            # Set up flags for dynamic (non-pre-rendered) mask display
            self.segmented_video_displayed = False
            self.has_prerendered_masks = False
            self.video_path = video_path
            self.original_video_path_for_resegment = video_path
            self.sam3_project_dir = os.path.abspath(project_dir)
            self.sam3_object_ids = set(int(k) for k in mapping)

            # Populate object names/colors from handoff
            object_colors_raw = handoff.get("object_colors", {})
            self.object_names = _AutoNameDict({int(k): v["name"] for k, v in mapping.items()})
            self.object_colors = _AutoColorDict({int(k): v for k, v in object_colors_raw.items()})

            # Load covered IDs — SAM2 objects declared done by SAM3 (read-only, no masks here;
            # their masks are the union of sam3_sub_ids computed at sam2_process time).
            covered_raw = handoff.get("sam2_covered_ids", {})
            self.sam2_covered_ids = {int(k): v for k, v in covered_raw.items()}
            for cid, info in self.sam2_covered_ids.items():
                self.sam3_object_ids.add(cid)
                self.object_names[cid] = info.get("name", f"Object_{cid}")
                c = info.get("color")
                if c:
                    self.object_colors[cid] = c

            # Pre-populate self.masks placeholders for all frames/objects
            all_obj_ids = set(int(k) for k in mapping) | set(self.sam2_covered_ids)
            self.masks = {
                frame_idx: {oid: None for oid in all_obj_ids}
                for frame_idx in range(num_frames)
            }

            all_ids_in_handoff = set(int(k) for k in mapping) | set(self.sam2_covered_ids)
            self.max_object_id = max(all_ids_in_handoff) if all_ids_in_handoff else 1
            self.object_var.set(self.current_object_id)
            self.object_spinbox.config(to=self.max_object_id)

            # Set mask_export_dir to sam2_results subdir for SAM2-added objects
            sam2_results_subdir = handoff.get("sam2_results_subdir", "sam2_results")
            sam2_masks_dir = os.path.join(project_dir, sam2_results_subdir, "masks")
            os.makedirs(sam2_masks_dir, exist_ok=True)
            self.mask_export_dir = sam2_masks_dir

            self.results_output_dir = project_dir
            self.loaded_from_results = True

            # Load video frames
            self.load_video_frames()

            # Clear undo/redo and dirty state
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.annotations_dirty = False
            self.updated_objects.clear()

            self.update_object_list()
            self.display_current_frame()
            self._enable_flash_mask_button()
            self._update_refine_button_state()

            num_objects = len(self.sam3_object_ids)
            self._show_auto_dismiss(
                "SAM3 Project Loaded",
                f"Loaded {num_objects} SAM3 object(s) from:\n{project_dir}\n\n"
                f"Masks are read directly from SAM3 paths.\n"
                f"New SAM2 objects will be saved to:\n{sam2_masks_dir}",
                timeout_ms=5000)

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load SAM3 project: {str(e)}")
            traceback.print_exc()

    def _load_annotations_from_metadata(self, metadata):
        """Load annotations from processing metadata into UI state"""
        if "original_annotations" not in metadata:
            print("WARNING: No original_annotations found in metadata")
            return

        annotations = metadata["original_annotations"]

        # Load object names: {int: str}
        object_names_raw = annotations.get("object_names", {})
        self.object_names = _AutoNameDict({int(k): v for k, v in object_names_raw.items()})

        # Load object colors: {int: [R,G,B]}
        object_colors_raw = annotations.get("object_colors", {})
        self.object_colors = _AutoColorDict({int(k): v for k, v in object_colors_raw.items()})

        # Load click points: [(x, y, is_positive, obj_id, frame_idx), ...]
        annotations_list = annotations.get("annotations", [])
        for ann in annotations_list:
            self.click_points.append([
                ann["x"],
                ann["y"],
                ann["is_positive"],
                ann["object_id"],
                ann["frame_index"]
            ])

        # Update annotated frames set
        self.annotated_frames = set(ann["frame_index"] for ann in annotations_list)

        # Prune phantom entries: old files pre-populated all 100 object_names/colors.
        # Keep only IDs that have actual click points.
        used_ids = {pt[3] for pt in self.click_points}
        phantom_ids = set(self.object_names.keys()) - used_ids
        if phantom_ids:
            print(f"Pruning {len(phantom_ids)} unused object entries from metadata "
                  f"(old pre-populated Object_1..100 names).")
            for pid in phantom_ids:
                self.object_names.pop(pid, None)
                self.object_colors.pop(pid, None)

        # Update max object ID
        if used_ids:
            self.max_object_id = max(used_ids)
        elif self.object_names:
            self.max_object_id = max(self.object_names.keys())
        else:
            self.max_object_id = 1

        print(f"Loaded annotations: {len(self.click_points)} points, "
              f"{len(self.object_names)} objects, "
              f"{len(self.annotated_frames)} annotated frames")

    def _get_original_video_for_resegmentation(self):
        """Get original video path for re-segmentation when working with loaded results"""
        print("Debug: self.original_video_path_for_resegment=", self.original_video_path_for_resegment)
        # Automatically use stored path from metadata if available and valid
        if self.original_video_path_for_resegment and os.path.exists(self.original_video_path_for_resegment):
            print(f"Using original video from metadata: {self.original_video_path_for_resegment}")
            return self.original_video_path_for_resegment

        # Only prompt user if no valid path is stored
        messagebox.showinfo(
            "Select Original Video",
            "Re-segmentation requires the original video (not the segmented version).\n\n"
            "Original video path not found in metadata. Please select the original video file."
        )

        original_video_path = filedialog.askopenfilename(
            title="Select Original Video for Re-segmentation",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            initialdir=self.last_dir_video or self.working_dir
        )

        if original_video_path:
            self.last_dir_video = os.path.dirname(original_video_path)
        return original_video_path if original_video_path else None

    def _initialize_original_frame_source(self):
        """
        Initialize frame source for the original video when displaying segmented results.

        This allows segment_current_frame_only to load clean frames from the original video
        instead of from the pre-segmented video (which would cause double overlay).
        """
        if not self.original_video_path_for_resegment or not os.path.exists(self.original_video_path_for_resegment):
            return

        try:
            self.original_frame_source = HybridFrameSource(
                self.original_video_path_for_resegment,
                None  # Could pass frame_dir if we have extracted frames
            )
            print(f"  Initialized original frame source for segment_current_frame_only: {self.original_video_path_for_resegment}")
        except Exception as e:
            print(f"  Warning: Could not initialize original frame source: {e}")
            self.original_frame_source = None

    def _count_custom_objects(self):
        """Count objects with custom names (not default Object_N format)"""
        return sum(1 for obj_id, name in self.object_names.items()
                   if not name.startswith("Object_"))

    def _reset_flash_state(self):
        """Reset flash animation state"""
        self.flash_in_progress = False
        self.flash_obj_id = None
        self.flash_white_on = False
        self.flash_points_in_progress = False
        self.flash_points_on = False
        self.flash_overlap_in_progress = False
        self.flash_overlap_on = False
        self.flash_overlap_computed = None

    def _enable_flash_mask_button(self):
        """Enable Flash Mask button when segmentation is available"""
        self.has_segmentation = True
        if hasattr(self, 'flash_mask_button'):
            self.flash_mask_button.config(state='normal')
        if hasattr(self, 'flash_overlap_button'):
            self.flash_overlap_button.config(state='normal')

    def _should_ignore_keyboard_shortcut(self):
        """Check if keyboard shortcuts should be ignored (e.g., when typing in Entry widgets)"""
        focused_widget = self.root.focus_get()
        # Ignore shortcuts when an Entry widget has focus
        return isinstance(focused_widget, tk.Entry)

    def _on_panel_click(self, event):
        """Handle clicks on main panels to remove focus from entry widgets"""
        if hasattr(self, 'canvas'):
            self.canvas.focus_set()

    def _handle_flash_shortcut(self):
        """Handle 'f' key shortcut: flash mask and/or points depending on availability"""
        if self._should_ignore_keyboard_shortcut():
            return
        if hasattr(self, 'flash_mask_button') and str(self.flash_mask_button.cget('state')) == 'normal':
            self.flash_selected_object_mask()
        if hasattr(self, 'flash_points_button') and str(self.flash_points_button.cget('state')) == 'normal':
            self.flash_frame_points()

    def _handle_toggle_removal_shortcut(self):
        """Handle 'r' key shortcut for toggling removal mode (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.toggle_point_removal_mode()

    def _handle_play_shortcut(self):
        """Handle Space bar shortcut for play/pause (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.toggle_play()

    def _handle_prev_frame_shortcut(self):
        """Handle Left arrow key for previous frame (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.prev_frame()

    def _handle_next_frame_shortcut(self):
        """Handle Right arrow key for next frame (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.next_frame()

    def _handle_prev_object_shortcut(self):
        """Handle Up arrow key for previous object (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.prev_object()

    def _handle_next_object_shortcut(self):
        """Handle Down arrow key for next object (only if not typing in text field)"""
        if not self._should_ignore_keyboard_shortcut():
            self.next_object()

    def _handle_keyboard_zoom(self, delta):
        """Handle Ctrl++/Ctrl+- keyboard zoom, anchored to current cursor position over canvas."""
        if self._should_ignore_keyboard_shortcut():
            return
        if not self.frames or self.current_frame is None:
            return

        # Get pointer position in screen coordinates
        ptr_x = self.root.winfo_pointerx()
        ptr_y = self.root.winfo_pointery()

        # Convert to canvas-relative coordinates
        canvas_root_x = self.canvas.winfo_rootx()
        canvas_root_y = self.canvas.winfo_rooty()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        canvas_x = ptr_x - canvas_root_x
        canvas_y = ptr_y - canvas_root_y

        # If cursor is outside the canvas, anchor to canvas center
        if canvas_x < 0 or canvas_x >= canvas_width or canvas_y < 0 or canvas_y >= canvas_height:
            canvas_x = canvas_width // 2
            canvas_y = canvas_height // 2

        # Build a minimal event-like object for _handle_zoom
        class _FakeEvent:
            pass
        ev = _FakeEvent()
        ev.x = canvas_x
        ev.y = canvas_y

        self._handle_zoom(ev, delta)

    def _handle_home_shortcut(self):
        """Jump to first object in the visible category list"""
        if self._should_ignore_keyboard_shortcut():
            return
        children = self.object_tree.get_children()
        if children:
            obj_id = int(self.object_tree.item(children[0], "text"))
            self.object_var.set(obj_id)
            self.on_object_change()

    def _handle_end_shortcut(self):
        """Jump to last object in the visible category list that has a custom name"""
        if self._should_ignore_keyboard_shortcut():
            return
        generic_pattern = re.compile(r'^Object_\d+$')
        children = self.object_tree.get_children()
        target_item = None
        for item in children:
            name = self.object_tree.item(item, "values")[0]
            if not generic_pattern.match(name):
                target_item = item
        if target_item:
            obj_id = int(self.object_tree.item(target_item, "text"))
            self.object_var.set(obj_id)
            self.on_object_change()

    def _handle_annotated_frame_shortcut(self, event, direction):
        """Handle PageUp/PageDown keys to jump between annotated frames (only if not typing)"""
        if self._should_ignore_keyboard_shortcut():
            return
        if direction == 'prev':
            self.jump_to_prev_annotated_frame()
        else:
            self.jump_to_next_annotated_frame()

    def flash_selected_object_mask(self):
        """Flash the mask for the currently selected object with white color"""
        current_obj_id = self.current_object_id
        if current_obj_id == 0:
            messagebox.showinfo("Info", "Please select an object first")
            return

        # Verify mask_export_dir is set (required for loading masks from disk)
        if not hasattr(self, 'mask_export_dir') or self.mask_export_dir is None:
            messagebox.showinfo("Flash Not Available",
                "No mask data available for flash.\n\n"
                "Flash requires either:\n"
                "1. A completed segmentation, or\n"
                "2. Loaded pre-segmented results with mask files")
            return

        # Try to load mask if not in memory or if it's a placeholder None value
        # Check BOTH temporary and permanent masks
        mask_exists = False
        mask = None

        # Priority 1: Check temporary masks (in-memory)
        if (self.current_frame_idx in self.temp_masks and
            current_obj_id in self.temp_masks.get(self.current_frame_idx, {})):
            mask = self.temp_masks[self.current_frame_idx][current_obj_id]
            mask_exists = (mask is not None)

        # Priority 2: Check permanent masks
        if not mask_exists:
            mask_exists = (self.current_frame_idx in self.masks and
                          current_obj_id in self.masks.get(self.current_frame_idx, {}) and
                          self.masks[self.current_frame_idx].get(current_obj_id) is not None)

        if not mask_exists:
            # Try loading from disk (for permanent masks only)
            if self.current_frame_idx in self.masks:
                mask = self._load_mask(self.current_frame_idx, current_obj_id)
                if mask is not None:
                    # Temporarily add to self.masks for flash animation
                    if self.current_frame_idx not in self.masks:
                        self.masks[self.current_frame_idx] = {}
                    self.masks[self.current_frame_idx][current_obj_id] = mask
                    mask_exists = True

        if not mask_exists:
            messagebox.showinfo("Info", "No mask found for selected object on current frame")
            return

        # Initialize flash state; start False so first toggle → ON (mask visible first)
        self.flash_in_progress = True
        self.flash_obj_id = current_obj_id
        self.flash_white_on = False

        # Flash 3 times: white for 0.3s, normal for 0.3s
        flash_count = [0]  # Use list to make it mutable in nested function
        max_flashes = 3

        def flash_step():
            if flash_count[0] >= max_flashes:
                # Restore original state
                self._reset_flash_state()
                self.display_current_frame()
                return

            # Toggle white on/off
            self.flash_white_on = not self.flash_white_on
            self.display_current_frame()

            # If we just turned white off, increment flash count
            if not self.flash_white_on:
                flash_count[0] += 1

            # Schedule next toggle after 300ms
            self.root.after(300, flash_step)

        # Start flashing
        flash_step()

    def flash_frame_points(self):
        """Flash annotation points for the current object on the current frame"""
        current_obj_id = self.current_object_id
        frame_points = [p for p in self.click_points
                        if p[3] == current_obj_id and p[4] == self.current_frame_idx]
        if not frame_points:
            return  # No points on this frame, do nothing

        self.flash_points_in_progress = True
        self.flash_points_on = True

        flash_count = [0]
        max_flashes = 3

        def flash_step():
            if flash_count[0] >= max_flashes:
                self.flash_points_in_progress = False
                self.flash_points_on = False
                self.display_current_frame()
                return

            self.flash_points_on = not self.flash_points_on
            self.display_current_frame()

            if not self.flash_points_on:
                flash_count[0] += 1

            self.root.after(300, flash_step)

        flash_step()

    def _handle_overlap_shortcut(self):
        """Handle 'o' key shortcut: flash overlap regions if segmentation is available"""
        if self._should_ignore_keyboard_shortcut():
            return
        if hasattr(self, 'flash_overlap_button') and str(self.flash_overlap_button.cget('state')) == 'normal':
            self.flash_overlap_regions()

    def _compute_overlap_mask_for_frame(self, frame_idx):
        """
        Compute a per-pixel boolean mask of regions claimed by 2 or more objects on frame_idx.

        Loads masks from disk (via _load_mask) or memory, so it works regardless of whether
        the displayed video is prerendered or original.  Only looks at the permanent self.masks
        dict (temp_masks are single-object previews and don't represent multi-object conflicts).

        Returns:
            np.ndarray (H, W) bool, or None if fewer than 2 objects have masks on this frame.
        """
        if frame_idx not in self.masks:
            return None

        obj_ids = list(self.masks[frame_idx].keys())
        if len(obj_ids) < 2:
            return None

        loaded = []
        for obj_id in obj_ids:
            mask = self._load_mask(frame_idx, obj_id)
            if mask is not None:
                loaded.append(mask)

        if len(loaded) < 2:
            return None

        # Sum across all masks; pixels with count >= 2 are the overlap regions
        # Use the shape of the first mask; _load_mask returns consistent shapes
        h, w = loaded[0].shape[:2]
        count = np.zeros((h, w), dtype=np.int32)
        for mask in loaded:
            if mask.shape[:2] != (h, w):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8))
                mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
                mask = np.array(mask_pil)
            count[mask > 0] += 1

        overlap_mask = count >= 2
        if not np.any(overlap_mask):
            return None  # No actual overlap pixels
        return overlap_mask

    def flash_overlap_regions(self):
        """Flash pixels that are claimed by 2+ objects on the current frame with a red overlay."""
        if not hasattr(self, 'mask_export_dir') or self.mask_export_dir is None:
            messagebox.showinfo("Flash Overlap Not Available",
                "No mask data available.\n\n"
                "Run or load a segmentation first.")
            return

        overlap_mask = self._compute_overlap_mask_for_frame(self.current_frame_idx)
        if overlap_mask is None:
            messagebox.showinfo("No Overlap",
                "No pixels are claimed by multiple objects on this frame.\n\n"
                "Either fewer than 2 objects are segmented here, or there is no overlap.")
            return

        # Cache the computed mask for the duration of the animation
        self.flash_overlap_computed = overlap_mask

        self.flash_overlap_in_progress = True
        self.flash_overlap_on = False  # First toggle → ON so overlay appears first

        flash_count = [0]
        max_flashes = 3

        def flash_step():
            if flash_count[0] >= max_flashes:
                self.flash_overlap_in_progress = False
                self.flash_overlap_on = False
                self.flash_overlap_computed = None
                self.display_current_frame()
                return

            self.flash_overlap_on = not self.flash_overlap_on
            self.display_current_frame()

            if not self.flash_overlap_on:
                flash_count[0] += 1

            self.root.after(300, flash_step)

        flash_step()

    def undo_last_point(self, event=None):
        """Undo the most recent annotation operation (add or remove)"""
        if not self.undo_stack:
            self.status_label.config(text="No operations to undo")
            return

        # Get last operation
        operation = self.undo_stack.pop()
        action = operation['action']

        if action == 'clear_frame':
            # Undo clear_frame: restore all removed points at their original indices
            points = operation['points']
            indices = operation['indices']
            obj_id = operation['obj_id']
            frame_idx = operation['frame_idx']
            # Insert in ascending index order so each insertion lands at the correct position
            for idx, point in sorted(zip(indices, points), key=lambda t: t[0]):
                self.click_points.insert(idx, point)
            self.annotated_frames.add(frame_idx)
            self.redo_stack.append(operation)
            self._mark_object_dirty(obj_id)
            self.current_object_id = obj_id
            self.object_var.set(obj_id)
            self.object_name_var.set(self.object_names.get(obj_id, f"Object_{obj_id}"))
            self.update_object_color_display()
            self.current_frame_idx = frame_idx
            self.frame_var.set(frame_idx)
            self.update_object_list()
            self.update_points_display()
            self.display_current_frame()
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            self.status_label.config(text=f"Undid clear frame for {obj_name} at frame {frame_idx + 1} ({len(points)} points restored)")
            return

        point = operation['point']
        x, y, is_positive, obj_id, frame_idx = point

        if action == 'add':
            # Undo addition: remove from end (LIFO)
            if not self.click_points or self.click_points[-1] != point:
                # Edge case: point not at end (shouldn't happen normally)
                try:
                    self.click_points.remove(point)
                except ValueError:
                    self.status_label.config(text="Cannot undo: point not found")
                    return
            else:
                self.click_points.pop()

            status_prefix = "Undid addition of"

            # Update annotated frames if frame now empty
            remaining_points_on_frame = [p for p in self.click_points if p[4] == frame_idx]
            if not remaining_points_on_frame:
                self.annotated_frames.discard(frame_idx)

        elif action == 'remove':
            # Undo removal: restore point at original index
            index = operation['index']
            self.click_points.insert(index, point)
            self.annotated_frames.add(frame_idx)
            status_prefix = "Undid removal of"

        # Add operation to redo stack
        self.redo_stack.append(operation)
        self._mark_object_dirty(obj_id)

        # Switch to the object that the point belonged to
        self.current_object_id = obj_id
        self.object_var.set(obj_id)
        self.object_name_var.set(self.object_names.get(obj_id, f"Object_{obj_id}"))
        self.update_object_color_display()

        # Jump to the frame where the point was
        self.current_frame_idx = frame_idx
        self.frame_var.set(frame_idx)

        # Update UI
        self.update_object_list()
        self.update_points_display()
        self.display_current_frame()

        # Show status
        point_type = "positive" if is_positive else "negative"
        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
        self.status_label.config(text=f"{status_prefix} {point_type} point for {obj_name} at frame {frame_idx + 1}")

    def redo_last_point(self, event=None):
        """Redo the last undone operation (add or remove)"""
        if not self.redo_stack:
            self.status_label.config(text="No operations to redo")
            return

        # Get operation from redo stack
        operation = self.redo_stack.pop()
        action = operation['action']

        if action == 'clear_frame':
            # Redo clear_frame: remove all points again
            points = operation['points']
            obj_id = operation['obj_id']
            frame_idx = operation['frame_idx']
            for point in points:
                try:
                    self.click_points.remove(point)
                except ValueError:
                    pass
            remaining_frames = {p[4] for p in self.click_points}
            self.annotated_frames = self.annotated_frames.intersection(remaining_frames)
            self.undo_stack.append(operation)
            self._mark_object_dirty(obj_id)
            self.current_object_id = obj_id
            self.object_var.set(obj_id)
            self.object_name_var.set(self.object_names.get(obj_id, f"Object_{obj_id}"))
            self.update_object_color_display()
            self.current_frame_idx = frame_idx
            self.frame_var.set(frame_idx)
            self.update_object_list()
            self.update_points_display()
            self.display_current_frame()
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            self.status_label.config(text=f"Redid clear frame for {obj_name} at frame {frame_idx + 1} ({len(points)} points removed)")
            return

        point = operation['point']
        x, y, is_positive, obj_id, frame_idx = point

        if action == 'add':
            # Redo addition: append to end
            self.click_points.append(point)
            self.annotated_frames.add(frame_idx)
            status_prefix = "Redid addition of"

        elif action == 'remove':
            # Redo removal: find and remove point
            try:
                self.click_points.remove(point)
            except ValueError:
                self.status_label.config(text="Cannot redo: point not found")
                return

            status_prefix = "Redid removal of"

            # Update annotated frames if frame now empty
            remaining_points_on_frame = [p for p in self.click_points if p[4] == frame_idx]
            if not remaining_points_on_frame:
                self.annotated_frames.discard(frame_idx)

        # Add operation back to undo stack
        self.undo_stack.append(operation)
        self._mark_object_dirty(obj_id)

        # Switch to the object that the point belongs to
        self.current_object_id = obj_id
        self.object_var.set(obj_id)
        self.object_name_var.set(self.object_names.get(obj_id, f"Object_{obj_id}"))
        self.update_object_color_display()

        # Jump to the frame where the point is
        self.current_frame_idx = frame_idx
        self.frame_var.set(frame_idx)

        # Update UI
        self.update_object_list()
        self.update_points_display()
        self.display_current_frame()

        # Show status
        point_type = "positive" if is_positive else "negative"
        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
        self.status_label.config(text=f"{status_prefix} {point_type} point for {obj_name} at frame {frame_idx + 1}")

    def _show_auto_dismiss(self, title, message, timeout_ms=4000):
        """Show an informational dialog that auto-closes after timeout_ms. User can also click OK."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        tk.Label(dialog, text=message, justify=tk.LEFT, padx=20, pady=12).pack()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=(0, 10))
        tk.Button(btn_frame, text="OK", command=dialog.destroy, width=8).pack()

        dialog.update_idletasks()
        px = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, px)}+{max(0, py)}")

        dialog.after(timeout_ms, lambda: dialog.destroy() if dialog.winfo_exists() else None)

    def _handle_annotation_frame_mismatch(self, annotation_data):
        """Handle frame count mismatch between saved annotations and current video"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Handle Frame Mismatch")
        dialog.geometry("500x350")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        saved_frames = annotation_data.get("total_frames", 0)
        current_frames = len(self.frames)

        ttk.Label(main_frame, text="Frame Count Mismatch Options", 
                    font=('Arial', 14, 'bold')).pack(pady=(0, 15))

        info_text = f"Saved annotations: {saved_frames} frames\n" \
                    f"Current video: {current_frames} frames\n\n" \
                    f"Choose how to handle this mismatch:"

        ttk.Label(main_frame, text=info_text, justify=tk.LEFT).pack(pady=(0, 15))

        option_var = tk.StringVar(value="skip")

        ttk.Radiobutton(main_frame, text="Skip invalid frame indices (safest)", 
                        variable=option_var, value="skip").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="Scale frame indices proportionally", 
                        variable=option_var, value="scale").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(main_frame, text="Cancel import", 
                        variable=option_var, value="cancel").pack(anchor=tk.W, pady=2)

        def apply_option():
            option = option_var.get()
            dialog.destroy()
            
            if option == "cancel":
                return
            elif option == "skip":
                self._import_annotations_skip_invalid(annotation_data)
            elif option == "scale":
                self._import_annotations_scale_indices(annotation_data)

        ttk.Button(main_frame, text="Apply", command=apply_option).pack(pady=(20, 0))

    def _import_annotations_skip_invalid(self, annotation_data):
        """Import annotations, skipping those with invalid frame indices"""
        imported_count = 0
        skipped_count = 0
        
        for annotation in annotation_data["annotations"]:
            try:
                frame_idx = annotation["frame_index"]
                
                # Skip if frame index is out of bounds
                if frame_idx >= len(self.frames):
                    skipped_count += 1
                    continue
                
                x = annotation["x"]
                y = annotation["y"]
                is_positive = annotation["is_positive"]
                obj_id = annotation["object_id"]
                obj_name = annotation.get("object_name", f"Object_{obj_id}")
                
                # Add the annotation point
                self.click_points.append((x, y, is_positive, obj_id, frame_idx))
                
                # Update object names and colors
                if obj_id not in self.object_names:
                    self.object_names[obj_id] = obj_name
                    if obj_id not in self.object_colors:
                        if "object_colors" in annotation_data and str(obj_id) in annotation_data["object_colors"]:
                            self.object_colors[obj_id] = annotation_data["object_colors"][str(obj_id)]
                        else:
                            self.object_colors[obj_id] = self._get_next_color()
                
                # Track annotated frames
                self.annotated_frames.add(frame_idx)
                imported_count += 1
                
            except KeyError as e:
                skipped_count += 1
                continue
        
        # Update UI
        self.update_points_display()
        self.update_object_list()
        self.display_current_frame()
        
        self.annotations_dirty = False
        self.updated_objects.clear()
        self._show_auto_dismiss("Import Complete",
                        f"Imported {imported_count} annotations\n"
                        f"Skipped {skipped_count} annotations with invalid frame indices",
                        timeout_ms=4000)

    def _get_next_color(self):
        """Get the color for the next object ID (auto-generated)."""
        next_id = self.max_object_id + 1
        return self.object_colors[next_id]

    def toggle_multi_frame_annotation(self):
        """Multi-frame annotation mode is always enabled"""
        # Multi-frame annotation is always active, no need to toggle
        self.multi_frame_label.config(text="MULTI-FRAME ANNOTATION ACTIVE")
        self.status_label.config(text="Multi-frame mode: Navigate to different frames and add points, then segment")

    def toggle_point_removal_mode(self):
        """Toggle point removal mode for removing individual annotation points"""
        self.point_removal_mode = not self.point_removal_mode

        if self.point_removal_mode:
            # Change button color to indicate active state
            self.remove_point_button.config(bg='#DC143C', activebackground='#FF6347')  # Red
            self.status_label.config(text="Point removal mode: Click on annotation points to remove them")
        else:
            # Reset button color to normal
            self.remove_point_button.config(bg='#404040', activebackground='#505050')
            self.status_label.config(text="Point removal mode disabled")
    
    def remove_point_at_location(self, x, y, frame_idx):
        """Remove the closest annotation point to the given location"""
        if not self.click_points:
            return False
        
        # Find points on the current frame
        frame_points = [(i, point) for i, point in enumerate(self.click_points) 
                       if point[4] == frame_idx]  # point[4] is frame_idx
        
        if not frame_points:
            return False
        
        # Find the closest point to the click location
        min_distance = float('inf')
        closest_point_idx = None
        
        for point_idx, (px, py, is_pos, obj_id, f_idx) in frame_points:
            # Calculate distance from click to point
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_point_idx = point_idx
        
        # Only remove if click is close enough to a point (within 20 pixels)
        if closest_point_idx is not None and min_distance <= 20:
            removed_point = self.click_points.pop(closest_point_idx)
            px, py, is_pos, obj_id, f_idx = removed_point
            self._mark_object_dirty(obj_id)

            # Store removal operation for undo
            operation = {
                'action': 'remove',
                'point': removed_point,
                'index': closest_point_idx  # Original position in list
            }
            self.undo_stack.append(operation)
            self.redo_stack.clear()

            # Update annotated frames if no more points on this frame
            remaining_points_on_frame = [p for p in self.click_points if p[4] == frame_idx]
            if not remaining_points_on_frame:
                self.annotated_frames.discard(frame_idx)

            # Update object list and display
            self.update_object_list()
            self.display_current_frame()

            # Show confirmation (using ORIGINAL coordinates in the message)
            point_type = "positive" if is_pos else "negative"
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            self.status_label.config(text=f"Removed {point_type} point for {obj_name} at ({px:.0f}, {py:.0f})")

            return True

        return False

    def update_object_name(self, event=None):
        """Update the name of the current object"""
        new_name = self.object_name_var.get().strip()
        if new_name:
            self.object_names[self.current_object_id] = new_name
            self.update_object_list()
        # Remove focus from entry after updating
        self.canvas.focus_set()

    def cancel_object_name_edit(self, event=None):
        """Cancel object name editing and restore original name"""
        # Reset to current object's name (cancel any unsaved changes)
        self.object_name_var.set(self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}"))
        # Remove focus from entry
        self.canvas.focus_set()

    def on_object_change(self, event=None):
        obj_id = self.object_var.get()
        if obj_id >= 1:
            self.current_object_id = obj_id
            self.object_name_var.set(self.object_names[obj_id])
            self.update_object_color_display()
            self.update_object_list()
            if self.frames:
                self.display_current_frame()

    def prev_object(self):
        """Go to previous object in the list"""
        if self.current_object_id > 1:
            self.object_var.set(self.current_object_id - 1)
            self.on_object_change()  # Explicitly trigger the update

    def next_object(self):
        """Go to next object in the list"""
        self.object_var.set(self.current_object_id + 1)
        self.on_object_change()

    def update_object_color_display(self):
        """Update the color indicator for the current object"""
        if self.current_object_id in self.sam2_covered_ids:
            # SAM3-covered object: read-only here, shown greyed out
            self.object_color_label.config(background='#808080')
        else:
            color = self.object_colors[self.current_object_id]
            self.object_color_label.config(background=self._rgb_to_hex(color))
    def add_new_object(self):
        """Add a new object for segmentation"""
        self.max_object_id += 1
        self.current_object_id = self.max_object_id
        self.object_var.set(self.current_object_id)
        self.object_spinbox.config(to=self.max_object_id)
        self.object_name_var.set(self.object_names[self.current_object_id])
        self.update_object_color_display()
        self.update_object_list()
        self.status_label.config(text=f"Added object {self.current_object_id}: {self.object_names[self.current_object_id]}")
            
    @staticmethod
    def _truncate_path(path, max_len=50):
        """Truncate a long path from the front, preserving directory boundaries at the end."""
        if len(path) <= max_len:
            return path
        parts = path.replace("\\", "/").split("/")
        result = parts[-1]  # always keep the filename
        for part in reversed(parts[:-1]):
            candidate = part + "/" + result
            if len("…/" + candidate) > max_len:
                break
            result = candidate
        return "…/" + result

    def load_video(self):
        """Load video file and extract frames"""
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ],
            initialdir=self.last_dir_video or self.working_dir
        )

        if not file_path:
            return

        # Clean up any existing lazy loading after the dialog, not before,
        # to avoid OpenCV release interrupting Tkinter's grab on the dialog.
        if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
            self.video_cap_lazy.release()
            self.video_cap_lazy = None

        self.last_dir_video = os.path.dirname(file_path)

        try:
            self.video_path = file_path
            # Store video path for display on canvas
            self.video_display_text = self._truncate_path(file_path)

            # CRITICAL FIX: Reset prerendered flags when loading new original video
            # This ensures flash works after previously viewing a segmented video
            self.segmented_video_displayed = False
            self.has_prerendered_masks = False
            self.original_video_path_for_resegment = None

            # Clean up original frame source when loading new video
            if hasattr(self, 'original_frame_source') and self.original_frame_source is not None:
                try:
                    self.original_frame_source.close()
                except:
                    pass
                self.original_frame_source = None

            # Clear temporary masks when loading new video
            self._clear_temporary_masks()

            # Reset display zoom when loading new video
            self.display_zoom_level = 1.0
            self.zoom_center_img_x = 0.5
            self.zoom_center_img_y = 0.5

            self.load_video_frames()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def load_video_frames(self):
        """Extract frames from video with multiple backend fallbacks"""
        try:
            # Try different OpenCV backends in order of preference
            backends = [
                (cv2.CAP_FFMPEG, "FFMPEG"),
                (cv2.CAP_GSTREAMER, "GStreamer"),
                (cv2.CAP_ANY, "Auto")
            ]

            self.video_cap = None
            successful_backend = None

            for backend, name in backends:
                try:
                    self.video_cap = cv2.VideoCapture(self.video_path, backend)
                    if self.video_cap.isOpened():
                        successful_backend = name
                        break
                    else:
                        self.video_cap.release()
                except Exception as e:
                    print(f"Failed to open with {name} backend: {e}")
                    continue

            if not self.video_cap or not self.video_cap.isOpened():
                raise ValueError(
                    "Could not open video file with any backend.\n"
                    "Please install required codecs or convert video to MP4 format."
                )

            self.status_label.config(text=f"Video loaded using {successful_backend} backend")

            self.frames = []
            # Only clear masks and click_points when loading a fresh video
            # When loading a segmented video (via import_masks), masks and click_points are already populated
            if not self.segmented_video_displayed:
                self.masks = {}
                self.click_points = []
            self.mask_cache = {}  # Clear mask cache to prevent stale data from previous video

            # Only reset these flags and clear original video path when loading a fresh video
            # When loading a segmented video (via import_masks), these are already set correctly before calling load_video_frames()
            if not self.segmented_video_displayed:
                self.segmented_video_displayed = False  # Reset flag when loading new original video
                self.has_prerendered_masks = False  # Reset prerendered masks flag
                self.original_video_path_for_resegment = None  # Clear any previously stored original video path

            # Reset Flash Mask button state when loading new video
            self.has_segmentation = False
            if hasattr(self, 'flash_mask_button'):
                self.flash_mask_button.config(state='disabled')

            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Store video properties
            self.video_props = {
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'fps': fps if fps > 0 else 30
            }
            self.video_fps = self.video_props['fps']  # Store FPS for playback

            # Reset slider zoom to full range for new video
            self.slider_zoom_level.set(1)

            # Handle lazy loading if enabled
            if self.lazy_load_var.get():
                self._setup_lazy_loading(total_frames, width, height, self.video_props['fps'])
                return

            # Eager loading: load all frames
            self.status_label.config(text=f"Loading {total_frames} frames...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()

            frame_count = 0

            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                frame_count += 1

                # Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                if frame_count % 10 == 0:
                    self.root.update_idletasks()

            self.video_cap.release()
            self.progress_bar.pack_forget()

            if self.frames:
                self.current_frame_idx = 0
                self.frame_slider.config(to=len(self.frames)-1)
                self.display_current_frame()

                self.status_label.config(text=f"Video loaded: {len(self.frames)} frames @ {self.video_fps:.1f} FPS")

                self.update_object_list()
            else:
                raise ValueError("No frames could be extracted from video")

        except Exception as e:
            self.progress_bar.pack_forget()
            raise e

    def _setup_lazy_loading(self, total_frames, width, height, fps):
        """Setup lazy loading for very large videos"""
        try:
            # Keep video capture open for lazy loading
            self.video_cap_lazy = cv2.VideoCapture(self.video_path)
            if not self.video_cap_lazy.isOpened():
                raise ValueError("Could not open video file for lazy loading")

            # Store video properties for lazy loading
            self.video_props = {
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'fps': fps
            }

            # Initialize frame cache (empty frames list - all frames, no skipping)
            self.frames = [None] * total_frames
            self.frame_cache = {}  # Cache for loaded frames

            # Initialize UI
            self.current_frame_idx = 0
            self.frame_slider.config(to=total_frames-1)
            self.display_current_frame()

            self.status_label.config(text=f"Lazy loading: {total_frames} frames @ {fps:.1f} FPS")

            self.update_object_list()

        except Exception as e:
            if self.video_cap_lazy:
                self.video_cap_lazy.release()
            raise e

    def _load_frame_lazy(self, frame_idx):
        """Load a specific frame on demand"""
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]

        if not self.video_cap_lazy or not hasattr(self, 'video_props'):
            return None

        try:
            # Seek to the frame, then verify and advance if seek landed short.
            # cv2.CAP_PROP_POS_FRAMES seeking is approximate for compressed video:
            # OpenCV seeks to the nearest preceding keyframe, so the actual position
            # after the seek can be earlier than requested. Reading forward the
            # remaining frames (at most one GOP, ~15-60 frames) corrects this.
            self.video_cap_lazy.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            actual = int(self.video_cap_lazy.get(cv2.CAP_PROP_POS_FRAMES))
            _seek_guard = 0
            while actual < frame_idx:
                ret, _ = self.video_cap_lazy.read()
                _seek_guard += 1
                if not ret or _seek_guard > 500:
                    # 500 frames is far beyond any realistic GOP; if we're still
                    # short the codec is not reporting position reliably — bail
                    # rather than spinning forever on the Tkinter main thread.
                    break
                actual = int(self.video_cap_lazy.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.video_cap_lazy.read()

            if not ret:
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Cache the frame (limit cache size to prevent memory issues)
            if len(self.frame_cache) > 50:  # Keep only last 50 frames in cache
                # Remove oldest frames
                oldest_keys = sorted(self.frame_cache.keys())[:10]
                for key in oldest_keys:
                    del self.frame_cache[key]

            self.frame_cache[frame_idx] = frame_rgb
            return frame_rgb

        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return None

    def _get_frame_dimensions(self):
        """
        Get frame dimensions (height, width) that works in all loading modes.

        Returns:
            tuple: (height, width) or None if unavailable

        Priority order:
        1. From video_props (most reliable)
        2. From loaded frame (eager mode or cached frame)
        3. From lazy load (load first frame on demand)
        4. From video_cap metadata (fallback)
        """
        # 1. Check video_props
        if hasattr(self, 'video_props') and self.video_props:
            if 'height' in self.video_props and 'width' in self.video_props:
                return (self.video_props['height'], self.video_props['width'])

        # 2. Check if we have frames loaded (eager mode)
        if self.frames and len(self.frames) > 0:
            if self.frames[0] is not None:  # Not None in eager mode
                return self.frames[0].shape[:2]

            # 3. Lazy mode: Try to load first frame
            if hasattr(self, '_load_frame_lazy'):
                first_frame = self._load_frame_lazy(0)
                if first_frame is not None:
                    return first_frame.shape[:2]

        # 4. Fallback to video capture metadata
        video_cap = None
        if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
            video_cap = self.video_cap_lazy
        elif hasattr(self, 'video_cap') and self.video_cap:
            video_cap = self.video_cap

        if video_cap:
            try:
                height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if height > 0 and width > 0:
                    return (height, width)
            except:
                pass

        # Could not determine dimensions
        print("Warning: Could not determine frame dimensions")
        return None

    def _get_session_cache_dir(self, video_path):
        """
        Get or create session-based cache directory for this video.
        Reuses cache within session if same video, creates new temp dir otherwise.

        Args:
            video_path: Path to video file

        Returns:
            str: Absolute path to cache directory
        """
        try:
            # Get video metadata for hash
            abs_path = os.path.abspath(video_path)
            file_size = os.path.getsize(abs_path)
            mtime = os.path.getmtime(abs_path)

            # Create video hash (identifies video uniquely)
            cache_key = f"{abs_path}|{file_size}|{int(mtime)}"
            video_hash = hashlib.md5(cache_key.encode()).hexdigest()

            # Check if we can reuse existing session cache
            if self.session_cache_dir and self.session_video_hash == video_hash:
                # Same video, reuse existing cache
                if os.path.exists(self.session_cache_dir):
                    print(f"Reusing session cache: {self.session_cache_dir}")
                    return self.session_cache_dir

            # Different video or no cache yet - clean up old cache if exists
            if self.session_cache_dir and os.path.exists(self.session_cache_dir):
                print(f"Cleaning up old session cache: {self.session_cache_dir}")
                shutil.rmtree(self.session_cache_dir, ignore_errors=True)

            # Create new temp directory for this session
            self.session_cache_dir = tempfile.mkdtemp(prefix='sam2_frames_')
            self.session_video_hash = video_hash
            print(f"Created session cache: {self.session_cache_dir}")

            return self.session_cache_dir

        except Exception as e:
            print(f"Error getting session cache dir: {e}")
            # Fallback to simple temp directory
            return tempfile.mkdtemp(prefix='sam2_frames_')

    def _validate_frame_cache(self, cache_dir, expected_count):
        """
        Check if cached frames are valid and complete.

        Args:
            cache_dir: Directory to check
            expected_count: Expected number of frame files

        Returns:
            bool: True if cache is valid, False otherwise
        """
        try:
            if not os.path.exists(cache_dir):
                return False

            # Count frame files
            frame_files = sorted([f for f in os.listdir(cache_dir)
                                 if f.endswith('.jpg') and f[:-4].isdigit()])

            if len(frame_files) != expected_count:
                print(f"Cache incomplete: {len(frame_files)} frames, expected {expected_count}")
                return False

            # Validate a few sample frames are readable
            samples = [0, len(frame_files)//2, len(frame_files)-1]
            for idx in samples:
                if idx < len(frame_files):
                    frame_path = os.path.join(cache_dir, frame_files[idx])
                    test_frame = cv2.imread(frame_path)
                    if test_frame is None:
                        print(f"Cache corrupted: cannot read {frame_files[idx]}")
                        return False

            return True

        except Exception as e:
            print(f"Error validating cache: {e}")
            return False

    def _calculate_segmentation_quality_metrics(self):
        """
        Calculate inter-frame changes and background ratios.
        Called after segmentation completes.
        Uses the calculate_quality_metrics utility from utils.py.
        """
        if not self.masks:
            return

        # Get frame dimensions
        dims = self._get_frame_dimensions()
        if dims is None:
            print("Cannot calculate metrics: frame dimensions unknown")
            return

        num_frames = len(self.frames)

        # Use the utility function from utils.py
        self.inter_frame_changes, self.background_ratios, self.overlap_ratios = calculate_quality_metrics(
            masks=self.masks,
            load_mask_func=self._load_mask,
            frame_dimensions=dims,
            num_frames=num_frames
        )

        # Update visualization
        self._update_quality_visualizations()

    def _render_quality_backgrounds(self):
        """
        Pre-render background colorbars as images for the current visible range.
        In zoom mode, only renders frames in the visible slider window.
        This is much faster than creating hundreds of rectangles on every update.
        """
        if not self.change_viz_canvas or not self.bg_viz_canvas or not self.overlap_viz_canvas:
            return False

        if not self.inter_frame_changes or not self.background_ratios:
            return False

        # Get canvas dimensions
        canvas_width = self.change_viz_canvas.winfo_width()
        canvas_height = 15

        if canvas_width <= 1:  # Canvas not yet rendered
            self.root.update()
            canvas_width = self.change_viz_canvas.winfo_width()

        if canvas_width <= 1:
            return False

        total_metric_frames = len(self.inter_frame_changes)
        if total_metric_frames == 0:
            return False

        # Get the visible range based on zoom level
        range_start, range_end, zoom_level = self._get_quality_viz_range()

        # Clamp range to available metrics
        range_start = max(0, min(range_start, total_metric_frames - 1))
        range_end = max(0, min(range_end, total_metric_frames - 1))

        visible_frames = range_end - range_start + 1
        if visible_frames <= 0:
            return False

        try:
            from PIL import Image, ImageDraw

            # Create image for inter-frame changes
            change_img = Image.new('RGB', (canvas_width, canvas_height))
            change_draw = ImageDraw.Draw(change_img)

            pixels_per_frame = canvas_width / visible_frames

            # Iterate only over the visible range
            for i, frame_idx in enumerate(range(range_start, range_end + 1)):
                x0 = int(i * pixels_per_frame)
                x1 = int((i + 1) * pixels_per_frame)

                change_ratio = self.inter_frame_changes[frame_idx]
                # Color: white (no change) to red (high change)
                intensity = int(255 * (1 - change_ratio))
                color = (255, intensity, intensity)

                change_draw.rectangle([x0, 0, x1, canvas_height], fill=color)

            self.change_viz_image = ImageTk.PhotoImage(change_img)

            # Create image for background ratios
            bg_img = Image.new('RGB', (canvas_width, canvas_height))
            bg_draw = ImageDraw.Draw(bg_img)

            for i, frame_idx in enumerate(range(range_start, range_end + 1)):
                x0 = int(i * pixels_per_frame)
                x1 = int((i + 1) * pixels_per_frame)

                bg_ratio = self.background_ratios[frame_idx]
                # Color: green (low bg) to yellow (high bg)
                if bg_ratio < 0.5:
                    g = 255
                    r = int(510 * bg_ratio)
                else:
                    r = 255
                    g = int(255 * (2 - 2 * bg_ratio))

                color = (r, g, 0)
                bg_draw.rectangle([x0, 0, x1, canvas_height], fill=color)

            self.bg_viz_image = ImageTk.PhotoImage(bg_img)

            # Create image for overlap ratios
            overlap_img = Image.new('RGB', (canvas_width, canvas_height))
            overlap_draw = ImageDraw.Draw(overlap_img)

            if self.overlap_ratios:
                for i, frame_idx in enumerate(range(range_start, range_end + 1)):
                    x0 = int(i * pixels_per_frame)
                    x1 = int((i + 1) * pixels_per_frame)

                    overlap_ratio = self.overlap_ratios[frame_idx]
                    # Color: white (no overlap) to blue (high overlap)
                    # Scale the ratio for better visibility (assume max useful overlap is ~0.5)
                    scaled_ratio = min(overlap_ratio * 2, 1.0)
                    intensity = int(255 * (1 - scaled_ratio))
                    color = (intensity, intensity, 255)

                    overlap_draw.rectangle([x0, 0, x1, canvas_height], fill=color)
            else:
                # No overlap data available (backward compatibility for old results)
                # Show light gray to indicate metric not calculated - no expensive recalculation
                overlap_draw.rectangle([0, 0, canvas_width, canvas_height], fill=(220, 220, 220))

            self.overlap_viz_image = ImageTk.PhotoImage(overlap_img)

            # Store the rendered range
            self.quality_viz_range_start = range_start
            self.quality_viz_range_end = range_end
            self.quality_viz_zoom_level = zoom_level

            self.quality_bg_rendered = True
            return True

        except Exception as e:
            print(f"WARNING: Failed to render quality backgrounds: {e}")
            return False

    def _update_quality_indicator(self):
        """
        Update only the blue frame indicator line (fast operation).
        Should be called on every frame change.

        If the visible range has changed (due to zoom or pan), triggers a full
        re-render instead. Otherwise, just repositions the blue indicator line.
        """
        if not self.change_viz_canvas or not self.bg_viz_canvas or not self.overlap_viz_canvas:
            return

        if not self.quality_bg_rendered:
            # Backgrounds not cached yet, do full render
            self._update_quality_visualizations()
            return

        # Check if the visible range has changed (zoom or pan)
        if self._quality_range_changed():
            # Range changed, need full re-render
            self._update_quality_visualizations()
            return

        # Get canvas dimensions
        canvas_width = self.change_viz_canvas.winfo_width()
        canvas_height = 15

        if canvas_width <= 1:
            return

        # Use the rendered range for positioning
        visible_frames = self.quality_viz_range_end - self.quality_viz_range_start + 1
        if visible_frames <= 0:
            return

        # Clear and redraw backgrounds
        self.change_viz_canvas.delete("all")
        self.bg_viz_canvas.delete("all")
        self.overlap_viz_canvas.delete("all")

        # Draw cached background images
        self.change_viz_canvas.create_image(0, 0, anchor='nw', image=self.change_viz_image)
        self.bg_viz_canvas.create_image(0, 0, anchor='nw', image=self.bg_viz_image)
        if self.overlap_viz_image:
            self.overlap_viz_canvas.create_image(0, 0, anchor='nw', image=self.overlap_viz_image)

        # Draw current frame indicator (positioned relative to visible range)
        if hasattr(self, 'current_frame_idx'):
            # Only draw if current frame is within visible range
            if (self.quality_viz_range_start <= self.current_frame_idx <= self.quality_viz_range_end):
                # Calculate position relative to visible range
                relative_pos = self.current_frame_idx - self.quality_viz_range_start
                pixels_per_frame = canvas_width / visible_frames
                x_pos = relative_pos * pixels_per_frame

                self.change_viz_canvas.create_line(
                    x_pos, 0, x_pos, canvas_height,
                    fill='blue', width=2
                )
                self.bg_viz_canvas.create_line(
                    x_pos, 0, x_pos, canvas_height,
                    fill='blue', width=2
                )
                self.overlap_viz_canvas.create_line(
                    x_pos, 0, x_pos, canvas_height,
                    fill='blue', width=2
                )

    def _get_quality_viz_range(self):
        """
        Get the frame range that should be displayed in quality visualizations.

        Returns:
            tuple: (range_start, range_end, zoom_level)
        """
        if not hasattr(self, 'slider_zoom_level') or not self.frames:
            total = len(self.inter_frame_changes) if self.inter_frame_changes else 0
            return (0, max(0, total - 1), 1)

        zoom_level = self.slider_zoom_level.get()
        total_frames = len(self.frames)

        if zoom_level == 1:
            # Full range mode - show all frames
            return (0, total_frames - 1, 1)
        else:
            # Zoomed mode - get current slider window bounds
            window_start = int(self.frame_slider.cget('from'))
            window_end = int(self.frame_slider.cget('to'))
            return (window_start, window_end, zoom_level)

    def _quality_range_changed(self):
        """
        Check if the visible range for quality visualization has changed.

        Returns:
            bool: True if range changed and re-render is needed, False otherwise
        """
        range_start, range_end, zoom_level = self._get_quality_viz_range()

        # Apply the same clamping as _render_quality_backgrounds() to ensure consistency
        if self.inter_frame_changes:
            total_metric_frames = len(self.inter_frame_changes)
            range_start = max(0, min(range_start, total_metric_frames - 1))
            range_end = max(0, min(range_end, total_metric_frames - 1))

        # Check if anything changed
        if (range_start != self.quality_viz_range_start or
            range_end != self.quality_viz_range_end or
            zoom_level != self.quality_viz_zoom_level):
            return True

        return False

    def _update_quality_visualizations(self):
        """
        Full update of quality visualizations (render backgrounds + indicator).
        Called when metrics are first loaded or calculated.
        """
        # Invalidate cache
        self.quality_bg_rendered = False

        # Render backgrounds
        if self._render_quality_backgrounds():
            # Update indicator
            self._update_quality_indicator()
        else:
            # Fallback to old method if caching fails
            self._update_quality_visualizations_fallback()

    def _update_quality_visualizations_fallback(self):
        """
        Fallback method using canvas rectangles (slower but always works).
        """
        if not self.change_viz_canvas or not self.bg_viz_canvas or not self.overlap_viz_canvas:
            return

        # Clear canvases
        self.change_viz_canvas.delete("all")
        self.bg_viz_canvas.delete("all")
        self.overlap_viz_canvas.delete("all")

        if not self.inter_frame_changes or not self.background_ratios:
            return

        # Get canvas dimensions
        canvas_width = self.change_viz_canvas.winfo_width()
        canvas_height = 15

        if canvas_width <= 1:  # Canvas not yet rendered
            self.root.update()
            canvas_width = self.change_viz_canvas.winfo_width()

        num_frames = len(self.inter_frame_changes)
        if num_frames == 0:
            return

        # Calculate pixels per frame
        pixels_per_frame = max(1, canvas_width / num_frames)

        # Render inter-frame changes
        for i, change_ratio in enumerate(self.inter_frame_changes):
            x0 = i * pixels_per_frame
            x1 = (i + 1) * pixels_per_frame

            # Color: white (no change) to red (high change)
            intensity = int(255 * (1 - change_ratio))
            color = f'#{255:02x}{intensity:02x}{intensity:02x}'

            self.change_viz_canvas.create_rectangle(
                x0, 0, x1, canvas_height,
                fill=color, outline=''
            )

        # Render background ratios
        for i, bg_ratio in enumerate(self.background_ratios):
            x0 = i * pixels_per_frame
            x1 = (i + 1) * pixels_per_frame

            # Color: green (low bg) to yellow (high bg)
            # Low BG (mostly objects) = good = green
            # High BG (mostly empty) = yellow/warning
            if bg_ratio < 0.5:
                # 0-50% bg: green
                g = 255
                r = int(510 * bg_ratio)
            else:
                # 50-100% bg: yellow to orange
                r = 255
                g = int(255 * (2 - 2 * bg_ratio))

            color = f'#{r:02x}{g:02x}00'

            self.bg_viz_canvas.create_rectangle(
                x0, 0, x1, canvas_height,
                fill=color, outline=''
            )

        # Render overlap ratios (if available)
        if self.overlap_ratios:
            for i, overlap_ratio in enumerate(self.overlap_ratios):
                x0 = i * pixels_per_frame
                x1 = (i + 1) * pixels_per_frame

                # Color: white (no overlap) to blue (high overlap)
                # Scale the ratio for better visibility
                scaled_ratio = min(overlap_ratio * 2, 1.0)
                intensity = int(255 * (1 - scaled_ratio))
                color = f'#{intensity:02x}{intensity:02x}{255:02x}'

                self.overlap_viz_canvas.create_rectangle(
                    x0, 0, x1, canvas_height,
                    fill=color, outline=''
                )
        else:
            # No data available (backward compatibility) - show light gray
            self.overlap_viz_canvas.create_rectangle(
                0, 0, canvas_width, canvas_height,
                fill='#dcdcdc', outline=''
            )

        # Draw current frame indicator
        if hasattr(self, 'current_frame_idx'):
            x_pos = self.current_frame_idx * pixels_per_frame
            self.change_viz_canvas.create_line(
                x_pos, 0, x_pos, canvas_height,
                fill='blue', width=2
            )
            self.bg_viz_canvas.create_line(
                x_pos, 0, x_pos, canvas_height,
                fill='blue', width=2
            )
            if self.overlap_ratios:
                self.overlap_viz_canvas.create_line(
                    x_pos, 0, x_pos, canvas_height,
                    fill='blue', width=2
                )

    def _save_quality_metrics(self, output_dir):
        """Save quality metrics to quality_metrics.npz in output directory.
        Uses the save_quality_metrics utility from utils.py."""
        return save_quality_metrics(
            output_dir=output_dir,
            inter_frame_changes=self.inter_frame_changes,
            background_ratios=self.background_ratios,
            overlap_ratios=self.overlap_ratios if self.overlap_ratios else None
        )

    def _load_quality_metrics(self, output_dir):
        """Load quality metrics from quality_metrics.npz if available.
        Uses the load_quality_metrics utility from utils.py."""
        inter_frame_changes, background_ratios, overlap_ratios = load_quality_metrics(output_dir)

        if inter_frame_changes is None or background_ratios is None:
            self.inter_frame_changes = []
            self.background_ratios = []
            self.overlap_ratios = []
            return False

        self.inter_frame_changes = inter_frame_changes
        self.background_ratios = background_ratios
        # Handle backward compatibility: old files may not have overlap_ratios
        # If missing, set to empty list (canvas will show gray, no recalculation)
        self.overlap_ratios = overlap_ratios if overlap_ratios is not None else []
        self._update_quality_visualizations()  # Refresh color bars
        return True

    def _on_viz_canvas_click(self, event):
        """
        Handle clicks on visualization canvas to jump to frame.
        Maps click position to the currently visible range (respects zoom).
        """
        if not self.inter_frame_changes:
            return

        canvas_width = event.widget.winfo_width()

        # Use the currently rendered visible range
        range_start = self.quality_viz_range_start
        range_end = self.quality_viz_range_end
        visible_frames = range_end - range_start + 1

        if visible_frames <= 0:
            return

        # Calculate which frame was clicked (relative to visible range)
        click_ratio = event.x / canvas_width
        relative_frame = int(click_ratio * visible_frames)
        target_frame = range_start + relative_frame
        target_frame = max(range_start, min(target_frame, range_end))

        # Jump to frame
        self.current_frame_idx = target_frame
        self.frame_var.set(target_frame)
        self.display_current_frame()
        # Note: don't call _update_quality_visualizations here as it may
        # be called by display_current_frame() via _update_quality_indicator()

    def _get_session_cache_key(self, video_path, frame_indices):
        """
        Generate a cache key for session-level frame reuse.

        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to extract

        Returns:
            str: Cache key based on video path, size, mtime, and frame range
        """
        try:
            # Get video file metadata
            abs_path = os.path.abspath(video_path)
            file_size = os.path.getsize(abs_path)
            mtime = os.path.getmtime(abs_path)

            # Create key from path, size, mtime, and frame range
            frame_range = f"{min(frame_indices)}-{max(frame_indices)}" if frame_indices else "all"
            cache_key = f"{abs_path}|{file_size}|{mtime:.0f}|{frame_range}"

            return cache_key
        except Exception as e:
            print(f"Warning: Could not generate cache key: {e}")
            return None

    def display_current_frame(self):
        """Display current video frame with overlays"""
        if not self.frames:
            return

        # Update Flash Points button state based on whether active object has points on this frame
        if hasattr(self, 'flash_points_button'):
            has_pts = any(p[3] == self.current_object_id and p[4] == self.current_frame_idx
                         for p in self.click_points)
            self.flash_points_button.config(state='normal' if has_pts else 'disabled')

        # Special case: If we have prerendered masks AND a temp mask for this frame,
        # load the original frame so we can display the temp mask cleanly
        if (self.has_prerendered_masks and
            self.current_frame_idx in self.temp_mask_frames and
            self.original_frame_source is not None):
            # Load from original video to avoid showing prerendered masks
            frame = self.original_frame_source.get_frame(self.current_frame_idx)
            if frame is None:
                # Fallback to prerendered if original load fails
                print(f"Warning: Could not load frame {self.current_frame_idx} from original video, using prerendered")
                if self.lazy_load_var.get() and hasattr(self, 'video_props'):
                    frame = self._load_frame_lazy(self.current_frame_idx)
                else:
                    if self.current_frame_idx >= len(self.frames):
                        return
                    frame = self.frames[self.current_frame_idx]
            if frame is None:
                return
            self.current_frame = frame.copy()
        else:
            # Normal case: Handle lazy loading (works for both original and segmented videos)
            if self.lazy_load_var.get() and hasattr(self, 'video_props'):
                # Load frame on demand
                frame = self._load_frame_lazy(self.current_frame_idx)
                if frame is None:
                    return
                self.current_frame = frame.copy()
            else:
                # Regular loading
                if self.current_frame_idx >= len(self.frames) or self.frames[self.current_frame_idx] is None:
                    return
                self.current_frame = self.frames[self.current_frame_idx].copy()
        display_frame = self.current_frame.copy()

        # Add annotation indicator for multi-frame annotation mode
        if self.multi_frame_annotation_mode and self.current_frame_idx in self.annotated_frames:
            # Add blue border for annotated frames
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (0, 165, 255), 8)
        
        # Apply mask overlay if enabled and masks exist
        # Check for masks to display (temp masks have priority over permanent)
        masks_to_display = None
        use_disk_cache = False

        if self.current_frame_idx in self.temp_masks:
            # Use temporary masks (from single-frame segmentation)
            masks_to_display = self.temp_masks[self.current_frame_idx]
            use_disk_cache = False  # Temp masks are in memory
        elif self.current_frame_idx in self.masks:
            # Use permanent masks (from full video segmentation)
            masks_to_display = self.masks[self.current_frame_idx]
            use_disk_cache = not self.has_prerendered_masks

        if masks_to_display:
            # OPTIMIZATION: If displaying pre-rendered segmented video, skip mask loading
            # The frames already have masks baked in from the segmented video
            # EXCEPT: If we have temp masks for this frame (from segment_current_frame_only)
            if self.has_prerendered_masks and self.current_frame_idx not in self.temp_mask_frames:
                # Frames already contain mask overlay - normally nothing to do
                # But if flash is in progress, we need to apply white overlay
                if self.flash_in_progress and self.flash_white_on and self.flash_obj_id is not None:
                    # Load the mask for the flashing object from disk
                    mask = self._load_mask(self.current_frame_idx, self.flash_obj_id)
                    if mask is not None:
                        if len(mask.shape) == 2:
                            # Resize mask if needed
                            if mask.shape != (display_frame.shape[0], display_frame.shape[1]):
                                from PIL import Image as PILImage
                                mask_pil = PILImage.fromarray(mask)
                                mask_pil = mask_pil.resize((display_frame.shape[1], display_frame.shape[0]), PILImage.NEAREST)
                                mask = np.array(mask_pil)

                            # Convert mask to boolean
                            mask_bool = mask > 0

                            # Create white overlay
                            white_overlay = np.zeros_like(display_frame)
                            white_overlay[mask_bool] = [255, 255, 255]

                            # Blend white overlay on top of existing frame
                            alpha = 0.7
                            display_frame = cv2.addWeighted(display_frame, 1-alpha, white_overlay, alpha, 0)
                pass
            else:
                # Original video mode: load masks from disk/memory and overlay
                # Collect all mask data first (to avoid cumulative blending bug)
                mask_data_list = []
                for obj_id in masks_to_display.keys():
                    # Load mask from disk OR memory depending on source
                    if use_disk_cache:
                        mask = self._load_mask(self.current_frame_idx, obj_id)
                    else:
                        # Temp masks are already in memory
                        mask = masks_to_display[obj_id]
                    if mask is None:
                        continue
                    # If flash is in progress, only show the flashing object
                    if self.flash_in_progress and obj_id != self.flash_obj_id:
                        continue

                    if len(mask.shape) == 2:  # Single channel mask

                        # Get object color
                        obj_color = self.object_colors.get(obj_id, [255, 255, 255])

                        # Check if mask needs resizing
                        if mask.shape != (display_frame.shape[0], display_frame.shape[1]):
                            print(f"  WARNING: Mask size {mask.shape} doesn't match frame size "
                                  f"({display_frame.shape[0]}, {display_frame.shape[1]})")
                            print(f"  Resizing mask...")
                            from PIL import Image as PILImage
                            mask_pil = PILImage.fromarray(mask)
                            mask_pil = mask_pil.resize((display_frame.shape[1], display_frame.shape[0]), PILImage.NEAREST)
                            mask = np.array(mask_pil)
                            print(f"  Resized mask to: {mask.shape}")

                        # Convert mask to boolean
                        mask_bool = mask > 0

                        # If flashing, override color to white
                        if self.flash_in_progress and obj_id == self.flash_obj_id and self.flash_white_on:
                            obj_color = [255, 255, 255]

                        mask_data_list.append((mask_bool, obj_color, obj_id))

                # FIXED: Single-pass blending to avoid cumulative darkening
                if mask_data_list:
                    height, width = display_frame.shape[:2]
                    combined_overlay = np.zeros((height, width, 3), dtype=np.float32)
                    overlap_count = np.zeros((height, width), dtype=np.int32)

                    for mask_bool, obj_color, obj_id in mask_data_list:
                        # Add color to overlapping regions (accumulate for averaging)
                        combined_overlay[mask_bool] += obj_color
                        overlap_count[mask_bool] += 1

                    # Average colors where masks overlap
                    mask_pixels = overlap_count > 0
                    combined_overlay[mask_pixels] /= overlap_count[mask_pixels, np.newaxis]
                    combined_overlay = combined_overlay.astype(np.uint8)

                    # Single blend operation - fixes cumulative darkening bug
                    alpha = self.mask_opacity_var.get() if hasattr(self, 'mask_opacity_var') else 0.4

                    display_frame = cv2.addWeighted(display_frame, 1-alpha, combined_overlay, alpha, 0)
        
        # Overlap flash overlay - applied after all normal mask rendering so it shows on top
        if self.flash_overlap_in_progress and self.flash_overlap_on and self.flash_overlap_computed is not None:
            overlap_mask = self.flash_overlap_computed
            # Resize overlap mask if it doesn't match the current display frame dimensions
            fh, fw = display_frame.shape[:2]
            if overlap_mask.shape != (fh, fw):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(overlap_mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((fw, fh), PILImage.NEAREST)
                overlap_mask = np.array(mask_pil) > 0
            # Bright red-orange overlay to make conflict regions stand out
            red_overlay = np.zeros_like(display_frame)
            red_overlay[overlap_mask] = [255, 60, 0]  # BGR: orange-red
            display_frame = cv2.addWeighted(display_frame, 0.4, red_overlay, 0.6, 0)

        # Draw click points only for the current frame
        flash_pts = self.flash_points_in_progress and self.flash_points_on
        for i, (x, y, is_positive, obj_id, frame_idx) in enumerate(self.click_points):
            if frame_idx != self.current_frame_idx:
                continue
            # Only draw points for current object or if showing all
            if obj_id == self.current_object_id or not hasattr(self, 'current_object_id'):
                obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                radius = 16 if flash_pts else 10
                line_length = 11 if flash_pts else 7
                line_thickness = 3 if flash_pts else 2

                if flash_pts:
                    # Draw white halo behind point for visibility
                    cv2.circle(display_frame, (int(x), int(y)), radius + 3, (255, 255, 255), 3)

                if is_positive:
                    color = tuple(obj_color)
                    # Draw unfilled circle in object color
                    cv2.circle(display_frame, (int(x), int(y)), radius, color, line_thickness)
                    # Draw + symbol
                    cv2.line(display_frame,
                            (int(x), int(y - line_length)),
                            (int(x), int(y + line_length)),
                            color, line_thickness)
                    cv2.line(display_frame,
                            (int(x - line_length), int(y)),
                            (int(x + line_length), int(y)),
                            color, line_thickness)
                else:
                    # Draw unfilled black circle with white minus line
                    cv2.circle(display_frame, (int(x), int(y)), radius, (0, 0, 0), line_thickness)
                    cv2.line(display_frame,
                            (int(x - line_length), int(y)),
                            (int(x + line_length), int(y)),
                            (255, 255, 255), line_thickness)

        # Gaze position overlay (drawn in native image coords; scaled with the frame)
        if self.gaze_points_by_frame:
            gaze_pts = self.gaze_points_by_frame.get(self.current_frame_idx)
            if gaze_pts:
                gaze_overlay.draw_gaze_marker(display_frame, gaze_pts)

        # Convert to PIL and display
        pil_image = Image.fromarray(display_frame)

        # Scale image to fit canvas (with zoom support)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size

            # Calculate base scale to fit canvas while maintaining aspect ratio
            scale_w = (canvas_width - 20) / img_width
            scale_h = (canvas_height - 20) / img_height
            base_scale = min(scale_w, scale_h)

            # Apply display zoom
            self.scale_factor = base_scale * self.display_zoom_level

            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)

            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        self.display_frame = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")

        # Position image: center if not zoomed, otherwise keep zoom center at canvas center
        if self.display_zoom_level <= 1.0:
            # Center the image on canvas (normal mode).
            # Set scrollregion to the full canvas widget so xview_moveto(0) keeps image centered.
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.display_frame)
            self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        else:
            # Position image so zoom center point stays at canvas center (zoomed mode)
            # The zoom center (as fraction of image) should be at canvas center
            img_pos_x = canvas_width // 2 - int(self.zoom_center_img_x * new_width)
            img_pos_y = canvas_height // 2 - int(self.zoom_center_img_y * new_height)

            # Create image at calculated position (using NW anchor for absolute positioning)
            self.canvas.create_image(img_pos_x, img_pos_y, anchor=tk.NW, image=self.display_frame)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # Update canvas overlay labels (always at canvas-widget corners, outside frame area)
        if hasattr(self, '_overlay_label_tl'):
            # Top-left: video filename
            if getattr(self, 'segmented_video_displayed', False) and getattr(self, 'original_video_path_for_resegment', None):
                vid_text = self._truncate_path(self.original_video_path_for_resegment)
            elif getattr(self, 'video_path', None):
                vid_text = self._truncate_path(self.video_path)
            else:
                vid_text = ""
            self._overlay_label_tl.config(text=vid_text)

            # Top-right: current active category with its color
            current_obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            current_obj_text = f"{current_obj_name} (ID: {self.current_object_id})"
            obj_rgb = self.object_colors.get(self.current_object_id, [255, 255, 255])
            obj_hex = f'#{obj_rgb[0]:02x}{obj_rgb[1]:02x}{obj_rgb[2]:02x}'
            self._overlay_label_tr.config(text=current_obj_text, fg=obj_hex)

            # Bottom-left: results folder (only when viewing segmented results)
            if self.results_output_dir:
                results_text = f"Results: {self._truncate_path(self.results_output_dir)}"
            else:
                results_text = ""
            self._overlay_label_bl.config(text=results_text)

            # Raise overlay labels above canvas items so they stay visible
            self._overlay_label_tl.lift()
            self._overlay_label_tr.lift()
            self._overlay_label_bl.lift()
        
        # Update frame info
        self.frame_var.set(self.current_frame_idx)
        frame_text = f"{self.current_frame_idx + 1}/{len(self.frames)}"
        if self.multi_frame_annotation_mode and self.current_frame_idx in self.annotated_frames:
            frame_text += " (annotated)"
        self.frame_label.config(text=frame_text)

        # Update quality visualization indicators (fast indicator update only)
        if self.inter_frame_changes and self.background_ratios:
            self._update_quality_indicator()

    def on_canvas_resize(self, event):
        """Handle canvas resize events with debouncing"""
        # Only respond to actual size changes, not other configure events
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Check if size actually changed
        if hasattr(self, '_last_canvas_size'):
            if (canvas_width, canvas_height) == self._last_canvas_size:
                return  # Size didn't change, ignore this event

        self._last_canvas_size = (canvas_width, canvas_height)

        # Cancel any pending resize callback
        if hasattr(self, '_resize_after_id'):
            self.root.after_cancel(self._resize_after_id)

        # Schedule display update after 100ms delay (debounce)
        # This prevents excessive updates during active resizing
        self._resize_after_id = self.root.after(100, self._handle_resize)

    def _handle_resize(self):
        """Actually handle the resize after debounce delay"""
        if self.frames:
            self.display_current_frame()

    def _on_quality_viz_resize(self, event):
        """Handle quality visualization canvas resize events with debouncing"""
        # Only respond to actual width changes (height is fixed at 15)
        canvas_width = event.widget.winfo_width()

        # Check if width actually changed
        if hasattr(self, '_last_quality_viz_width'):
            if canvas_width == self._last_quality_viz_width:
                return  # Width didn't change, ignore this event

        self._last_quality_viz_width = canvas_width

        # Cancel any pending resize callback
        if hasattr(self, '_quality_viz_resize_after_id'):
            self.root.after_cancel(self._quality_viz_resize_after_id)

        # Schedule update after 100ms delay (debounce)
        # This prevents excessive updates during active resizing
        self._quality_viz_resize_after_id = self.root.after(100, self._handle_quality_viz_resize)

    def _handle_quality_viz_resize(self):
        """Actually handle the quality viz resize after debounce delay"""
        # Invalidate cached backgrounds and trigger re-render
        if hasattr(self, 'inter_frame_changes') and self.inter_frame_changes:
            self.quality_bg_rendered = False
            self._update_quality_visualizations()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel: frame navigation, annotated frame jump, or zoom"""
        if not self.frames:
            return

        # Detect scroll direction (cross-platform)
        if event.num == 5 or event.delta < 0:  # Scroll down
            delta = -1
        else:  # Scroll up (event.num == 4 or event.delta > 0)
            delta = 1

        # Check modifier keys (use both event.state bits and hasattr for robustness)
        # Control key detection (Ctrl on Windows/Linux, Control on Mac)
        ctrl_pressed = (event.state & 0x4) != 0

        # Command key detection (Mac) - try multiple methods
        # On Mac, Command might be 0x8, 0x10, or available via other means
        cmd_pressed = (event.state & 0x10) != 0  # More reliable for Mac Command

        # Shift key detection
        shift_pressed = (event.state & 0x1) != 0

        if ctrl_pressed or cmd_pressed:
            # Zoom mode: zoom in/out centered on cursor
            self._handle_zoom(event, delta)
        elif shift_pressed:
            # Annotated frame navigation mode
            if delta > 0:
                self.jump_to_prev_annotated_frame()
            else:
                self.jump_to_next_annotated_frame()
        else:
            # Normal frame navigation mode: prev/next frame (respecting slider zoom)
            if delta > 0:
                self.prev_frame()
            else:
                self.next_frame()

    def _handle_zoom(self, event, delta):
        """Handle zoom in/out centered on cursor position"""
        if not self.frames or self.current_frame is None:
            return

        # Get cursor position in canvas coordinates
        canvas_x = event.x
        canvas_y = event.y

        # Get current canvas and image dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Get current image dimensions
        img_height, img_width = self.current_frame.shape[:2]

        # Calculate base scale (fit to canvas)
        scale_w = (canvas_width - 20) / img_width
        scale_h = (canvas_height - 20) / img_height
        base_scale = min(scale_w, scale_h)

        # Convert cursor position to image coordinates (fraction 0.0-1.0)
        # Current display dimensions
        current_display_width = img_width * base_scale * self.display_zoom_level
        current_display_height = img_height * base_scale * self.display_zoom_level

        # Current image position (centered if zoom=1, or offset if zoomed)
        if self.display_zoom_level <= 1.0:
            # Centered
            img_left = (canvas_width - current_display_width) / 2
            img_top = (canvas_height - current_display_height) / 2
        else:
            # Positioned to keep zoom center fixed
            img_left = canvas_width / 2 - self.zoom_center_img_x * current_display_width
            img_top = canvas_height / 2 - self.zoom_center_img_y * current_display_height

        # Calculate cursor position relative to image (fraction)
        cursor_img_x = (canvas_x - img_left) / current_display_width
        cursor_img_y = (canvas_y - img_top) / current_display_height

        # Clamp to image bounds
        cursor_img_x = max(0.0, min(1.0, cursor_img_x))
        cursor_img_y = max(0.0, min(1.0, cursor_img_y))

        # Calculate new zoom level
        zoom_factor = 1.15  # 15% zoom per scroll step
        if delta > 0:  # Zoom in
            new_zoom = self.display_zoom_level * zoom_factor
        else:  # Zoom out
            new_zoom = self.display_zoom_level / zoom_factor

        # Enforce zoom limits
        max_zoom = 10.0  # Maximum 10x zoom
        new_zoom = max(0.1, min(max_zoom, new_zoom))

        # Auto-reset when zooming out to fit-or-smaller
        if new_zoom <= 1.0:
            self.display_zoom_level = 1.0
            self.zoom_center_img_x = 0.5
            self.zoom_center_img_y = 0.5
            self.display_current_frame()
            return

        # Update zoom level and center
        self.display_zoom_level = new_zoom
        self.zoom_center_img_x = cursor_img_x
        self.zoom_center_img_y = cursor_img_y

        # Redraw with new zoom
        self.display_current_frame()

    def on_canvas_click(self, event):
        """Handle left mouse click (positive point or point removal)"""
        # Remove focus from any text entry widgets
        self.canvas.focus_set()

        if self.point_removal_mode:
            self.handle_point_removal_click(event)
        else:
            self.add_click_point(event, is_positive=True)

    def on_canvas_right_click(self, event):
        """Handle right mouse click (negative point or point removal)"""
        # Remove focus from any text entry widgets
        self.canvas.focus_set()

        if self.point_removal_mode:
            self.handle_point_removal_click(event)
        else:
            self.add_click_point(event, is_positive=False)
    
    def handle_point_removal_click(self, event):
        """Handle click in point removal mode"""
        if not self.frames or not self.current_frame_idx < len(self.frames):
            return
            
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        if self.scale_factor > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Calculate image position (account for zoom and centering)
            img_display_width = int(self.current_frame.shape[1] * self.scale_factor)
            img_display_height = int(self.current_frame.shape[0] * self.scale_factor)

            if self.display_zoom_level <= 1.0:
                # Centered positioning (normal mode)
                offset_x = (canvas_width - img_display_width) // 2
                offset_y = (canvas_height - img_display_height) // 2
            else:
                # Zoomed positioning (keep zoom center at canvas center)
                offset_x = canvas_width // 2 - int(self.zoom_center_img_x * img_display_width)
                offset_y = canvas_height // 2 - int(self.zoom_center_img_y * img_display_height)

            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor

            # Check if click is within image bounds
            if 0 <= img_x < self.current_frame.shape[1] and 0 <= img_y < self.current_frame.shape[0]:
                # Try to remove a point at this location
                if not self.remove_point_at_location(img_x, img_y, self.current_frame_idx):
                    self.status_label.config(text="No annotation point found near this location")
        else:
            self.status_label.config(text="Cannot remove points: invalid scale factor")
        
    def add_click_point(self, event, is_positive=True):
        """Add click point for segmentation"""
        if not self.frames or not self.current_frame_idx < len(self.frames):
            return

        if self.current_object_id in self.sam3_object_ids:
            self.status_label.config(
                text=f"Object {self.current_object_id} was segmented by SAM3 - select a different object to annotate.")
            return

        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        if self.scale_factor > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Calculate image position (account for zoom and centering)
            img_display_width = int(self.current_frame.shape[1] * self.scale_factor)
            img_display_height = int(self.current_frame.shape[0] * self.scale_factor)

            if self.display_zoom_level <= 1.0:
                # Centered positioning (normal mode)
                offset_x = (canvas_width - img_display_width) // 2
                offset_y = (canvas_height - img_display_height) // 2
            else:
                # Zoomed positioning (keep zoom center at canvas center)
                offset_x = canvas_width // 2 - int(self.zoom_center_img_x * img_display_width)
                offset_y = canvas_height // 2 - int(self.zoom_center_img_y * img_display_height)

            # Convert to current frame coordinates
            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor

            # Ensure coordinates are within current frame bounds
            img_height, img_width = self.current_frame.shape[:2]
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                # Store addition operation for undo
                operation = {
                    'action': 'add',
                    'point': (img_x, img_y, is_positive, self.current_object_id, self.current_frame_idx)
                }
                self.undo_stack.append(operation)
                self.redo_stack.clear()

                # Store frame-aware point
                self.click_points.append((img_x, img_y, is_positive,
                                        self.current_object_id, self.current_frame_idx))
                self._mark_object_dirty(self.current_object_id)

                # Track annotated frames in multi-frame annotation mode
                if self.multi_frame_annotation_mode:
                    self.annotated_frames.add(self.current_frame_idx)
                
                self.update_points_display()
                self.update_object_list()
                self.display_current_frame()
                
    def update_points_display(self):
        """Update the points display label"""
        if self.click_points:
            # Count points by object
            object_counts = {}
            for _, _, is_pos, obj_id, _ in self.click_points:
                if obj_id not in object_counts:
                    object_counts[obj_id] = {'pos': 0, 'neg': 0}
                if is_pos:
                    object_counts[obj_id]['pos'] += 1
                else:
                    object_counts[obj_id]['neg'] += 1
            
            # Create summary text
            total_points = len(self.click_points)
            current_obj_points = object_counts.get(self.current_object_id, {'pos': 0, 'neg': 0})
            obj_name = self.object_names.get(self.current_object_id, f"Obj{self.current_object_id}")
            points_text = f"Total: {total_points} | {obj_name}: +{current_obj_points['pos']}, -{current_obj_points['neg']}"
            
            # Add multi-frame annotation info
            if self.multi_frame_annotation_mode and self.annotated_frames:
                points_text += f" | Annotated frames: {len(self.annotated_frames)}"
        else:
            points_text = "No points (Left: +, Right: -)"
            if self.multi_frame_annotation_mode:
                points_text += " | Multi-frame mode active"
        
        self.points_label.config(text=points_text)
            
    def clear_current_frame_points(self):
        """Clear all click points for the current object on the current frame only"""
        removed = [p for p in self.click_points
                   if p[3] == self.current_object_id and p[4] == self.current_frame_idx]
        if not removed:
            return

        # Record original indices for undo restoration
        indices = [self.click_points.index(p) for p in removed]

        self.click_points = [p for p in self.click_points
                             if not (p[3] == self.current_object_id and p[4] == self.current_frame_idx)]

        # Update annotated frames set - remove frames that no longer have any points
        remaining_frames = {p[4] for p in self.click_points}
        self.annotated_frames = self.annotated_frames.intersection(remaining_frames)

        self._mark_object_dirty(self.current_object_id)

        # Push bulk operation to undo stack so it can be undone with Ctrl+Z
        self.undo_stack.append({
            'action': 'clear_frame',
            'points': removed,
            'indices': indices,
            'obj_id': self.current_object_id,
            'frame_idx': self.current_frame_idx,
        })
        self.redo_stack.clear()

        self.update_points_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()

    def clear_points(self):
        """Clear all click points for the current object only"""
        had_points = any(p[3] == self.current_object_id for p in self.click_points)
        # Filter out points for current object
        self.click_points = [p for p in self.click_points if p[3] != self.current_object_id]

        # Update annotated frames set - remove frames that no longer have any points
        remaining_frames = {p[4] for p in self.click_points}
        self.annotated_frames = self.annotated_frames.intersection(remaining_frames)

        # Clear undo/redo history (bulk operation, not individually undoable)
        if had_points:
            self._mark_object_dirty(self.current_object_id)
        self.undo_stack.clear()
        self.redo_stack.clear()

        self.update_points_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()
    
    def clear_current_object(self):
        """Clear points and masks for current object only"""
        had_points = any(p[3] == self.current_object_id for p in self.click_points)
        # Remove points for current object
        self.click_points = [p for p in self.click_points if p[3] != self.current_object_id]
        if had_points:
            self._mark_object_dirty(self.current_object_id)
        
        # Remove masks for current object
        for frame_idx in self.masks:
            if self.current_object_id in self.masks[frame_idx]:
                del self.masks[frame_idx][self.current_object_id]
        
        self.update_points_display()
        self.update_object_list()
        if self.frames:
            self.display_current_frame()
            
    def on_mask_opacity_change(self, value=None):
        """Handle mask opacity slider change"""
        opacity = self.mask_opacity_var.get()
        # Update label to show percentage
        self.opacity_label.config(text=f"{int(opacity * 100)}%")
        
        # Redraw frame if masks are visible
        if self.frames:
            self.display_current_frame()
            
    def prev_frame(self):
        """Go to previous frame (jump by nav step size)"""
        if self.frames and self.current_frame_idx > 0:
            self._reset_flash_state()
            self.current_frame_idx = max(0, self.current_frame_idx - self.nav_step_size.get())
            self.display_current_frame()

    def next_frame(self):
        """Go to next frame (jump by nav step size)"""
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self._reset_flash_state()
            self.current_frame_idx = min(len(self.frames) - 1, self.current_frame_idx + self.nav_step_size.get())
            self.display_current_frame()

    def _start_continuous_nav(self, direction):
        """Start continuous frame navigation on button hold"""
        # Cancel any existing navigation
        self._stop_continuous_nav()

        # Record start time for threshold detection
        self._nav_start_time = time.time()
        self._nav_direction = direction

        # Initial single frame move (immediate response)
        if direction == "prev":
            self.prev_frame()
        else:
            self.next_frame()

        # Schedule check for continuous navigation after threshold
        self._nav_check_id = self.root.after(300, self._check_continuous_nav)

    def _check_continuous_nav(self):
        """Check if button is still held and start continuous navigation"""
        # If we get here, button has been held for 300ms - start continuous mode
        self._continuous_nav()

    def _continuous_nav(self):
        """Continuously navigate frames while button is held"""
        if not hasattr(self, '_nav_direction'):
            return

        # Move to next/prev frame
        if self._nav_direction == "prev":
            if self.frames and self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.display_current_frame()
        else:  # next
            if self.frames and self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.display_current_frame()

        # Schedule next move (faster rate during continuous nav)
        self._nav_repeat_id = self.root.after(50, self._continuous_nav)

    def _stop_continuous_nav(self):
        """Stop continuous frame navigation"""
        # Cancel pending navigation callbacks
        if hasattr(self, '_nav_check_id'):
            self.root.after_cancel(self._nav_check_id)
            delattr(self, '_nav_check_id')

        if hasattr(self, '_nav_repeat_id'):
            self.root.after_cancel(self._nav_repeat_id)
            delattr(self, '_nav_repeat_id')

        if hasattr(self, '_nav_direction'):
            delattr(self, '_nav_direction')

        if hasattr(self, '_nav_start_time'):
            delattr(self, '_nav_start_time')

    def reset_video(self):
        """Reset to first frame, slider zoom, playback speed, and display zoom"""
        if self.frames:
            self.current_frame_idx = 0
            self.set_slider_zoom(1)
            self.set_playback_speed(1.0)
            self.display_zoom_level = 1.0
            self.zoom_center_img_x = 0.5
            self.zoom_center_img_y = 0.5
            self.display_current_frame()

    def jump_to_prev_annotated_frame(self):
        """Jump to the previous frame with annotations for the current object"""
        # Get frames with annotations for current object only
        current_obj_frames = set()
        for _, _, _, obj_id, frame_idx in self.click_points:
            if obj_id == self.current_object_id:
                current_obj_frames.add(frame_idx)
        
        if not current_obj_frames:
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("No Annotations", f"No annotated frames found for {obj_name}.")
            return
        
        sorted_annotated = sorted(list(current_obj_frames))
        prev_frames = [f for f in sorted_annotated if f < self.current_frame_idx]
        
        if prev_frames:
            self.current_frame_idx = prev_frames[-1]
            self.display_current_frame()
        else:
            self.current_frame_idx = sorted_annotated[-1]
            self.display_current_frame()
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            self.status_label.config(text=f"Wrapped to last annotated frame for {obj_name}")
    
    def jump_to_next_annotated_frame(self):
        """Jump to the next frame with annotations for the current object"""
        # Get frames with annotations for current object only
        current_obj_frames = set()
        for _, _, _, obj_id, frame_idx in self.click_points:
            if obj_id == self.current_object_id:
                current_obj_frames.add(frame_idx)
        
        if not current_obj_frames:
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            messagebox.showinfo("No Annotations", f"No annotated frames found for {obj_name}.")
            return
        
        sorted_annotated = sorted(list(current_obj_frames))
        next_frames = [f for f in sorted_annotated if f > self.current_frame_idx]
        
        if next_frames:
            self.current_frame_idx = next_frames[0]
            self.display_current_frame()
        else:
            self.current_frame_idx = sorted_annotated[0]
            self.display_current_frame()
            obj_name = self.object_names.get(self.current_object_id, f"Object_{self.current_object_id}")
            self.status_label.config(text=f"Wrapped to first annotated frame for {obj_name}")
            
    def set_slider_zoom(self, zoom_level):
        """Set slider zoom level for precise navigation"""
        if not self.frames:
            return

        self.slider_zoom_level.set(zoom_level)
        total_frames = len(self.frames)

        if zoom_level == 1:
            # Full range mode
            self.slider_window_center = self.current_frame_idx
            self.frame_slider.config(from_=0, to=total_frames - 1)
            self.frame_var.set(self.current_frame_idx)
            self.zoom_info_label.config(text="(Full range)")
        else:
            # Zoomed mode - slider shows window around current frame
            window_size = max(100, total_frames // zoom_level)
            self.slider_window_center = self.current_frame_idx

            # Calculate window boundaries
            half_window = window_size // 2
            window_start = max(0, self.current_frame_idx - half_window)
            window_end = min(total_frames - 1, self.current_frame_idx + half_window)

            # Adjust if at boundaries
            if window_end - window_start < window_size:
                if window_start == 0:
                    window_end = min(total_frames - 1, window_size)
                elif window_end == total_frames - 1:
                    window_start = max(0, total_frames - window_size)

            # Update slider range to show window
            self.frame_slider.config(from_=window_start, to=window_end)
            self.frame_var.set(self.current_frame_idx)
            self.zoom_info_label.config(text=f"(Frames {window_start}-{window_end})")

        self.status_label.config(text=f"Slider zoom: {zoom_level}x")
        self._update_zoom_button_highlight()

        # Re-render quality visualizations for new zoom range
        if self.inter_frame_changes and self.background_ratios:
            self._update_quality_visualizations()

    def set_playback_speed(self, speed):
        """Set playback speed multiplier"""
        self.playback_speed.set(speed)
        speed_labels = {0.25: "Very Slow", 0.5: "Slow", 1.0: "Normal", 2.0: "Fast", 4.0: "Very Fast"}
        label_text = speed_labels.get(speed, f"{speed}x")
        self.speed_info_label.config(text=f"({label_text})")
        self.status_label.config(text=f"Playback speed: {speed}x")
        self._update_speed_button_highlight()

    def _update_zoom_button_highlight(self):
        """Update zoom button styling to show current selection with prominent colors"""
        current_zoom = self.slider_zoom_level.get()
        for level, button in self.zoom_buttons.items():
            if level == current_zoom:
                button.configure(style='Selected.TButton')  # Bright teal/green background
            else:
                button.configure(style='TButton')  # Normal dark gray background


    def _update_speed_button_highlight(self):
        """Update speed button styling to show current selection with prominent colors"""
        current_speed = self.playback_speed.get()
        for speed, button in self.speed_buttons.items():
            if speed == current_speed:
                button.configure(style='Selected.TButton')  # Bright teal/green background
            else:
                button.configure(style='TButton')  # Normal dark gray background

    def set_nav_step_size(self, step):
        """Set navigation step size (frames per arrow/scroll/play tick)"""
        self.nav_step_size.set(step)
        self._update_step_button_highlight()

    def _update_step_button_highlight(self):
        """Update step size button styling to show current selection"""
        current_step = self.nav_step_size.get()
        for step, button in self.step_buttons.items():
            if step == current_step:
                button.configure(style='Selected.TButton')
            else:
                button.configure(style='TButton')

    def on_slider_change(self, value):
        """Handle frame slider change"""
        # Remove focus from any text entry widgets
        if hasattr(self, 'canvas'):
            self.canvas.focus_set()

        if self.frames and not self.playing:
            # Reset flash state when manually changing frames
            self._reset_flash_state()

            new_frame_idx = int(float(value))
            zoom_level = self.slider_zoom_level.get()

            # In zoomed mode, update window center as user navigates
            # ONLY apply delay when user is manually dragging the slider
            if zoom_level > 1 and self.slider_manual_change:
                total_frames = len(self.frames)
                window_size = max(100, total_frames // zoom_level)
                half_window = window_size // 2

                # Check if we're near window boundaries, schedule delayed window shift if needed
                current_slider_min = int(self.frame_slider.cget('from'))
                current_slider_max = int(self.frame_slider.cget('to'))

                # If navigating near edges, schedule delayed recenter
                if (new_frame_idx - current_slider_min < window_size // 10 or
                    current_slider_max - new_frame_idx < window_size // 10):

                    # Cancel any pending jump
                    if self.zoom_jump_scheduled:
                        self.root.after_cancel(self.zoom_jump_scheduled)

                    # Schedule jump after 500ms delay
                    self.zoom_jump_scheduled = self.root.after(30,
                        lambda: self._execute_zoom_jump(new_frame_idx, window_size, total_frames))

            self.current_frame_idx = new_frame_idx
            self.display_current_frame()

    def _execute_zoom_jump(self, new_frame_idx, window_size, total_frames):
        """Execute delayed zoom window jump with notification"""
        half_window = window_size // 2

        # Get old range for notification
        old_min = int(self.frame_slider.cget('from'))
        old_max = int(self.frame_slider.cget('to'))

        # Recenter window on new_frame_idx
        self.slider_window_center = new_frame_idx
        window_start = max(0, new_frame_idx - half_window)
        window_end = min(total_frames - 1, new_frame_idx + half_window)

        # Adjust window at boundaries
        if window_end - window_start < window_size:
            if window_start == 0:
                window_end = min(total_frames - 1, window_size)
            elif window_end == total_frames - 1:
                window_start = max(0, total_frames - window_size)

        # Update slider range
        self.frame_slider.config(from_=window_start, to=window_end)

        # Update zoom info label
        zoom_level = self.slider_zoom_level.get()
        self.zoom_info_label.config(text=f"(Frames {window_start}-{window_end})")

        # Flash notification
        # self._flash_zoom_jump_notification(old_min, old_max, window_start, window_end)

        # Re-render quality visualizations for new window
        if self.inter_frame_changes and self.background_ratios:
            self._update_quality_visualizations()

        self.zoom_jump_scheduled = None

    def _flash_zoom_jump_notification(self, old_min, old_max, new_min, new_max):
        """Flash status label to notify user of zoom window jump"""
        import time

        # Throttle notifications (max 1 per second)
        current_time = time.time()
        if current_time - self.last_jump_notification < 1.0:
            return
        self.last_jump_notification = current_time

        # Show jump notification
        old_text = self.status_label.cget('text')
        old_bg = self.status_label.cget('background')

        jump_msg = f"Zoom window shifted: [{old_min}-{old_max}] → [{new_min}-{new_max}]"

        # Flash 3 times (similar to flash_mask feature)
        def flash_cycle(count):
            if count <= 0:
                self.status_label.config(text=old_text, background=old_bg)
                return

            # Toggle between highlight and normal
            if count % 2 == 1:
                self.status_label.config(text=jump_msg, background='yellow')
            else:
                self.status_label.config(text=jump_msg, background=old_bg)

            self.root.after(300, lambda: flash_cycle(count - 1))

        flash_cycle(6)  # 3 full cycles (on/off)

    def toggle_play(self):
        """Toggle video playback"""
        if not self.frames:
            return

        # Stop backward playback if running
        if self.playing_backward:
            self.playing_backward = False
            self.play_backward_button.config(text="Play Rev")

        self.playing = not self.playing
        if self.playing:
            # Reset flash state when starting playback
            self._reset_flash_state()
            self.play_button.config(text="Pause")
            threading.Thread(target=self.play_video, daemon=True).start()
        else:
            self.play_button.config(text="Play")
            
    def play_video(self):
        """Play video in separate thread"""
        # Frame step = nav_step_size * speed (for speed >= 1x, skip frames to achieve speedup).
        # For speed < 1x: use nav_step_size with a longer delay to slow down.
        speed = self.playback_speed.get()
        step_size = self.nav_step_size.get()
        base_delay = 1.0 / self.video_fps  # Delay for original FPS
        if speed >= 1.0:
            frame_step = max(1, round(speed)) * step_size
            adjusted_delay = base_delay
        else:
            frame_step = step_size
            adjusted_delay = base_delay / speed

        while self.playing and self.frames:
            if self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx = min(
                    self.current_frame_idx + frame_step,
                    len(self.frames) - 1
                )
                self.root.after(0, self.display_current_frame)

                # Update slider position in zoomed mode
                if self.slider_zoom_level.get() > 1:
                    self.root.after(0, lambda: self.frame_var.set(self.current_frame_idx))

                threading.Event().wait(adjusted_delay)
            else:
                self.playing = False
                self.root.after(0, lambda: self.play_button.config(text="Play"))
                break
    
    def toggle_play_backward(self):
        """Toggle reverse video playback"""
        if not self.frames:
            return

        # Stop forward playback if running
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")

        self.playing_backward = not self.playing_backward
        if self.playing_backward:
            self._reset_flash_state()
            self.play_backward_button.config(text="Pause Rev")
            threading.Thread(target=self.play_video_backward, daemon=True).start()
        else:
            self.play_backward_button.config(text="Play Rev")

    def play_video_backward(self):
        """Play video in reverse in a separate thread"""
        speed = self.playback_speed.get()
        step_size = self.nav_step_size.get()
        base_delay = 1.0 / self.video_fps
        if speed >= 1.0:
            frame_step = max(1, round(speed)) * step_size
            adjusted_delay = base_delay
        else:
            frame_step = step_size
            adjusted_delay = base_delay / speed

        while self.playing_backward and self.frames:
            if self.current_frame_idx > 0:
                self.current_frame_idx = max(0, self.current_frame_idx - frame_step)
                self.root.after(0, self.display_current_frame)

                if self.slider_zoom_level.get() > 1:
                    self.root.after(0, lambda: self.frame_var.set(self.current_frame_idx))

                threading.Event().wait(adjusted_delay)
            else:
                self.playing_backward = False
                self.root.after(0, lambda: self.play_backward_button.config(text="Play Rev"))
                break

    def _detect_available_gpus(self):
        """Detect available GPUs and return list of options"""
        gpu_options = []
        
        try:
            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_options.append("auto")  # Let PyTorch choose
                gpu_options.append("cpu")   # Force CPU
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    gpu_options.append(f"cuda:{i} ({gpu_name} - {gpu_memory:.1f}GB)")
            else:
                gpu_options.append("cpu")
                gpu_options.append("auto")
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            gpu_options = ["cpu", "auto"]
        
        return gpu_options
    
    def _get_selected_device(self):
        """Get the selected device string and convert to PyTorch device"""
        selected = self.selected_gpu.get()

        if selected == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif selected == "cpu":
            return "cpu"
        elif "cuda:" in selected:
            # Extract cuda device from display string like "cuda:0 (GPU Name - 24.0GB)"
            # This handles both plain "cuda:0" and formatted "cuda:0 (...)" strings
            device_part = selected.split("cuda:")[1].split(" ")[0]
            return f"cuda:{device_part}"
        else:
            return "cpu"
    
    def _show_three_button_dialog(self, title, message, button1_text, button2_text, button3_text="Cancel"):
        """
        Show a custom dialog with 3 buttons

        Returns:
            0: Button 1 clicked
            1: Button 2 clicked
            None: Button 3 (Cancel) clicked or dialog closed
        """
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("500x250")
        dialog.resizable(False, False)

        # Result variable
        result = [None]

        # Message frame
        message_frame = ttk.Frame(dialog, padding=20)
        message_frame.pack(fill=tk.BOTH, expand=True)

        message_label = ttk.Label(message_frame, text=message, wraplength=450, justify=tk.LEFT)
        message_label.pack(expand=True)

        # Button frame
        button_frame = ttk.Frame(dialog, padding=(20, 0, 20, 20))
        button_frame.pack(fill=tk.X)

        def on_button1():
            result[0] = 0
            dialog.destroy()

        def on_button2():
            result[0] = 1
            dialog.destroy()

        def on_button3():
            result[0] = None
            dialog.destroy()

        ttk.Button(button_frame, text=button1_text, command=on_button1, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=button2_text, command=on_button2, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=button3_text, command=on_button3, width=15).pack(side=tk.RIGHT, padx=5)

        # Handle window close button
        dialog.protocol("WM_DELETE_WINDOW", on_button3)

        # Wait for dialog to close
        self.root.wait_window(dialog)

        return result[0]

    def on_gpu_selection_change(self, event=None):
        """Handle GPU selection change"""
        device = self._get_selected_device()
        self.gpu_device = device
        self._update_gpu_info_display()
        
        # Update status
        self.status_label.config(text=f"GPU selection changed to: {device}")
        
        # If model is already loaded, warn user that they need to reload
        if hasattr(self, 'sam2_model') and self.sam2_model is not None:
            messagebox.showwarning(
                "Model Reload Required",
                f"GPU selection changed to: {device}\n\n"
                f"The SAM2 model is currently loaded on a different device.\n"
                f"Please reload the model to use the new GPU selection."
            )
    
    def _update_gpu_info_display(self):
        """Update the GPU info display"""
        try:
            device = self._get_selected_device()
            
            if device == "cpu":
                info_text = "Using CPU (slower but works on any system)"
            elif device == "cuda" or device == "cuda:0":
                if torch and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    info_text = f"Auto-selected: {gpu_name} ({gpu_memory:.1f}GB)"
                else:
                    info_text = "Auto-selected: CPU (CUDA not available)"
            elif device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if torch and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    info_text = f"Selected: {gpu_name} ({gpu_memory:.1f}GB)"
                else:
                    info_text = f"Selected: GPU {gpu_id} (not available)"
            else:
                info_text = f"Selected: {device}"
                
            self.gpu_info_label.config(text=info_text)
            
        except Exception as e:
            self.gpu_info_label.config(text=f"Error: {str(e)}")

    def _detect_available_models(self, model_type=None):
        """Detect available SAM2/SAM3 models from checkpoints

        Args:
            model_type: 'SAM2' or 'SAM3'. If None, defaults to current selection.
        """
        from pathlib import Path

        # Determine which model type to detect
        if model_type is None:
            # Check if attributes exist (may be called during initialization)
            if hasattr(self, 'sam3_available') and hasattr(self, 'model_type_var'):
                model_type = self.model_type_var.get() if self.sam3_available else "SAM2"
            else:
                model_type = "SAM2"  # Default to SAM2 during initialization

        # SAM3 has one or two variants depending on which checkpoints are downloaded
        if model_type == "SAM3":
            sam3_checkpoint_dir = Path(self.checkpoint_dir).parent.parent / "sam3" / "checkpoints"
            models = ["auto"]
            if sam3_checkpoint_dir.exists():
                if (sam3_checkpoint_dir / "sam3.pt").exists():
                    models.append("SAM3|sam3.pt")
            return models

        # SAM2 detection (original logic)
        checkpoint_dir = Path(self.checkpoint_dir)
        if not checkpoint_dir.exists():
            return ["auto"]

        models = []

        # Model mapping: checkpoint filename -> (display name, config path)
        model_mapping = {
            # SAM2.1 models (preferred)
            "sam2.1_hiera_tiny.pt": ("SAM2.1 Tiny (fastest)", "sam2.1/sam2.1_hiera_t.yaml"),
            "sam2.1_hiera_small.pt": ("SAM2.1 Small", "sam2.1/sam2.1_hiera_s.yaml"),
            "sam2.1_hiera_base_plus.pt": ("SAM2.1 Base+", "sam2.1/sam2.1_hiera_b+.yaml"),
            "sam2.1_hiera_large.pt": ("SAM2.1 Large (best)", "sam2.1/sam2.1_hiera_l.yaml"),

            # SAM2 legacy models
            "sam2_hiera_tiny.pt": ("SAM2 Tiny", "sam2/sam2_hiera_t.yaml"),
            "sam2_hiera_small.pt": ("SAM2 Small", "sam2/sam2_hiera_s.yaml"),
            "sam2_hiera_base_plus.pt": ("SAM2 Base+", "sam2/sam2_hiera_b+.yaml"),
            "sam2_hiera_large.pt": ("SAM2 Large", "sam2/sam2_hiera_l.yaml"),
        }

        models.append("auto")  # Auto-selection option

        for checkpoint_file, (display_name, config_path) in model_mapping.items():
            checkpoint_path = checkpoint_dir / checkpoint_file
            full_config_path = Path(self.config_dir) / config_path

            if checkpoint_path.exists() and full_config_path.exists():
                models.append(f"{display_name}|{checkpoint_file}|{config_path}")

        return models if len(models) > 1 else ["auto"]

    def _auto_select_best_model(self):
        """Auto-select best available model based on GPU memory"""
        models = self.available_models[1:]  # Skip "auto" option

        if not models:
            return None

        # Preference: SAM2.1 models first, then larger models for better quality
        # Adjust based on available GPU memory
        preference_order = [
            # SAM2.1 models (preferred)
            "sam2.1_hiera_large.pt",       # Best quality
            "sam2.1_hiera_base_plus.pt",   # High quality
            "sam2.1_hiera_small.pt",       # Good balance
            "sam2.1_hiera_tiny.pt",        # Fastest
            # SAM2 legacy models (fallback)
            "sam2_hiera_large.pt",
            "sam2_hiera_base_plus.pt",
            "sam2_hiera_small.pt",
            "sam2_hiera_tiny.pt",
        ]

        # Check GPU memory and adjust preference
        try:
            if torch and torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Adjust based on GPU memory
                if gpu_mem_gb >= 8:
                    # Prefer large models (best quality)
                    pass  # Keep original order
                elif gpu_mem_gb >= 4:
                    # Prefer base+ and small models
                    preference_order = [p for p in preference_order if 'large' not in p]
                else:
                    # Low memory: prefer small and tiny models
                    preference_order = [p for p in preference_order if 'large' not in p and 'base' not in p]
        except:
            # Default to base+ if can't detect GPU
            preference_order = [p for p in preference_order if 'large' not in p]

        # Find first available model in preference order
        for preferred in preference_order:
            for model in models:
                if preferred in model:
                    return model

        # Fallback to first available
        return models[0]

    def _format_model_list(self):
        """Format model list for display in combobox"""
        formatted = ["auto"]  # Match internal value for consistency
        for model in self.available_models[1:]:  # Skip first "auto"
            display_name = model.split('|')[0]
            formatted.append(display_name)
        return formatted

    def on_model_type_change(self):
        """Handle model type change between SAM2 and SAM3"""
        new_type = self.model_type_var.get()

        # Warn if model is already loaded
        if self.model_loaded:
            result = messagebox.askokcancel(
                "Change Model Type",
                f"Switching to {new_type} will clear the current model and all annotations.\n\n"
                "Click OK to continue or Cancel to keep current model.",
                icon='warning'
            )
            if not result:
                # Revert to previous selection
                old_type = "SAM3" if new_type == "SAM2" else "SAM2"
                self.model_type_var.set(old_type)
                return

            # Clear current model
            self.sam2_model = None
            self.model_loaded = False
            self.using_sam3 = False
            self.inference_state = None
            self.current_model_info = None
            self.model_status_label.config(text="No model loaded", foreground='red')

        # Update model dropdown to show appropriate models for selected type
        self.available_models = self._detect_available_models(model_type=new_type)
        self.model_combo['values'] = self._format_model_list()

        # Auto-select the first available model (auto or the only model for SAM3)
        if self.available_models:
            self.selected_model.set(self._format_model_list()[0])

        status_msg = f"{new_type} selected. Load a model to begin."
        if new_type == "SAM3":
            status_msg += " (Single variant - uses HuggingFace model)"
        self.status_label.config(text=status_msg)

        

    def on_model_selection_change(self, event=None):
        """Handle model selection change"""
        # Get selected display name
        selected_display = self.model_combo.get()

        if selected_display == "auto":
            self.selected_model.set("auto")
        else:
            # Store only the display name so the combo shows a clean label.
            # The full internal "Display|ckpt|cfg" string is looked up from
            # self.available_models at load time.
            self.selected_model.set(selected_display)

        # Warn user if model already loaded
        if hasattr(self, 'sam2_model') and self.sam2_model is not None:
            response = messagebox.askokcancel(
                "Model Change",
                "Changing the model requires reloading.\n\n"
                "This will clear current segmentation state.\n\n"
                "Click OK to continue or Cancel to keep current model.",
                icon='warning'
            )
            if response:
                self.load_sam2_model()
            else:
                # Revert selection
                if self.current_model_info:
                    display_name = self.current_model_info.split('|')[0]
                    self.model_combo.set(display_name)

    def _ensure_model_dtype_consistency(self):
        """Ensure model is using consistent dtypes to avoid BFloat16/Float mismatches"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Set model to float32 precision
                if hasattr(self.sam2_model, 'model'):
                    self.sam2_model.model = self.sam2_model.model.to(dtype=torch.float32)
                    self.sam2_model.model.eval()
                
                # Force all model parameters to float32
                if hasattr(self.sam2_model, 'model'):
                    for param in self.sam2_model.model.parameters():
                        param.data = param.data.to(dtype=torch.float32)
                
                # Disable autocast to prevent mixed precision issues
                if hasattr(torch, 'autocast'):
                    # This will be handled in the inference calls
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not ensure dtype consistency: {e}")
    
    def _disable_autocast_for_inference(self):
        """Disable autocast during inference to prevent dtype mismatches"""
        try:
            # Get the device type for autocast
            device = self._get_selected_device()
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            
            # Create a context manager that disables autocast
            if hasattr(torch, 'autocast'):
                # In PyTorch 1.10+, use torch.autocast with device_type
                return torch.autocast(device_type=device_type, enabled=False)
            elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                # Fallback for older PyTorch versions with CUDA autocast
                return torch.amp.autocast(enabled=False)
            else:
                # Fallback to no_grad
                return torch.no_grad()
        except Exception as e:
            print(f"Warning: Could not create autocast context: {e}")
            # Return a simple no_grad context as fallback
            return torch.no_grad()
    
    def _force_model_float32(self):
        """Force all model components to float32 to prevent dtype mismatches"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Force all model parameters to float32
                if hasattr(self.sam2_model, 'model'):
                    for name, param in self.sam2_model.model.named_parameters():
                        if param.dtype != torch.float32:
                            param.data = param.data.to(dtype=torch.float32)
                    
                    # Also force buffers to float32
                    for name, buffer in self.sam2_model.model.named_buffers():
                        if buffer.dtype != torch.float32:
                            buffer.data = buffer.data.to(dtype=torch.float32)
                
                # Force any other model components
                for attr_name in dir(self.sam2_model):
                    if not attr_name.startswith('_'):
                        attr = getattr(self.sam2_model, attr_name)
                        if hasattr(attr, 'to') and hasattr(attr, 'dtype'):
                            try:
                                if attr.dtype != torch.float32:
                                    setattr(self.sam2_model, attr_name, attr.to(dtype=torch.float32))
                            except:
                                pass  # Skip if not convertible
                
                # Force the entire model to float32 recursively
                self._recursive_float32_conversion(self.sam2_model)
                                
        except Exception as e:
            print(f"Warning: Could not force model to float32: {e}")
    
    def _recursive_float32_conversion(self, obj):
        """Recursively convert all tensors in an object to float32"""
        try:
            if hasattr(obj, 'to') and hasattr(obj, 'dtype'):
                if obj.dtype != torch.float32:
                    obj.data = obj.data.to(dtype=torch.float32)
            elif hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        self._recursive_float32_conversion(attr_value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    self._recursive_float32_conversion(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    self._recursive_float32_conversion(value)
        except Exception as e:
            pass  # Skip if conversion fails
    
    def _patch_model_for_float32(self):
        """Patch the model to force float32 operations"""
        try:
            if hasattr(self, 'sam2_model') and self.sam2_model is not None:
                # Patch the model's forward method to force float32
                if hasattr(self.sam2_model, 'model'):
                    original_forward = self.sam2_model.model.forward
                    
                    def float32_forward(*args, **kwargs):
                        # Convert all tensor inputs to float32
                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                if arg.dtype != torch.float32:
                                    new_args.append(arg.to(dtype=torch.float32))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)
                        
                        # Convert tensor kwargs to float32
                        new_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor):
                                if value.dtype != torch.float32:
                                    new_kwargs[key] = value.to(dtype=torch.float32)
                                else:
                                    new_kwargs[key] = value
                            else:
                                new_kwargs[key] = value
                        
                        # Call original forward with float32 tensors
                        return original_forward(*new_args, **new_kwargs)
                    
                    # Replace the forward method
                    self.sam2_model.model.forward = float32_forward
                    
                    print("Model forward method patched for float32")
                    
        except Exception as e:
            print(f"Warning: Could not patch model for float32: {e}")
    
    def _disable_mixed_precision_globally(self):
        """Disable mixed precision globally to prevent dtype issues"""
        try:
            # Set environment variables to disable mixed precision
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Disable autocast globally
            if hasattr(torch, 'autocast'):
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                
            print("Mixed precision disabled globally")
            
        except Exception as e:
            print(f"Warning: Could not disable mixed precision globally: {e}")
    
    def _force_float32_context(self):
        """Context manager that forces all operations to float32"""
        class Float32Context:
            def __init__(self, parent):
                self.parent = parent
                self.original_autocast = None
                
            def __enter__(self):
                # Disable autocast completely
                if hasattr(torch, 'autocast'):
                    self.original_autocast = torch.autocast(enabled=False)
                    self.original_autocast.__enter__()
                
                # Force model to float32
                self.parent._force_model_float32()
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore autocast
                if self.original_autocast:
                    self.original_autocast.__exit__(exc_type, exc_val, exc_tb)
        
        return Float32Context(self)
    
    def show_frame_points(self):
        """Show a list of annotation points on the current frame"""
        if not self.click_points:
            messagebox.showinfo("No Points", "No annotation points found on any frame.")
            return
        
        # Get points on current frame
        current_frame_points = [point for point in self.click_points 
                              if point[4] == self.current_frame_idx]
        
        if not current_frame_points:
            messagebox.showinfo("No Points", f"No annotation points found on frame {self.current_frame_idx + 1}.")
            return
        
        # Create a simple dialog showing the points
        points_info = f"Annotation points on frame {self.current_frame_idx + 1}:\n\n"
        
        for i, (x, y, is_pos, obj_id, frame_idx) in enumerate(current_frame_points):
            point_type = "Positive" if is_pos else "Negative"
            obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
            points_info += f"{i+1}. {point_type} point for {obj_name} at ({x:.0f}, {y:.0f})\n"
        
        points_info += f"\nTotal: {len(current_frame_points)} points on this frame"
        points_info += f"\nTotal: {len(self.click_points)} points on all frames"
        
        messagebox.showinfo("Frame Points", points_info)
    
    def _prepare_tensors_for_inference(self, points, labels):
        """Prepare points and labels tensors with consistent dtype and device"""
        try:
            device = self._get_selected_device()
            
            # Convert to tensors with consistent dtype
            if isinstance(points, (list, np.ndarray)):
                points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            else:
                points_tensor = points.to(dtype=torch.float32, device=device)
                
            if isinstance(labels, (list, np.ndarray)):
                labels_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
            else:
                labels_tensor = labels.to(dtype=torch.int64, device=device)
            
            # Ensure tensors are contiguous and have correct shape
            points_tensor = points_tensor.contiguous()
            labels_tensor = labels_tensor.contiguous()
            
            # Ensure points are 2D (N, 2) and labels are 1D (N,)
            if len(points_tensor.shape) == 1:
                points_tensor = points_tensor.unsqueeze(0)
            if len(labels_tensor.shape) == 0:
                labels_tensor = labels_tensor.unsqueeze(0)
            
            return points_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error preparing tensors: {e}")
            # Fallback to original values
            return points, labels
    
    

    def segment_video(self):
        """Segment video using SAM2/SAM3 model via VideoSegmenter."""
        if not self.frames:
            messagebox.showwarning("Warning", "Please load a video first")
            return

        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("Warning", "Please load a SAM model first")
            return

        if not self.click_points:
            messagebox.showwarning("Warning", "Please add some click points first")
            return

        # Validate annotations
        is_valid, error_msg = self._validate_annotations_before_segmentation()
        if not is_valid:
            messagebox.showerror("Invalid Annotations",
                            f"{error_msg}\n\n"
                            f"This usually happens when:\n"
                            f"1. Annotations were saved with different video settings\n"
                            f"2. Video was reloaded with frame skipping enabled\n\n"
                            f"Please reload the video with the same settings used when creating annotations, "
                            f"or create new annotations for the current video.")
            return

        # Clear any temporary single-frame masks before full segmentation
        self._clear_temporary_masks()

        # Ask for output directory for results
        initial_dir = os.path.dirname(self.last_export_dir) if self.last_export_dir else self.working_dir
        output_base_dir = filedialog.askdirectory(
            title="Select Output Directory for Segmentation Results (Cancel to abort)",
            initialdir=initial_dir
        )

        if not output_base_dir:
            return  # User cancelled

        # Save this directory for next time
        self._save_last_export_dir(output_base_dir)

        # Create output directory structure
        try:
            os.makedirs(output_base_dir, exist_ok=True)
            masks_output_dir = os.path.join(output_base_dir, "masks")
            os.makedirs(masks_output_dir, exist_ok=True)
            print(f"Output directory: {output_base_dir}")
            print(f"Masks will be saved to: {masks_output_dir}")
        except Exception as e:
            messagebox.showerror("Directory Error", f"Failed to create output directories: {e}")
            return

        # Check if only-updated mode is applicable
        only_updated = False
        unchanged_ids: set = set()
        prev_masks_dir = None
        if (self.updated_objects
                and hasattr(self, 'mask_export_dir')
                and self.mask_export_dir
                and os.path.isdir(self.mask_export_dir)):
            all_obj_ids = set(self.object_names.keys())
            candidates_for_reuse = all_obj_ids - self.updated_objects
            if candidates_for_reuse:
                answer = messagebox.askyesnocancel(
                    "Partial Re-segmentation",
                    f"{len(self.updated_objects)} object(s) have updated annotations "
                    f"(IDs: {', '.join(str(i) for i in sorted(self.updated_objects))}).\n\n"
                    f"Re-segment only updated objects and reuse existing masks for "
                    f"unchanged object(s) (IDs: {', '.join(str(i) for i in sorted(candidates_for_reuse))})"
                    f"?\n\n"
                    f"[Yes] Re-segment updated objects only\n"
                    f"[No] Re-segment all objects\n"
                    f"[Cancel] Abort"
                )
                if answer is None:  # Cancel
                    return
                if answer:  # Yes
                    only_updated = True
                    unchanged_ids = candidates_for_reuse
                    prev_masks_dir = self.mask_export_dir
                    print(f"Partial re-segmentation: updating {sorted(self.updated_objects)}, "
                          f"reusing masks for {sorted(unchanged_ids)}")

        # Determine output mask format.
        # Priority order:
        # 1. Refinement mode (prev_masks_dir set): auto-detect from previous masks dir.
        # 2. Chosen output dir already has results: warn + force existing format.
        # 3. Fresh empty dir: ask the user.
        _chosen_masks_dir = Path(masks_output_dir)
        _existing_npz = list(_chosen_masks_dir.glob("masks_f*.npz")) if _chosen_masks_dir.exists() else []
        _existing_png = list(_chosen_masks_dir.glob("mask_f*.png")) if _chosen_masks_dir.exists() else []
        _chosen_has_results = bool(_existing_npz or _existing_png)

        if prev_masks_dir and os.path.isdir(prev_masks_dir):
            # Refinement: preserve the format of the previous result
            _has_npz = any(Path(prev_masks_dir).glob("masks_f*.npz"))
            _has_png = any(Path(prev_masks_dir).glob("mask_f*.png"))
            _mask_format = "npz" if (_has_npz and not _has_png) else "png"
            print(f"Mask format auto-detected from previous results: {_mask_format.upper()}")
        elif _chosen_has_results:
            # Output dir already contains results — force existing format and warn
            _mask_format = "npz" if (_existing_npz and not _existing_png) else "png"
            proceed = messagebox.askyesno(
                "Directory Already Has Results",
                f"The selected output directory already contains segmentation results "
                f"({_mask_format.upper()} format).\n\n"
                f"Continuing will overwrite the existing files using the same format "
                f"({_mask_format.upper()}).\n\n"
                f"Continue?"
            )
            if not proceed:
                return
            print(f"Mask format forced to match existing results: {_mask_format.upper()}")
        else:
            _use_npz = messagebox.askyesno(
                "Mask Output Format",
                "Save masks as NPZ bundles?\n\n"
                "[Yes] NPZ  - one compressed file per frame (~5x smaller)\n"
                "[No]  PNG  - one file per object per frame (default)"
            )
            _mask_format = "npz" if _use_npz else "png"
            print(f"Mask format chosen by user: {_mask_format.upper()}")

        try:
            self.status_label.config(text="Preparing for segmentation...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()

            # Check if we need to use original video for re-segmentation
            segmentation_video_path = None
            if self.segmented_video_displayed:
                segmentation_video_path = self._get_original_video_for_resegmentation()
                if not segmentation_video_path:
                    return  # User cancelled

            source_video = segmentation_video_path or self.video_path
            total_frames = len(self.frames)
            frames_to_process = list(range(0, total_frames))
            print(f"Processing full video: {len(frames_to_process)} frames at original resolution")

            # Get or reuse session-based cache directory for frames
            temp_dir = self._get_session_cache_dir(source_video)
            export_dir = masks_output_dir

            try:
                # Calculate coordinate scaling for re-segmentation
                coord_scale_x = 1.0
                coord_scale_y = 1.0
                if segmentation_video_path:
                    cap_temp = cv2.VideoCapture(segmentation_video_path)
                    orig_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap_temp.release()

                    dims = self._get_frame_dimensions()
                    if dims is None:
                        raise ValueError("Cannot determine segmented video dimensions")
                    seg_height, seg_width = dims
                    coord_scale_x = orig_width / seg_width
                    coord_scale_y = orig_height / seg_height

                    print(f"Re-segmentation coordinate scaling: {coord_scale_x:.3f}x (width), {coord_scale_y:.3f}x (height)")

                # Convert click_points to PointAnnotation objects with coordinate scaling
                annotations = []
                for x, y, is_pos, obj_id, frame_idx in self.click_points:
                    if frame_idx not in frames_to_process:
                        continue
                    if only_updated and obj_id not in self.updated_objects:
                        continue

                    # Get sequential index for SAM
                    sequential_frame_idx = frames_to_process.index(frame_idx)

                    # Apply coordinate scaling for re-segmentation
                    scaled_x = x * coord_scale_x
                    scaled_y = y * coord_scale_y

                    annotations.append(PointAnnotation(
                        x=scaled_x,
                        y=scaled_y,
                        is_positive=bool(is_pos),
                        object_id=obj_id,
                        frame_idx=sequential_frame_idx,
                        object_name=self.object_names.get(obj_id, f"Object_{obj_id}")
                    ))

                # Create segmentation config
                config = SegmentationConfig(
                    output_dir=output_base_dir,
                    masks_subdir="masks",
                    frame_dir=temp_dir,  # Use session cache
                    enable_backward_propagation=True,
                    frames_to_keep=20,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    calculate_quality_metrics=True,
                    cleanup_temp_frames=False,  # Don't cleanup session cache
                    exclusive_masks=self.exclusive_masks_var.get(),
                    mask_format=_mask_format
                )

                # Create progress callback
                progress_callback = TkProgressCallback(
                    self.root, self.progress_var, self.status_label
                )

                # Get device
                device = self._get_selected_device()
                self.device = device  # Store for use in export functions

                # Use the bfloat16 setting from model initialization
                use_bfloat16 = getattr(self, 'use_bfloat16', False)

                # Create VideoSegmenter
                segmenter = VideoSegmenter(
                    predictor=self.sam2_model,
                    device=device,
                    use_bfloat16=use_bfloat16
                )
                segmenter.set_progress_callback(progress_callback)

                # Run segmentation
                result = segmenter.segment(
                    video_path=source_video,
                    annotations=annotations,
                    object_names=self.object_names,
                    object_colors=self.object_colors,
                    config=config
                )

                # Store inference state for potential reuse
                self.inference_state = result.inference_state

                # Convert result to UI's expected format
                mask_metadata = []
                self.masks = {}
                for frame_idx, frame_data in result.masks_metadata.items():
                    for obj_id, mask_info in frame_data.items():
                        entry = {
                            'frame_idx': frame_idx,
                            'obj_id': obj_id,
                            'mask_file': mask_info['filename'],
                            'object_name': mask_info['name'],
                            'color': mask_info['color']
                        }
                        if 'npz_key' in mask_info:
                            entry['npz_key'] = mask_info['npz_key']
                        mask_metadata.append(entry)
                        if frame_idx not in self.masks:
                            self.masks[frame_idx] = {}
                        self.masks[frame_idx][obj_id] = None  # Placeholder

                # Merge unchanged masks from previous directory (only-updated mode)
                if only_updated and unchanged_ids and prev_masks_dir:
                    import re as _re
                    import shutil as _shutil
                    _prev_path = Path(prev_masks_dir)
                    _new_path = Path(export_dir)
                    _merged = 0
                    _png_pat = _re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)\.png$")
                    _npz_key_pat = _re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)$")

                    # Scan previous masks dir for unchanged-object data.
                    # Store (obj_name, src_png_or_None, lazy_npz_ref_or_None) per (fidx, oid).
                    _to_restore: dict = {}

                    # PNG scan
                    for _fp in sorted(_prev_path.glob("mask_f*.png")):
                        _m = _png_pat.match(_fp.name)
                        if not _m:
                            continue
                        _oid = int(_m.group(3))
                        if _oid not in unchanged_ids:
                            continue
                        _fidx, _oname = int(_m.group(1)), _m.group(2)
                        _to_restore.setdefault(_fidx, {})[_oid] = (_oname, _fp, None)

                    # NPZ scan — lazy (path, key) refs, never load arrays during scan
                    for _npz_fp in sorted(_prev_path.glob("masks_f*.npz")):
                        _fn_m = _re.match(r"^masks_f(\d{6})\.npz$", _npz_fp.name)
                        if not _fn_m:
                            continue
                        _fidx = int(_fn_m.group(1))
                        try:
                            _data = np.load(str(_npz_fp))
                            for _key in _data.files:
                                _km = _npz_key_pat.match(_key)
                                if not _km:
                                    continue
                                _oid = int(_km.group(3))
                                if _oid not in unchanged_ids:
                                    continue
                                if _fidx in _to_restore and _oid in _to_restore[_fidx]:
                                    continue  # PNG wins
                                _oname = _km.group(2)
                                _to_restore.setdefault(_fidx, {})[_oid] = (_oname, None, (_npz_fp, _key))
                        except Exception:
                            continue

                    # Write unchanged masks directly in the output format (_mask_format).
                    # PNG format: copy PNG → PNG.
                    # NPZ format: accumulate per-frame arrays, write one NPZ bundle per frame
                    #             (segmenter already wrote the updated objects as NPZ, so we
                    #              need to add unchanged objects into those same bundles or new ones).
                    #
                    # For NPZ output: group all unchanged objects by frame, then merge with the
                    # already-written NPZ bundle for that frame (which holds updated objects).
                    # One frame in RAM at a time → memory efficient.

                    if _mask_format == "npz":
                        # Group unchanged objects by frame
                        _by_frame: dict = {}  # fidx -> {oid: (oname, src_png, lazy_ref)}
                        for _fidx, _obj_map in _to_restore.items():
                            for _oid, _entry in _obj_map.items():
                                _by_frame.setdefault(_fidx, {})[_oid] = _entry

                        for _fidx, _obj_map in sorted(_by_frame.items()):
                            _npz_out = _new_path / f"masks_f{_fidx:06d}.npz"
                            # Load existing bundle (updated objects written by segmenter)
                            _bundle: dict = {}
                            if _npz_out.exists():
                                try:
                                    _existing = np.load(str(_npz_out))
                                    _bundle = {k: _existing[k] for k in _existing.files}
                                except Exception:
                                    pass
                            # Add each unchanged object
                            for _oid, (_oname, _src_png, _lazy_ref) in _obj_map.items():
                                _npz_key = f"mask_f{_fidx:06d}_{_oname}_id{_oid}"
                                if _src_png is not None:
                                    _arr = cv2.imread(str(_src_png), cv2.IMREAD_GRAYSCALE)
                                    if _arr is not None:
                                        _bundle[_npz_key] = _arr
                                elif _lazy_ref is not None:
                                    _src_npz_fp, _src_key = _lazy_ref
                                    try:
                                        _sd = np.load(str(_src_npz_fp))
                                        if _src_key in _sd.files:
                                            _bundle[_npz_key] = (_sd[_src_key] > 0).astype(np.uint8) * 255
                                    except Exception:
                                        pass
                                if _npz_key in _bundle:
                                    mask_metadata.append({
                                        'frame_idx': _fidx, 'obj_id': _oid,
                                        'mask_file': f"masks_f{_fidx:06d}.npz",
                                        'npz_key': _npz_key,
                                        'object_name': self.object_names.get(_oid, _oname),
                                        'color': self.object_colors.get(_oid, (255, 0, 0)),
                                    })
                                    if _fidx not in self.masks:
                                        self.masks[_fidx] = {}
                                    self.masks[_fidx][_oid] = None
                                    _merged += 1
                            # Write merged bundle
                            if _bundle:
                                np.savez_compressed(str(_npz_out), **_bundle)
                    else:
                        # PNG format: copy unchanged PNG → output dir as PNG
                        for _fidx, _obj_map in _to_restore.items():
                            for _oid, (_oname, _src_png, _lazy_ref) in _obj_map.items():
                                if _src_png is not None:
                                    _dst_name = f"mask_f{_fidx:06d}_{_oname}_id{_oid}.png"
                                    _dst = _new_path / _dst_name
                                    if _prev_path.resolve() != _new_path.resolve() and not _dst.exists():
                                        _shutil.copy2(str(_src_png), str(_dst))
                                    mask_metadata.append({
                                        'frame_idx': _fidx, 'obj_id': _oid,
                                        'mask_file': _dst_name,
                                        'object_name': self.object_names.get(_oid, _oname),
                                        'color': self.object_colors.get(_oid, (255, 0, 0)),
                                    })
                                    if _fidx not in self.masks:
                                        self.masks[_fidx] = {}
                                    self.masks[_fidx][_oid] = None
                                    _merged += 1
                                # NPZ source with PNG output: load and write as PNG
                                elif _lazy_ref is not None:
                                    _src_npz_fp, _src_key = _lazy_ref
                                    try:
                                        _sd = np.load(str(_src_npz_fp))
                                        if _src_key in _sd.files:
                                            _arr = (_sd[_src_key] > 0).astype(np.uint8) * 255
                                            _dst_name = f"mask_f{_fidx:06d}_{_oname}_id{_oid}.png"
                                            cv2.imwrite(str(_new_path / _dst_name), _arr)
                                            mask_metadata.append({
                                                'frame_idx': _fidx, 'obj_id': _oid,
                                                'mask_file': _dst_name,
                                                'object_name': self.object_names.get(_oid, _oname),
                                                'color': self.object_colors.get(_oid, (255, 0, 0)),
                                            })
                                            if _fidx not in self.masks:
                                                self.masks[_fidx] = {}
                                            self.masks[_fidx][_oid] = None
                                            _merged += 1
                                    except Exception:
                                        pass

                    print(f"Reused {_merged} masks for unchanged objects {sorted(unchanged_ids)} from {_prev_path}")

                # Store export directory
                self.mask_export_dir = export_dir

                # Get quality metrics from result.
                # If only-updated mode was used, result.quality_metrics only covers the updated
                # objects; force recalculation from the full merged self.masks instead.
                quality_metrics = None if only_updated else result.quality_metrics
                propagation_success = len(mask_metadata) > 0

                # Save metadata and temporary directory
                metadata_path = os.path.join(export_dir, "segmentation_metadata.json")
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump({
                        'video_path': self.video_path,
                        'total_frames': len(self.frames),
                        'objects': {obj_id: self.object_names.get(obj_id, f"Object {obj_id}")
                                   for obj_id in self.object_names.keys()},
                        'masks': mask_metadata
                    }, f, indent=2)

                # Store export directory for later use (video export, cleanup)
                self.mask_export_dir = export_dir

                # Only show success messages if propagation actually completed
                if propagation_success and len(mask_metadata) > 0:
                    print(f"\nMasks saved to temporary directory: {export_dir}")
                    print(f"Metadata saved to: {metadata_path}")
                elif len(mask_metadata) > 0:
                    print(f"\n[WARNING] Propagation incomplete. Partial results ({len(mask_metadata)} masks) saved to: {export_dir}")
                    print(f"Metadata saved to: {metadata_path}")
                else:
                    print(f"\n[ERROR] No masks generated. Check errors above.")

                # For backward compatibility, populate self.masks with empty dicts
                # (actual masks will be loaded from disk on demand)
                if not hasattr(self, 'masks'):
                    self.masks = {}
                for item in mask_metadata:
                    frame_idx = item['frame_idx']
                    obj_id = item['obj_id']
                    if frame_idx not in self.masks:
                        self.masks[frame_idx] = {}
                    # Store a placeholder - actual mask will be loaded from disk when needed
                    self.masks[frame_idx][obj_id] = None

                self.progress_bar.pack_forget()

                # Count results
                total_masks = sum(len(frame_masks) for frame_masks in self.masks.values())
                unique_objects = set()
                for frame_masks in self.masks.values():
                    unique_objects.update(frame_masks.keys())

                # Only proceed with export if propagation succeeded and we have masks
                if propagation_success and total_masks > 0:
                    self.status_label.config(text=f"Segmentation complete! Generated {total_masks} masks for {len(unique_objects)} objects")
                    # Multi-frame annotation mode stays active for continued annotation
                    self.updated_objects.clear()  # All objects are now up to date

                    # Enable Flash Mask button now that segmentation is complete
                    self._enable_flash_mask_button()

                    self.update_object_list()
                    self.display_current_frame()

                    # Export segmented video and metadata
                    try:
                        # Prepare annotations data (matching sam2_process.py format)
                        annotations_data = {
                            "video_path": self.video_path,
                            "total_frames": len(self.frames),
                            "object_names": self.object_names,
                            "object_colors": self.object_colors,
                            "annotations": [
                                {
                                    "frame_index": frame_idx,
                                    "object_id": obj_id,
                                    "x": x,
                                    "y": y,
                                    "is_positive": is_pos
                                }
                                for x, y, is_pos, obj_id, frame_idx in self.click_points
                            ]
                        }

                        # Export segmented video with overlays
                        self.status_label.config(text="Exporting segmented video...")
                        self.root.update()
                        video_exported = self._export_segmented_video(output_base_dir, masks_output_dir, source_video_path=source_video, overlay_opacity=self.mask_opacity_var.get())

                        # Save processing metadata (matching sam2_process.py format)
                        metadata_path = os.path.join(output_base_dir, "processing_metadata.json")
                        with open(metadata_path, 'w') as f:
                            json.dump({
                                "processing_info": {
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "total_frames_processed": len(frames_to_process),
                                    "total_masks_generated": total_masks,
                                    "objects_detected": list(self.object_names.keys()),
                                    "overlay_opacity": self.mask_opacity_var.get()
                                },
                                "file_paths": {
                                    # Use original_video_path_for_resegment if available (when re-segmenting loaded results),
                                    # otherwise use video_path (when segmenting freshly loaded video)
                                    "original_video_path": str(Path(self.original_video_path_for_resegment).resolve()) if self.original_video_path_for_resegment else (str(Path(self.video_path).resolve()) if self.video_path else None),
                                    "segmented_video_filename": "segmented_video.avi",
                                    "metadata_filename": "processing_metadata.json"
                                },
                                "original_annotations": annotations_data
                            }, f, indent=2)

                        self.saved_opacity = self.mask_opacity_var.get()

                        print(f"[SUCCESS] Results saved to: {output_base_dir}")
                        print(f"   - Masks: {masks_output_dir}/ ({total_masks} files)")
                        print(f"   - Video: {output_base_dir}/segmented_video.avi")
                        print(f"   - Metadata: {metadata_path}")

                        # Preserve original video path for re-segmentation before reloading
                        self.original_video_path_for_resegment = source_video

                        # Reload the segmented video into UI for playback
                        self.status_label.config(text="Loading segmented video into UI...")
                        self.root.update()

                        try:
                            segmented_video_path = os.path.join(output_base_dir, "segmented_video.avi")
                            if os.path.exists(segmented_video_path):
                                # Clean up existing lazy loading state before reloading
                                if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
                                    self.video_cap_lazy.release()
                                    self.video_cap_lazy = None
                                if hasattr(self, 'frame_cache'):
                                    self.frame_cache = {}

                                # Load segmented video using standard path (respects lazy loading preference)
                                # This is consistent with how import_masks() loads videos (lines 1167-1175)
                                print(f"Loading segmented video: {segmented_video_path}")
                                self.video_path = segmented_video_path
                                self.segmented_video_displayed = True
                                self.has_prerendered_masks = True  # Frames have masks baked in

                                # load_video_frames() handles:
                                # - Setting up video_props and video_cap_lazy if lazy loading enabled
                                # - Loading all frames into self.frames if lazy loading disabled
                                # - Updating frame_slider, current_frame_idx
                                # - Calling display_current_frame()
                                self.load_video_frames()

                                # Initialize original frame source for segment_current_frame_only
                                self._initialize_original_frame_source()
                            else:
                                print("WARNING: Segmented video not found, keeping current frames")

                        except Exception as reload_error:
                            print(f"Warning: Could not reload segmented video: {reload_error}")
                            traceback.print_exc()
                            # Continue anyway - segmentation succeeded even if reload failed

                        # Get quality metrics from segmentation result
                        if quality_metrics is not None:
                            # Handle both old (2-value) and new (3-value) format
                            if len(quality_metrics) == 3:
                                self.inter_frame_changes, self.background_ratios, self.overlap_ratios = quality_metrics
                            else:
                                # Backward compatibility: old format without overlap_ratios
                                self.inter_frame_changes, self.background_ratios = quality_metrics
                                self.overlap_ratios = []
                            self._update_quality_visualizations()
                        else:
                            # Fallback to disk-based calculation if metrics weren't calculated
                            self._calculate_segmentation_quality_metrics()

                        # Save quality metrics to disk
                        self._save_quality_metrics(output_base_dir)

                    except Exception as e:
                        print(f"[WARNING] Failed to export results: {e}")
                        traceback.print_exc()

                    # Enable Refine button now that segmentation is complete
                    self._update_refine_button_state()

                    self._show_auto_dismiss("Success",
                                      f"Segmentation completed!\n"
                                      f"Objects: {len(unique_objects)}\n"
                                      f"Total masks: {total_masks}\n"
                                      f"Frames processed: {len(frames_to_process)}\n\n"
                                      f"Results saved to:\n{output_base_dir}", timeout_ms=6000)
                elif total_masks > 0:
                    # Masks were generated but propagation had errors
                    self.status_label.config(text=f"Partial results: {total_masks} masks generated with errors")
                    print(f"[INFO] Skipping video export due to propagation errors.")
                    messagebox.showwarning("Partial Success",
                                          f"Segmentation completed with errors.\n"
                                          f"Total masks: {total_masks}\n\n"
                                          f"Partial results saved to:\n{export_dir}\n\n"
                                          f"Video export skipped due to errors.")
                else:
                    self.status_label.config(text="No masks generated")
                    messagebox.showwarning("Warning", "No masks were generated. Try different points.")
                
            finally:
                # Session cache is preserved for reuse within session only
                # It will be cleaned up when app closes or different video is processed
                pass
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Segmentation failed")
            traceback.print_exc()
            messagebox.showerror("Segmentation Error", f"Segmentation failed: {str(e)}")

    def _validate_annotations_before_segmentation(self):
        """Validate that annotations have valid frame indices for current video"""
        if not self.click_points:
            return False, "No annotation points found"

        invalid_points = []
        for i, (x, y, is_pos, obj_id, frame_idx) in enumerate(self.click_points):
            if frame_idx >= len(self.frames):
                invalid_points.append((i, frame_idx))

        if invalid_points:
            return False, f"Found {len(invalid_points)} annotations with invalid frame indices"

        return True, "All annotations valid"

    def _update_refine_button_state(self):
        """Enable/disable the Refine button based on current state."""
        if not self.refine_button:
            return

        # Requirements for refinement:
        # 1. Have masks loaded (either from segmentation or import)
        # 2. Have a video loaded
        # 3. Have a model loaded
        can_refine = (
            self.masks and
            self.frames and
            self.model_loaded and
            self.sam2_model is not None
        )

        if can_refine:
            self.refine_button.configure(state='normal', text="Refine Range")
        else:
            # Provide helpful text indicating what's missing
            if not (self.masks and self.frames):
                reason = "Refine Range (no video)"
            elif not (self.model_loaded and self.sam2_model is not None):
                reason = "Refine Range (load model first)"
            else:
                reason = "Refine Range"
            self.refine_button.configure(state='disabled', text=reason)

    def _validate_refine_inputs(self):
        """
        Validate frame range inputs for refinement.

        User inputs are 1-indexed (matching the UI display).
        Returns 0-indexed values for internal use.

        Returns:
            (is_valid, start_frame_0indexed, end_frame_0indexed, error_message)
        """
        start_str = self.refine_frame_start_var.get().strip()
        end_str = self.refine_frame_end_var.get().strip()

        # Check if both fields are provided
        if not start_str or not end_str:
            return False, None, None, "Please enter both start and end frame numbers."

        # Parse integers (user provides 1-indexed values)
        try:
            start_1based = int(start_str)
            end_1based = int(end_str)
        except ValueError:
            return False, None, None, "Frame numbers must be valid integers."

        total_frames = len(self.frames)

        # Validate range bounds (in 1-based terms for user-friendly messages)
        if start_1based < 1:
            return False, None, None, f"Start frame must be at least 1 (got {start_1based})."
        if end_1based > total_frames:
            return False, None, None, f"End frame {end_1based} exceeds video length ({total_frames} frames)."
        if start_1based > end_1based:
            return False, None, None, f"Start frame ({start_1based}) must be <= end frame ({end_1based})."

        # Convert to 0-indexed for internal use
        start = start_1based - 1
        end = end_1based - 1

        return True, start, end, None

    def _prepare_annotations_for_range(self, start_frame, end_frame):
        """
        Filter and adjust annotations for the specified frame range.

        Args:
            start_frame: Start of range (inclusive, 0-indexed)
            end_frame: End of range (inclusive, 0-indexed)

        Returns:
            List of PointAnnotation objects with adjusted frame indices
        """
        from segment import PointAnnotation

        annotations = []
        for x, y, is_positive, obj_id, frame_idx in self.click_points:
            if start_frame <= frame_idx <= end_frame:
                # Adjust frame index: relative to extracted range (SAM expects 0-based)
                adjusted_frame_idx = frame_idx - start_frame
                annotations.append(PointAnnotation(
                    x=x,
                    y=y,
                    is_positive=is_positive,
                    object_id=obj_id,
                    frame_idx=adjusted_frame_idx
                ))

        return annotations

    def refine_segmentation(self):
        """
        Main entry point for refining segmentation in a specific frame range.
        Re-segments only the specified range and updates masks.
        """
        # Basic checks
        if not self.frames:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("Warning", "Please load a SAM model first.")
            return

        if not self.masks:
            messagebox.showwarning("Warning", "No existing segmentation found. Use 'Segment Video' first.")
            return

        # Validate inputs
        is_valid, start_frame, end_frame, error_msg = self._validate_refine_inputs()
        if not is_valid:
            messagebox.showwarning("Invalid Range", error_msg)
            return

        # Check for annotations in the range
        annotations_in_range = self._prepare_annotations_for_range(start_frame, end_frame)
        if not annotations_in_range:
            messagebox.showwarning("No Annotations",
                                  f"No annotation points found in frames {start_frame + 1}-{end_frame + 1}.\n\n"
                                  f"Add annotation points in this range before refining.")
            return

        # Determine which objects in the range need re-segmentation.
        # When updated_objects is tracked, only re-segment objects whose annotations
        # have changed; unchanged objects keep their existing masks from self.masks.
        obj_ids_in_range = {a.object_id for a in annotations_in_range}
        unchanged_obj_ids_in_range: set = set()
        if self.updated_objects:
            updated_in_range = obj_ids_in_range & self.updated_objects
            unchanged_in_range = obj_ids_in_range - self.updated_objects
            if not updated_in_range:
                messagebox.showwarning(
                    "No Updated Annotations",
                    f"No updated annotation points found in frames "
                    f"{start_frame + 1}-{end_frame + 1}.\n\n"
                    f"All objects in this range are unchanged since the last segmentation. "
                    f"No re-segmentation needed."
                )
                return
            # Filter to only re-segment the updated objects
            annotations_in_range = [a for a in annotations_in_range
                                     if a.object_id in updated_in_range]
            unchanged_obj_ids_in_range = unchanged_in_range
            print(f"Partial refinement: re-segmenting {sorted(updated_in_range)}, "
                  f"keeping existing masks for {sorted(unchanged_in_range)}")

        # Get output directory (use existing results dir if available)
        if self.results_output_dir and os.path.exists(self.results_output_dir):
            output_dir = self.results_output_dir
        else:
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Refined Results",
                initialdir=os.path.dirname(self.last_export_dir) if self.last_export_dir else self.working_dir
            )
            if not output_dir:
                return

        # Always regenerate the video after refinement so the display stays in sync with masks
        total_frames = len(self.frames)
        is_partial_refinement = (start_frame > 0 or end_frame < total_frames - 1)
        regenerate_video = True

        # Warn when the range covers the entire video
        if not is_partial_refinement:
            updated_obj_names = ", ".join(
                self.object_names.get(oid, f"Object_{oid}")
                for oid in sorted({a.object_id for a in annotations_in_range})
            )
            if not unchanged_obj_ids_in_range:
                # All objects updated over full range — equivalent to Segment Video
                warn_msg = (
                    "You've selected the entire video and all objects will be re-segmented.\n\n"
                    "This is equivalent to clicking 'Segment Video'.\n\n"
                    "Continue with Refine Range anyway?"
                )
            else:
                # Only some objects updated over full range — still valid partial refinement
                warn_msg = (
                    f"You've selected the entire video.\n\n"
                    f"Only updated objects will be re-segmented: {updated_obj_names}\n\n"
                    "Continue?"
                )
            if not messagebox.askyesno("Full Video Range", warn_msg):
                return

        num_frames = end_frame - start_frame + 1
        num_annotations = len(annotations_in_range)

        # Build confirmation message (include partial-object info when applicable)
        partial_obj_info = ""
        if unchanged_obj_ids_in_range:
            reseg_names = ", ".join(
                self.object_names.get(oid, f"Object_{oid}")
                for oid in sorted(obj_ids_in_range - unchanged_obj_ids_in_range)
            )
            keep_names = ", ".join(
                self.object_names.get(oid, f"Object_{oid}")
                for oid in sorted(unchanged_obj_ids_in_range)
            )
            partial_obj_info = (
                f"Re-segment: {reseg_names}\n"
                f"Keep existing masks: {keep_names}\n"
            )

        confirm_msg = (
            f"Refine Segmentation\n\n"
            f"Frame range: {start_frame + 1} to {end_frame + 1} ({num_frames} frames)\n"
            f"Annotations in range: {num_annotations}\n"
            + (partial_obj_info)
            + f"Output directory: {output_dir}\n"
            f"Regenerate video: Yes (always)\n\n"
            f"This will overwrite existing masks for re-segmented objects in this range.\n"
            f"Continue?"
        )

        if not messagebox.askyesno("Confirm Refinement", confirm_msg):
            return

        # Run the refinement
        self._run_range_segmentation(start_frame, end_frame, annotations_in_range, output_dir, regenerate_video)

    def _run_range_segmentation(self, start_frame, end_frame, annotations, output_dir, regenerate_video):
        """
        Execute VideoSegmenter for a specific frame range.

        Args:
            start_frame: Start frame index (0-indexed, inclusive)
            end_frame: End frame index (0-indexed, inclusive)
            annotations: List of PointAnnotation with adjusted frame indices (0-based within range)
            output_dir: Directory to save results
            regenerate_video: Whether to regenerate the video after refinement
        """
        import tempfile
        import traceback
        from segment import VideoSegmenter, SegmentationConfig

        try:
            self.status_label.config(text="Preparing for range segmentation...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()

            # Get original video path (not the segmented video being displayed)
            original_video = self._get_original_video_for_resegmentation()
            if not original_video:
                self.progress_bar.pack_forget()
                return

            # Validate original video exists
            if not os.path.exists(original_video):
                messagebox.showerror("Error", f"Original video not found: {original_video}")
                self.progress_bar.pack_forget()
                return

            # Create dedicated temp directory for refinement (separate from session cache)
            refine_temp_dir = tempfile.mkdtemp(prefix='sam2_refine_')
            masks_dir = os.path.join(output_dir, "masks")

            # Progress callback for UI updates
            class UIProgressCallback:
                def __init__(self, ui):
                    self.ui = ui
                    self.current_phase = ""
                    self.total_steps = 0

                def on_progress(self, phase, current, total, message):
                    if total > 0:
                        progress = (current / total) * 100
                        self.ui.progress_var.set(progress)
                    self.ui.status_label.config(text=message)
                    self.ui.root.update()

                def on_phase_start(self, phase, total_steps):
                    self.current_phase = phase
                    self.total_steps = total_steps
                    self.ui.status_label.config(text=f"Starting {phase}...")
                    self.ui.root.update()

                def on_phase_complete(self, phase):
                    self.ui.status_label.config(text=f"Completed {phase}")
                    self.ui.root.update()

            try:
                # Create VideoSegmenter with the existing predictor
                # Use the bfloat16 setting from model initialization
                device = self._get_selected_device()
                self.device = device  # Store for use in export functions
                segmenter = VideoSegmenter(
                    predictor=self.sam2_model,
                    device=device,
                    use_bfloat16=getattr(self, 'use_bfloat16', False)
                )
                segmenter.set_progress_callback(UIProgressCallback(self))

                # Detect mask format from existing output masks so refinement stays
                # consistent with the format chosen during the original segmentation.
                _masks_dir = Path(output_dir) / "masks"
                _has_npz = _masks_dir.exists() and any(_masks_dir.glob("masks_f*.npz"))
                _has_png = _masks_dir.exists() and any(_masks_dir.glob("mask_f*.png"))
                _refine_mask_format = "npz" if (_has_npz and not _has_png) else "png"

                # Configure segmentation for the range
                # frame_range is (start, end) where end is exclusive
                # frame_offset ensures output mask files use global frame indices
                config = SegmentationConfig(
                    output_dir=output_dir,
                    frame_dir=refine_temp_dir,
                    frame_range=(start_frame, end_frame + 1),  # end is exclusive
                    frame_offset=start_frame,  # Output files use global indices
                    enable_backward_propagation=True,
                    frames_to_keep=20,
                    calculate_quality_metrics=False,  # We'll update metrics separately
                    cleanup_temp_frames=True,
                    exclusive_masks=self.exclusive_masks_var.get(),
                    mask_format=_refine_mask_format
                )

                self.status_label.config(text="Running range segmentation...")
                self.root.update()

                # Run segmentation
                result = segmenter.segment(
                    video_path=original_video,
                    annotations=annotations,
                    object_names=self.object_names,
                    object_colors=self.object_colors,
                    config=config
                )

                # Update self.masks for the refined range
                self.status_label.config(text="Updating masks...")
                self.root.update()

                for frame_idx, obj_masks in result.masks_metadata.items():
                    # frame_idx from result should already be global (due to frame_offset)
                    if frame_idx not in self.masks:
                        self.masks[frame_idx] = {}
                    for obj_id, mask_meta in obj_masks.items():
                        self.masks[frame_idx][obj_id] = mask_meta

                # Mark re-segmented objects as no longer updated and refresh object list
                refined_obj_ids = {a.object_id for a in annotations}
                self.updated_objects -= refined_obj_ids
                self.update_object_list()

                # Update quality metrics for the range (plus boundary frames)
                # _update_quality_metrics_for_range reads ALL mask files on disk for the
                # range (both newly written and pre-existing unchanged objects), so
                # quality metrics correctly reflect the full scene.
                self._update_quality_metrics_for_range(start_frame, end_frame, masks_dir)

                # Regenerate video if requested
                if regenerate_video:
                    self.status_label.config(text="Regenerating video...")
                    self.root.update()
                    self._regenerate_video_with_stitching(start_frame, end_frame, output_dir, masks_dir, original_video)

                    # Reload the newly generated segmented video for display
                    self.status_label.config(text="Loading refined video...")
                    self.root.update()

                    segmented_video_path = os.path.join(output_dir, "segmented_video.avi")
                    if os.path.exists(segmented_video_path):
                        # Save current frame position to restore it after reload
                        saved_frame_idx = self.current_frame_idx

                        # Clean up any existing lazy loading state
                        if hasattr(self, 'video_cap_lazy') and self.video_cap_lazy:
                            self.video_cap_lazy.release()
                            self.video_cap_lazy = None

                        # Clear frame cache to prevent stale data
                        if hasattr(self, 'frame_cache'):
                            self.frame_cache = {}

                        # Update to use the segmented video for display
                        self.video_path = segmented_video_path
                        self.segmented_video_displayed = True
                        self.has_prerendered_masks = True

                        # Reload video frames from the new segmented video
                        self.load_video_frames()

                        # Initialize original frame source for segment_current_frame_only
                        self._initialize_original_frame_source()

                        # Re-enable Flash Mask button (load_video_frames disables it)
                        self._enable_flash_mask_button()

                        # Restore frame position
                        if 0 <= saved_frame_idx < len(self.frames):
                            self.current_frame_idx = saved_frame_idx
                            self.frame_slider.set(saved_frame_idx)

                # Clear mask cache for affected frames (force reload from disk)
                for frame_idx in range(start_frame, end_frame + 1):
                    keys_to_remove = [k for k in self.mask_cache if k[0] == frame_idx]
                    for key in keys_to_remove:
                        del self.mask_cache[key]

                # Refresh display
                self.display_current_frame()

                self.progress_bar.pack_forget()
                self.status_label.config(text=f"Refined frames {start_frame}-{end_frame}")

                self._show_auto_dismiss("Refinement Complete",
                                   f"Successfully refined frames {start_frame} to {end_frame}.\n\n"
                                   f"Masks updated: {end_frame - start_frame + 1} frames\n"
                                   f"Video regenerated: {'Yes' if regenerate_video else 'No'}", timeout_ms=5000)

            except Exception as e:
                self.progress_bar.pack_forget()
                self.status_label.config(text="Refinement failed")
                traceback.print_exc()
                messagebox.showerror("Refinement Error", f"Failed to refine segmentation: {str(e)}")

            finally:
                # Clean up temp directory
                try:
                    import shutil
                    if os.path.exists(refine_temp_dir):
                        shutil.rmtree(refine_temp_dir)
                except Exception:
                    pass

        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Refinement failed")
            traceback.print_exc()
            messagebox.showerror("Refinement Error", f"Unexpected error: {str(e)}")

    def _update_quality_metrics_for_range(self, start_frame, end_frame, masks_dir):
        """
        Update quality metrics for the refined frame range and its boundaries.

        Args:
            start_frame: Start of refined range (inclusive)
            end_frame: End of refined range (inclusive)
            masks_dir: Directory containing mask files
        """
        import numpy as np
        from pathlib import Path

        total_frames = len(self.frames)

        # Extend range by 1 frame on each side for boundary calculations
        metrics_start = max(0, start_frame - 1)
        metrics_end = min(total_frames - 1, end_frame + 1)

        # Load existing quality metrics from disk to preserve values outside refined range
        if self.results_output_dir:
            loaded = self._load_quality_metrics(self.results_output_dir)
            if not loaded:
                # If loading fails, initialize with zeros
                if not self.inter_frame_changes or len(self.inter_frame_changes) != total_frames - 1:
                    self.inter_frame_changes = [0.0] * (total_frames - 1)
                if not self.background_ratios or len(self.background_ratios) != total_frames:
                    self.background_ratios = [0.0] * total_frames
                if not self.overlap_ratios or len(self.overlap_ratios) != total_frames:
                    self.overlap_ratios = [0.0] * total_frames
        else:
            # No output dir, ensure arrays exist
            if not self.inter_frame_changes or len(self.inter_frame_changes) != total_frames - 1:
                self.inter_frame_changes = [0.0] * (total_frames - 1)
            if not self.background_ratios or len(self.background_ratios) != total_frames:
                self.background_ratios = [0.0] * total_frames
            if not self.overlap_ratios or len(self.overlap_ratios) != total_frames:
                self.overlap_ratios = [0.0] * total_frames

        # Load masks for the extended range and recalculate metrics
        masks_path = Path(masks_dir)
        prev_combined_mask = None

        for frame_idx in range(metrics_start, metrics_end + 1):
            # Load all object masks for this frame (PNG or NPZ) and combine them
            combined_mask = None
            overlap_count = None
            frame_masks = load_all_masks_for_frame(str(masks_path), frame_idx)

            for _obj_id, _obj_name, mask in frame_masks:
                if combined_mask is None:
                    combined_mask = (mask > 0).astype(np.uint8)
                    overlap_count = (mask > 0).astype(np.int32)
                else:
                    combined_mask = np.logical_or(combined_mask, mask > 0).astype(np.uint8)
                    overlap_count += (mask > 0).astype(np.int32)

            # Calculate background ratio for this frame
            if combined_mask is not None:
                total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
                foreground_pixels = np.sum(combined_mask > 0)
                background_ratio = 1.0 - (foreground_pixels / total_pixels)
                self.background_ratios[frame_idx] = background_ratio

                # Calculate overlap ratio (excess assignments beyond first)
                excess_overlaps = np.where(overlap_count > 0, overlap_count - 1, 0)
                overlap_ratio = np.sum(excess_overlaps) / total_pixels
                self.overlap_ratios[frame_idx] = overlap_ratio
            else:
                # No masks on this frame — fully background, no overlap
                self.background_ratios[frame_idx] = 1.0
                self.overlap_ratios[frame_idx] = 0.0

            # Calculate inter-frame change (comparing to previous frame).
            # Must handle the case where a mask fully disappears (combined_mask is None)
            # or fully appears (prev_combined_mask is None), both of which are large changes.
            if frame_idx > 0:
                if prev_combined_mask is not None and combined_mask is not None:
                    changed_pixels = np.sum(prev_combined_mask != combined_mask)
                    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
                    change_ratio = changed_pixels / total_pixels
                elif prev_combined_mask is not None:
                    # Mask disappeared completely — every foreground pixel changed
                    changed_pixels = np.sum(prev_combined_mask > 0)
                    total_pixels = prev_combined_mask.shape[0] * prev_combined_mask.shape[1]
                    change_ratio = changed_pixels / total_pixels
                elif combined_mask is not None:
                    # Mask appeared from nothing — every foreground pixel is new
                    changed_pixels = np.sum(combined_mask > 0)
                    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
                    change_ratio = changed_pixels / total_pixels
                else:
                    change_ratio = 0.0  # Both frames had no masks
                # inter_frame_changes[i] represents change between frame i and i+1
                change_idx = frame_idx - 1
                if 0 <= change_idx < len(self.inter_frame_changes):
                    self.inter_frame_changes[change_idx] = change_ratio

            prev_combined_mask = combined_mask

        # Save updated metrics to disk
        if self.results_output_dir:
            self._save_quality_metrics(self.results_output_dir)

        # Refresh quality visualizations
        self._update_quality_visualizations()

    def _regenerate_video_with_stitching(self, start_frame, end_frame, output_dir, masks_dir, original_video_path):
        """
        Smart video regeneration: attempt to stitch segments if possible, otherwise full re-export.

        Args:
            start_frame: Start of refined range (inclusive)
            end_frame: End of refined range (inclusive)
            output_dir: Directory containing output files
            masks_dir: Directory containing mask files
            original_video_path: Path to the original video
        """
        import tempfile
        from pathlib import Path

        total_frames = len(self.frames)
        existing_video = os.path.join(output_dir, "segmented_video.avi")

        # Check if we can use stitching approach
        # Requirements: start > 0 AND end < total-1 AND ffmpeg available AND existing video exists
        can_stitch = (
            start_frame > 0 and
            end_frame < total_frames - 1 and
            self._has_ffmpeg() and
            os.path.exists(existing_video)
        )

        if can_stitch:
            try:
                self.status_label.config(text="Stitching video segments...")
                self.root.update()
                success = self._stitch_video_segments(
                    start_frame, end_frame, output_dir, masks_dir,
                    original_video_path, existing_video
                )
                if success:
                    return
                # Fall through to full re-export if stitching failed
                print("Video stitching failed, falling back to full re-export")
            except Exception as e:
                print(f"Video stitching error: {e}, falling back to full re-export")

        # Fallback: full video re-export
        self.status_label.config(text="Re-exporting entire video...")
        self.root.update()
        self._full_video_reexport(output_dir, masks_dir, original_video_path)

    def _has_ffmpeg(self):
        """Check if ffmpeg is available in the system PATH."""
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _stitch_video_segments(self, start_frame, end_frame, output_dir, masks_dir,
                               original_video_path, existing_video_path):
        """
        Stitch video using ffmpeg: pre-segment + rendered middle + post-segment.

        Args:
            start_frame: Start of refined range
            end_frame: End of refined range
            output_dir: Directory for output
            masks_dir: Directory containing masks
            original_video_path: Original video for frame extraction
            existing_video_path: Existing segmented video for pre/post segments

        Returns:
            True if stitching succeeded, False otherwise
        """
        import subprocess
        import tempfile
        from pathlib import Path

        total_frames = len(self.frames)

        # Get video properties
        cap = cv2.VideoCapture(existing_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            fps = 30

        temp_dir = tempfile.mkdtemp(prefix='sam2_stitch_')

        try:
            segments = []

            # 1. Extract pre-segment (frames 0 to start_frame-1).
            # -frames:v stops decoding after start_frame frames, avoiding the full-video
            # decode that select='lt(n,X)' would require.
            # setpts=PTS-STARTPTS resets PTS to 0 so the concat demuxer adds offsets
            # correctly (avoids duplicate-frame inflation from PTS mismatch).
            if start_frame > 0:
                pre_segment = os.path.join(temp_dir, "pre_segment.avi")
                cmd_pre = [
                    'ffmpeg', '-y', '-i', existing_video_path,
                    '-vf', 'setpts=PTS-STARTPTS',
                    '-frames:v', str(start_frame),
                    '-vsync', '0',
                    '-c:v', 'mjpeg', '-q:v', '5',
                    pre_segment
                ]
                result = subprocess.run(cmd_pre, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(pre_segment):
                    segments.append(pre_segment)
                    print(f"Pre-segment created: frames 0-{start_frame-1}")
                else:
                    print(f"Pre-segment extraction failed: {result.stderr}")
                    return False

            # 2. Render middle segment (refined frames with new masks)
            middle_segment = os.path.join(temp_dir, "middle_segment.avi")
            if not self._render_frame_range_to_video(
                start_frame, end_frame, masks_dir, original_video_path,
                middle_segment, width, height, fps
            ):
                return False
            segments.append(middle_segment)

            # 3. Extract post-segment (frames end_frame+1 to end).
            # Fast input seeking (-ss before -i) jumps directly to ~1 s before the
            # target frame rather than decoding from the start (the old select approach
            # had to decode all end_frame+1 frames just to skip them).
            # For x264: -ss always lands on an I-frame so all decoded frames are valid.
            # Timestamp-based select='gte(t,T)' trims to exact target; setpts resets PTS.
            if end_frame < total_frames - 1:
                post_segment = os.path.join(temp_dir, "post_segment.avi")
                target_pts = (end_frame + 1) / fps      # exact target time (s)
                seek_time  = max(0.0, target_pts - 1.0) # seek ~1 s before (stays before keyframe)
                epsilon    = 0.5 / fps                   # half-frame tolerance for float precision
                cmd_post = [
                    'ffmpeg', '-y',
                    '-ss', f'{seek_time:.6f}',           # fast input seek
                    '-i', existing_video_path,
                    '-vf', f"select='gte(t\\,{target_pts - epsilon:.6f})',setpts=PTS-STARTPTS",
                    '-vsync', '0',
                    '-c:v', 'mjpeg', '-q:v', '5',
                    post_segment
                ]
                result = subprocess.run(cmd_post, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(post_segment):
                    segments.append(post_segment)
                    print(f"Post-segment created: frames {end_frame+1}-{total_frames-1}")
                else:
                    print(f"Post-segment extraction failed: {result.stderr}")
                    return False

            # 4. Create concat list file
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")

            print(f"Concatenating {len(segments)} segments:")
            for i, seg in enumerate(segments):
                print(f"  Segment {i+1}: {os.path.basename(seg)}")

            # 5. Concatenate segments.
            # All three segments are MJPEG AVI so -c:v copy skips re-encoding entirely,
            # making concatenation nearly instantaneous regardless of video length.
            output_video = os.path.join(output_dir, "segmented_video.avi")
            temp_output = os.path.join(temp_dir, "concat_output.avi")
            cmd_concat = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c:v', 'copy',
                temp_output
            ]
            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Concatenation failed: {result.stderr}")
                return False

            print(f"Concatenation successful, output: {temp_output}")

            # Move temp output to final location (replacing original)
            import shutil
            shutil.move(temp_output, output_video)

            return True

        except Exception as e:
            print(f"Stitching error: {e}")
            return False

        finally:
            # Clean up temp directory
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

    def _render_frame_range_to_video(self, start_frame, end_frame, masks_dir,
                                     original_video_path, output_path,
                                     width, height, fps):
        """
        Render a range of frames with mask overlays to a video file.

        Args:
            start_frame: Start frame index
            end_frame: End frame index
            masks_dir: Directory containing mask files
            original_video_path: Path to original video for frame data
            output_path: Output video path
            width: Video width
            height: Video height
            fps: Frames per second

        Returns:
            True if successful, False otherwise
        """
        from pathlib import Path
        import numpy as np
        from utils import parse_mask_filename

        try:
            # Open original video
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                return False

            # Create video writer (use MJPEG to match pre/post segments)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Verify writer was created successfully
            if not out.isOpened():
                print(f"Failed to create VideoWriter for {output_path}")
                cap.release()
                return False

            masks_path = Path(masks_dir)
            overlay_opacity = self.saved_opacity

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames_written = 0
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Failed to read frame {frame_idx} from original video")
                    break

                # Resize if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                # Overlay masks for this frame using single-pass blending to avoid
                # cumulative darkening when multiple objects overlap.
                combined_overlay = frame.copy()
                has_any_mask = False
                combined_mask = np.zeros((height, width), dtype=bool)
                for obj_id, _obj_name, mask in load_all_masks_for_frame(str(masks_path), frame_idx):
                    if mask.shape[0] != height or mask.shape[1] != width:
                        mask = cv2.resize(mask, (width, height))

                    color = self.object_colors.get(obj_id, [255, 0, 0])
                    if isinstance(color, list):
                        color = tuple(color)

                    mask_bool = mask > 0
                    combined_overlay[mask_bool] = [color[2], color[1], color[0]]  # BGR
                    combined_mask |= mask_bool
                    has_any_mask = True

                if has_any_mask:
                    frame = cv2.addWeighted(frame, 1 - overlay_opacity, combined_overlay, overlay_opacity, 0)

                out.write(frame)
                frames_written += 1

            cap.release()
            out.release()

            print(f"Middle segment: wrote {frames_written} frames (expected {end_frame - start_frame + 1}) to {output_path}")
            return os.path.exists(output_path) and frames_written > 0

        except Exception as e:
            print(f"Frame range rendering error: {e}")
            return False

    def _full_video_reexport(self, output_dir, masks_dir, original_video_path):
        """
        Fallback: complete video re-export when stitching is not possible.

        Args:
            output_dir: Directory for output
            masks_dir: Directory containing masks
            original_video_path: Original video for frames
        """
        # Use existing export functionality
        try:
            fps = self.video_fps if hasattr(self, 'video_fps') else 30
            self._export_segmented_video(
                output_dir=output_dir,
                masks_dir=masks_dir,
                fps=fps,
                overlay_opacity=self.saved_opacity,
                source_video_path=original_video_path
            )
        except Exception as e:
            print(f"Full video re-export failed: {e}")
            messagebox.showwarning("Video Export Warning",
                                  f"Could not regenerate video: {e}\n\n"
                                  f"Masks have been updated. You can manually export the video later.")

    def _export_segmented_video(self, output_dir, masks_dir, fps=30, overlay_opacity=0.4, source_video_path=None):
        """
        Export segmented video with mask overlays and text labels.
        Uses shared export_segmented_video from utils.py for consistency with CLI.

        Args:
            output_dir: Directory to save the output video
            masks_dir: Directory containing mask PNG files
            fps: Frames per second for output video
            overlay_opacity: Opacity for mask overlay (0.0 to 1.0)
            source_video_path: Path to original video file (used when re-segmenting)
        """
        # Determine video source
        if source_video_path and os.path.exists(source_video_path):
            video_path = source_video_path
        elif hasattr(self, 'video_path') and self.video_path:
            video_path = self.video_path
        else:
            print("ERROR: No video source available for export")
            return False

        # Get session cache dir for frame reuse (if available)
        frame_dir = None
        if hasattr(self, '_get_session_cache_dir'):
            try:
                cache_dir = self._get_session_cache_dir()
                if cache_dir and os.path.exists(cache_dir):
                    # Check if frames exist in cache
                    frame_files = list(Path(cache_dir).glob("*.jpg"))
                    if frame_files:
                        frame_dir = cache_dir
            except Exception:
                pass  # Fall back to video-only source

        # Create hybrid frame source for efficiency
        try:
            frame_source = HybridFrameSource(video_path, frame_dir)
        except ValueError as e:
            print(f"ERROR: Could not create frame source: {e}")
            return False

        # Define mask data callback for UI state
        def get_mask_data_from_ui(frame_idx):
            """Load mask data from UI state (self.masks)."""
            result = []
            if frame_idx not in self.masks or not self.masks[frame_idx]:
                return result

            for obj_id in self.masks[frame_idx].keys():
                # Load mask from disk
                mask = self._load_mask(frame_idx, obj_id)
                if mask is None:
                    continue

                mask_bool = (mask > 0).astype(bool)
                color = self.object_colors.get(obj_id, [255, 0, 0])  # Default red
                # Ensure color is a tuple for consistency
                if isinstance(color, list):
                    color = tuple(color)
                name = self.object_names.get(obj_id, f"Object_{obj_id}")

                result.append((mask_bool, color, name, obj_id))

            return result

        output_path = Path(output_dir) / "segmented_video.avi"

        try:
            # Auto-enable GPU overlay if inference device is CUDA
            use_gpu_overlay = self.device.startswith("cuda") if self.device else False

            success = export_segmented_video(
                frame_source=frame_source,
                masks_dir=str(masks_dir),
                get_mask_data=get_mask_data_from_ui,
                output_path=str(output_path),
                fps=fps,
                overlay_opacity=overlay_opacity,
                compress=True,
                crf=23,
                use_gpu=use_gpu_overlay,
                gpu_device=self.device if use_gpu_overlay else None,
            )
            return success
        finally:
            frame_source.close()

    def _start_background_video_export(self):
        """Start background video export"""
        # Get export settings from user first
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Background Video Export Settings")
        export_dialog.geometry("400x500")
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Variables for export settings
        export_format = tk.StringVar(value="mp4")
        overlay_opacity = tk.DoubleVar(value=0.4)
        show_object_names = tk.BooleanVar(value=True)
        show_object_ids = tk.BooleanVar(value=False)
        show_boundaries = tk.BooleanVar(value=True)
        fps_var = tk.DoubleVar(value=30.0)
        quality_var = tk.StringVar(value="medium")
        export_mode = tk.StringVar(value="overlay")
        
        # UI Elements (simplified for background export)
        main_frame = ttk.Frame(export_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Background Video Export", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        # Export mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Export Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="Original + Mask Overlay", 
                        variable=export_mode, value="overlay").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Masks Only (Black Background)", 
                        variable=export_mode, value="masks_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Side by Side", 
                        variable=export_mode, value="side_by_side").pack(anchor=tk.W)
        
        # Video settings
        video_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        # FPS setting
        fps_frame = ttk.Frame(video_frame)
        fps_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=60, 
                                textvariable=fps_var, width=10,
                                bg='#404040', fg='white', insertbackground='white')
        fps_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Quality setting
        quality_frame = ttk.Frame(video_frame)
        quality_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        quality_combo = ttk.Combobox(quality_frame, textvariable=quality_var,
                                    values=["low", "medium", "high"], state="readonly", width=10)
        quality_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Format setting
        format_frame = ttk.Frame(video_frame)
        format_frame.pack(fill=tk.X)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=export_format,
                                   values=["mp4", "avi", "mov"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_background_export():
            export_dialog.destroy()
            self._perform_background_video_export(export_format.get(), overlay_opacity.get(), 
                                                show_object_names.get(), show_object_ids.get(), 
                                                show_boundaries.get(), fps_var.get(), 
                                                quality_var.get(), export_mode.get())
            
        def cancel_export():
            export_dialog.destroy()
            
        ttk.Button(button_frame, text="Start Background Export", command=start_background_export).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=cancel_export).pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        export_dialog.wait_window()
    
    def _start_foreground_video_export(self):
        """Start foreground video export (original behavior)"""
            
        # Get export settings from user
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Video Settings")
        export_dialog.geometry("400x500")
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Variables for export settings
        export_format = tk.StringVar(value="mp4")
        overlay_opacity = tk.DoubleVar(value=0.4)
        show_object_names = tk.BooleanVar(value=True)
        show_object_ids = tk.BooleanVar(value=False)
        show_boundaries = tk.BooleanVar(value=True)
        fps_var = tk.DoubleVar(value=30.0)
        quality_var = tk.StringVar(value="medium")
        
        # Export modes
        export_mode = tk.StringVar(value="overlay")
        
        # UI Elements
        main_frame = ttk.Frame(export_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_frame, text="Video Export Settings", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        # Export mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Export Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="Original + Mask Overlay", 
                        variable=export_mode, value="overlay").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Masks Only (Black Background)", 
                        variable=export_mode, value="masks_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Side by Side", 
                        variable=export_mode, value="side_by_side").pack(anchor=tk.W)
        
        # Overlay settings
        overlay_frame = ttk.LabelFrame(main_frame, text="Overlay Settings", padding=10)
        overlay_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(overlay_frame, text="Mask Opacity:").pack(anchor=tk.W)
        opacity_scale = ttk.Scale(overlay_frame, from_=0.1, to=1.0, 
                                 variable=overlay_opacity, orient=tk.HORIZONTAL)
        opacity_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(overlay_frame, text="Show Object Names", 
                        variable=show_object_names).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Show Object IDs", 
                        variable=show_object_ids).pack(anchor=tk.W)
        ttk.Checkbutton(overlay_frame, text="Show Mask Boundaries", 
                        variable=show_boundaries).pack(anchor=tk.W)
        
        # Video settings
        video_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        # FPS setting
        fps_frame = ttk.Frame(video_frame)
        fps_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=60, 
                                textvariable=fps_var, width=10,
                                bg='#404040', fg='white', insertbackground='white')
        fps_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Quality setting
        quality_frame = ttk.Frame(video_frame)
        quality_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        quality_combo = ttk.Combobox(quality_frame, textvariable=quality_var,
                                    values=["low", "medium", "high"], state="readonly", width=10)
        quality_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Format setting
        format_frame = ttk.Frame(video_frame)
        format_frame.pack(fill=tk.X)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=export_format,
                                   values=["mp4", "avi", "mov"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_export():
            export_dialog.destroy()
            self._perform_video_export(export_format.get(), overlay_opacity.get(), 
                                      show_object_names.get(), show_object_ids.get(), 
                                      show_boundaries.get(), fps_var.get(), 
                                      quality_var.get(), export_mode.get())
            
        def cancel_export():
            export_dialog.destroy()
            
        ttk.Button(button_frame, text="Export", command=start_export).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=cancel_export).pack(side=tk.RIGHT)
        
        # Wait for dialog to close
        export_dialog.wait_window()

    def _perform_video_export(self, export_format, overlay_opacity, show_object_names, 
                             show_object_ids, show_boundaries, fps, quality, export_mode):
        """Perform the actual video export based on settings"""
        
        # Get output file path with folder creation
        file_path = self._get_export_file_path_with_creation(
            title="Save Video As",
            default_name=f"sam2_video.{export_format}",
            file_types=[
                (f"{export_format.upper()} files", f"*.{export_format}"),
                ("All files", "*.*")
            ],
            default_ext=f".{export_format}",
            last_dir_attr='last_dir_video_export'
        )
        
        if not file_path:
            return
            
        try:
            self.status_label.config(text="Exporting video...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Setup video writer
            height, width = self.frames[0].shape[:2]
            
            # Adjust dimensions based on export mode
            if export_mode == "side_by_side":
                output_width = width * 2
                output_height = height
            else:
                output_width = width
                output_height = height
            
            # Video codec settings
            if export_format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif export_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif export_format == "mov":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out = cv2.VideoWriter(file_path, fourcc, fps, 
                                (output_width, output_height))
            
            if not out.isOpened():
                raise ValueError("Could not open video writer")
            
            # Only export frames that were processed during segmentation
            if hasattr(self, 'processing_range') and self.processing_range:
                frames_to_export = self.processing_range
                total_frames = len(frames_to_export)
                self.status_label.config(text=f"Exporting {total_frames} processed frames...")
            else:
                frames_to_export = list(range(len(self.frames)))
                total_frames = len(self.frames)
            
            for export_idx, frame_idx in enumerate(frames_to_export):
                frame = self.frames[frame_idx]
                # Create output frame based on mode
                if export_mode == "overlay":
                    output_frame = self._create_overlay_frame(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries
                    )
                elif export_mode == "masks_only":
                    output_frame = self._create_masks_only_frame(
                        frame, frame_idx, show_object_names,
                        show_object_ids, show_boundaries
                    )
                elif export_mode == "side_by_side":
                    output_frame = self._create_side_by_side_frame(
                        frame, frame_idx, overlay_opacity,
                        show_object_names, show_object_ids,
                        show_boundaries
                    )
                
                # Convert RGB to BGR for OpenCV
                if len(output_frame.shape) == 3:
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                else:
                    output_frame_bgr = output_frame
                
                out.write(output_frame_bgr)
                
                # Update progress
                self.progress_var.set((export_idx + 1) / total_frames * 100)
                self.root.update()
            
            self.progress_bar.pack_forget()
            self.status_label.config(text=f"Masks exported to {os.path.basename(file_path)}")
            
            # Determine if this was a limited export
            if hasattr(self, 'processing_range') and self.processing_range and len(self.processing_range) < len(self.frames):
                range_info = f" (frames {min(self.processing_range)+1}-{max(self.processing_range)+1} of {len(self.frames)})"
            else:
                range_info = ""
            
            # Show brief success message (auto-dismisses after 5s)
            self._show_auto_dismiss("Export Complete",
                              f"Video exported successfully!\n"
                              f"Location: {file_path}\n"
                              f"Frames: {total_frames}{range_info}\n"
                              f"FPS: {fps}\n"
                              f"Format: {export_format.upper()}", timeout_ms=5000)
                              
        except Exception as e:
            self.progress_bar.pack_forget()
            raise e
        
    def _get_export_file_path_with_creation(self, title, default_name, file_types, default_ext,
                                            last_dir_attr='last_dir_video_export'):
        """Get file path for export with directory creation support.

        last_dir_attr: name of the instance attribute used to remember the last directory
        for this dialog type (e.g. 'last_dir_annotations', 'last_dir_video_export').
        """
        initial = getattr(self, last_dir_attr, None) or self.working_dir
        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=file_types,
            initialfile=default_name,
            initialdir=initial
        )

        if file_path:
            setattr(self, last_dir_attr, os.path.dirname(file_path))
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        return file_path

    def _rgb_to_hex(self, rgb):
        """Convert RGB color list to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def load_sam2_model(self):
        """Load SAM2 or SAM3 model with correct video predictor initialization"""
        try:
            # Determine which model type to load
            model_type = self.model_type_var.get() if self.sam3_available else "SAM2"
            self.status_label.config(text=f"Loading {model_type} model...")
            self.model_status_label.config(text="Loading...", foreground='orange')
            self.root.update()

            # Select and validate device FIRST (works for both SAM2 and SAM3)
            device = self._get_selected_device()

            # Validate device selection and provide feedback
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
                if not torch or not torch.cuda.is_available():
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
                elif gpu_id >= torch.cuda.device_count():
                    device = "cpu"
                    self.status_label.config(text=f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPU(s) found), using CPU...")
                else:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    self.status_label.config(text=f"Using GPU {gpu_id} ({gpu_name}) for inference...")
            elif device == "cuda":
                if torch and torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    self.status_label.config(text=f"Using CUDA GPU ({gpu_name}) for inference...")
                else:
                    device = "cpu"
                    self.status_label.config(text="CUDA not available, using CPU...")
            else:  # cpu
                self.status_label.config(text="Using CPU for inference (slower)...")

            self.root.update()

            # Load model based on type
            if model_type == "SAM3":
                # SAM3 / SAM3.1 loading
                # IMPORTANT: Must patch SAM3 modules BEFORE importing model builders
                # because the import triggers loading of modules with hardcoded "cuda"
                from utils import patch_sam3_modules_for_device
                patch_sam3_modules_for_device(device)

                # Determine which SAM3 variant to load from the model dropdown selection.
                # selected_model holds the display name; look up the full internal string
                # ("Display|ckpt_file") from available_models for variant detection.
                model_selection = self.selected_model.get()
                _full_model_info = next(
                    (m for m in self.available_models[1:] if m.split('|')[0] == model_selection),
                    self.available_models[1] if len(self.available_models) > 1 else ""
                )
                try:
                    from sam3.model_builder import build_sam3_video_model
                except ImportError:
                    raise ImportError(
                        "SAM3 not found. Please install SAM3:\n"
                        "1. Run setup.py and choose to install SAM3\n"
                        "2. Follow HuggingFace authentication steps\n"
                        "3. Download checkpoints from https://huggingface.co/facebook/sam3"
                    )

                self.status_label.config(text="Building SAM3 model (may download from HuggingFace)...")
                self.root.update()

                # Parse checkpoint path from full internal model info string
                parts = _full_model_info.split('|')
                ckpt_filename = parts[1] if len(parts) >= 2 else "sam3.pt"
                sam3_checkpoint_dir = os.path.join(
                    os.path.dirname(os.path.dirname(self.checkpoint_dir)), "sam3", "checkpoints"
                )
                ckpt_path = os.path.join(sam3_checkpoint_dir, ckpt_filename)

                if os.path.exists(ckpt_path):
                    sam3_model = build_sam3_video_model(checkpoint_path=ckpt_path, device=device)
                else:
                    sam3_model = build_sam3_video_model(device=device)

                # Extract the predictor using SAM2-compatible interface
                self.sam2_model = sam3_model.tracker
                self.sam2_model.backbone = sam3_model.detector.backbone

                self.using_sam3 = True
                display_name = "SAM3 (HuggingFace)"

            else:
                # SAM2 loading logic
                # Determine model to load
                model_selection = self.selected_model.get()

                if model_selection == "auto":
                    model_info = self._auto_select_best_model()
                    if not model_info:
                        raise ValueError("No models available. Please run setup.py first to download models.")
                else:
                    model_info = next(
                        (m for m in self.available_models[1:] if m.split('|')[0] == model_selection),
                        None
                    )
                    if not model_info:
                        raise ValueError(f"Model not found: {model_selection}")

                # Parse model info: "Display Name|checkpoint_file|config_path"
                parts = model_info.split('|')
                if len(parts) != 3:
                    raise ValueError(f"Invalid model selection: {model_info}")

                display_name, checkpoint_file, config_path = parts

                # Build full paths using new structure
                sam2_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_file)
                model_cfg = os.path.join(self.config_dir, config_path)

                # Hydra on Linux strips leading '/', so prepend extra '/' for absolute paths
                if model_cfg.startswith('/'):
                    model_cfg = '/' + model_cfg

                # Check files exist
                if not os.path.exists(sam2_checkpoint):
                    raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}\n\nPlease run setup.py to download models.")
                if not os.path.exists(model_cfg):
                    raise FileNotFoundError(f"Config not found: {model_cfg}\n\nPlease ensure SAM2 is properly installed.")

                # Enable lazy loading BEFORE building the predictor.
                # This monkey-patches SAM2's load_video_frames_from_jpg_images() so
                # that init_state() loads frames on-demand (LRU cache) instead of
                # eagerly loading all frames into RAM (~178GB for 36K frames → ~2GB).
                enable_lazy_loading(cache_size=20, enable_sam3=True)

                # Import the correct builder for VIDEO segmentation
                from sam2.build_sam import build_sam2_video_predictor
                self.using_sam3 = False

                # Build the VIDEO predictor for SAM2
                # Use context manager to prevent hardcoded CUDA allocations when:
                # - CPU is selected (device == "cpu")
                # - A specific GPU is selected (device == "cuda:N" where N >= 0)
                # SAM2 has hardcoded "cuda" allocations that default to cuda:0,
                # which causes OOM errors on CPU or device mismatches on other GPUs.
                # By temporarily making CUDA unavailable, we skip these allocations
                # and let .to(device) properly place everything on the correct device.
                if should_disable_cuda_for_device(device):
                    with DisableCUDADuringInit():
                        self.sam2_model = build_sam2_video_predictor(
                            config_file=model_cfg,
                            ckpt_path=sam2_checkpoint,
                            device=device
                        )
                else:
                    self.sam2_model = build_sam2_video_predictor(
                        config_file=model_cfg,
                        ckpt_path=sam2_checkpoint,
                        device=device
                    )

            # Setup autocast context for BFloat16 (following SAM2 notebook pattern)
            # Use utility function to automatically detect GPU capabilities
            from utils import setup_precision_context
            self.autocast_context, self.use_bfloat16, precision_msg = setup_precision_context(device)

            # Enter the autocast context if enabled
            if self.autocast_context:
                self.autocast_context.__enter__()

            print(precision_msg)

            self.model_loaded = True
            # Store model info (for SAM2 only; SAM3 doesn't use this)
            if not self.using_sam3:
                self.current_model_info = model_info
            else:
                self.current_model_info = "SAM3|sam3_hiera_l.pt|sam3_hiera_l.yaml"

            # Update status display
            dtype_str = "BF16+TF32" if self.autocast_context else "FP32"
            model_type_display = model_type if self.sam3_available else "SAM2"
            self.model_status_label.config(
                text=f"{display_name} ({device.upper()}/{dtype_str})",
                foreground='green'
            )
            self.status_label.config(text=f"{model_type_display} loaded: {display_name} on {device.upper()}")

            # Test that the model has the required methods.
            if not hasattr(self.sam2_model, 'init_state'):
                raise AttributeError("Model does not have 'init_state' method. Check SAM2 installation.")
            if not hasattr(self.sam2_model, 'add_new_points'):
                raise AttributeError("Model does not have 'add_new_points' method. Check SAM2 installation.")

            # Update refine button state now that model is loaded
            self._update_refine_button_state()

        except Exception as e:
            self.model_loaded = False
            self.model_status_label.config(text="Load Failed", foreground='red')
            traceback.print_exc()

            # Determine which model type failed
            model_type = self.model_type_var.get() if self.sam3_available else "SAM2"

            error_msg = f"Failed to load {model_type} model:\n\n{str(e)}\n\nPossible solutions:\n"
            if model_type == "SAM3":
                error_msg += "1. Run setup.py and install SAM3\n"
                error_msg += "2. Authenticate with HuggingFace: huggingface-cli login\n"
                error_msg += "3. Request access at https://huggingface.co/facebook/sam3\n"
                error_msg += "4. Check Python ≥3.12, PyTorch ≥2.7, CUDA ≥12.6\n"
                error_msg += "5. Try switching to SAM2 instead"
            else:
                error_msg += "1. Run setup.py to download models and install SAM2\n"
                error_msg += "2. Check that sam2/ directory exists with checkpoints\n"
                error_msg += "3. Verify SAM2 package is installed (pip list | grep sam2)\n"
                error_msg += "4. Try a different model from the dropdown"

            messagebox.showerror("Model Load Error", error_msg)

    # =========================================================================
    # Single-Frame Segmentation Methods
    # =========================================================================

    def _clear_temporary_masks(self):
        """Clear all temporary single-frame segmentation masks."""
        self.temp_masks.clear()
        self.temp_mask_frames.clear()

    def _load_sam2_image_predictor(self):
        """
        Lazily load SAM2ImagePredictor (cached in instance variable).

        Returns:
            SAM2ImagePredictor instance or None if failed
        """
        if self.sam2_image_predictor is not None:
            return self.sam2_image_predictor

        if not self.model_loaded or not self.sam2_model:
            return None

        try:
            from segment import load_sam2_image_predictor

            self.sam2_image_predictor = load_sam2_image_predictor(
                self.sam2_model,
                device=self.device
            )

            return self.sam2_image_predictor

        except Exception as e:
            print(f"Failed to load SAM2 image predictor: {e}")
            traceback.print_exc()
            return None

    def _load_sam3_image_predictor(self):
        """
        Lazily load SAM3 image model and processor (cached in instance variables).

        Returns:
            Tuple of (model, processor) or (None, None) if failed
        """
        if self.sam3_image_model is not None and self.sam3_processor is not None:
            return self.sam3_image_model, self.sam3_processor

        if not self.model_loaded or not self.using_sam3:
            return None, None

        try:
            from segment import load_sam3_image_predictor

            # Get checkpoint path if available
            checkpoint_path = getattr(self, 'sam3_checkpoint_path', None)

            # BPE tokenizer path (note: sam3/sam3/assets - package is nested in repo)
            bpe_path = os.path.join(get_project_root(), "sam_models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")

            self.sam3_image_model, self.sam3_processor = load_sam3_image_predictor(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                device=self.device
            )

            return self.sam3_image_model, self.sam3_processor

        except Exception as e:
            print(f"Failed to load SAM3 image predictor: {e}")
            traceback.print_exc()
            return None, None

    def _get_current_frame_points(self):
        """
        Extract annotation points for current frame from UI state.

        Returns:
            Dict[obj_id, (points_array, labels_array)] or None if no points
        """
        frame_points_by_obj = {}

        for x, y, is_positive, obj_id, frame_idx in self.click_points:
            if frame_idx != self.current_frame_idx:
                continue

            if obj_id not in frame_points_by_obj:
                frame_points_by_obj[obj_id] = {'coords': [], 'labels': []}

            frame_points_by_obj[obj_id]['coords'].append([x, y])
            frame_points_by_obj[obj_id]['labels'].append(1 if is_positive else 0)

        if not frame_points_by_obj:
            return None

        # Convert to numpy arrays
        result = {}
        for obj_id, data in frame_points_by_obj.items():
            result[obj_id] = (
                np.array(data['coords'], dtype=np.float32),
                np.array(data['labels'], dtype=np.int32)
            )

        return result

    def segment_current_frame_only(self):
        """
        Segment only the current frame using SAM2ImagePredictor or SAM3 image model.

        Creates temporary in-memory masks that:
        - Appear when viewing the frame
        - Persist across frame navigation
        - Are cleared when running full video segmentation
        - Do NOT save to disk

        NOTE: Results may differ from full video propagation due to lack of temporal context.
        This is for quick annotation quality evaluation, not final mask generation.
        """
        # Validation checks
        if not self.frames:
            messagebox.showwarning("No Video", "Please load a video first")
            return

        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("No Model", "Please load a SAM model first")
            return

        # Get annotation points for current frame
        frame_points = self._get_current_frame_points()
        if not frame_points:
            messagebox.showinfo("No Points",
                f"No annotation points on frame {self.current_frame_idx + 1}.\n\n"
                f"Please add annotation points first.")
            return

        try:
            # Show appropriate status message (SAM3 first load may take 5-10s)
            if self.using_sam3 and self.sam3_image_model is None:
                self.status_label.config(text="Loading SAM3 image model (first time, may take 5-10s)...")
            else:
                self.status_label.config(text="Segmenting current frame...")
            self.root.update()

            # Get current frame
            # If we have a prerendered segmented video loaded, use the original video source
            # to avoid double-overlaying masks
            if self.has_prerendered_masks and self.original_frame_source is not None:
                current_frame_bgr = self.original_frame_source.get_frame(self.current_frame_idx)
                if current_frame_bgr is None:
                    raise RuntimeError(f"Failed to load frame {self.current_frame_idx} from original video")
            else:
                # Regular case: load from current video (handle both lazy and eager loading)
                if self.lazy_load_var.get() and hasattr(self, 'video_props'):
                    current_frame_bgr = self._load_frame_lazy(self.current_frame_idx)
                else:
                    current_frame_bgr = self.frames[self.current_frame_idx]

                if current_frame_bgr is None:
                    raise RuntimeError("Failed to load current frame")

            # Convert BGR to RGB
            current_frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)

            # Branch based on model type
            if self.using_sam3:
                # SAM3: Load image predictor and segment
                model, processor = self._load_sam3_image_predictor()
                if model is None or processor is None:
                    raise RuntimeError("Failed to initialize SAM3 image predictor")

                from segment import segment_frame_sam3
                temp_masks_this_frame = segment_frame_sam3(model, processor, current_frame_rgb, frame_points)
            else:
                # SAM2: Load image predictor and segment
                image_predictor = self._load_sam2_image_predictor()
                if image_predictor is None:
                    raise RuntimeError("Failed to initialize SAM2 image predictor")

                from segment import segment_frame_sam2
                temp_masks_this_frame = segment_frame_sam2(image_predictor, current_frame_rgb, frame_points)

            # Store temporary masks in memory
            self.temp_masks[self.current_frame_idx] = temp_masks_this_frame
            self.temp_mask_frames.add(self.current_frame_idx)

            # Refresh display to show new masks
            self.display_current_frame()

            self.status_label.config(
                text=f"Segmented frame {self.current_frame_idx + 1} "
                     f"({len(temp_masks_this_frame)} object(s))"
            )

        except Exception as e:
            messagebox.showerror("Segmentation Failed",
                f"Failed to segment current frame:\n\n{str(e)}")
            print(f"Error in segment_current_frame_only: {e}")
            traceback.print_exc()
            self.status_label.config(text="Segmentation failed")

    def _monitor_memory(self, frame_idx, total_frames):
        """Monitor and log GPU/RAM memory usage (every 100 frames)"""
        # Only monitor every 100 frames to reduce overhead
        if frame_idx % 100 != 0:
            return

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            # Log to console
            print(f"  GPU Memory - Frame {frame_idx}/{total_frames}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {peak:.2f}GB peak")

            # Reset peak stats every 100 frames to track incremental growth
            torch.cuda.reset_peak_memory_stats()

        # Monitor RAM
        try:
            import psutil
            ram_used = psutil.virtual_memory().used / 1024**3
            ram_total = psutil.virtual_memory().total / 1024**3
            ram_percent = psutil.virtual_memory().percent
            print(f"  RAM: {ram_used:.2f}/{ram_total:.2f} GB ({ram_percent:.1f}%)")
        except ImportError:
            pass  # psutil not available, skip RAM monitoring

    def _export_mask_to_disk(self, frame_idx, obj_id, mask, output_dir, metadata_list):
        """
        Export mask to disk immediately and store only metadata.
        Uses the export_mask_to_disk utility from utils.py.

        Args:
            frame_idx: Frame index
            obj_id: Object ID
            mask: Binary mask (H, W) numpy array
            output_dir: Directory to save masks
            metadata_list: List to append metadata to
        """
        # Get object metadata first (needed for filename)
        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
        obj_color = self.object_colors.get(obj_id, (255, 0, 0))

        # Use the utility function from utils.py
        metadata = export_mask_to_disk_util(
            mask=mask,
            output_dir=output_dir,
            frame_idx=frame_idx,
            obj_id=obj_id,
            obj_name=obj_name,
            obj_color=obj_color
        )
        metadata_list.append(metadata)

        # Explicitly delete mask array to free memory immediately
        del mask

    def _cleanup_frame_cache(self, current_frame_idx, inference_state):
        """
        Clean up old frames from inference state to prevent memory growth

        SAM2 only needs last 6-16 frames for temporal memory attention.
        Keeping more than necessary wastes GPU memory.

        Args:
            current_frame_idx: Current frame being processed
            inference_state: SAM2 inference state object
        """
        frames_to_keep = 20  # Fixed value, always keep last 20 frames

        # Get non-conditioning frames from inference state
        # This is where SAM2 stores frame features and outputs
        if hasattr(inference_state, 'non_cond_frame_outputs'):
            non_cond = inference_state.non_cond_frame_outputs

            # Find frames older than the cache window
            old_frames = [f for f in non_cond.keys() if f < current_frame_idx - frames_to_keep]

            # Delete old frame outputs
            for old_frame in old_frames:
                del non_cond[old_frame]

            if old_frames and current_frame_idx % 50 == 0:
                print(f"  Cleaned up {len(old_frames)} old frames from cache (keeping last {frames_to_keep})")

        # Periodic GPU memory cleanup
        if current_frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_mask(self, frame_idx, obj_id):
        """
        Load mask from disk with LRU caching.
        Uses the load_mask utility from utils.py for the actual disk I/O.

        Args:
            frame_idx: Frame index
            obj_id: Object ID

        Returns:
            mask: Binary mask array (H, W) or None if not found
        """
        # Check cache first
        cache_key = (frame_idx, obj_id)
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        # SAM2 covered-by-SAM3 IDs: mask is the pixel-wise union of the referenced
        # SAM3 sub-instance masks (mirrors sam2_process.py covered-union computation).
        covered_info = self.sam2_covered_ids.get(obj_id)
        if covered_info is not None:
            sub_ids = [int(s) for s in covered_info.get("sam3_sub_ids", [])]
            union = None
            for sub_id in sub_ids:
                sub_mask = self._load_mask(frame_idx, sub_id)
                if sub_mask is None:
                    continue
                sub_bin = (sub_mask > 0)
                union = sub_bin if union is None else (union | sub_bin)
            if union is None:
                return None
            mask = (union.astype(np.uint8) * 255)
            self._cache_mask(cache_key, mask)
            return mask

        # Check SAM3 native mask paths first
        if obj_id in self.sam3_mask_dirs:
            mask_dir, pattern = self.sam3_mask_dirs[obj_id]
            path = os.path.join(mask_dir, pattern.format(frame=frame_idx))
            if os.path.exists(path):
                if path.endswith('.npz'):
                    try:
                        data = np.load(path)
                        mask = data["mask"]
                        if mask is not None:
                            self._cache_mask(cache_key, mask)
                        return mask
                    except Exception:
                        return None
                else:
                    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        self._cache_mask(cache_key, mask)
                    return mask
            return None

        if not hasattr(self, 'mask_export_dir') or self.mask_export_dir is None:
            print(f"WARNING: No mask export directory found for frame {frame_idx}, obj {obj_id}")
            return None

        # Get object name for filename lookup
        obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")

        # Use the utility function from utils.py
        mask = load_mask_from_disk(
            mask_dir=self.mask_export_dir,
            frame_idx=frame_idx,
            obj_id=obj_id,
            obj_name=obj_name
        )

        if mask is not None:
            self._cache_mask(cache_key, mask)

        return mask

    def _cache_mask(self, cache_key, mask):
        """Store mask in cache with LRU eviction"""
        if mask is not None:
            self.mask_cache[cache_key] = mask
            # LRU eviction if cache exceeds size limit
            if len(self.mask_cache) > self.mask_cache_size:
                oldest_key = next(iter(self.mask_cache))
                del self.mask_cache[oldest_key]

    def _get_config_file_path(self):
        """Get path to config file for storing persistent settings"""
        config_dir = os.path.expanduser("~/.sam2_ui")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config.json")

    def _load_last_export_dir(self):
        """Load last export directory from config file"""
        try:
            config_path = self._get_config_file_path()
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('last_export_dir', None)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
        return None

    def _save_last_export_dir(self, export_dir):
        """Save last export directory to config file"""
        try:
            # Update in-memory variable (fixes within-session persistence)
            self.last_export_dir = export_dir

            config_path = self._get_config_file_path()
            config = {}
            # Load existing config if it exists
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except:
                    pass
            # Update last export dir
            config['last_export_dir'] = export_dir
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="SAM2 Video Annotation Tool")
    parser.add_argument(
        "project", nargs="?", default=None,
        help="Path to a SAM2 result directory to load on startup (segmented video + processing_metadata.json)",
    )
    parser.add_argument(
        "--project", dest="project_opt", default=None,
        help="Path to a SAM2 result directory to load on startup (alternative to positional arg)",
    )
    parser.add_argument(
        "--working-dir", "-w",
        default=None,
        help="Starting directory for all file/folder dialogs (load video, import/export annotations, etc.)"
    )
    args = parser.parse_args()
    project_dir = args.project_opt or args.project

    working_dir = None
    if args.working_dir:
        working_dir = os.path.abspath(args.working_dir)
        if not os.path.isdir(working_dir):
            print(f"Warning: --working-dir '{working_dir}' does not exist, using current directory")
            working_dir = None

    root = tk.Tk()
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    app = SAM2VideoUI(root, working_dir=working_dir)

    if project_dir:
        root.update()  # ensure window is realized before loading (status bar, etc.)
        app.last_dir_results = os.path.dirname(os.path.abspath(project_dir))
        app._load_results_from_dir(project_dir)

    def on_closing():
        # Check for active exports and segmentation
        active_exports = len(app.active_exports) if app.active_exports else 0
        active_segmentation = len(app.active_segmentation) if app.active_segmentation else 0
        total_active = active_exports + active_segmentation
        
        if total_active > 0:
            result = messagebox.askyesnocancel(
                "Active Background Tasks Detected",
                f"You have {total_active} background task(s) running:\n"
                f"- {active_exports} export(s)\n"
                f"- {active_segmentation} segmentation(s)\n\n"
                f"What would you like to do?\n\n"
                f"Yes: Save task status and quit (tasks will continue)\n"
                f"No: Quit without saving (tasks will be lost)\n"
                f"Cancel: Stay in application"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes - save task status and let tasks continue
                if hasattr(app, '_handle_background_tasks_save_on_exit'):
                    app._handle_background_tasks_save_on_exit()
                # Don't stop workers - let them continue
            # else: No - user wants to quit without saving, just exit

        # Warn if there are unsaved annotation changes
        if getattr(app, 'annotations_dirty', False) and app.click_points:
            keep_open = messagebox.askyesno(
                "Unsaved Annotations",
                "You have annotation changes that have not been exported.\n\n"
                "Close anyway? (Changes will be lost)"
            )
            if not keep_open:
                return  # User chose to stay and export first

        # Clean up video capture if lazy loading
        if hasattr(app, 'video_cap_lazy') and app.video_cap_lazy:
            app.video_cap_lazy.release()

        # Clean up session-based frame cache
        if hasattr(app, 'session_cache_dir') and app.session_cache_dir:
            if os.path.exists(app.session_cache_dir):
                print(f"Cleaning up session cache: {app.session_cache_dir}")
                shutil.rmtree(app.session_cache_dir, ignore_errors=True)

        # CRITICAL: Always destroy root at the end
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
