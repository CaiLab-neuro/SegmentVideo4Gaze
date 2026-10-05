"""
Utility functions for SAM2 Video UI.

This module contains reusable functions for mask I/O operations,
quality metrics calculation, and other common utilities.
"""

import os
import re
import numpy as np
import cv2
from typing import Dict, List, Tuple, Callable, Optional, Any

# Try to import torch for CUDA utilities
try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# CUDA Utilities
# =============================================================================

class DisableCUDADuringInit:
    """
    Context manager to temporarily make torch.cuda.is_available() return False.

    This is needed because SAM2 has hardcoded CUDA allocations during model init:
    - position_encoding.py: warmup_cache allocates on cuda:0
    - transformer.py: freqs_cis is moved to "cuda" (defaults to cuda:0)

    By making CUDA appear unavailable during init, these allocations are skipped,
    and the subsequent .to(device) call properly places everything on the correct device.

    Usage:
        # When using CPU or a specific GPU other than cuda:0
        with DisableCUDADuringInit():
            model = build_sam2_video_predictor(config, checkpoint, device="cpu")

        # Or with the helper function:
        if should_disable_cuda_for_device(device):
            with DisableCUDADuringInit():
                model = build_sam2_video_predictor(...)
    """
    def __init__(self):
        self._original_is_available = None

    def __enter__(self):
        if torch is not None:
            self._original_is_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch is not None and self._original_is_available is not None:
            torch.cuda.is_available = self._original_is_available
        return False


def should_disable_cuda_for_device(device: str) -> bool:
    """
    Determine if CUDA should be temporarily disabled during model initialization.

    SAM2 has hardcoded "cuda" allocations that default to cuda:0, which causes
    OOM errors on CPU or device mismatches on other GPUs. This function returns
    True if the device selection requires disabling CUDA during init.

    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0", "cuda:1")

    Returns:
        True if CUDA should be disabled during model init, False otherwise
    """
    return device == "cpu" or (device.startswith("cuda:") and device != "cuda:0")


def supports_bfloat16(device: str) -> bool:
    """
    Check if a device supports native BFloat16 operations.

    BFloat16 is only supported on NVIDIA GPUs with compute capability >= 8.0
    (Ampere architecture and newer: RTX 30xx+, RTX 40xx+, A100, H100, etc.).

    Older GPUs (V100, RTX 20xx, GTX series) can run BFloat16 via software
    emulation but performance is worse than Float32.

    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0", "cuda:1")

    Returns:
        True if the device has native BFloat16 support, False otherwise
    """
    if torch is None or device == "cpu" or not torch.cuda.is_available():
        return False

    try:
        # Extract GPU ID from device string
        gpu_id = 0 if device == "cuda" else int(device.split(":")[1])

        # Check compute capability (Ampere = 8.0+)
        compute_cap = torch.cuda.get_device_properties(gpu_id).major
        return compute_cap >= 8
    except Exception:
        # If we can't determine, assume no support
        return False


def setup_precision_context(device: str, force_bfloat16: bool = False):
    """
    Setup the appropriate precision context for a given device.

    Automatically enables BFloat16 on Ampere+ GPUs and TF32 for matrix operations.
    Returns a context manager and flags for the precision settings.

    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0", "cuda:1")
        force_bfloat16: If True, enable BFloat16 even on unsupported GPUs (not recommended)

    Returns:
        Tuple of (autocast_context, use_bfloat16: bool, info_message: str)
        - autocast_context: Context manager to enter (or None)
        - use_bfloat16: Boolean flag indicating if BFloat16 is active
        - info_message: Human-readable string describing the setup

    Example:
        >>> autocast_ctx, use_bf16, msg = setup_precision_context("cuda:0")
        >>> print(msg)
        >>> if autocast_ctx:
        >>>     autocast_ctx.__enter__()
    """
    if device == "cpu" or torch is None or not torch.cuda.is_available():
        return None, False, f"Model loaded on {device} (CPU mode, no autocast)"

    try:
        # Extract GPU ID and get properties
        gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        compute_cap = gpu_props.major
        gpu_name = gpu_props.name

        # Check BFloat16 support
        has_bfloat16 = compute_cap >= 8

        if has_bfloat16 or force_bfloat16:
            # Enable BFloat16 autocast
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

            # Enable TF32 for Ampere+ (same requirement as BFloat16)
            if compute_cap >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                msg = f"Model loaded on {device} with BFloat16 autocast and TF32 enabled (compute capability {compute_cap}.x)"
            else:
                msg = f"Model loaded on {device} with BFloat16 autocast (FORCED on compute capability {compute_cap}.x - may be slower)"

            return autocast_ctx, True, msg
        else:
            # Use Float32 for older GPUs
            msg = f"Model loaded on {device} with Float32 ({gpu_name}, compute capability {compute_cap}.x - BFloat16 not supported)"
            return None, False, msg

    except Exception as e:
        # Fallback to Float32 if we can't detect capabilities
        return None, False, f"Model loaded on {device} with Float32 (capability detection failed: {e})"


# Global variable to store the target device for SAM3
_SAM3_TARGET_DEVICE = None


def set_sam3_target_device(device: str):
    """
    Set the target device for SAM3 model building.

    This must be called BEFORE importing SAM3 modules or building the model.

    Args:
        device: Target device string (e.g., "cpu", "cuda:0", "cuda:1")
    """
    global _SAM3_TARGET_DEVICE
    _SAM3_TARGET_DEVICE = device
    print(f"  [SAM3] Target device set to: {device}")


def get_sam3_target_device() -> str:
    """Get the target device for SAM3, defaulting to cuda:0 if not set."""
    global _SAM3_TARGET_DEVICE
    if _SAM3_TARGET_DEVICE is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return _SAM3_TARGET_DEVICE


def patch_sam3_modules_for_device(device: str):
    """
    Comprehensive monkey-patch for SAM3 to fix ALL hardcoded CUDA references.

    This patches modules at the CLASS level so that model construction uses
    the correct device. Must be called BEFORE building the SAM3 model.

    Args:
        device: Target device string (e.g., "cpu", "cuda:0", "cuda:1")

    Patches applied:
        1. PositionEmbeddingSine.__init__ - precompute uses hardcoded "cuda"
        2. TransformerDecoder._get_coords - uses hardcoded "cuda" for coord cache
        3. Various .cuda() calls throughout the codebase (runtime patches)
    """
    set_sam3_target_device(device)

    # Patch 1: PositionEmbeddingSine (called during model construction)
    _patch_position_embedding_sine(device)

    # Patch 2: TransformerDecoder._get_coords (decoder.py line 281)
    _patch_transformer_decoder(device)

    # Patch 3: VL Combiner (vl_combiner.py)
    _patch_vl_combiner()

    # Patch 4: IO Utils (io_utils.py)
    _patch_io_utils()

    # Patch 5: Sam3VideoPredictor
    _patch_sam3_video_predictor()

    # Patch 6: RoPEAttention (transformer.py line 285)
    _patch_rope_attention(device)

    # Patch 7: Runtime patches (for inference)
    _patch_sam3_tracker_predictor()
    _patch_sam3_tracker_base()

    # Patch 8: SAM3.1 multiplex model (VideoTrackingDynamicMultiplex.init_state)
    _patch_sam3_multiplex_init_state(device)

    print(f"  [SAM3] All modules patched for device: {device}")


def _patch_position_embedding_sine(device: str):
    """Patch PositionEmbeddingSine to use the target device instead of hardcoded cuda."""
    try:
        from sam3.model import position_encoding
    except ImportError:
        return

    if hasattr(position_encoding.PositionEmbeddingSine, '_original_init'):
        return  # Already patched

    original_init = position_encoding.PositionEmbeddingSine.__init__
    position_encoding.PositionEmbeddingSine._original_init = original_init

    def patched_init(self, num_pos_feats, temperature=10000, normalize=True,
                     scale=None, precompute_resolution=None):
        import math
        super(position_encoding.PositionEmbeddingSine, self).__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        # PATCHED: Use target device instead of hardcoded "cuda"
        target_device = get_sam3_target_device()
        if precompute_resolution is not None:
            precompute_sizes = [
                (precompute_resolution // 4, precompute_resolution // 4),
                (precompute_resolution // 8, precompute_resolution // 8),
                (precompute_resolution // 16, precompute_resolution // 16),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in precompute_sizes:
                tensors = torch.zeros((1, 1) + size, device=target_device)
                self.forward(tensors)
                self.cache[size] = self.cache[size].clone().detach()

    position_encoding.PositionEmbeddingSine.__init__ = patched_init
    print("  [patch_sam3_for_device] Patched PositionEmbeddingSine.__init__")


def _patch_transformer_decoder(device: str):
    """Patch TransformerDecoder._get_coords to use target device instead of hardcoded cuda."""
    try:
        from sam3.model import decoder
    except ImportError:
        return

    if not hasattr(decoder, 'TransformerDecoder'):
        return

    if hasattr(decoder.TransformerDecoder, '_device_patched'):
        return  # Already patched

    # Patch the static _get_coords method to use target device
    original_get_coords = decoder.TransformerDecoder._get_coords

    @staticmethod
    def patched_get_coords(H, W, device):
        # If device is "cuda" (hardcoded), replace with target device
        if device == "cuda":
            device = get_sam3_target_device()
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    decoder.TransformerDecoder._get_coords = patched_get_coords
    decoder.TransformerDecoder._device_patched = True
    print("  [patch_sam3_for_device] Patched TransformerDecoder._get_coords")


def _patch_vl_combiner():
    """Patch VLCombiner to use inference_state device instead of hardcoded cuda."""
    try:
        from sam3.model import vl_combiner
    except ImportError:
        return

    # The VLCombiner has device="cuda" as default in method signatures
    # We need to patch the call sites or change defaults
    # For now, we'll document that users should pass device explicitly
    pass  # VL combiner patches handled at call sites


def _patch_io_utils():
    """Patch io_utils.py to use correct device instead of hardcoded .cuda()."""
    try:
        from sam3.model import io_utils
    except ImportError:
        return

    if hasattr(io_utils, '_device_patched'):
        return

    # Patch load_image_as_tensor if it exists
    if hasattr(io_utils, 'load_image_as_tensor'):
        original_load = io_utils.load_image_as_tensor

        def patched_load(*args, **kwargs):
            result = original_load(*args, **kwargs)
            target_device = get_sam3_target_device()
            if hasattr(result, 'to'):
                return result.to(target_device)
            return result

        io_utils.load_image_as_tensor = patched_load

    io_utils._device_patched = True
    print("  [patch_sam3_for_device] Patched io_utils")


def _patch_sam3_video_predictor():
    """Patch Sam3VideoPredictor to use correct device."""
    try:
        from sam3.model import sam3_video_predictor
    except ImportError:
        return

    if hasattr(sam3_video_predictor, '_device_patched'):
        return

    # The predictor has .cuda() calls that need to be device-aware
    # These will be handled by the init_state patches
    sam3_video_predictor._device_patched = True


def _patch_rope_attention(device: str):
    """Patch RoPEAttention.__init__ to use target device instead of hardcoded cuda.

    In transformer.py line 285, there's:
        device = torch.device("cuda") if torch.cuda.is_available() else None
    This causes OOM when cuda:0 is full but we want cuda:1.
    """
    try:
        from sam3.sam import transformer
        from sam3.sam import rope
    except ImportError:
        return

    if not hasattr(transformer, 'RoPEAttention'):
        return

    if hasattr(transformer.RoPEAttention, '_device_patched'):
        return  # Already patched

    original_init = transformer.RoPEAttention.__init__
    transformer.RoPEAttention._original_init = original_init

    def patched_init(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        use_rope_real=False,
        **kwargs,
    ):
        # Call parent Attention.__init__
        transformer.Attention.__init__(self, *args, **kwargs)
        self.use_rope_real = use_rope_real
        from functools import partial
        self.compute_cis = partial(
            rope.compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        # PATCHED: Use target device instead of hardcoded "cuda"
        target_device = get_sam3_target_device()
        device_obj = torch.device(target_device) if target_device else None
        self.freqs_cis = self.compute_cis(
            end_x=feat_sizes[0], end_y=feat_sizes[1], device=device_obj
        )
        if self.use_rope_real:
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag
        self.rope_k_repeat = rope_k_repeat

    transformer.RoPEAttention.__init__ = patched_init
    transformer.RoPEAttention._device_patched = True
    print("  [patch_sam3_for_device] Patched RoPEAttention.__init__")


def patch_sam3_for_device():
    """
    Monkey-patch SAM3 to fix hardcoded .cuda() calls in runtime code.

    NOTE: This function is DEPRECATED in favor of patch_sam3_modules_for_device(device),
    which patches both construction-time AND runtime issues. If you've already called
    patch_sam3_modules_for_device, calling this function is a no-op.

    SAM3 has several hardcoded .cuda() calls that cause crashes when using
    devices other than cuda:0 (e.g., cuda:1, cuda:2). This function patches
    the affected methods to use the correct device from the inference state.

    Call this ONCE after importing SAM3 but BEFORE running any inference.

    Patches:
        1. Sam3TrackerPredictor._get_image_feature: Line 1024
           - Original: image = inference_state["images"][frame_idx].cuda()
           - Patched: image = inference_state["images"][frame_idx].to(inference_state["device"])

        2. Sam3TrackerBase._prepare_memory_conditioned_features: Lines 657, 664
           - Original: feats = prev["maskmem_features"].cuda(non_blocking=True)
           - Patched: feats = prev["maskmem_features"].to(device, non_blocking=True)
           - Original: maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()
           - Patched: maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
    """
    # Patch 1: Sam3TrackerPredictor._get_image_feature
    _patch_sam3_tracker_predictor()

    # Patch 2: Sam3TrackerBase._prepare_memory_conditioned_features
    _patch_sam3_tracker_base()


def _patch_sam3_tracker_predictor():
    """Patch Sam3TrackerPredictor methods to fix hardcoded device issues.

    Patches:
    - _get_image_feature: fixes hardcoded .cuda() call
    - init_state: fixes hardcoded torch.device("cuda") for storage_device
    """
    try:
        from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
    except ImportError:
        return  # SAM3 not installed

    # Check if already patched
    if hasattr(Sam3TrackerPredictor, '_original_get_image_feature'):
        return

    # Store original method
    Sam3TrackerPredictor._original_get_image_feature = Sam3TrackerPredictor._get_image_feature

    def _patched_get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame (patched for device flexibility)."""
        # Look up in the cache
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            if self.backbone is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached. "
                    "Please run inference on this frame first."
                )
            else:
                # Cache miss -- we will run inference on a single image
                # PATCHED: Use the correct device from inference_state instead of hardcoded .cuda()
                image = inference_state["images"][frame_idx].to(inference_state["device"]).float().unsqueeze(0)
                backbone_out = self.forward_image(image)
                # Cache the most recent frame's feature (for repeated interactions with
                # a frame; we can use an LRU cache for more frames in the future).
                inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        if "tracker_backbone_out" in backbone_out:
            backbone_out = backbone_out["tracker_backbone_out"]  # get backbone output

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            feat = feat.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["backbone_fpn"][i] = feat
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    Sam3TrackerPredictor._get_image_feature = _patched_get_image_feature
    print("  [patch_sam3_for_device] Patched Sam3TrackerPredictor._get_image_feature")

    # Also patch init_state to fix storage_device hardcoding
    if not hasattr(Sam3TrackerPredictor, '_original_init_state'):
        Sam3TrackerPredictor._original_init_state = Sam3TrackerPredictor.init_state

        @torch.inference_mode()
        def _patched_init_state(
            self,
            video_height=None,
            video_width=None,
            num_frames=None,
            video_path=None,
            cached_features=None,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        ):
            """Initialize inference state (patched to use self.device instead of hardcoded cuda)."""
            from collections import OrderedDict
            from sam3.model.utils.sam2_utils import load_video_frames

            inference_state = {}
            inference_state["offload_video_to_cpu"] = offload_video_to_cpu
            inference_state["offload_state_to_cpu"] = offload_state_to_cpu
            inference_state["device"] = self.device

            # PATCHED: Use self.device instead of hardcoded torch.device("cuda")
            if offload_state_to_cpu:
                inference_state["storage_device"] = torch.device("cpu")
            else:
                inference_state["storage_device"] = self.device  # <-- THE FIX

            if video_path is not None:
                images, video_height, video_width = load_video_frames(
                    video_path=video_path,
                    image_size=self.image_size,
                    offload_video_to_cpu=offload_video_to_cpu,
                    async_loading_frames=async_loading_frames,
                    compute_device=inference_state["storage_device"],
                )
                inference_state["images"] = images
                inference_state["num_frames"] = len(images)
                inference_state["video_height"] = video_height
                inference_state["video_width"] = video_width
            else:
                inference_state["video_height"] = video_height
                inference_state["video_width"] = video_width
                inference_state["num_frames"] = num_frames

            inference_state["point_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"] = {}
            inference_state["cached_features"] = (
                {} if cached_features is None else cached_features
            )
            inference_state["constants"] = {}
            inference_state["obj_id_to_idx"] = OrderedDict()
            inference_state["obj_idx_to_id"] = OrderedDict()
            inference_state["obj_ids"] = []
            inference_state["output_dict"] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["first_ann_frame_idx"] = None
            inference_state["output_dict_per_obj"] = {}
            inference_state["temp_output_dict_per_obj"] = {}
            inference_state["consolidated_frame_inds"] = {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            }
            inference_state["tracking_has_started"] = False
            inference_state["frames_already_tracked"] = {}
            self.clear_all_points_in_video(inference_state)
            return inference_state

        Sam3TrackerPredictor.init_state = _patched_init_state
        print("  [patch_sam3_for_device] Patched Sam3TrackerPredictor.init_state")


def _patch_sam3_tracker_base():
    """Patch Sam3TrackerBase._prepare_memory_conditioned_features to fix hardcoded .cuda() calls."""
    try:
        from sam3.model.sam3_tracker_base import Sam3TrackerBase
        from sam3.model.sam3_tracker_utils import select_closest_cond_frames
    except ImportError:
        return  # SAM3 not installed

    # Check if already patched
    if hasattr(Sam3TrackerBase, '_original_prepare_memory_conditioned_features'):
        return

    # Store original method
    Sam3TrackerBase._original_prepare_memory_conditioned_features = \
        Sam3TrackerBase._prepare_memory_conditioned_features

    def _patched_prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        """Fuse the current frame's visual feature map with previous memory (patched for device flexibility)."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame and use_prev_mem_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []

            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0

            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]

            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if self.use_memory_selection:
                    if t_rel > len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[-t_rel]
                else:
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            prev_frame_idx = ((frame_idx - 2) // r) * r
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        else:
                            prev_frame_idx = -(-(frame_idx + 2) // r) * r
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames

                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                # PATCHED: Use .to(device) instead of hardcoded .cuda()
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=bool)
                )

                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                # PATCHED: Use .to(device) instead of hardcoded .cuda()
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    # add a spatial embedding for the conditioning frame
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                # Temporal positional encoding
                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            # Optionally, select only a subset of spatial memory frames during training
            if (
                self.training
                and self.prob_to_dropout_spatial_mem > 0
                and self.rng.random() < self.prob_to_dropout_spatial_mem
            ):
                num_spatial_mem_keep = self.rng.integers(len(to_cat_prompt) + 1)
                keep = self.rng.choice(
                    range(len(to_cat_prompt)), num_spatial_mem_keep, replace=False
                ).tolist()
                to_cat_prompt = [to_cat_prompt[i] for i in keep]
                to_cat_prompt_mask = [to_cat_prompt_mask[i] for i in keep]
                to_cat_prompt_pos_embed = [to_cat_prompt_pos_embed[i] for i in keep]

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)

            # First add those object pointers from selected conditioning frames
            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [
                (
                    (frame_idx - t) * tpos_sign_mul,
                    out["obj_ptr"],
                    True,  # is_selected_cond_frame
                )
                for t, out in ptr_cond_outputs.items()
            ]

            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if not self.use_memory_selection:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                else:
                    if -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]

                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )
                # expand to batch size
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)  # "to_cat_prompt_mask" is not used
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        # Step 2: Concatenate the memories and forward through the transformer encoder
        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None  # For now, we always masks are zeros anyways
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    Sam3TrackerBase._prepare_memory_conditioned_features = _patched_prepare_memory_conditioned_features
    print("  [patch_sam3_for_device] Patched Sam3TrackerBase._prepare_memory_conditioned_features")


def _patch_sam3_multiplex_init_state(device: str):
    """Patch VideoTrackingMultiplexDemo._get_image_feature to use the target device.

    _get_image_feature hardcodes .cuda() on the raw image tensor. We replace
    it with .to(inference_state["device"]).

    The inference_state device fields themselves are fixed in segment.py after
    calling VideoTrackingMultiplexDemo.init_state() directly (necessary because
    Sam3VideoTrackingMultiplexDemo.init_state — what the builder returns — does not
    accept a video_path at all and is meant for pre-computed feature caching).
    """
    try:
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo
    except ImportError:
        return  # SAM3.1 not installed

    if hasattr(VideoTrackingMultiplexDemo, '_get_image_feature_device_patched'):
        return

    def patched_get_image_feature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            # PATCHED: use inference_state["device"] instead of hardcoded .cuda()
            dev = inference_state.get("device", torch.device("cuda"))
            image = inference_state["images"][frame_idx].to(dev).float().unsqueeze(0)
            from sam3.model.data_misc import NestedTensor
            backbone_out = self.forward_image(
                NestedTensor(tensors=image, mask=None),
                need_sam3_out=False,  # SAM3 detector outputs are top-level tensors that break the cloning loop; tracking only needs interactive + sam2_backbone_out
                need_interactive_out=True,
                need_propagation_out=True,
            )
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        features = self._prepare_backbone_features(backbone_out)
        return image, features

    VideoTrackingMultiplexDemo._get_image_feature = patched_get_image_feature
    VideoTrackingMultiplexDemo._get_image_feature_device_patched = True
    print("  [patch_sam3_for_device] Patched VideoTrackingMultiplexDemo._get_image_feature")


# =============================================================================
# Mask Filename Utilities
# =============================================================================

def generate_mask_filename(frame_idx: int, obj_id: int, obj_name: str) -> str:
    """
    Generate standardized mask filename.

    Args:
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name (will be sanitized)

    Returns:
        Filename string in format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
    """
    return f"mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png"


def generate_legacy_mask_filename(frame_idx: int, obj_id: int) -> str:
    """
    Generate legacy mask filename for backward compatibility.

    Args:
        frame_idx: Frame index
        obj_id: Object ID

    Returns:
        Filename string in legacy format: frame_{frame_idx:05d}_obj_{obj_id}.png
    """
    return f"frame_{frame_idx:05d}_obj_{obj_id}.png"


def parse_mask_filename(filename: str) -> Optional[Tuple[int, str, int]]:
    """
    Parse mask filename to extract frame index, object name, and object ID.

    Args:
        filename: Mask filename to parse

    Returns:
        Tuple of (frame_idx, obj_name, obj_id) or None if parsing fails
    """
    # Try new format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
    match = re.match(r'mask_f(\d{6})_(.+)_id(\d+)\.png', filename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))

    # Try legacy format: frame_{frame_idx:05d}_obj_{obj_id}.png
    match = re.match(r'frame_(\d{5})_obj_(\d+)\.png', filename)
    if match:
        return int(match.group(1)), f"Object_{match.group(2)}", int(match.group(2))

    return None


# =============================================================================
# Mask I/O Functions
# =============================================================================

def save_mask(
    mask: np.ndarray,
    output_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: str
) -> str:
    """
    Save mask to disk as PNG file.

    Args:
        mask: Binary mask (H, W) numpy array
        output_dir: Directory to save mask
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name

    Returns:
        Path to saved mask file
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(output_dir, mask_filename)

    cv2.imwrite(mask_path, mask)
    return mask_path


def load_mask(
    mask_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Load mask from disk.

    Tries multiple filename patterns for compatibility:
    1. New pattern with exact name
    2. Legacy pattern
    3. Pattern matching by frame and ID (name mismatch fallback)

    Args:
        mask_dir: Directory containing mask files
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name (optional, defaults to "Object_{obj_id}")

    Returns:
        Mask as numpy array (H, W) or None if not found
    """
    if obj_name is None:
        obj_name = f"Object_{obj_id}"

    # Try 1: Exact name match with new pattern
    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(mask_dir, mask_filename)

    if os.path.exists(mask_path):
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Try 2: Legacy pattern (for backward compatibility)
    mask_filename_legacy = generate_legacy_mask_filename(frame_idx, obj_id)
    mask_path_legacy = os.path.join(mask_dir, mask_filename_legacy)

    if os.path.exists(mask_path_legacy):
        print(f"WARNING: Using legacy mask pattern for frame {frame_idx}, obj {obj_id}")
        return cv2.imread(mask_path_legacy, cv2.IMREAD_GRAYSCALE)

    # Try 3: Pattern matching by frame and ID (name mismatch fallback)
    if os.path.isdir(mask_dir):
        for filename in os.listdir(mask_dir):
            match = re.match(rf'mask_f{frame_idx:06d}_(.+)_id{obj_id}\.png', filename)
            if match:
                found_name = match.group(1)
                print(f"WARNING: Mask name mismatch for obj {obj_id}. "
                      f"Expected '{obj_name}', found '{found_name}'. Using found mask.")
                mask_path_fallback = os.path.join(mask_dir, filename)
                return cv2.imread(mask_path_fallback, cv2.IMREAD_GRAYSCALE)

    # Try 4: NPZ bundle (masks_f{frame:06d}.npz with a key matching the object id)
    npz_path = os.path.join(mask_dir, f"masks_f{frame_idx:06d}.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            for key in data.files:
                m = re.match(rf'mask_f{frame_idx:06d}_(.+)_id{obj_id}$', key)
                if m:
                    return (data[key] > 0).astype(np.uint8) * 255
        except Exception:
            pass

    print(f"WARNING: Mask file not found for frame {frame_idx}, obj {obj_id}")
    return None


def load_all_masks_for_frame(
    mask_dir: str,
    frame_idx: int,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Load every object mask for one frame, merging PNG files and NPZ bundle.

    PNG takes priority over NPZ when both contain the same object ID.
    Returns a list of (obj_id, obj_name, mask_uint8) tuples where mask_uint8 is
    a (H, W) uint8 array with values 0 or 255.
    """
    mask_dir_path = Path(mask_dir)
    # Use dict keyed by obj_id so PNG can override NPZ on collision
    results: dict[int, Tuple[str, np.ndarray]] = {}

    # --- PNG files ---
    png_files = sorted(mask_dir_path.glob(f"mask_f{frame_idx:06d}_*.png"))
    for mask_file in png_files:
        parsed = parse_mask_filename(mask_file.name)
        if parsed is None:
            continue
        _, obj_name, obj_id = parsed
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            results[obj_id] = (obj_name, mask)

    # --- Per-object NPZ files (SAM3 link / covered-union path; array key 'mask') ---
    _perobj_npz_pat = re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)\.npz$")
    for mask_file in sorted(mask_dir_path.glob(f"mask_f{frame_idx:06d}_*.npz")):
        m = _perobj_npz_pat.match(mask_file.name)
        if m is None:
            continue
        obj_name, obj_id = m.group(2), int(m.group(3))
        if obj_id in results:
            continue  # PNG already loaded — PNG wins
        try:
            data = np.load(str(mask_file))
            if 'mask' in data.files:
                results[obj_id] = (obj_name, (data['mask'] > 0).astype(np.uint8) * 255)
        except Exception:
            pass

    # --- NPZ bundle (adds objects not found as PNG) ---
    npz_path = mask_dir_path / f"masks_f{frame_idx:06d}.npz"
    if npz_path.exists():
        _npz_key_pat = re.compile(r"^mask_f(\d{6})_(.+)_id(\d+)$")
        try:
            data = np.load(str(npz_path))
            for key in data.files:
                m = _npz_key_pat.match(key)
                if not m:
                    continue
                obj_id = int(m.group(3))
                if obj_id in results:
                    continue  # PNG already loaded — PNG wins
                obj_name = m.group(2)
                arr = (data[key] > 0).astype(np.uint8) * 255
                results[obj_id] = (obj_name, arr)
        except Exception:
            pass

    return [(oid, oname, mask) for oid, (oname, mask) in results.items()]


def export_mask_to_disk(
    mask: np.ndarray,
    output_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: str,
    obj_color: Tuple[int, int, int] = (255, 0, 0)
) -> Dict[str, Any]:
    """
    Export mask to disk and return metadata.

    Args:
        mask: Binary mask (H, W) numpy array
        output_dir: Directory to save mask
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name
        obj_color: Object color as (R, G, B) tuple

    Returns:
        Metadata dictionary containing mask information
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(output_dir, mask_filename)

    cv2.imwrite(mask_path, mask)

    metadata = {
        'frame_idx': frame_idx,
        'obj_id': obj_id,
        'mask_file': mask_filename,
        'mask_path': mask_path,
        'object_name': obj_name,
        'color': obj_color,
        'mask_shape': mask.shape
    }

    return metadata


# =============================================================================
# Quality Metrics Functions
# =============================================================================

def _mask_to_bool(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Binarize a mask (any non-zero value) and resize it to (height, width) if needed."""
    if mask.shape != (height, width):
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
        mask = np.array(mask_pil)
    return mask > 0


def _union_xor_change_count(prev_masks: Dict[int, np.ndarray],
                            cur_masks: Dict[int, np.ndarray],
                            height: int, width: int) -> int:
    """
    Count pixels where at least one object's mask changed between two frames.

    Computes |OR_k (M_k(prev) XOR M_k(cur))| over the union of object IDs present
    in either frame; an object missing from one frame counts as an empty mask there.
    Unlike diffing a single obj_id label map, this is not affected by overlapping
    masks (no last-writer-wins) or by obj_id overflowing the label dtype. Masks
    must already be boolean at (height, width).
    """
    changed = np.zeros((height, width), dtype=bool)
    for obj_id in prev_masks.keys() | cur_masks.keys():
        prev = prev_masks.get(obj_id)
        cur = cur_masks.get(obj_id)
        if prev is None:
            changed |= cur
        elif cur is None:
            changed |= prev
        else:
            changed |= prev ^ cur
    return int(np.count_nonzero(changed))

class IncrementalQualityMetricsCalculator:
    """
    Calculate quality metrics incrementally during video propagation.

    This class enables calculating inter-frame changes and background ratios
    during forward/backward propagation passes, avoiding the need to load
    all masks from disk after processing.

    Usage:
        calculator = IncrementalQualityMetricsCalculator((height, width), num_frames)

        # During forward propagation (frames 0 to N-1):
        for frame_idx, masks in forward_propagation():
            calculator.update_forward(frame_idx, masks)

        # During backward propagation (frames N-1 to 0):
        for frame_idx, masks in backward_propagation():
            calculator.update_backward(frame_idx, masks)

        # Get final results
        inter_frame_changes, background_ratios = calculator.get_results()
    """

    def __init__(self, frame_dimensions: Tuple[int, int], num_frames: int):
        """
        Initialize the incremental calculator.

        Args:
            frame_dimensions: (height, width) of frames
            num_frames: Total number of frames in the video
        """
        self.height, self.width = frame_dimensions
        self.total_pixels = self.height * self.width
        self.num_frames = num_frames

        # Initialize arrays with zeros (will be filled during propagation)
        self.inter_frame_changes: List[float] = [0.0] * num_frames
        self.background_ratios: List[float] = [0.0] * num_frames
        self.overlap_ratios: List[float] = [0.0] * num_frames

        # State for forward pass: previous frame's per-object boolean masks
        self._forward_prev_masks: Optional[Dict[int, np.ndarray]] = None

        # State for backward pass: next frame's (in forward order) per-object masks
        self._backward_next_masks: Optional[Dict[int, np.ndarray]] = None

        # Track which pass has been run
        self._forward_completed = False
        self._backward_completed = False

    def _normalize_masks(self, object_masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Convert object masks to boolean arrays at frame resolution.

        Args:
            object_masks: Dictionary mapping obj_id -> mask array (H, W)

        Returns:
            Dictionary mapping obj_id -> bool array (H, W); None masks are dropped
        """
        return {
            obj_id: _mask_to_bool(mask, self.height, self.width)
            for obj_id, mask in object_masks.items()
            if mask is not None
        }

    def _calculate_background_ratio(self, bool_masks: Dict[int, np.ndarray]) -> float:
        """Calculate the ratio of pixels not covered by any object."""
        union = np.zeros((self.height, self.width), dtype=bool)
        for mask in bool_masks.values():
            union |= mask
        return 1.0 - (np.count_nonzero(union) / self.total_pixels)

    def _calculate_change_ratio(self, masks1: Dict[int, np.ndarray],
                                masks2: Dict[int, np.ndarray]) -> float:
        """Ratio of pixels where any object's mask changed (union of per-object XORs)."""
        return _union_xor_change_count(masks1, masks2, self.height, self.width) / self.total_pixels

    def _calculate_overlap_ratio(self, bool_masks: Dict[int, np.ndarray]) -> float:
        """
        Calculate the ratio of overlapping category assignments.

        For each pixel, count how many objects claim it. For pixels claimed by
        multiple objects, count the excess assignments (count - 1). For example:
        - Pixel in 1 object: contributes 0
        - Pixel in 2 objects: contributes 1
        - Pixel in 3 objects: contributes 2

        Args:
            bool_masks: Dictionary mapping obj_id -> bool array (H, W)

        Returns:
            Ratio of excess category assignments (0.0 = no overlap, higher = more overlap)
        """
        # Create a count map: how many objects claim each pixel
        overlap_count = np.zeros((self.height, self.width), dtype=np.int32)
        for mask in bool_masks.values():
            overlap_count += mask

        # For pixels with count > 0, subtract 1 to get the excess overlap count
        # (pixels in only 1 category don't count as overlap)
        excess_overlaps = np.where(overlap_count > 0, overlap_count - 1, 0)

        # Total excess assignments across all pixels, as ratio of total pixels
        return np.sum(excess_overlaps) / self.total_pixels

    def update_forward(self, frame_idx: int, object_masks: Dict[int, np.ndarray]) -> None:
        """
        Update metrics during forward propagation (frames 0 to N-1).

        Call this for each frame in forward order as masks are generated.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return

        bool_masks = self._normalize_masks(object_masks)

        # Calculate background ratio
        self.background_ratios[frame_idx] = self._calculate_background_ratio(bool_masks)

        # Calculate overlap ratio
        self.overlap_ratios[frame_idx] = self._calculate_overlap_ratio(bool_masks)

        # Calculate inter-frame change
        if self._forward_prev_masks is None:
            # First frame: no previous to compare
            self.inter_frame_changes[frame_idx] = 0.0
        else:
            self.inter_frame_changes[frame_idx] = self._calculate_change_ratio(
                self._forward_prev_masks, bool_masks
            )

        # Store current masks for next iteration (fresh arrays, no aliasing with caller)
        self._forward_prev_masks = bool_masks

    def update_backward(self, frame_idx: int, object_masks: Dict[int, np.ndarray]) -> None:
        """
        Update metrics during backward propagation (frames N-1 to 0).

        Call this for each frame in backward order. This updates the metrics
        calculated during forward pass with the final mask values.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return

        bool_masks = self._normalize_masks(object_masks)

        # Update background ratio with final mask
        self.background_ratios[frame_idx] = self._calculate_background_ratio(bool_masks)

        # Update overlap ratio with final mask
        self.overlap_ratios[frame_idx] = self._calculate_overlap_ratio(bool_masks)

        # Update inter-frame change for the NEXT frame (in forward order)
        # When processing frame i in backward order, we have:
        # - bool_masks: masks for frame i
        # - _backward_next_masks: masks for frame i+1 (from previous backward iteration)
        # inter_frame_changes[i+1] = change from frame i to frame i+1
        if self._backward_next_masks is not None and frame_idx + 1 < self.num_frames:
            self.inter_frame_changes[frame_idx + 1] = self._calculate_change_ratio(
                bool_masks, self._backward_next_masks
            )

        # Store current masks for next backward iteration
        self._backward_next_masks = bool_masks

        # Mark backward pass for frame 0 as completed
        if frame_idx == 0:
            self._backward_completed = True

    def update_single_pass(self, frame_idx: int, object_masks: Dict[int, np.ndarray],
                           reverse: bool = False) -> None:
        """
        Update metrics for a single propagation pass (forward or backward only).

        This is useful when only one direction propagation is performed.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
            reverse: True if processing in backward direction
        """
        if reverse:
            self.update_backward(frame_idx, object_masks)
        else:
            self.update_forward(frame_idx, object_masks)

    def get_results(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Get the calculated quality metrics.

        Returns:
            Tuple of (inter_frame_changes, background_ratios, overlap_ratios)
        """
        return self.inter_frame_changes, self.background_ratios, self.overlap_ratios

    def print_summary(self) -> None:
        """Print a summary of the calculated metrics."""
        print(f"Quality metrics calculated for {self.num_frames} frames")
        if len(self.inter_frame_changes) > 1:
            # Skip frame 0 for mean inter-frame change calculation
            mean_change = np.mean(self.inter_frame_changes[1:])
            print(f"  Mean inter-frame change: {mean_change:.3f}")
        print(f"  Mean background ratio: {np.mean(self.background_ratios):.3f}")
        print(f"  Mean overlap ratio: {np.mean(self.overlap_ratios):.3f}")


class QualityMetricsCalculator:
    """
    Calculate segmentation quality metrics: inter-frame changes and background ratios.

    This class is designed to be reusable across different contexts (UI, batch processing).
    """

    def __init__(self, frame_dimensions: Tuple[int, int], num_frames: int):
        """
        Initialize the calculator.

        Args:
            frame_dimensions: (height, width) of frames
            num_frames: Total number of frames in the video
        """
        self.height, self.width = frame_dimensions
        self.total_pixels = self.height * self.width
        self.num_frames = num_frames

        self.inter_frame_changes: List[float] = []
        self.background_ratios: List[float] = []
        self.overlap_ratios: List[float] = []

    def calculate(
        self,
        masks: Dict[int, Dict[int, Any]],
        load_mask_func: Callable[[int, int], Optional[np.ndarray]]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate quality metrics for all frames.

        Args:
            masks: Dictionary mapping frame_idx -> {obj_id -> mask_data}
            load_mask_func: Function(frame_idx, obj_id) -> numpy array mask or None

        Returns:
            Tuple of (inter_frame_changes, background_ratios, overlap_ratios)
        """
        print("Calculating segmentation quality metrics...")

        self.inter_frame_changes = []
        self.background_ratios = []
        self.overlap_ratios = []

        prev_masks: Optional[Dict[int, np.ndarray]] = None

        for frame_idx in range(self.num_frames):
            # Load this frame's per-object boolean masks
            cur_masks: Dict[int, np.ndarray] = {}
            if frame_idx in masks:
                for obj_id in masks[frame_idx]:
                    mask = load_mask_func(frame_idx, obj_id)
                    if mask is not None:
                        cur_masks[obj_id] = _mask_to_bool(mask, self.height, self.width)

            # Union of all objects and per-pixel overlap count
            union = np.zeros((self.height, self.width), dtype=bool)
            overlap_count = np.zeros((self.height, self.width), dtype=np.int32)
            for mask in cur_masks.values():
                union |= mask
                overlap_count += mask

            # Calculate background ratio
            object_pixels = np.count_nonzero(union)
            bg_ratio = 1.0 - (object_pixels / self.total_pixels)
            self.background_ratios.append(bg_ratio)

            # Calculate overlap ratio (excess assignments: count - 1 for pixels with count > 0)
            excess_overlaps = np.where(overlap_count > 0, overlap_count - 1, 0)
            overlap_ratio = np.sum(excess_overlaps) / self.total_pixels
            self.overlap_ratios.append(overlap_ratio)

            # Calculate inter-frame change
            if prev_masks is None:
                # First frame: no previous frame to compare
                self.inter_frame_changes.append(0.0)
            else:
                # Count pixels where any object's mask changed (union of per-object XORs)
                changed_pixels = _union_xor_change_count(prev_masks, cur_masks, self.height, self.width)
                change_ratio = changed_pixels / self.total_pixels
                self.inter_frame_changes.append(change_ratio)

            prev_masks = cur_masks

        print(f"Calculated metrics for {self.num_frames} frames")
        if len(self.inter_frame_changes) > 1:
            print(f"  Mean inter-frame change: {np.mean(self.inter_frame_changes[1:]):.3f}")
        print(f"  Mean background ratio: {np.mean(self.background_ratios):.3f}")
        print(f"  Mean overlap ratio: {np.mean(self.overlap_ratios):.3f}")

        return self.inter_frame_changes, self.background_ratios, self.overlap_ratios

    def calculate_grouped(
        self,
        masks: Dict[int, Dict[int, Any]],
        load_mask_func: Callable[[int, int], Optional[np.ndarray]],
        obj_id_to_group: Dict[int, Any],
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics with per-group (per-concept) inter-frame change
        and overlap ratios, plus a single global background ratio.

        Rationale:
          - Background ratio measures how much of the frame is segmented at all;
            a single global number is the meaningful summary.
          - Inter-frame change is only meaningful *within* a category; a jittery
            concept should not be averaged away by a stable one.  The per-concept
            track is normalized by that concept's peak single-frame area (max over
            all frames of |union of its instances|) rather than by the total frame
            pixel count, so concepts are comparable regardless of how much of the
            frame they cover.  The global 'inter_frame_changes' track keeps the
            old frame-pixel normalization for backward compatibility.
          - Overlap is measured per concept as a one-vs-rest cross-concept ratio:
            the fraction of that concept's segmented area that is also claimed by
            at least one *other* concept.  Within-concept overlap is not measured
            because SAM3 already enforces pixel-wise mutual exclusivity among the
            instances of a single concept (Sam3VideoInference._postprocess_output
            calls _apply_object_wise_non_overlapping_constraints unconditionally),
            so it is ~0 by construction.  A global excess-assignment overlap track
            is still returned under 'overlap_ratios' for backward compatibility.

        Args:
            masks: {frame_idx: {obj_id: mask_data}}
            load_mask_func: (frame_idx, obj_id) -> np.ndarray | None
            obj_id_to_group: {obj_id: group_key} (typically concept name)

        Returns a dict:
            'background_ratios':            [float] * num_frames  (global)
            'inter_frame_changes':          [float] * num_frames  (global aggregate,
                                             identical semantics to calculate(); kept
                                             for backward compatibility)
            'overlap_ratios':               [float] * num_frames  (global excess-
                                             assignment aggregate, backward compat)
            'group_names':                  [group_key, ...]  (stable order)
            'inter_frame_changes_by_group': {group_key: [float] * num_frames}
                                             pixels where any of the concept's
                                             instance masks changed between
                                             consecutive frames (union of per-
                                             instance XORs) / max single-frame
                                             union area of that concept
            'overlap_ratios_by_group':      {group_key: [float] * num_frames}
                                             one-vs-rest cross-concept ratio:
                                             |concept_area ∩ (claimed by another
                                             concept)| / |concept_area|; 0.0 when
                                             the concept is absent on that frame
        """
        print("Calculating segmentation quality metrics (per-concept)...")

        group_names = list(dict.fromkeys(
            obj_id_to_group[k] for k in sorted(obj_id_to_group)
        ))
        group_of = obj_id_to_group

        bg_ratios: List[float] = []
        inter_global: List[float] = []
        overlap_global: List[float] = []
        # Per-concept inter-frame change is stored as a raw changed-pixel count here
        # and normalized after the loop by that concept's *peak* single-frame area
        # (max over all frames of |union of the concept's instances|).  Dividing by a
        # per-concept constant makes concepts comparable regardless of size, without
        # the small-frame blow-up a per-frame-union denominator would cause.
        inter_raw_by_group: Dict[Any, List[float]] = {g: [] for g in group_names}
        max_union_area: Dict[Any, int] = {g: 0 for g in group_names}
        overlap_by_group: Dict[Any, List[float]] = {g: [] for g in group_names}

        prev_masks: Optional[Dict[int, np.ndarray]] = None

        for frame_idx in range(self.num_frames):
            # Load this frame's per-object boolean masks
            cur_masks: Dict[int, np.ndarray] = {}
            if frame_idx in masks:
                for obj_id in masks[frame_idx]:
                    mask = load_mask_func(frame_idx, obj_id)
                    if mask is not None:
                        cur_masks[obj_id] = _mask_to_bool(mask, self.height, self.width)

            global_union = np.zeros((self.height, self.width), dtype=bool)
            global_overlap_count = np.zeros((self.height, self.width), dtype=np.int32)
            group_union = {
                g: np.zeros((self.height, self.width), dtype=bool) for g in group_names
            }
            for obj_id, hit in cur_masks.items():
                global_union |= hit
                global_overlap_count += hit
                g = group_of.get(obj_id)
                if g in group_union:
                    group_union[g] |= hit

            # Number of *distinct concepts* covering each pixel (for one-vs-rest overlap)
            concept_coverage_count = np.zeros((self.height, self.width), dtype=np.int32)
            for g in group_names:
                concept_coverage_count += group_union[g]

            # Global background ratio
            object_pixels = np.count_nonzero(global_union)
            bg_ratios.append(1.0 - (object_pixels / self.total_pixels))

            # Global aggregates (backward compatible with calculate())
            excess = np.where(global_overlap_count > 0, global_overlap_count - 1, 0)
            overlap_global.append(float(np.sum(excess) / self.total_pixels))
            # Inter-frame change: union of per-object XORs, globally and per concept.
            # Each object's XOR is computed once and OR'd into both change maps; an
            # object missing from one frame counts as an empty mask there.
            if prev_masks is None:
                inter_global.append(0.0)
            else:
                global_changed = np.zeros((self.height, self.width), dtype=bool)
                group_changed = {
                    g: np.zeros((self.height, self.width), dtype=bool) for g in group_names
                }
                for obj_id in prev_masks.keys() | cur_masks.keys():
                    prev = prev_masks.get(obj_id)
                    cur = cur_masks.get(obj_id)
                    if prev is None:
                        diff = cur
                    elif cur is None:
                        diff = prev
                    else:
                        diff = prev ^ cur
                    global_changed |= diff
                    g = group_of.get(obj_id)
                    if g in group_changed:
                        group_changed[g] |= diff
                inter_global.append(np.count_nonzero(global_changed) / self.total_pixels)

            # Per-group metrics
            for g in group_names:
                union = group_union[g]
                area = int(np.count_nonzero(union))
                if area > max_union_area[g]:
                    max_union_area[g] = area
                if area == 0:
                    overlap_by_group[g].append(0.0)
                else:
                    contested = int(np.count_nonzero(union & (concept_coverage_count >= 2)))
                    overlap_by_group[g].append(contested / area)
                if prev_masks is None:
                    inter_raw_by_group[g].append(0.0)
                else:
                    inter_raw_by_group[g].append(float(np.count_nonzero(group_changed[g])))

            prev_masks = cur_masks

        # Normalize per-concept inter-frame change by the concept's peak area
        inter_by_group: Dict[Any, List[float]] = {}
        for g in group_names:
            denom = max_union_area[g]
            if denom > 0:
                inter_by_group[g] = [c / denom for c in inter_raw_by_group[g]]
            else:
                inter_by_group[g] = [0.0 for _ in inter_raw_by_group[g]]

        print(f"Calculated per-concept metrics for {self.num_frames} frames, "
              f"{len(group_names)} concept(s)")
        if len(bg_ratios) > 1:
            print(f"  Mean background ratio: {np.mean(bg_ratios):.3f}")

        return {
            'background_ratios': bg_ratios,
            'inter_frame_changes': inter_global,
            'overlap_ratios': overlap_global,
            'group_names': group_names,
            'inter_frame_changes_by_group': inter_by_group,
            'overlap_ratios_by_group': overlap_by_group,
        }


def calculate_quality_metrics(
    masks: Dict[int, Dict[int, Any]],
    load_mask_func: Callable[[int, int], Optional[np.ndarray]],
    frame_dimensions: Tuple[int, int],
    num_frames: int
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convenience function to calculate quality metrics without instantiating a class.

    Args:
        masks: Dictionary mapping frame_idx -> {obj_id -> mask_data}
        load_mask_func: Function(frame_idx, obj_id) -> numpy array mask or None
        frame_dimensions: (height, width) of frames
        num_frames: Total number of frames

    Returns:
        Tuple of (inter_frame_changes, background_ratios, overlap_ratios)
    """
    calculator = QualityMetricsCalculator(frame_dimensions, num_frames)
    return calculator.calculate(masks, load_mask_func)


def calculate_quality_metrics_grouped(
    masks: Dict[int, Dict[int, Any]],
    load_mask_func: Callable[[int, int], Optional[np.ndarray]],
    obj_id_to_group: Dict[int, Any],
    frame_dimensions: Tuple[int, int],
    num_frames: int
) -> Dict[str, Any]:
    """
    Convenience wrapper around QualityMetricsCalculator.calculate_grouped().

    Returns the dict documented on calculate_grouped(): a global background ratio
    and global aggregates plus per-group inter-frame-change / overlap ratios.
    """
    calculator = QualityMetricsCalculator(frame_dimensions, num_frames)
    return calculator.calculate_grouped(masks, load_mask_func, obj_id_to_group)


def save_quality_metrics(
    output_dir: str,
    inter_frame_changes: List[float],
    background_ratios: List[float],
    overlap_ratios: Optional[List[float]] = None
) -> bool:
    """
    Save quality metrics to quality_metrics.npz in output directory.

    Args:
        output_dir: Directory to save the metrics file
        inter_frame_changes: List of inter-frame change ratios
        background_ratios: List of background ratios
        overlap_ratios: List of overlap ratios (optional, for backward compatibility)

    Returns:
        True if saved successfully, False otherwise
    """
    if not inter_frame_changes or not background_ratios:
        print("No quality metrics to save")
        return False

    try:
        metrics_path = os.path.join(output_dir, "quality_metrics.npz")

        save_dict = {
            'inter_frame_changes': np.array(inter_frame_changes),
            'background_ratios': np.array(background_ratios),
            'frame_count': len(inter_frame_changes)
        }

        # Add overlap ratios if provided
        if overlap_ratios is not None:
            save_dict['overlap_ratios'] = np.array(overlap_ratios)

        np.savez_compressed(metrics_path, **save_dict)

        print(f"Saved quality metrics to {metrics_path}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to save quality metrics: {e}")
        return False


def save_quality_metrics_grouped(output_dir: str, result: Dict[str, Any]) -> bool:
    """
    Save the dict produced by calculate_quality_metrics_grouped() to
    quality_metrics.npz.

    The file stays backward compatible: the global keys ('inter_frame_changes',
    'background_ratios', 'overlap_ratios', 'frame_count') are written exactly as
    save_quality_metrics() writes them, so older readers keep working.  Per-concept
    data is added under extra keys that older readers simply ignore:
        'group_names'                  1-D unicode array
        'inter_frame_changes_by_group' 2-D array, shape (n_groups, n_frames)
        'overlap_ratios_by_group'      2-D array, shape (n_groups, n_frames)
    """
    inter = list(result.get('inter_frame_changes') or [])
    bg = list(result.get('background_ratios') or [])
    if not inter or not bg:
        print("No quality metrics to save")
        return False

    try:
        metrics_path = os.path.join(output_dir, "quality_metrics.npz")
        save_dict = {
            'inter_frame_changes': np.array(inter),
            'background_ratios': np.array(bg),
            'frame_count': len(inter),
        }
        overlap = result.get('overlap_ratios')
        if overlap is not None:
            save_dict['overlap_ratios'] = np.array(overlap)

        group_names = list(result.get('group_names') or [])
        if group_names:
            inter_bg = result.get('inter_frame_changes_by_group', {}) or {}
            overlap_bg = result.get('overlap_ratios_by_group', {}) or {}
            save_dict['group_names'] = np.array([str(g) for g in group_names])
            save_dict['inter_frame_changes_by_group'] = np.array(
                [inter_bg[g] for g in group_names], dtype=float)
            save_dict['overlap_ratios_by_group'] = np.array(
                [overlap_bg[g] for g in group_names], dtype=float)

        np.savez_compressed(metrics_path, **save_dict)
        print(f"Saved quality metrics to {metrics_path}"
              + (f" ({len(group_names)} concept(s))" if group_names else ""))
        return True
    except Exception as e:
        print(f"WARNING: Failed to save quality metrics: {e}")
        return False


# =============================================================================
# Video Compression Functions
# =============================================================================

def compress_video_with_ffmpeg(
    input_path: str,
    output_path: str,
    crf: int = 23,
    preset: str = 'medium',
    gop: int = 10
) -> bool:
    """
    Re-encode video with FFmpeg using scrubbing-friendly compression.

    Prefers short-GOP x264; falls back to MJPEG if unavailable.

    Args:
        input_path: Path to the input (uncompressed) video
        output_path: Path for the output (compressed) video
        crf: Quality setting (0-51, lower=better, 23=default)
        preset: Encoding speed preset (medium recommended)
        gop: GOP (Group of Pictures) size for keyframe interval

    Returns:
        bool: True if compression succeeded, False otherwise
    """
    import subprocess
    import shutil

    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print("WARNING: ffmpeg not found, skipping compression")
        shutil.move(input_path, output_path)
        return False

    def has_libx264():
        try:
            r = subprocess.run(
                [ffmpeg_path, '-encoders'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return 'libx264' in r.stdout
        except Exception:
            return False

    try:
        if has_libx264():
            print(f"Compressing with x264 (CRF={crf}, GOP={gop})")

            cmd = [
                ffmpeg_path,
                '-loglevel', 'error',
                '-i', input_path,

                # Video encoding
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', str(crf),

                # Scrubbing-friendly settings
                '-g', str(gop),
                '-keyint_min', str(gop),
                '-bf', '0',
                '-x264-params', 'scenecut=0',

                # Compatibility
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',

                '-y',
                output_path
            ]
        else:
            print("libx264 unavailable; falling back to MJPEG")

            cmd = [
                ffmpeg_path,
                '-loglevel', 'error',
                '-i', input_path,
                '-c:v', 'mjpeg',
                '-q:v', '5',   # 2–5 is reasonable
                '-pix_fmt', 'yuvj420p',
                '-y',
                output_path
            ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error:\n{result.stderr}")
            shutil.move(input_path, output_path)
            return False

        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        reduction = (1 - compressed_size / original_size) * 100

        print(
            f"Compressed: {original_size/1e6:.1f}MB → "
            f"{compressed_size/1e6:.1f}MB ({reduction:.1f}% reduction)"
        )

        os.remove(input_path)
        return True

    except Exception as e:
        print(f"WARNING: FFmpeg failed: {e}")
        if os.path.exists(input_path):
            shutil.move(input_path, output_path)
        return False


def load_quality_metrics(output_dir: str) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """
    Load quality metrics from quality_metrics.npz if available.

    Args:
        output_dir: Directory containing the metrics file

    Returns:
        Tuple of (inter_frame_changes, background_ratios, overlap_ratios), or (None, None, None) if not found/failed
        For backward compatibility, overlap_ratios may be None if not present in the file.
    """
    metrics_path = os.path.join(output_dir, "quality_metrics.npz")

    if not os.path.exists(metrics_path):
        print("No quality metrics file found (OK for older results)")
        return None, None, None

    try:
        data = np.load(metrics_path)

        inter_frame_changes = data['inter_frame_changes'].tolist()
        background_ratios = data['background_ratios'].tolist()

        # Load overlap ratios if available (backward compatibility)
        overlap_ratios = None
        if 'overlap_ratios' in data:
            overlap_ratios = data['overlap_ratios'].tolist()

        print(f"Loaded quality metrics: {len(inter_frame_changes)} values")
        if overlap_ratios is not None:
            print(f"  (including overlap ratios)")
        return inter_frame_changes, background_ratios, overlap_ratios
    except Exception as e:
        print(f"WARNING: Failed to load quality metrics: {e}")
        return None, None, None


def load_quality_metrics_grouped(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load quality_metrics.npz including per-concept metrics when present.

    Returns a dict with the same shape as calculate_quality_metrics_grouped()'s
    output, or None if the file is missing/unreadable.  Files written by the
    older (non-grouped) version load fine: 'group_names' and the *_by_group
    entries come back as None.
    """
    metrics_path = os.path.join(output_dir, "quality_metrics.npz")
    if not os.path.exists(metrics_path):
        print("No quality metrics file found (OK for older results)")
        return None

    try:
        data = np.load(metrics_path)
        out: Dict[str, Any] = {
            'inter_frame_changes': data['inter_frame_changes'].tolist(),
            'background_ratios': data['background_ratios'].tolist(),
            'overlap_ratios': (data['overlap_ratios'].tolist()
                               if 'overlap_ratios' in data else None),
            'group_names': None,
            'inter_frame_changes_by_group': None,
            'overlap_ratios_by_group': None,
        }

        if 'group_names' in data:
            names = [str(x) for x in data['group_names'].tolist()]
            out['group_names'] = names
            if 'inter_frame_changes_by_group' in data:
                arr = data['inter_frame_changes_by_group']
                out['inter_frame_changes_by_group'] = {
                    n: arr[i].tolist() for i, n in enumerate(names)
                }
            if 'overlap_ratios_by_group' in data:
                arr = data['overlap_ratios_by_group']
                out['overlap_ratios_by_group'] = {
                    n: arr[i].tolist() for i, n in enumerate(names)
                }
            print(f"Loaded quality metrics: {len(names)} concept(s), "
                  f"{len(out['inter_frame_changes'])} frames")
        else:
            print(f"Loaded quality metrics: "
                  f"{len(out['inter_frame_changes'])} values (no per-concept data)")
        return out
    except Exception as e:
        print(f"WARNING: Failed to load quality metrics: {e}")
        return None


# =============================================================================
# Frame Source Abstractions for Video Export
# =============================================================================

from abc import ABC, abstractmethod
from pathlib import Path


class FrameSource(ABC):
    """
    Abstract base class for providing frames during video export.

    Implementations can load frames from video files, extracted JPEG directories,
    or a combination of both (hybrid approach).
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of frames."""
        pass

    @abstractmethod
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a frame by index.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            Frame as numpy array (H, W, 3) in BGR format, or None if not available
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get frame dimensions.

        Returns:
            Tuple of (height, width)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the frame source."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VideoFileFrameSource(FrameSource):
    """
    Load frames directly from a video file.

    This is the traditional approach - reads frames on demand from the video.
    """

    def __init__(self, video_path: str):
        """
        Initialize video file frame source.

        Args:
            video_path: Path to the video file
        """
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._last_read_idx = -1

    def __len__(self) -> int:
        return self._total_frames

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= self._total_frames:
            return None

        # Seek if not sequential read
        if frame_idx != self._last_read_idx + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self._cap.read()
        if ret:
            self._last_read_idx = frame_idx
            return frame
        return None

    def get_dimensions(self) -> Tuple[int, int]:
        return (self._height, self._width)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ExtractedFramesSource(FrameSource):
    """
    Load frames from a directory of extracted JPEG/PNG images.

    This is faster than reading from video since JPEGs are already decoded
    and don't require sequential seeking.
    """

    def __init__(self, frame_dir: str):
        """
        Initialize extracted frames source.

        Args:
            frame_dir: Directory containing extracted frame images
        """
        self._frame_dir = Path(frame_dir)
        if not self._frame_dir.exists():
            raise ValueError(f"Frame directory does not exist: {frame_dir}")

        # Find all frame files (try both jpg and png)
        self._frame_files = sorted(self._frame_dir.glob("*.jpg"))
        if not self._frame_files:
            self._frame_files = sorted(self._frame_dir.glob("*.png"))

        if not self._frame_files:
            raise ValueError(f"No frame images found in: {frame_dir}")

        # Build frame index mapping (handle gaps in numbering)
        self._frame_index: Dict[int, Path] = {}
        for frame_path in self._frame_files:
            # Extract frame number from filename (e.g., "00001.jpg" -> 1)
            try:
                frame_num = int(frame_path.stem)
                self._frame_index[frame_num] = frame_path
            except ValueError:
                continue

        # Determine dimensions from first frame
        first_frame = cv2.imread(str(self._frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Cannot read first frame: {self._frame_files[0]}")
        self._height, self._width = first_frame.shape[:2]

        # Total frames is the maximum index + 1
        if self._frame_index:
            self._total_frames = max(self._frame_index.keys()) + 1
        else:
            self._total_frames = len(self._frame_files)

    def __len__(self) -> int:
        return self._total_frames

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= self._total_frames:
            return None

        frame_path = self._frame_index.get(frame_idx)
        if frame_path is None:
            return None

        return cv2.imread(str(frame_path))

    def get_dimensions(self) -> Tuple[int, int]:
        return (self._height, self._width)

    def has_frame(self, frame_idx: int) -> bool:
        """Check if a specific frame is available."""
        return frame_idx in self._frame_index

    def get_available_frames(self) -> set:
        """Get set of available frame indices."""
        return set(self._frame_index.keys())

    def close(self) -> None:
        # No resources to release for file-based source
        pass


class HybridFrameSource(FrameSource):
    """
    Try extracted frames first, fall back to video file.

    This is the most flexible option - uses fast JPEG loading when available,
    but falls back to reading from the original video for missing frames.

    Includes safety check to detect frame/video dimension mismatch.
    """

    def __init__(self, video_path: str, frame_dir: Optional[str] = None):
        """
        Initialize hybrid frame source.

        Args:
            video_path: Path to the original video file
            frame_dir: Optional path to extracted frames directory
        """
        self._video_source = VideoFileFrameSource(video_path)
        self._extracted_source: Optional[ExtractedFramesSource] = None
        self._available_extracted: set = set()

        if frame_dir and Path(frame_dir).exists():
            try:
                candidate_source = ExtractedFramesSource(frame_dir)

                # SAFETY CHECK: Validate frame dimensions match video
                video_h, video_w = self._video_source.get_dimensions()
                frame_h, frame_w = candidate_source.get_dimensions()

                if (frame_h, frame_w) != (video_h, video_w):
                    print(f"WARNING: Frame cache dimensions ({frame_w}x{frame_h}) "
                          f"don't match video ({video_w}x{video_h}). "
                          f"Ignoring cached frames.")
                else:
                    self._extracted_source = candidate_source
                    self._available_extracted = candidate_source.get_available_frames()
                    print(f"  Using {len(self._available_extracted)} cached frames from: {frame_dir}")
            except ValueError as e:
                print(f"WARNING: Could not use extracted frames: {e}")

    def __len__(self) -> int:
        return len(self._video_source)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        # Try extracted frames first if available
        if frame_idx in self._available_extracted and self._extracted_source:
            frame = self._extracted_source.get_frame(frame_idx)
            if frame is not None:
                return frame

        # Fall back to video file
        return self._video_source.get_frame(frame_idx)

    def get_dimensions(self) -> Tuple[int, int]:
        return self._video_source.get_dimensions()

    def uses_extracted_frames(self) -> bool:
        """Check if any extracted frames are being used."""
        return len(self._available_extracted) > 0

    def close(self) -> None:
        self._video_source.close()
        if self._extracted_source:
            self._extracted_source.close()


# =============================================================================
# Contrasting Text Color Utility
# =============================================================================

def get_contrasting_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Calculate contrasting text color (white or black) based on background luminance.

    Uses ITU-R BT.709 formula for relative luminance calculation.

    Args:
        bg_color: RGB color tuple (e.g., (255, 0, 0) for red)

    Returns:
        (R, G, B) tuple: (255, 255, 255) for white or (0, 0, 0) for black
    """
    if isinstance(bg_color, (list, tuple)) and len(bg_color) >= 3:
        r, g, b = bg_color[0], bg_color[1], bg_color[2]
    else:
        r, g, b = 255, 255, 255  # Default white

    # Calculate relative luminance (ITU-R BT.709)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Return white for dark colors, black for bright colors
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


# =============================================================================
# GPU Overlay Computation
# =============================================================================

def compute_overlay_gpu(
    frame: np.ndarray,
    mask_data_list: List[Tuple[np.ndarray, Tuple[int, int, int], str, int]],
    overlay_opacity: float = 0.4,
    device: str = 'cuda:3'
) -> Optional[np.ndarray]:
    """
    GPU-accelerated mask overlay computation.

    Moves frame and masks to GPU, performs accumulation and blending on GPU,
    returns result on CPU. Includes fallback to CPU on GPU errors.

    Args:
        frame: Input frame as numpy array (uint8, BGR)
        mask_data_list: List of (mask_bool, color_rgb, name, obj_id) tuples
        overlay_opacity: Opacity for overlay blending (0.0 to 1.0)
        device: GPU device string (e.g., 'cuda:3')

    Returns:
        Blended frame with overlays (uint8, BGR) or None if overlay failed
    """
    if not torch or not mask_data_list:
        return None

    try:
        with torch.no_grad():
            height, width = frame.shape[:2]

            # Convert frame to GPU tensor, normalize to [0, 1]
            frame_gpu = torch.from_numpy(frame).float().to(device) / 255.0

            # Initialize accumulators on GPU
            combined_overlay = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
            overlap_count = torch.zeros((height, width), dtype=torch.uint8, device=device)

            # Process each mask
            for mask_bool, color_rgb, name, obj_id in mask_data_list:
                # Convert mask to GPU
                mask_gpu = torch.from_numpy(mask_bool).to(device)

                # Convert RGB to BGR, normalize to [0, 1]
                color_bgr = torch.tensor(
                    (color_rgb[2], color_rgb[1], color_rgb[0]),
                    dtype=torch.float32,
                    device=device
                ) / 255.0

                # Accumulate color on masked regions
                combined_overlay[mask_gpu] += color_bgr
                overlap_count[mask_gpu] += 1

                # Clean up mask tensor immediately
                del mask_gpu

            # Average overlapping regions
            mask_pixels = overlap_count > 0
            if mask_pixels.any():
                combined_overlay[mask_pixels] /= overlap_count[mask_pixels].unsqueeze(1).float()

                # Blend on GPU
                alpha = overlay_opacity
                beta = 1.0 - alpha
                blended = (frame_gpu * beta + combined_overlay * alpha).clamp(0, 1)

                # Convert back to uint8 CPU array
                result = (blended * 255).byte().cpu().numpy()

                # Clean up GPU memory
                del frame_gpu, combined_overlay, overlap_count
                torch.cuda.empty_cache()

                return result

            # Clean up if no overlays
            del frame_gpu, combined_overlay, overlap_count
            torch.cuda.empty_cache()
            return None

    except RuntimeError as e:
        # GPU out of memory or other GPU error - return None for fallback to CPU
        print(f"GPU overlay error: {e}, falling back to CPU")
        torch.cuda.empty_cache()
        return None


def compute_overlay_cpu(
    frame: np.ndarray,
    mask_data_list: List[Tuple[np.ndarray, Tuple[int, int, int], str, int]],
    overlay_opacity: float = 0.4
) -> Optional[np.ndarray]:
    """
    CPU-based mask overlay computation (original implementation).

    Args:
        frame: Input frame as numpy array (uint8, BGR)
        mask_data_list: List of (mask_bool, color_rgb, name, obj_id) tuples
        overlay_opacity: Opacity for overlay blending (0.0 to 1.0)

    Returns:
        Blended frame with overlays (uint8, BGR) or None if overlay failed
    """
    if not mask_data_list:
        return None

    height, width = frame.shape[:2]

    # Create overlay with averaged colors for overlapping regions
    combined_overlay = np.zeros((height, width, 3), dtype=np.float32)
    overlap_count = np.zeros((height, width), dtype=np.int32)

    for mask_bool, color_rgb, name, obj_id in mask_data_list:
        # Convert RGB to BGR
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        combined_overlay[mask_bool] += color_bgr
        overlap_count[mask_bool] += 1

    # Average colors where masks overlap
    mask_pixels = overlap_count > 0
    if mask_pixels.any():
        combined_overlay[mask_pixels] /= overlap_count[mask_pixels, np.newaxis]
        combined_overlay = combined_overlay.astype(np.uint8)

        # Single blend operation
        result = cv2.addWeighted(frame, 1 - overlay_opacity,
                                combined_overlay, overlay_opacity, 0)
        return result

    return None


# =============================================================================
# Video Export Functions
# =============================================================================

def export_segmented_video(
    frame_source: FrameSource,
    masks_dir: str,
    get_mask_data: Callable[[int], List[Tuple[np.ndarray, Tuple[int, int, int], str, int]]],
    output_path: str,
    fps: float = 30.0,
    overlay_opacity: float = 0.4,
    compress: bool = True,
    crf: int = 23,
    use_gpu: bool = False,
    gpu_device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Export segmented video with mask overlays and text labels.

    This is the core export function that handles the common logic for both
    CLI and UI export workflows.

    Args:
        frame_source: FrameSource instance providing video frames
        masks_dir: Directory containing mask PNG files
        get_mask_data: Function that takes frame_idx and returns list of
                       (mask_array, rgb_color, name, obj_id) tuples
        output_path: Path for the output video file
        fps: Frames per second for output video
        overlay_opacity: Opacity for mask overlay (0.0 to 1.0)
        compress: Whether to compress with FFmpeg after writing
        crf: CRF value for FFmpeg compression (lower = better quality)
        use_gpu: Whether to use GPU-accelerated overlay computation
        gpu_device: GPU device string (e.g., 'cuda:0', 'cuda:3'). If None, defaults to 'cuda'
        progress_callback: Optional callback(current_frame, total_frames) for progress

    Returns:
        True if export succeeded, False otherwise
    """
    print("Exporting segmented video...")
    if use_gpu:
        if gpu_device is None:
            gpu_device = "cuda"
        print(f"  Using GPU overlay acceleration on device: {gpu_device}")

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = frame_source.get_dimensions()
    total_frames = len(frame_source)

    # Use temp file for compression workflow
    if compress:
        temp_path = output_path.parent / f"{output_path.stem}.temp.avi"
    else:
        temp_path = output_path.with_suffix(".avi")

    # Initialize video writer with MJPEG codec
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print("ERROR: Could not initialize video writer")
        return False

    processed_frames = 0

    try:
        for frame_idx in range(total_frames):
            frame = frame_source.get_frame(frame_idx)
            if frame is None:
                print(f"WARNING: Could not read frame {frame_idx}, skipping")
                continue

            # Ensure frame dimensions match expected
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            overlay_frame = frame.copy()

            # Get mask data for this frame
            mask_data_list = get_mask_data(frame_idx)

            if mask_data_list:
                # Preprocess masks to match frame dimensions
                preprocessed_masks = []
                for mask_bool, color_rgb, name, obj_id in mask_data_list:
                    # Resize mask if needed
                    if mask_bool.shape != (height, width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask_bool * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((width, height), PILImage.BILINEAR)
                        mask_bool = np.array(mask_pil) > 127

                    preprocessed_masks.append((mask_bool, color_rgb, name, obj_id))

                # Try GPU overlay computation if enabled
                if use_gpu and torch is not None:
                    blended = compute_overlay_gpu(frame, preprocessed_masks, overlay_opacity, gpu_device)
                    if blended is not None:
                        overlay_frame = blended
                    else:
                        # Fall back to CPU if GPU fails
                        blended = compute_overlay_cpu(frame, preprocessed_masks, overlay_opacity)
                        if blended is not None:
                            overlay_frame = blended
                else:
                    # Use CPU overlay computation
                    blended = compute_overlay_cpu(frame, preprocessed_masks, overlay_opacity)
                    if blended is not None:
                        overlay_frame = blended

                # Add text labels with smart contrast
                for mask_bool, color_rgb, name, obj_id in preprocessed_masks:
                    if mask_bool.any():
                        y_coords, x_coords = np.where(mask_bool)
                        if len(y_coords) > 0:
                            center_x = int(np.mean(x_coords))
                            center_y = int(np.mean(y_coords))

                            # Calculate contrasting text color (ITU-R BT.709)
                            text_color = get_contrasting_text_color(color_rgb)

                            cv2.putText(overlay_frame, name, (center_x, center_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            out.write(overlay_frame)
            processed_frames += 1

            if progress_callback:
                progress_callback(processed_frames, total_frames)
            elif processed_frames % 100 == 0:
                print(f"   Processed {processed_frames}/{total_frames} frames...")

    finally:
        out.release()

    # Compress with FFmpeg if requested
    if compress:
        print("Compressing video with FFmpeg...")
        final_path = output_path.with_suffix(".avi")
        success = compress_video_with_ffmpeg(str(temp_path), str(final_path), crf=crf)
        if success:
            print(f"OK: Exported and compressed segmented video to {final_path}")
        else:
            print(f"OK: Exported segmented video to {final_path} (compression skipped)")
    else:
        print(f"OK: Exported segmented video to {temp_path}")

    return True


def export_video_from_dict(
    video_path: str,
    masks_by_frame: Dict[int, Dict[int, Dict[str, Any]]],
    object_names: Dict[int, str],
    object_colors: Dict[int, Tuple[int, int, int]],
    output_dir: str,
    fps: float = 30.0,
    overlay_opacity: float = 0.4,
    compress: bool = True,
    crf: int = 23,
    frame_dir: Optional[str] = None,
    use_gpu: bool = False,
    gpu_device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Export segmented video from masks_by_frame dictionary (CLI usage).

    This is a convenience wrapper around export_segmented_video() for use
    with sam2_process.py workflow.

    Args:
        video_path: Path to the original video file
        masks_by_frame: Dict mapping frame_idx -> {obj_id -> mask_metadata}
        object_names: Dict mapping object_id -> name string
        object_colors: Dict mapping object_id -> (R, G, B) tuple
        output_dir: Directory for output files
        fps: Frames per second for output video
        overlay_opacity: Opacity for mask overlay (0.0 to 1.0)
        compress: Whether to compress with FFmpeg
        crf: CRF value for FFmpeg compression
        frame_dir: Optional path to extracted frames directory (for reuse)
        use_gpu: Whether to use GPU-accelerated overlay computation
        gpu_device: GPU device string (e.g., 'cuda:0', 'cuda:3'). If None, defaults to 'cuda'
        progress_callback: Optional progress callback

    Returns:
        True if export succeeded, False otherwise
    """
    masks_dir = Path(output_dir) / "masks"
    output_path = Path(output_dir) / "segmented_video.avi"

    # Create frame source - use hybrid if frame_dir provided
    frame_source = HybridFrameSource(video_path, frame_dir)

    def get_mask_data_from_dict(frame_idx: int) -> List[Tuple[np.ndarray, Tuple[int, int, int], str, int]]:
        """Load mask data from masks_by_frame dictionary."""
        result = []
        if frame_idx not in masks_by_frame:
            return result

        for obj_id, mask_data in masks_by_frame[frame_idx].items():
            # Load mask from disk (PNG or compressed NPZ bundle)
            mask_path = masks_dir / mask_data['filename']
            if mask_data['filename'].endswith('.npz'):
                try:
                    npz_data = np.load(str(mask_path))
                    npz_key = mask_data.get('npz_key', '')
                    if npz_key not in npz_data.files and 'mask' in npz_data.files:
                        # Per-object NPZ (SAM3 link / covered-union path) stores its
                        # single array under 'mask'; only bundle files carry npz_key.
                        npz_key = 'mask'
                    mask = (npz_data[npz_key] > 0).astype(np.uint8) * 255 if npz_key in npz_data.files else None
                except Exception:
                    mask = None
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"WARNING: Could not load mask: {mask_path}")
                continue

            mask_bool = (mask > 0).astype(bool)
            # Use stored name and color from masks_metadata (set during propagation)
            # These are the authoritative values - they may differ from input dicts
            name = mask_data.get('name', object_names.get(obj_id, f"Object_{obj_id}"))
            color = mask_data.get('color', object_colors.get(obj_id, (255, 0, 0)))

            result.append((mask_bool, color, name, obj_id))

        return result

    try:
        success = export_segmented_video(
            frame_source=frame_source,
            masks_dir=str(masks_dir),
            get_mask_data=get_mask_data_from_dict,
            output_path=str(output_path),
            fps=fps,
            overlay_opacity=overlay_opacity,
            compress=compress,
            crf=crf,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            progress_callback=progress_callback
        )
        return success
    finally:
        frame_source.close()
