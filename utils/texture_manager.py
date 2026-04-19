"""
TextureManager — Garment image processing for the VTON pipeline.

Responsibilities:
  - Background removal from garment images (GrabCut / white-thresh)
  - Garment texture extraction and normalisation
  - Returns a (H, W, 4) RGBA numpy array ready for TextureProjector.
"""

import io
import logging
import numpy as np

# Force-patch legacy NumPy attributes needed by chumpy
import numpy as _np_mod
for _attr in ['float', 'int', 'bool', 'object', 'str', 'complex']:
    if not hasattr(_np_mod, _attr):
        setattr(_np_mod, _attr, getattr(__builtins__, _attr, None) or eval(_attr))

import os
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class TextureManager:
    def __init__(self, uv_res: int = 1024, templates_root: str = 'templates'):
        self.uv_res = uv_res
        self.templates_root = templates_root
        logger.info(f"✓ TextureManager initialised (resolution: {uv_res}px)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_texture(self, mesh, garment_img) -> np.ndarray:
        """
        Process garment_img, remove its background, and return an RGBA
        numpy array (H, W, 4) suitable for texture atlas baking.

        Parameters
        ----------
        mesh        : unused in this implementation (reserved for future
                      UV-guided warping)
        garment_img : bytes | np.ndarray | PIL.Image

        Returns
        -------
        np.ndarray  shape (uv_res, uv_res, 4), dtype uint8
        """
        logger.info(f"⚡ TextureManager: processing garment @ {self.uv_res}px …")
        rgba = self._decode(garment_img)
        rgba = self._remove_background(rgba)
        rgba = self._resize(rgba)
        logger.info(f"✓ Garment texture ready: {rgba.shape}, "
                    f"non-transparent pixels={np.count_nonzero(rgba[:,:,3])}")
        return rgba

    def map_texture_bytes(self, mesh, garment_bytes: bytes) -> np.ndarray:
        """Convenience wrapper when the garment arrives as raw bytes."""
        return self.map_texture(mesh, garment_bytes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, image_data) -> np.ndarray:
        """Return an RGBA uint8 numpy array from any input type."""
        if isinstance(image_data, np.ndarray):
            arr = image_data.astype(np.uint8) if image_data.dtype != np.uint8 else image_data
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
            elif arr.shape[-1] == 3:
                alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
                arr = np.concatenate([arr, alpha], axis=-1)
            return arr

        if isinstance(image_data, Image.Image):
            return np.array(image_data.convert("RGBA"))

        if isinstance(image_data, (bytes, bytearray)):
            try:
                pil = Image.open(io.BytesIO(image_data)).convert("RGBA")
                return np.array(pil)
            except Exception as e:
                logger.warning(f"_decode failed on bytes: {e}. Returning magenta debug swatch.")
                out = np.zeros((self.uv_res, self.uv_res, 4), dtype=np.uint8)
                out[:, :, 0] = 255   # R
                out[:, :, 2] = 255   # B  → magenta = "I got garment bytes but couldn't decode"
                out[:, :, 3] = 255
                return out

        logger.warning(f"_decode: unknown type {type(image_data)}, returning transparent canvas")
        return np.zeros((self.uv_res, self.uv_res, 4), dtype=np.uint8)

    def _remove_background(self, rgba: np.ndarray) -> np.ndarray:
        """
        Two-stage background removal:
          1. White/near-white pixel threshold (fast, works for studio shots)
          2. GrabCut refinement on the remaining alpha mask
        Returns RGBA with background pixels set to alpha=0.
        """
        out = rgba.copy()
        rgb = rgba[:, :, :3]

        # Stage 1: white threshold (covers >90 % of garment product shots)
        white_mask = np.all(rgb > 240, axis=-1)
        out[white_mask, 3] = 0

        # Stage 2: GrabCut refinement (if cv2 available and image non-trivial)
        try:
            h, w = rgb.shape[:2]
            # Only run GrabCut if the image is large enough
            if h > 64 and w > 64:
                gc_mask = np.where(out[:, :, 3] > 0,
                                   cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                rect = (max(1, w // 8), max(1, h // 8),
                        w - w // 4, h - h // 4)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model,
                            5, cv2.GC_INIT_WITH_MASK)
                fg_mask = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
                out[~fg_mask, 3] = 0
        except Exception as gc_err:
            logger.debug(f"GrabCut skipped: {gc_err}")

        return out

    def _resize(self, rgba: np.ndarray) -> np.ndarray:
        """Resize to (uv_res, uv_res) maintaining RGBA."""
        pil = Image.fromarray(rgba, "RGBA")
        pil = pil.resize((self.uv_res, self.uv_res), Image.LANCZOS)
        return np.array(pil)
