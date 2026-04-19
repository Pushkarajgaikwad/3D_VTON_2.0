"""
TextureProjector — Semantically-correct SMPL UV atlas pipeline.
================================================================

PROBLEM FIXED:
  Previous code used a naive cylindrical UV projection (arctan2 of XZ),
  which painted the garment texture identically across every vertex on the
  mesh — face, hands, legs, torso all showed the same red swatch.

HOW IT WORKS NOW:
  1. Load the canonical SMPL body-part UV layout from smpl_uv.py.
     Each of the 6890 SMPL vertices maps to a distinct tile in a 4×3 atlas:
       head | neck | l_arm | r_arm
       torso_a | torso_b | l_hand | r_hand
       l_leg | r_leg | l_foot | r_foot

  2. Build a layered atlas:
       Layer 0 (whole atlas background): neutral skin tone
       Layer 1 (torso tiles):            garment texture  ← U2Net segmented
       Layer 2 (head tile, optional):    face blender output

  3. Bind UV + PBRMaterial + TextureVisuals and export GLB.

The atlas compositor uses pixel-space rectangle paste so blending is
CPU-only with zero GPU dependency.
"""

import io
import os
import gc
import logging
import traceback
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atlas geometry constants — must match smpl_uv.py PART_TILE layout
# ---------------------------------------------------------------------------

ATLAS_NCOLS = 4
ATLAS_NROWS = 3

# Tile (col, row) for semantic regions
TILE_HEAD    = (0, 0)
TILE_NECK    = (1, 0)
TILE_L_ARM   = (2, 0)
TILE_R_ARM   = (3, 0)
TILE_TORSO_A = (0, 1)   # front torso
TILE_TORSO_B = (1, 1)   # back torso
TILE_L_HAND  = (2, 1)
TILE_R_HAND  = (3, 1)
TILE_L_LEG   = (0, 2)
TILE_R_LEG   = (1, 2)
TILE_L_FOOT  = (2, 2)
TILE_R_FOOT  = (3, 2)

# Tiles that receive the garment (upper-body garment = torso + arms)
GARMENT_TILES = [TILE_TORSO_A, TILE_TORSO_B, TILE_L_ARM, TILE_R_ARM,
                 TILE_L_HAND, TILE_R_HAND]

# Default skin tone (RGBA)
SKIN_TONE = (210, 180, 140, 255)   # tan
NECK_TONE = (200, 170, 130, 255)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _to_pil_rgba(image_data) -> "Image.Image":
    """Convert bytes | np.ndarray | PIL.Image → PIL RGBA."""
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGBA")
    if isinstance(image_data, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image_data)).convert("RGBA")
        except Exception as e:
            logger.warning(f"Could not decode image bytes: {e}. Fallback.")
            return _solid(512, (255, 0, 0, 255))
    if isinstance(image_data, np.ndarray):
        arr = image_data
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        mode = "RGB" if arr.shape[-1] == 3 else "RGBA"
        return Image.fromarray(arr, mode=mode).convert("RGBA")
    logger.warning(f"Unknown image_data type {type(image_data)}. Fallback.")
    return _solid(512, (255, 0, 0, 255))


def _solid(size: int, color: tuple) -> "Image.Image":
    return Image.new("RGBA", (size, size), color=color)


def _tile_px_rect(col: int, row: int, atlas_size: int):
    """
    Return (x0, y0, x1, y1) pixel rectangle for this tile.
    atlas_size must be divisible by ATLAS_NCOLS and ATLAS_NROWS.
    """
    tw = atlas_size // ATLAS_NCOLS
    th = atlas_size // ATLAS_NROWS
    x0 = col * tw
    y0 = row * th
    return x0, y0, x0 + tw, y0 + th


# ---------------------------------------------------------------------------
# Atlas compositor
# ---------------------------------------------------------------------------

def build_atlas(garment_pil: "Image.Image",
                face_pil: "Image.Image | None" = None,
                atlas_size: int = 1024) -> "Image.Image":
    """
    Build a semantically-correct texture atlas.

    Layer order (each overwrites the previous in its region):
      1. Full-atlas skin base
      2. Garment image → torso + arm tiles
      3. Face image → head tile (optional)

    Parameters
    ----------
    garment_pil : RGBA PIL image of the garment (already BG-removed)
    face_pil    : optional RGBA PIL image for the face region
    atlas_size  : output atlas resolution (square)

    Returns
    -------
    RGBA PIL image of shape (atlas_size, atlas_size)
    """
    atlas = Image.new("RGBA", (atlas_size, atlas_size), color=SKIN_TONE)

    tw = atlas_size // ATLAS_NCOLS
    th = atlas_size // ATLAS_NROWS

    # ── Layer 1: skin base is already done (background fill) ──────────────

    # ── Layer 2: garment on torso + arm tiles ─────────────────────────────
    for (col, row) in GARMENT_TILES:
        x0, y0, x1, y1 = _tile_px_rect(col, row, atlas_size)
        region_w = x1 - x0
        region_h = y1 - y0

        if col in (0, 1) and row == 1:
            # Torso: use full garment, sample the vertical centre
            # (garment images are typically portrait; torso = middle 60%)
            gw, gh = garment_pil.size
            crop_top    = int(gh * 0.05)
            crop_bottom = int(gh * 0.65)
            garment_crop = garment_pil.crop((0, crop_top, gw, crop_bottom))
        elif col in (2, 3) and row == 0:
            # Arm tiles: use the sleeve portion (top 30% of garment)
            gw, gh = garment_pil.size
            # Left arm = left quarter, right arm = right quarter
            if col == 2:
                garment_crop = garment_pil.crop((0, 0, gw // 2, int(gh * 0.35)))
            else:
                garment_crop = garment_pil.crop((gw // 2, 0, gw, int(gh * 0.35)))
        else:
            # Hand tiles: sleeve cuff area
            gw, gh = garment_pil.size
            half = gw // 2
            if col == 2:
                garment_crop = garment_pil.crop((0, int(gh * 0.3), half, int(gh * 0.5)))
            else:
                garment_crop = garment_pil.crop((half, int(gh * 0.3), gw, int(gh * 0.5)))

        # Resize crop to tile size using high-quality resample
        garment_tile = garment_crop.resize((region_w, region_h), Image.LANCZOS)

        # Alpha-composite so BG-removed garment blends smoothly
        # Build a clean base patch
        base_patch = Image.new("RGBA", (region_w, region_h), color=SKIN_TONE)
        base_patch = Image.alpha_composite(base_patch, garment_tile)
        atlas.paste(base_patch, (x0, y0))

    # ── Layer 3: face on head tile ─────────────────────────────────────────
    if face_pil is not None:
        try:
            x0, y0, x1, y1 = _tile_px_rect(*TILE_HEAD, atlas_size)
            region_w = x1 - x0
            region_h = y1 - y0
            face_resized = face_pil.resize((region_w, region_h), Image.LANCZOS)
            base_patch = Image.new("RGBA", (region_w, region_h), color=SKIN_TONE)
            base_patch = Image.alpha_composite(base_patch, face_resized)
            atlas.paste(base_patch, (x0, y0))
        except Exception as e:
            logger.warning(f"Face paste failed: {e}")

    return atlas


# ---------------------------------------------------------------------------
# Debug atlas (for test script)
# ---------------------------------------------------------------------------

def build_debug_atlas(atlas_size: int = 1024) -> "Image.Image":
    """
    Return a multi-colour atlas map that makes UV island correctness
    visually obvious:
      head    = GREEN
      torso   = RED
      arms    = BLUE
      legs    = YELLOW
      hands   = MAGENTA
      feet    = CYAN
      neck    = ORANGE
    """
    atlas = Image.new("RGBA", (atlas_size, atlas_size), (30, 30, 30, 255))
    draw = ImageDraw.Draw(atlas)

    color_map = {
        TILE_HEAD:    (0,   220,  0,   255),   # GREEN
        TILE_NECK:    (255, 140,  0,   255),   # ORANGE
        TILE_L_ARM:   (0,   100, 255,  255),   # BLUE
        TILE_R_ARM:   (0,    80, 200,  255),   # BLUE-ish
        TILE_TORSO_A: (220,  30,  30,  255),   # RED
        TILE_TORSO_B: (180,  20,  20,  255),   # DARK RED
        TILE_L_HAND:  (220,   0, 220,  255),   # MAGENTA
        TILE_R_HAND:  (180,   0, 180,  255),   # DARK MAGENTA
        TILE_L_LEG:   (220, 220,   0,  255),   # YELLOW
        TILE_R_LEG:   (180, 180,   0,  255),   # DARK YELLOW
        TILE_L_FOOT:  (0,   220, 220,  255),   # CYAN
        TILE_R_FOOT:  (0,   180, 180,  255),   # DARK CYAN
    }

    for (col, row), color in color_map.items():
        x0, y0, x1, y1 = _tile_px_rect(col, row, atlas_size)
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=color)
        # Label
        label = f"c{col}r{row}"
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255, 200))

    return atlas


# ---------------------------------------------------------------------------
# UV loading
# ---------------------------------------------------------------------------

def load_smpl_uv(n_verts: int = 6890) -> np.ndarray:
    """
    Primary UV loader. Tries sources in order:
      1. smpl_uv_cache.npy (generated by smpl_uv.py on first run)
      2. smpl_uv.get_smpl_uv_per_vertex() (generates + caches on first call)

    Returns (n_verts, 2) float32 array.
    """
    from utils.smpl_uv import get_smpl_uv_per_vertex
    uv = get_smpl_uv_per_vertex(n_verts)
    logger.info(
        f"SMPL body-part UV loaded: shape={uv.shape}, "
        f"u=[{uv[:,0].min():.3f}, {uv[:,0].max():.3f}], "
        f"v=[{uv[:,1].min():.3f}, {uv[:,1].max():.3f}]"
    )
    return uv


# ---------------------------------------------------------------------------
# TextureProjector — public class
# ---------------------------------------------------------------------------

class TextureProjector:
    def __init__(self, atlas_size: int = 1024):
        self.atlas_size = atlas_size
        logger.info(
            "[OK] TextureProjector initialised "
            f"(SMPL body-part UV, {atlas_size}px atlas)"
        )

    def project_and_export(self,
                           mesh,
                           texture_image,
                           output_path: str,
                           skin_image=None,
                           face_image=None) -> str:
        """
        Build a semantically-layered atlas, bind it to the mesh with the
        canonical SMPL body-part UV, and export a textured GLB.

        Parameters
        ----------
        mesh          : trimesh.Trimesh or any object with .vertices/.faces
        texture_image : bytes | np.ndarray | PIL.Image  — the garment
        output_path   : destination .glb path
        skin_image    : unused (reserved; skin is drawn automatically)
        face_image    : optional face region PIL image
        """
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # ── 1. Normalise mesh to trimesh.Trimesh ──────────────────────────
        mesh, n_verts = self._to_trimesh(mesh)
        if mesh is None:
            logger.error("Could not convert mesh. Aborting export.")
            return output_path

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # ── 2. Load SMPL body-part UV ─────────────────────────────────────
        try:
            uv_coords = load_smpl_uv(n_verts=len(mesh.vertices))
        except Exception as e:
            logger.error(
                f"SMPL UV load failed: {e}\n{traceback.format_exc()}\n"
                "Falling back to cylindrical UV (will cause full-body paint)."
            )
            uv_coords = self._cylindrical_uv_fallback(mesh)

        # ── 3. Build atlas ────────────────────────────────────────────────
        try:
            garment_pil = _to_pil_rgba(texture_image) if texture_image is not None \
                          else _solid(self.atlas_size, (180, 180, 180, 255))
            face_pil    = _to_pil_rgba(face_image) if face_image is not None else None
            atlas       = build_atlas(garment_pil, face_pil, atlas_size=self.atlas_size)
            logger.info(f"Atlas built: {atlas.size} RGBA")
        except Exception as e:
            logger.warning(f"Atlas build failed ({e}), using debug atlas")
            atlas = build_debug_atlas(self.atlas_size)

        # ── 4. Bind PBRMaterial + TextureVisuals ──────────────────────────
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=atlas,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords,
            material=material,
        )
        logger.info("[OK] PBRMaterial + TextureVisuals bound to mesh")

        # ── 5. Export GLB ─────────────────────────────────────────────────
        try:
            glb_bytes = mesh.export(file_type="glb")
            with open(output_path, "wb") as f:
                f.write(glb_bytes)
            file_size = len(glb_bytes)
            logger.info(f"[OK] GLB exported: {output_path} ({file_size:,} bytes)")
            if file_size < 10_000:
                logger.warning(f"GLB suspiciously small ({file_size} bytes)")
        except Exception as e:
            logger.error(f"GLB export failed: {e}\n{traceback.format_exc()}")
        finally:
            gc.collect()

        return output_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_trimesh(self, mesh) -> "tuple[trimesh.Trimesh | None, int]":
        if isinstance(mesh, trimesh.Trimesh):
            return mesh, len(mesh.vertices)
        logger.info("Converting non-Trimesh object …")
        try:
            verts = mesh.verts_list()[0].cpu().numpy()
            faces = mesh.faces_list()[0].cpu().numpy()
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            return m, len(verts)
        except AttributeError:
            pass
        try:
            verts = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            return m, len(verts)
        except Exception as e:
            logger.error(f"Mesh conversion failed: {e}")
            return None, 0

    @staticmethod
    def _cylindrical_uv_fallback(mesh: trimesh.Trimesh) -> np.ndarray:
        """Last-resort cylindrical UV — only used if SMPL UV fails."""
        verts = mesh.vertices.copy()
        centered = verts - verts.mean(axis=0)
        x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
        u = (np.arctan2(x, z) + np.pi) / (2.0 * np.pi)
        y_min, y_max = y.min(), y.max()
        y_range = max(y_max - y_min, 1e-6)
        v = 1.0 - (y - y_min) / y_range
        return np.clip(np.stack([u, v], axis=1).astype(np.float32), 0.0, 1.0)
