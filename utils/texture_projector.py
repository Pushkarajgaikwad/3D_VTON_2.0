"""
TextureProjector — Bakes garment + skin into a texture atlas and exports
a properly textured GLB via trimesh PBRMaterial + TextureVisuals.

Previously this file had two code paths that both called `mesh.export()` 
raw (discarding all texture data).  This rewrite fixes that.
"""

import io
import os
import gc
import logging
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pil_rgba(image_data) -> Image.Image:
    """
    Convert any of: bytes, np.ndarray (H,W,3)/(H,W,4), or PIL Image
    into a PIL RGBA image.
    """
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGBA")

    if isinstance(image_data, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image_data)).convert("RGBA")
        except Exception as e:
            logger.warning(f"Could not decode image bytes: {e}. Using red fallback.")
            return _red_fallback()

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

    logger.warning(f"Unknown image_data type {type(image_data)}. Using red fallback.")
    return _red_fallback()


def _red_fallback(size: int = 512) -> Image.Image:
    """Bright red 512×512 RGBA — unmistakable in any viewer."""
    img = Image.new("RGBA", (size, size), color=(255, 0, 0, 255))
    return img


def _build_atlas(garment_img: Image.Image,
                 skin_img: Image.Image | None = None,
                 atlas_size: int = 1024) -> Image.Image:
    """
    Bake garment (top half) and skin (bottom half) into a single square atlas.
    If skin_img is None the garment fills the full atlas.
    """
    atlas = Image.new("RGBA", (atlas_size, atlas_size), (0, 0, 0, 0))

    if skin_img is None:
        garment_resized = garment_img.resize((atlas_size, atlas_size), Image.LANCZOS)
        atlas.paste(garment_resized, (0, 0))
    else:
        half = atlas_size // 2
        garment_resized = garment_img.resize((atlas_size, half), Image.LANCZOS)
        skin_resized = skin_img.resize((atlas_size, half), Image.LANCZOS)
        atlas.paste(garment_resized, (0, 0))
        atlas.paste(skin_resized, (0, half))

    return atlas


def _generate_uv_for_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Generate per-vertex UV coordinates via cylindrical/spherical projection.

    Trimesh's TextureVisuals requires UV shape (N_vertices, 2) where
    N_vertices == len(mesh.vertices).  This is the correct shape.
    """
    verts = mesh.vertices.copy()
    centered = verts - verts.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]

    # Cylindrical U from azimuth angle
    u = (np.arctan2(x, z) + np.pi) / (2.0 * np.pi)

    # V from normalized height
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if (y_max - y_min) > 1e-6 else 1.0
    v = (y - y_min) / y_range        # 0 = bottom, 1 = top
    v = 1.0 - v                       # flip: 0 = top of texture = top of body

    uv = np.stack([u, v], axis=1).astype(np.float32)
    return np.clip(uv, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TextureProjector:
    def __init__(self, atlas_size: int = 1024):
        self.atlas_size = atlas_size
        logger.info("✓ TextureProjector initialised (PBR atlas pipeline)")

    def project_and_export(self,
                           mesh,
                           texture_image,
                           output_path: str,
                           skin_image=None) -> str:
        """
        Bake texture_image (+ optional skin_image) into a single atlas,
        bind it to the mesh via PBRMaterial + TextureVisuals, and export GLB.

        Parameters
        ----------
        mesh          : trimesh.Trimesh  (may come from PIFuHD / SMPL)
        texture_image : bytes | np.ndarray | PIL.Image — the garment texture
        output_path   : destination .glb path
        skin_image    : optional bytes | np.ndarray | PIL.Image — face/skin map
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # ── 1. Ensure we have a trimesh.Trimesh (not a Scene or custom Meshes) ──
        if not isinstance(mesh, trimesh.Trimesh):
            logger.info("Converting non-Trimesh mesh object to trimesh.Trimesh …")
            try:
                # Handle pytorch3d Meshes-like objects
                verts = mesh.verts_list()[0].cpu().numpy()
                faces = mesh.faces_list()[0].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            except AttributeError:
                try:
                    # Handle objects with .vertices / .faces attributes
                    mesh = trimesh.Trimesh(
                        vertices=np.array(mesh.vertices),
                        faces=np.array(mesh.faces),
                        process=False
                    )
                except Exception as conv_err:
                    logger.error(f"Cannot convert mesh to Trimesh: {conv_err}. Exporting bare.")
                    raw_bytes = mesh.export(file_type="glb") if hasattr(mesh, "export") else b""
                    with open(output_path, "wb") as f:
                        f.write(raw_bytes)
                    return output_path

        logger.info(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # ── 2. Build the texture atlas ───────────────────────────────────────
        try:
            garment_pil = _to_pil_rgba(texture_image)
            skin_pil    = _to_pil_rgba(skin_image) if skin_image is not None else None
            atlas       = _build_atlas(garment_pil, skin_pil, atlas_size=self.atlas_size)
            logger.info(f"Texture atlas built: {atlas.size} RGBA")
        except Exception as atlas_err:
            logger.warning(f"Atlas build failed ({atlas_err}), using red fallback")
            atlas = _red_fallback(self.atlas_size)

        # ── 3. Generate UV coordinates ───────────────────────────────────────
        uv_coords = _generate_uv_for_mesh(mesh)
        logger.info(f"UV coords: shape={uv_coords.shape}, "
                    f"u=[{uv_coords[:,0].min():.3f}, {uv_coords[:,0].max():.3f}], "
                    f"v=[{uv_coords[:,1].min():.3f}, {uv_coords[:,1].max():.3f}]")

        # ── 4. Bind material ─────────────────────────────────────────────────
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
        logger.info("✓ PBRMaterial + TextureVisuals bound to mesh")

        # ── 5. Export GLB ────────────────────────────────────────────────────
        try:
            glb_bytes = mesh.export(file_type="glb")
            with open(output_path, "wb") as f:
                f.write(glb_bytes)
            file_size = len(glb_bytes)
            logger.info(f"✓ GLB exported: {output_path} ({file_size:,} bytes)")
            if file_size < 10_000:
                logger.warning(
                    f"GLB is suspiciously small ({file_size} bytes). "
                    "Texture may not have been embedded."
                )
        except Exception as export_err:
            logger.error(f"GLB export failed: {export_err}. Writing raw mesh.")
            raw_bytes = mesh.export(file_type="glb")
            with open(output_path, "wb") as f:
                f.write(raw_bytes)
        finally:
            gc.collect()

        return output_path
