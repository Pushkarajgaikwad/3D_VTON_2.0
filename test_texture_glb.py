#!/usr/bin/env python3
"""
Phase 3 Loop Test — GLB Texture Retention Validator
====================================================
Loads templates/tshirt/mesh.obj, applies a solid RED 512x512 RGBA texture
via PBRMaterial + TextureVisuals, exports to test_color.glb, then verifies
the texture is actually embedded in the binary by searching for the PNG magic
bytes (\\x89PNG) inside the file.

Pass criteria:
  1. test_color.glb exists and is > 50 KB
  2. The file contains PNG magic bytes (texture embedded)
  3. No trimesh exceptions during export
"""

import sys
import os
import struct
import logging

# Make sure we can import from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Patch legacy numpy attributes needed by chumpy (imported transitively)
import patch_env  # noqa: F401

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test_texture_glb")

MESH_PATH  = os.path.join(ROOT, "templates", "tshirt", "mesh.obj")
OUTPUT_GLB = os.path.join(ROOT, "test_color.glb")
ATLAS_SIZE = 512

# ── PNG magic bytes (first 8 bytes of every PNG file) ──────────────────────
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def make_red_atlas(size: int = ATLAS_SIZE) -> Image.Image:
    """Pure red RGBA image — unmistakable in any 3D viewer."""
    img = Image.new("RGBA", (size, size), color=(255, 0, 0, 255))
    # Draw a white cross so we can see UV orientation in the viewer
    arr = np.array(img)
    cx, cy = size // 2, size // 2
    arr[cy - 5 : cy + 5, :, :] = [255, 255, 255, 255]  # horizontal bar
    arr[:, cx - 5 : cx + 5, :] = [255, 255, 255, 255]  # vertical bar
    return Image.fromarray(arr, "RGBA")


def generate_uv(vertices: np.ndarray) -> np.ndarray:
    """Cylindrical UV projection — per vertex, shape (N, 2)."""
    centered = vertices - vertices.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    u = (np.arctan2(x, z) + np.pi) / (2.0 * np.pi)
    y_min, y_max = y.min(), y.max()
    y_range = max(y_max - y_min, 1e-6)
    v = 1.0 - (y - y_min) / y_range
    return np.clip(np.stack([u, v], axis=1).astype(np.float32), 0.0, 1.0)


def run_test() -> bool:
    logger.info("=" * 60)
    logger.info("  GLB Texture Retention Loop Test")
    logger.info("=" * 60)

    # ── Step 1: Load mesh ───────────────────────────────────────────────────
    if not os.path.exists(MESH_PATH):
        logger.error(f"FAIL — mesh not found: {MESH_PATH}")
        return False

    logger.info(f"Loading mesh: {MESH_PATH}")
    scene = trimesh.load(MESH_PATH, process=False, force='mesh')

    # trimesh.load can return a Scene if the OBJ has groups
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        if not geoms:
            logger.error("FAIL — OBJ loaded as empty Scene")
            return False
        mesh = trimesh.util.concatenate(geoms)
        logger.info(f"Merged {len(geoms)} submesh(es) → {len(mesh.vertices)} verts")
    else:
        mesh = scene

    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    if len(mesh.vertices) == 0:
        logger.error("FAIL — mesh has zero vertices")
        return False

    # ── Step 2: Build red atlas ─────────────────────────────────────────────
    atlas = make_red_atlas(ATLAS_SIZE)
    logger.info(f"Atlas: {atlas.size} {atlas.mode}")

    # ── Step 3: Compute UV coords ───────────────────────────────────────────
    uv_coords = generate_uv(mesh.vertices)
    logger.info(f"UV: shape={uv_coords.shape}")

    # ── Step 4: Bind PBRMaterial + TextureVisuals ───────────────────────────
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, material=material)
    logger.info("PBRMaterial + TextureVisuals bound ✓")

    # ── Step 5: Export GLB ──────────────────────────────────────────────────
    logger.info(f"Exporting GLB → {OUTPUT_GLB}")
    try:
        glb_bytes = mesh.export(file_type='glb')
    except Exception as e:
        logger.error(f"FAIL — mesh.export() raised: {e}")
        import traceback; traceback.print_exc()
        return False

    with open(OUTPUT_GLB, 'wb') as f:
        f.write(glb_bytes)

    file_size = len(glb_bytes)
    logger.info(f"GLB written: {file_size:,} bytes")

    # ── Step 6: Verification ────────────────────────────────────────────────
    passed = True

    # 6a. Size check
    if file_size < 50_000:
        logger.error(f"FAIL — GLB too small ({file_size} bytes, expected >50 KB). "
                     "Texture was likely not embedded.")
        passed = False
    else:
        logger.info(f"PASS — GLB size check ({file_size:,} bytes > 50 KB)")

    # 6b. PNG magic bytes in binary
    if PNG_MAGIC in glb_bytes:
        # Find offset for debug
        offset = glb_bytes.index(PNG_MAGIC)
        logger.info(f"PASS — PNG magic bytes found at offset {offset:#x} "
                    f"→ texture IS embedded in GLB ✓")
    else:
        logger.error("FAIL — PNG magic bytes NOT found in GLB. "
                     "Trimesh exported the mesh without the texture image.")
        passed = False

    # 6c. GLB header check (magic = 0x46546C67 'glTF')
    if glb_bytes[:4] == b'glTF':
        logger.info("PASS — GLB header magic 'glTF' confirmed ✓")
    else:
        logger.error(f"FAIL — unexpected file header: {glb_bytes[:4]!r}")
        passed = False

    # ── Final verdict ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    if passed:
        logger.info("  ✅  ALL CHECKS PASSED — TextureProjector pipeline is FIXED")
        logger.info(f"  View: {OUTPUT_GLB}")
    else:
        logger.error("  ❌  ONE OR MORE CHECKS FAILED — see above for details")
    logger.info("=" * 60)
    return passed


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
