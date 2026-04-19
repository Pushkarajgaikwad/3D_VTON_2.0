#!/usr/bin/env python3
"""
test_uv_mapping.py — UV Island Correctness Validator
=====================================================

Loop test for Phase 3. Verifies that the SMPL body-part UV layout is
semantically correct: head vertices do NOT fall in the torso UV tile,
and torso vertices DO fall in the torso UV tile.

Debug atlas colours:
  GREEN  = head tile
  RED    = torso tiles
  BLUE   = arm tiles
  YELLOW = leg tiles
  MAGENTA= hand tiles
  CYAN   = foot tiles
  ORANGE = neck tile

Pass criteria:
  1. test_uv.glb exists and is > 50 KB with embedded PNG texture
  2. Head vertices (e.g. index 6260) are mapped to head UV tile (u < 0.25, v < 0.333)
  3. Torso vertices (e.g. index 1000) are mapped to torso UV tile (0.0 <= u < 0.5, 0.333 <= v < 0.667)
  4. Leg vertices (e.g. index 100) are NOT in the torso UV tile
  5. No two body-part groups share the same mean UV coordinate (segmentation sanity)
"""

import sys
import os
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import patch_env  # noqa: F401

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_uv_mapping")

MESH_PATH   = os.path.join(ROOT, "templates", "tshirt", "mesh.obj")
OUTPUT_GLB  = os.path.join(ROOT, "test_uv.glb")
PNG_MAGIC   = b'\x89PNG\r\n\x1a\n'

# Known vertex indices for body-part checks (SMPL canonical segmentation)
KNOWN_HEAD_VERT    = 6500   # in range 6260-6890 → head tile
KNOWN_TORSO_VERT   = 750    # in range 500-1700  → torso_a tile
KNOWN_LEG_VERT     = 100    # in range 0-900      → left_leg tile (NOT torso)
KNOWN_L_ARM_VERT   = 1800   # in range 1700-2100  → left_arm tile


def _load_mesh() -> trimesh.Trimesh:
    scene = trimesh.load(MESH_PATH, process=False, force='mesh')
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = scene
    return mesh


def _verify_uv_tile(uv: np.ndarray, vert_idx: int,
                    u_min: float, u_max: float,
                    v_min: float, v_max: float,
                    label: str) -> bool:
    u, v = float(uv[vert_idx, 0]), float(uv[vert_idx, 1])
    inside = (u_min <= u < u_max) and (v_min <= v < v_max)
    status = "PASS" if inside else "FAIL"
    logger.info(
        f"  [{status}] {label} vertex {vert_idx}: "
        f"UV=({u:.3f},{v:.3f}), "
        f"expected tile u=[{u_min},{u_max}) v=[{v_min},{v_max})"
    )
    return inside


def run_test() -> bool:
    logger.info("=" * 62)
    logger.info("  UV Island Correctness Validator")
    logger.info("=" * 62)

    passed = True

    # ── Step 1: Load mesh ─────────────────────────────────────────────────
    if not os.path.exists(MESH_PATH):
        logger.error(f"FAIL — mesh not found: {MESH_PATH}")
        return False

    mesh = _load_mesh()
    n_verts = len(mesh.vertices)
    logger.info(f"Mesh: {n_verts} vertices, {len(mesh.faces)} faces")

    if n_verts < 6890:
        logger.warning(
            f"Mesh has only {n_verts} vertices (< 6890). "
            "UV island checks will be clipped to available indices."
        )

    # ── Step 2: Generate SMPL body-part UV ───────────────────────────────
    from utils.smpl_uv import get_smpl_uv_per_vertex, get_smpl_part_map

    uv = get_smpl_uv_per_vertex(n_verts=n_verts)
    logger.info(
        f"UV shape: {uv.shape}, "
        f"u=[{uv[:,0].min():.3f},{uv[:,0].max():.3f}], "
        f"v=[{uv[:,1].min():.3f},{uv[:,1].max():.3f}]"
    )

    part_map = get_smpl_part_map()

    # ── Step 3: Semantic tile bounds ──────────────────────────────────────
    # Atlas is 4 cols × 3 rows → each tile = 0.25 wide × 0.333 tall
    TW = 1.0 / 4
    TH = 1.0 / 3

    # Tile bounds: (u_min, u_max, v_min, v_max)
    TILES = {
        "head":    (0*TW, 1*TW, 0*TH, 1*TH),
        "neck":    (1*TW, 2*TW, 0*TH, 1*TH),
        "torso_a": (0*TW, 1*TW, 1*TH, 2*TH),
        "torso_b": (1*TW, 2*TW, 1*TH, 2*TH),
        "l_leg":   (0*TW, 1*TW, 2*TH, 3*TH),
        "r_leg":   (1*TW, 2*TW, 2*TH, 3*TH),
        "l_arm":   (2*TW, 3*TW, 0*TH, 1*TH),
    }

    logger.info("\n--- UV Tile Assignment Checks ---")

    # Check 1: Head vertex → head tile
    if KNOWN_HEAD_VERT < n_verts:
        ok = _verify_uv_tile(uv, KNOWN_HEAD_VERT, *TILES["head"], "HEAD")
        passed = passed and ok
    else:
        logger.warning(f"Skipping head check (vert {KNOWN_HEAD_VERT} >= n_verts {n_verts})")

    # Check 2: Torso vertex → torso_a tile
    if KNOWN_TORSO_VERT < n_verts:
        ok = _verify_uv_tile(uv, KNOWN_TORSO_VERT, *TILES["torso_a"], "TORSO")
        passed = passed and ok
    else:
        logger.warning(f"Skipping torso check (vert {KNOWN_TORSO_VERT} >= n_verts {n_verts})")

    # Check 3: Leg vertex NOT in torso tile (critical anti-overlap check)
    if KNOWN_LEG_VERT < n_verts:
        u, v = float(uv[KNOWN_LEG_VERT, 0]), float(uv[KNOWN_LEG_VERT, 1])
        in_torso = (0 <= u < 2*TW) and (1*TH <= v < 2*TH)
        if in_torso:
            logger.error(
                f"  [FAIL] LEG vertex {KNOWN_LEG_VERT}: UV=({u:.3f},{v:.3f}) "
                "is INSIDE the torso tile — full-body paint bug still active!"
            )
            passed = False
        else:
            # Should be in leg tile
            ok = _verify_uv_tile(uv, KNOWN_LEG_VERT, *TILES["l_leg"], "LEG (not torso)")
            passed = passed and ok

    # Check 4: Arm vertex → l_arm tile
    if KNOWN_L_ARM_VERT < n_verts:
        ok = _verify_uv_tile(uv, KNOWN_L_ARM_VERT, *TILES["l_arm"], "L_ARM")
        passed = passed and ok

    # Check 5: Segmentation sanity — each part must have distinct mean UV
    logger.info("\n--- Body-Part UV Centroid Sanity ---")
    part_centroids = {}
    for part_idx in range(12):
        mask = np.where(part_map[:n_verts] == part_idx)[0]
        if len(mask) > 0:
            centroid = uv[mask].mean(axis=0)
            part_centroids[part_idx] = centroid
            logger.info(f"  Part {part_idx:2d}: {len(mask):4d} verts, "
                        f"centroid UV=({centroid[0]:.3f},{centroid[1]:.3f})")

    # All centroids should be distinct (no two parts share the same mean tile)
    centroids_list = list(part_centroids.values())
    all_distinct = True
    for i in range(len(centroids_list)):
        for j in range(i+1, len(centroids_list)):
            dist = np.linalg.norm(centroids_list[i] - centroids_list[j])
            if dist < 0.01:
                logger.error(
                    f"  [FAIL] Parts {i} and {j} have nearly identical UV centroids "
                    f"(dist={dist:.6f}) — UV not segmented correctly!"
                )
                all_distinct = False
                passed = False
    if all_distinct:
        logger.info("  [PASS] All 12 body-part centroids are distinct")

    # ── Step 4: Build debug atlas ──────────────────────────────────────────
    from utils.texture_projector import build_debug_atlas
    atlas = build_debug_atlas(atlas_size=1024)
    logger.info(f"\nDebug atlas: {atlas.size}")

    # ── Step 5: Bind + export ──────────────────────────────────────────────
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    logger.info(f"Exporting debug GLB: {OUTPUT_GLB}")
    glb_bytes = mesh.export(file_type="glb")
    with open(OUTPUT_GLB, "wb") as f:
        f.write(glb_bytes)

    # ── Step 6: Binary verification ───────────────────────────────────────
    logger.info("\n--- GLB Binary Verification ---")
    file_size = len(glb_bytes)
    if file_size > 50_000:
        logger.info(f"  [PASS] GLB size: {file_size:,} bytes")
    else:
        logger.error(f"  [FAIL] GLB too small: {file_size} bytes")
        passed = False

    if glb_bytes[:4] == b'glTF':
        logger.info("  [PASS] GLB header 'glTF' OK")
    else:
        logger.error(f"  [FAIL] Bad GLB header: {glb_bytes[:4]!r}")
        passed = False

    if PNG_MAGIC in glb_bytes:
        offset = glb_bytes.index(PNG_MAGIC)
        logger.info(f"  [PASS] PNG texture embedded at offset {offset:#x}")
    else:
        logger.error("  [FAIL] No PNG magic bytes found in GLB")
        passed = False

    # ── Final verdict ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 62)
    if passed:
        logger.info("  ALL CHECKS PASSED — UV Island mapping is CORRECT")
        logger.info(f"  Debug GLB: {OUTPUT_GLB}")
        logger.info("  Open in any glTF viewer to visually verify:")
        logger.info("    Head = GREEN, Torso = RED, Arms = BLUE, Legs = YELLOW")
    else:
        logger.error("  ONE OR MORE CHECKS FAILED — see above for details")
    logger.info("=" * 62)

    return passed


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
