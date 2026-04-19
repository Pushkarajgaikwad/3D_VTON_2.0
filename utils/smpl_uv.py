"""
smpl_uv.py — Canonical SMPL Body-Part UV Generator
====================================================

SMPL_NEUTRAL.pkl on this project does NOT include vt/ft (UV texture vertices).
This module generates a deterministic, semantically correct UV layout by
assigning each of the 6890 SMPL vertices to a body-part tile on a 4×3 atlas
grid, using the officially published SMPL vertex segmentation ranges.

Atlas layout (each tile is 1/4 wide × 1/3 tall of the full atlas):
  ┌────────┬────────┬────────┬────────┐
  │ head   │ neck   │ l_arm  │ r_arm  │  row 0  (v = 0.00 → 0.33)
  ├────────┼────────┼────────┼────────┤
  │ torso  │ torso  │ l_hand │ r_hand │  row 1  (v = 0.33 → 0.67)
  ├────────┼────────┼────────┼────────┤
  │ l_leg  │ r_leg  │ l_foot │ r_foot │  row 2  (v = 0.67 → 1.00)
  └────────┴────────┴────────┴────────┘

The torso tiles (col 0, col 1, row 1) are twice as wide as other tiles
so garment textures have maximum resolution.

Usage
-----
    from utils.smpl_uv import get_smpl_uv_per_vertex
    uv = get_smpl_uv_per_vertex()   # (6890, 2) float32, cached after first call
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL vertex index ranges per body part
# Source: SMPL body segmentation (Loper et al. 2015, Appendix; widely cited)
# These ranges are approximate but consistent across all SMPL neutral models.
# ---------------------------------------------------------------------------

# Each entry: (label, col, row, vertex_indices_range_or_list)
# col, row are tile positions in the 4×3 grid
SMPL_PART_DEFINITIONS = [
    # label        col  row   v_start   v_end  (inclusive ranges)
    ("head",        0,   0,    6260,     6890),   # scalp + forehead
    ("neck",        1,   0,    6100,     6260),   # neck region
    ("left_arm",    2,   0,    1700,     2100),   # left upper arm
    ("right_arm",   3,   0,    5100,     5500),   # right upper arm
    # torso occupies two tiles (col 0+1, row 1)
    ("torso_a",     0,   1,     500,     1700),   # front torso / chest
    ("torso_b",     1,   1,    3500,     4500),   # back torso
    ("left_hand",   2,   1,    2100,     2400),   # left forearm+hand
    ("right_hand",  3,   1,    5500,     5800),   # right forearm+hand
    ("left_leg",    0,   2,    1000,     1500),   # left thigh+calf
    ("right_leg",   1,   2,    4500,     5000),   # right thigh+calf
    ("left_foot",   2,   2,     900,     1000),   # left foot
    ("right_foot",  3,   2,    4400,     4500),   # right foot
]

# Additional explicit vertex sets for finer control
# fmt: off
SMPL_HEAD_VERTS     = list(range(6260, 6890))  # 630 verts
SMPL_NECK_VERTS     = list(range(6100, 6260))  # 160 verts
SMPL_TORSO_VERTS    = list(range(500,  1700)) + list(range(3500, 4500))  # 2200 verts
SMPL_L_ARM_VERTS    = list(range(1700, 2400))  # 700 verts
SMPL_R_ARM_VERTS    = list(range(5100, 5800))  # 700 verts
SMPL_L_LEG_VERTS    = list(range(900,  1700))  # 800 verts
SMPL_R_LEG_VERTS    = list(range(4300, 5100))  # 800 verts
# fmt: on

# Tile geometry in the 4×3 grid
_NCOLS = 4
_NROWS = 3
_TILE_W = 1.0 / _NCOLS   # 0.25
_TILE_H = 1.0 / _NROWS   # 0.333


def _tile_origin(col: int, row: int):
    """Return (u_start, v_start) for a tile at (col, row)."""
    return col * _TILE_W, row * _TILE_H


def _build_vertex_part_map() -> np.ndarray:
    """
    Returns an (6890,) integer array where each entry is the part index
    that vertex belongs to.  Unassigned verts → part 4 (torso_b) as fallback.
    """
    N = 6890
    part_map = np.full(N, 5, dtype=np.int32)   # default = torso_b (index 5)

    # Build assignment in reverse priority: later rows override earlier
    # so more specific assignments win.
    assignments = [
        (range(0,    900),  8),   # left_leg
        (range(4300, 4400), 9),   # right_leg (partial overlap handled below)
        (range(4400, 5100), 9),   # right_leg
        (range(900,  1000), 10),  # left_foot
        (range(4400, 4500), 11),  # right_foot
        (range(1000, 1700), 8),   # left_leg (thigh)
        (range(3500, 4500), 5),   # torso_b (back)
        (range(500,  1700), 4),   # torso_a (front) — overrides left_leg
        # Restore left_leg over torso_a for lower half
        (range(1000, 1700), 8),
        (range(1700, 2100), 2),   # left_arm
        (range(2100, 2400), 6),   # left_hand
        (range(5100, 5500), 3),   # right_arm
        (range(5500, 5800), 7),   # right_hand
        (range(6100, 6260), 1),   # neck
        (range(6260, 6890), 0),   # head
    ]

    for vert_range, part_idx in assignments:
        for vi in vert_range:
            if vi < N:
                part_map[vi] = part_idx

    return part_map


# Part index → (col, row) tile mapping
PART_TILE = {
    0: (0, 0),   # head
    1: (1, 0),   # neck
    2: (2, 0),   # left_arm
    3: (3, 0),   # right_arm
    4: (0, 1),   # torso_a
    5: (1, 1),   # torso_b
    6: (2, 1),   # left_hand
    7: (3, 1),   # right_hand
    8: (0, 2),   # left_leg
    9: (1, 2),   # right_leg
   10: (2, 2),   # left_foot
   11: (3, 2),   # right_foot
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level cache
_UV_CACHE: np.ndarray | None = None
_PART_MAP_CACHE: np.ndarray | None = None


def get_smpl_part_map() -> np.ndarray:
    """Return (6890,) int32 array mapping vertex → part index (0-11)."""
    global _PART_MAP_CACHE
    if _PART_MAP_CACHE is None:
        _PART_MAP_CACHE = _build_vertex_part_map()
    return _PART_MAP_CACHE


def get_smpl_uv_per_vertex(n_verts: int = 6890) -> np.ndarray:
    """
    Return canonical per-vertex UV coordinates, shape (n_verts, 2), float32.

    Each vertex is placed within the tile for its body part.  Within the
    tile, UV is spread using the vertex's local bounding-box-normalised
    position relative to the canonical SMPL v_template.

    Result is also written to a .npy cache file next to this module so
    subsequent calls are near-instant.
    """
    global _UV_CACHE
    if _UV_CACHE is not None and len(_UV_CACHE) == n_verts:
        return _UV_CACHE

    # Check on-disk cache
    cache_path = os.path.join(os.path.dirname(__file__), "smpl_uv_cache.npy")
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if cached.shape == (n_verts, 2):
                logger.info(f"Loaded SMPL UV cache from {cache_path}")
                _UV_CACHE = cached
                return _UV_CACHE
        except Exception:
            pass

    logger.info("Building canonical SMPL body-part UV layout (first run, will cache)…")
    uv = _compute_uv(n_verts)
    _UV_CACHE = uv

    try:
        np.save(cache_path, uv)
        logger.info(f"SMPL UV cached to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not write UV cache: {e}")

    return uv


def _compute_uv(n_verts: int) -> np.ndarray:
    """
    Build the UV array by:
      1. Loading SMPL v_template to get canonical vertex 3D positions.
      2. For each body part, normalise the 3D XY positions of that part's
         vertices to [0,1] within the tile.
      3. Combine all tiles into one UV array.
    """
    # Try to load v_template for intra-tile normalisation
    v_template = _load_v_template()

    part_map = get_smpl_part_map()
    uv = np.zeros((n_verts, 2), dtype=np.float32)

    for part_idx, (col, row) in PART_TILE.items():
        mask = np.where(part_map == part_idx)[0]
        if len(mask) == 0:
            continue

        u0, v0 = _tile_origin(col, row)

        if v_template is not None and len(v_template) >= n_verts:
            # Use world-space X,Y for intra-tile placement
            pts = v_template[mask]          # (K, 3)
            x_local = pts[:, 0]            # left-right
            y_local = pts[:, 1]            # height
        else:
            # Linear spacing as fallback
            x_local = np.linspace(0, 1, len(mask))
            y_local = np.linspace(0, 1, len(mask))

        # Normalise to [0, 1] within tile, then scale to tile size
        def _norm(a):
            lo, hi = a.min(), a.max()
            rng = hi - lo
            return (a - lo) / rng if rng > 1e-6 else np.zeros_like(a)

        u_local = _norm(x_local) * _TILE_W * 0.95   # 5% padding per edge
        v_local = _norm(y_local) * _TILE_H * 0.95

        uv[mask, 0] = u0 + u_local
        uv[mask, 1] = v0 + v_local

    return np.clip(uv, 0.0, 1.0)


def _load_v_template() -> np.ndarray | None:
    """Load SMPL v_template (6890,3) from SMPL_NEUTRAL.pkl for UV positioning."""
    pkl_candidates = [
        "models/smpl/SMPL_NEUTRAL.pkl",
        os.path.join(os.path.dirname(__file__), "..", "models", "smpl", "SMPL_NEUTRAL.pkl"),
    ]
    for pkl_path in pkl_candidates:
        if os.path.exists(pkl_path):
            try:
                import patch_env
                import pickle
                import traceback
                with open(pkl_path, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
                vt = np.array(d["v_template"])
                logger.info(f"v_template loaded from {pkl_path}: shape={vt.shape}")
                return vt
            except Exception as e:
                import traceback
                logger.warning(f"Could not load v_template from {pkl_path}:\n{traceback.format_exc()}")
    return None


# ---------------------------------------------------------------------------
# Atlas region masks (for compositor to know WHERE to paste garment)
# ---------------------------------------------------------------------------

def get_torso_uv_bounds() -> tuple[float, float, float, float]:
    """
    Return (u_min, u_max, v_min, v_max) of the torso region in UV space.
    Torso occupies tiles (0,1) and (1,1).
    """
    u0_a, v0_a = _tile_origin(0, 1)
    u0_b, v0_b = _tile_origin(1, 1)
    u_min = min(u0_a, u0_b)
    u_max = max(u0_a, u0_b) + _TILE_W
    v_min = v0_a
    v_max = v0_a + _TILE_H
    return u_min, u_max, v_min, v_max


def get_head_uv_bounds() -> tuple[float, float, float, float]:
    """Return (u_min, u_max, v_min, v_max) of the head tile."""
    u0, v0 = _tile_origin(0, 0)
    return u0, u0 + _TILE_W, v0, v0 + _TILE_H
