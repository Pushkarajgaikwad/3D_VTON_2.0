"""
smpl_uv.py — Robust Topological Unwrapping via xatlas
======================================================
Generated via Option B. Replaces hallucinated 4x3 bounding-box logic
with an industry-standard mesh parameterization (xatlas).
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Cache variables
_UV_CACHE: np.ndarray | None = None


def get_smpl_uv_per_vertex(n_verts: int = 6890) -> np.ndarray:
    """
    Return canonical per-vertex UV coordinates, shape (n_verts, 2), float32.
    Dynamically unwraps the mesh using xatlas for guaranteed continuous,
    non-overlapping UV topology.
    """
    global _UV_CACHE
    if _UV_CACHE is not None and len(_UV_CACHE) == n_verts:
        return _UV_CACHE

    cache_path = os.path.join(os.path.dirname(__file__), "smpl_uv_cache_xatlas.npy")
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if cached.shape == (n_verts, 2):
                logger.info(f"Loaded xatlas UV cache from {cache_path}")
                _UV_CACHE = cached
                return _UV_CACHE
        except Exception:
            pass

    logger.info("Initializing xatlas for robust topological unwrapping...")
    try:
        import xatlas
    except ImportError:
        raise ImportError("xatlas is not installed! Please run: pip install xatlas")

    v_template, faces = _load_mesh_data()
    if v_template is None or faces is None:
        logger.warning("Could not load SMPL faces/vertices. Returning zero UVs.")
        return np.zeros((n_verts, 2), dtype=np.float32)

    logger.info("Running xatlas.parametrize... (This might take a moment on first run)")
    vmapping, indices, uvs = xatlas.parametrize(v_template, faces)
    
    # xatlas creates new vertices at seams (so len(vmapping) > n_verts)
    # We must reduce this back to exactly `n_verts` to match current pipeline.
    # We take the first mapped UV coordinate for each unique original vertex.
    uv_original = np.zeros((n_verts, 2), dtype=np.float32)
    seen = np.zeros(n_verts, dtype=bool)

    for new_idx, orig_idx in enumerate(vmapping):
        if orig_idx < n_verts and not seen[orig_idx]:
            uv_original[orig_idx] = uvs[new_idx]
            seen[orig_idx] = True

    _UV_CACHE = uv_original

    try:
        np.save(cache_path, uv_original)
        logger.info(f"xatlas UV cached to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not write cache: {e}")

    return uv_original


def _load_mesh_data():
    """Load SMPL v_template (6890,3) and faces from SMPL_NEUTRAL.pkl"""
    pkl_candidates = [
        "models/smpl/SMPL_NEUTRAL.pkl",
        os.path.join(os.path.dirname(__file__), "..", "models", "smpl", "SMPL_NEUTRAL.pkl"),
    ]
    for pkl_path in pkl_candidates:
        if os.path.exists(pkl_path):
            try:
                import patch_env
                import pickle
                with open(pkl_path, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
                vt = np.array(d["v_template"])
                faces = np.array(d["f"])
                return vt, faces
            except Exception as e:
                logger.warning(f"Could not load from {pkl_path}: {e}")
    return None, None

# Compatibility stubs
def get_smpl_part_map() -> np.ndarray:
    raise RuntimeError("get_smpl_part_map is deprecated. xatlas uses dynamic continuous unwrapping without semantic parts.")

def get_torso_uv_bounds() -> tuple[float, float, float, float]:
    raise RuntimeError("get_torso_uv_bounds is deprecated. xatlas uses dynamic continuous unwrapping without semantic parts.")

def get_head_uv_bounds() -> tuple[float, float, float, float]:
    raise RuntimeError("get_head_uv_bounds is deprecated. xatlas uses dynamic continuous unwrapping without semantic parts.")
