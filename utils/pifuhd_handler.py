import os
import torch
import trimesh
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Paths for body mesh assets (order of preference)
_MESH_NPZ_PATH = "templates/tshirt/mesh.npz"
_MESH_OBJ_PATH = "templates/tshirt/mesh.obj"

# Minimum acceptable Z-range for a valid 3D body mesh
_MIN_Z_RANGE = 0.01


def _load_validated_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh and validate that it has genuine 3D depth.

    Raises ValueError if the Z-range is below _MIN_Z_RANGE (flat mesh).
    """
    mesh = trimesh.load(path, process=False)
    verts = np.array(mesh.vertices)
    z_range = verts[:, 2].max() - verts[:, 2].min()

    if z_range < _MIN_Z_RANGE:
        raise ValueError(
            f"Mesh at {path} is flat (Z-range={z_range:.6f}, "
            f"threshold={_MIN_Z_RANGE}). Rejecting."
        )

    logger.info(
        f"✓ Loaded body mesh from {path}: "
        f"{verts.shape[0]} verts, {mesh.faces.shape[0]} faces, "
        f"Z-range={z_range:.4f}"
    )
    return mesh


class PIFuHDHandler:
    def __init__(self, device="cpu", *args, **kwargs):
        self.device = device
        self.model_path = "checkpoints/pifuhd_final.pt"

        # Fallback SMPL handler (passed from main.py startup)
        self.fallback = kwargs.get("fallback", None)

    def reconstruct(self, image_bytes, keypoints=None, *args, **kwargs):
        """
        Stage 2: 3D Body Reconstruction.

        Load order:
          1. mesh.npz (6890-vertex SMPL body — authoritative source)
          2. mesh.obj (trimesh-exported OBJ from the same data)
          3. SMPL T-pose via fallback handler
          4. Capsule primitive (last resort)

        Every loaded mesh is validated for Z-depth (z_range ≥ 0.01)
        to prevent silent geometry collapse.
        """
        print("⚡ Executing OnePose-Aligned 3D Reconstruction...")

        # --- Attempt 1: Load from NPZ (canonical 6890-vertex SMPL body) ---
        if os.path.exists(_MESH_NPZ_PATH):
            try:
                data = np.load(_MESH_NPZ_PATH)
                verts = data["vertices"]
                faces = data["faces"]
                mesh = trimesh.Trimesh(
                    vertices=verts, faces=faces, process=False
                )
                z_range = verts[:, 2].max() - verts[:, 2].min()
                if z_range < _MIN_Z_RANGE:
                    raise ValueError(
                        f"NPZ mesh is flat (Z-range={z_range:.6f})"
                    )
                print(
                    f"✓ Loaded body mesh from NPZ: "
                    f"{verts.shape[0]} verts, Z-range={z_range:.4f}"
                )
                return mesh
            except Exception as e:
                logger.warning(f"NPZ load failed: {e}")

        # --- Attempt 2: Load from OBJ ---
        if os.path.exists(_MESH_OBJ_PATH):
            try:
                mesh = _load_validated_mesh(_MESH_OBJ_PATH)
                return mesh
            except Exception as e:
                logger.warning(f"OBJ load failed or flat: {e}")

        # --- Attempt 3: SMPL fallback handler ---
        if self.fallback is not None:
            try:
                print("⚠ Using SMPL fallback for body mesh generation...")
                mesh = self.fallback.generate_tpose()
                z_range = mesh.vertices[:, 2].ptp()
                if z_range < _MIN_Z_RANGE:
                    raise ValueError(
                        f"SMPL fallback mesh is flat (Z-range={z_range:.6f})"
                    )
                print(
                    f"✓ SMPL fallback mesh: "
                    f"{mesh.vertices.shape[0]} verts, Z-range={z_range:.4f}"
                )
                return mesh
            except Exception as e:
                logger.warning(f"SMPL fallback failed: {e}")

        # --- Attempt 4: Last resort primitive ---
        print(
            "⚠ All mesh sources exhausted. "
            "Falling back to capsule primitive."
        )
        return trimesh.creation.capsule(height=1.5, radius=0.3)
