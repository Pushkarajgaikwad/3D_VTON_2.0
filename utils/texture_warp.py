import numpy as np
import logging

logger = logging.getLogger(__name__)


class TextureWarpEngine:
    def __init__(self, texture_resolution=1024):
        self.res = texture_resolution

    # ------------------------------------------------------------------ #
    # Z-depth safety utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inject_synthetic_depth(vertices: np.ndarray) -> np.ndarray:
        """
        Inject a cylindrical depth profile based on vertex Y-coordinates.

        The human body is roughly cylindrical along the Y (height) axis.
        This maps the X-coordinate spread at each height slice into a
        synthetic Z offset, producing a non-flat mesh that Stage 5 can
        export as a valid 3D GLB.
        """
        v = vertices.copy()
        # Normalise X to [-1, 1] range
        x_min, x_max = v[:, 0].min(), v[:, 0].max()
        x_range = x_max - x_min
        if x_range < 1e-6:
            x_range = 1.0
        x_norm = 2.0 * (v[:, 0] - x_min) / x_range - 1.0

        # Cylindrical Z: sqrt(1 - x^2) scaled to ~15% of body height
        y_range = v[:, 1].max() - v[:, 1].min()
        depth_scale = max(y_range * 0.15, 0.05)
        v[:, 2] = np.sqrt(np.clip(1.0 - x_norm ** 2, 0, 1)) * depth_scale

        logger.info(
            f"Synthetic depth injected: Z-range now "
            f"{v[:, 2].min():.4f}..{v[:, 2].max():.4f}"
        )
        return v

    @staticmethod
    def _synthetic_body_fallback(keypoints) -> dict:
        """
        Generate a minimal cylindrical body proxy when no mesh is available.
        Returns a person_repr dict with 6890 synthetic vertices.
        """
        n = 6890
        # Generate a vertical cylinder of points
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        heights = np.linspace(-0.9, 0.9, n // 100 + 1)
        verts = []
        for h in heights:
            # Vary radius by height (wider at shoulders/hips)
            t = (h + 0.9) / 1.8  # normalise to [0, 1]
            radius = 0.15 + 0.08 * np.sin(np.pi * t)  # torso shape
            for a in theta:
                verts.append([radius * np.cos(a), h, radius * np.sin(a)])
                if len(verts) >= n:
                    break
            if len(verts) >= n:
                break

        vertices = np.array(verts[:n], dtype=np.float64)
        logger.warning(
            f"Using synthetic body fallback: {len(vertices)} verts, "
            f"Z-range={vertices[:, 2].ptp():.4f}"
        )
        return {
            "vertices": vertices,
            "keypoints": keypoints,
            "center": np.mean(vertices, axis=0),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_person_repr(self, body_mesh, keypoints):
        """
        Build the person representation dict for the warping stage.

        Includes a strict Z-collapse assertion: if the mesh Z-range is
        below 0.01, synthetic depth is injected to prevent a flat GLB.
        """
        # Case 1: No mesh at all → full synthetic fallback
        if body_mesh is None:
            logger.warning("body_mesh is None — activating synthetic body fallback")
            return self._synthetic_body_fallback(keypoints)

        vertices = np.array(body_mesh.vertices)

        # Case 2: Mesh exists but is flat → inject depth
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        if z_range < 0.01:
            logger.warning(
                f"⚠ Z-COLLAPSE DETECTED in body mesh: Z-range = {z_range:.6f}. "
                f"Injecting synthetic depth to prevent flat GLB export."
            )
            vertices = self._inject_synthetic_depth(vertices)

        return {
            "vertices": vertices,
            "keypoints": keypoints,
            "center": np.mean(vertices, axis=0),
        }

    def warp(self, garment_bytes, person_repr):
        """Return the garment image as a texture (placeholder for TPS warp)."""
        return garment_bytes
