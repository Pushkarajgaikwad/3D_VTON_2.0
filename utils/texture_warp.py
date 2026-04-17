import numpy as np

class TextureWarpEngine:
    def __init__(self, texture_resolution=1024):
        self.res = texture_resolution

    def build_person_repr(self, body_mesh, keypoints):
        # Ensure we aren't dealing with a None object
        if body_mesh is None:
            return {"vertices": np.zeros((6890, 3)), "keypoints": keypoints}
            
        vertices = np.array(body_mesh.vertices)
        return {
            "vertices": vertices,
            "keypoints": keypoints,
            "center": np.mean(vertices, axis=0)
        }

    def warp(self, garment_bytes, person_repr):
        # Logic to return the garment image as a texture
        return garment_bytes 
