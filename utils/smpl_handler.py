import torch
import trimesh
import numpy as np
from smplx import SMPL

class SMPLHandler:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        # The NumPy patch in main.py allows this to work now
        self.model = SMPL(model_path, model_type='smpl', gender='neutral').to(device)
        self.faces = self.model.faces
        print("✓ SUCCESS: SMPL High-Fidelity Infrastructure Ready.")

    def generate_tpose(self):
        betas = torch.zeros(1, 10).to(self.device)
        full_pose = torch.zeros(1, 72).to(self.device)
        with torch.no_grad():
            output = self.model(betas=betas, body_pose=full_pose[:, 3:], global_orient=full_pose[:, :3])
        vertices = output.vertices[0].detach().cpu().numpy()
        return trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
