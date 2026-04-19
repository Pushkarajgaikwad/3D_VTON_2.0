import trimesh
import trimesh.visual
import trimesh.visual.material
import numpy as np
from PIL import Image

def test_export():
    # Load the base mesh
    mesh_path = "templates/tshirt/mesh.obj"
    mesh = trimesh.load(mesh_path, process=False)
    
    # Create a red texture image
    red_img = np.zeros((256, 256, 3), dtype=np.uint8)
    red_img[:, :, 0] = 255
    pil_texture = Image.fromarray(red_img)

    # Spherical UV mapping
    vertices = np.array(mesh.vertices)
    centered = vertices - vertices.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x**2 + z**2))
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    uv_coords = np.stack([u, v], axis=1).astype(np.float64)
    uv_coords = np.clip(uv_coords, 0.0, 1.0)

    # Create Material
    material = trimesh.visual.material.SimpleMaterial(image=pil_texture)
    
    # Another option for Simple vs PBR
    # material = trimesh.visual.material.PBRMaterial(
    #     baseColorTexture=pil_texture,
    #     baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    # )

    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv_coords,
        material=material,
    )

    # Export
    out_path = "test_color_output.glb"
    mesh.export(out_path)
    print(f"Exported to {out_path}")

    # Re-load and verify
    reloaded = trimesh.load(out_path, process=False)
    
    if hasattr(reloaded.visual, 'material') and hasattr(reloaded.visual.material, 'image'):
        if reloaded.visual.material.image is not None:
            print("SUCCESS: GLB retained image texture.")
        elif getattr(reloaded.visual.material, 'baseColorTexture', None) is not None:
            print("SUCCESS: GLB retained PBR baseColorTexture.")
        else:
            print("FAILURE: Material has no image.")
    elif hasattr(reloaded.visual, 'uv') and len(reloaded.visual.uv) > 0:
        print("PARTIAL SUCCESS: UV map is there, but no texture image attached.")
    else:
        print("FAILURE: No material or visual attached.")
        
if __name__ == '__main__':
    test_export()
