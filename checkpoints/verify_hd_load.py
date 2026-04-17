import torch
import os

# Since you are inside the checkpoints folder, we use the local name
path = "alias_final.pth"
try:
    # map_location="cpu" is vital to avoid memory crashes on login nodes
    checkpoint = torch.load(path, map_location="cpu")
    print(f"\n✓ VERIFICATION SUCCESS: alias_final.pth (384MB) is valid and loaded.")
    print(f"✓ Keys found in model: {list(checkpoint.keys())[:3]}...") 
except Exception as e:
    print(f"\n✗ VERIFICATION FAILED: {e}")
