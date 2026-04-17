import sys
import numpy as np

# Force patches for check
try:
    import numpy
    numpy.float = float
    numpy.int = int
    print(f"✓ NumPy version: {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy Patch Error: {e}")

print("\n--- Testing FaceExtractor ---")
try:
    import mediapipe as mp
    from utils.face_extractor import FaceExtractor
    test_face = FaceExtractor()
    print("✓ SUCCESS: FaceExtractor initialized.")
except Exception as e:
    print(f"✗ FAILED: FaceExtractor Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing PIFuHD Weights ---")
import os
path = "checkpoints/pifuhd_final.pth"
if os.path.exists(path):
    print(f"✓ Found PIFuHD 1.5GB weights at {path} ({os.path.getsize(path)/1e9:.2f} GB)")
else:
    print(f"✗ MISSING: Weights not found at {path}")
