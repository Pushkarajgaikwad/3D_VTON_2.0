"""Quick smoke test: verify SMPL .pkl loads via our chumpy shim without TypeError."""
import patch_env
import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
import chumpy

print(f"NumPy version: {np.__version__}")
print(f"Ch has __reduce_ex__: {hasattr(chumpy.Ch, '__reduce_ex__')}")

pkl_path = "models/smpl/SMPL_NEUTRAL.pkl"
with open(pkl_path, "rb") as f:
    model = pickle.load(f, encoding="latin1")

print(f"SMPL keys: {list(model.keys())[:8]}...")
verts = np.array(model.get("v_template", []))
print(f"v_template shape: {verts.shape}")
print("SUCCESS: SMPL .pkl loaded without TypeError")
