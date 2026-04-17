import numpy as np
import sys
import builtins

# This is the "Proper Fix" for Chumpy errors
# It injects legacy attributes where Chumpy expects them
legacy_attrs = {
    'float': float, 'int': int, 'bool': bool,
    'object': object, 'str': str, 'complex': complex
}

for attr, val in legacy_attrs.items():
    if not hasattr(np, attr):
        setattr(np, attr, val)

# Force-patch the 'chumpy' module itself if it's already loaded
if 'chumpy' in sys.modules:
    import chumpy
    chumpy.np = np

print("✓ Environment Patch: Legacy attributes globally injected.")
