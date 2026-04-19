"""
Environment patches for Python 3.11+ / NumPy 1.26.x compatibility.

1. Injects legacy numpy attributes (np.float, np.int, etc.) removed in NumPy 1.24+.
2. Monkey-patches copyreg._reconstructor to safely allocate ndarray subclasses
   (fixes the 'object.__new__(Ch) is not safe' TypeError).
3. Injects a custom __setstate__ on chumpy.Ch to handle legacy dict-based pickle
   state (fixes '__setstate__() argument 1 must be 4-item sequence, not dict').
"""

import numpy as np
import sys
import copyreg

# ---------------------------------------------------------------------------
# 1. Inject legacy numpy scalar aliases
# ---------------------------------------------------------------------------
legacy_attrs = {
    'float': float, 'int': int, 'bool': bool,
    'object': object, 'str': str, 'complex': complex,
}

for attr, val in legacy_attrs.items():
    if not hasattr(np, attr):
        setattr(np, attr, val)

# ---------------------------------------------------------------------------
# 2. Global copyreg interceptor for ndarray subclass unpickling
#
#    Legacy SMPL .pkl pickle streams contain:
#        (copyreg._reconstructor, (Ch, object, None))
#
#    Python 3.11+ rejects object.__new__() for ndarray subclasses.
#    Fix: if cls is an ndarray subclass, use np.ndarray.__new__() instead.
# ---------------------------------------------------------------------------
_original_reconstructor = copyreg._reconstructor


def _safe_reconstructor(cls, base, state):
    if base is object and isinstance(cls, type) and issubclass(cls, np.ndarray):
        obj = np.ndarray.__new__(cls, (0,), dtype=np.float64)
        return obj
    return _original_reconstructor(cls, base, state)


copyreg._reconstructor = _safe_reconstructor

# ---------------------------------------------------------------------------
# 3. Inject __setstate__ on chumpy.Ch to absorb legacy dict state
#
#    After _reconstructor allocates the Ch ndarray, pickle calls
#    __setstate__(state) with the output of old chumpy's __getstate__()
#    — a dict of computation graph metadata.
#
#    np.ndarray.__setstate__ strictly requires a 4-or-5-item tuple
#    (shape, dtype, fortran, data), so it crashes on dicts.
#
#    Fix: intercept __setstate__. If state is a dict, extract any
#    embedded array data. If state is a proper ndarray tuple, forward
#    to the parent. Silently absorb anything else.
# ---------------------------------------------------------------------------

import chumpy


def _ch_setstate(self, state):
    """
    Custom __setstate__ for chumpy.Ch that handles both:
      - dict state (legacy chumpy metadata from old .pkl files)
      - tuple state (standard ndarray serialisation)
    """
    if isinstance(state, tuple):
        # Standard ndarray state: (shape, dtype, fortran, data)
        # or (version, shape, dtype, fortran, data)
        try:
            np.ndarray.__setstate__(self, state)
        except Exception:
            pass
        return

    if not isinstance(state, dict):
        return  # Unknown state type — silently ignore

    # -- Dict state from old chumpy's __getstate__ --
    # Try common keys where array data might be stored.
    arr = None
    for key in ('x', '_data', 'data', 'value', 'a'):
        if key in state:
            try:
                candidate = np.asarray(state[key])
                if candidate.size > 0:
                    arr = candidate
                    break
            except Exception:
                continue

    # Scan all values for any ndarray-like object
    if arr is None:
        for v in state.values():
            if isinstance(v, np.ndarray) and v.size > 0:
                arr = v
                break
            if hasattr(v, '__array__'):
                try:
                    candidate = np.asarray(v)
                    if candidate.size > 0:
                        arr = candidate
                        break
                except Exception:
                    continue

    if arr is not None:
        # Reconstruct the ndarray contents via the proper tuple protocol.
        raw_bytes = arr.tobytes()
        np.ndarray.__setstate__(
            self,
            (1, arr.shape, arr.dtype, False, raw_bytes),
        )


# Inject onto the Ch class so it's active before any pickle.load() call
chumpy.Ch.__setstate__ = _ch_setstate

print("✓ Environment Patch: Legacy attrs + copyreg interceptor + Ch.__setstate__ active.")
