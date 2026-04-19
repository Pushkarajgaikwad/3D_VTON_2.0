"""
Minimal chumpy compatibility shim for Python 3.11+.

The original SMPL .pkl files were serialised with chumpy arrays.
This stub converts them to plain numpy arrays on load so that
smplx can read the model without requiring the real chumpy package.

CRITICAL FIX (Python 3.11 / NumPy ≥1.24):
    The original __reduce__ returned ``(Ch, (data,))`` which caused
    ``copyreg._reconstructor`` to call ``object.__new__(Ch)``.
    Python 3.11 rejects this for ndarray subclasses with:
        TypeError: object.__new__(Ch) is not safe, use numpy.ndarray.__new__()
    The fix uses __reduce_ex__ with a safe reconstruction helper that
    returns a plain np.ndarray — which is all smplx/SMPL needs.
"""

import numpy as np


# ---- Safe pickle reconstruction (module-level for pickle to find) ---- #

def _reconstruct_ch(data_bytes, shape, dtype_str):
    """
    Reconstruct a chumpy ``Ch`` object from pickle as a plain ``np.ndarray``.

    Returning ndarray instead of Ch is safe because:
      - smplx only calls ``.r`` or ``np.asarray()`` on chumpy objects,
        both of which work identically on plain ndarrays.
      - Avoiding the Ch subclass entirely sidesteps ALL ndarray subclass
        allocation safety checks introduced in Python 3.11+ / NumPy 1.24+.
    """
    arr = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).copy()
    return arr.reshape(shape)


class Ch(np.ndarray):
    """
    Drop-in stub for chumpy.Ch.
    Behaves like a numpy array so smplx can use it directly.
    """

    def __new__(cls, x=None, *args, **kwargs):
        if x is None:
            arr = np.array([])
        elif hasattr(x, '__array__'):
            arr = np.asarray(x)
        else:
            try:
                arr = np.array(x)
            except Exception:
                arr = np.array([])
        return arr.view(cls)

    # smplx / SMPL code sometimes accesses .r to get the numpy value
    @property
    def r(self):
        return np.asarray(self)

    def __reduce_ex__(self, protocol):
        # Safe reconstruction: serialize as raw bytes + metadata,
        # reconstruct as plain np.ndarray (not Ch) to avoid
        # object.__new__(Ch) strictness in Python 3.11+.
        plain = np.asarray(self)
        return (
            _reconstruct_ch,
            (plain.tobytes(), plain.shape, plain.dtype.str),
        )

    # Silence attribute errors for chumpy-specific methods
    def __getattr__(self, name):
        raise AttributeError(name)


def array(x, *args, **kwargs):
    return np.array(x)


# Aliases used by some SMPL pkl files
zeros  = np.zeros
ones   = np.ones
arange = np.arange


class reordering_csc_matrix:
    """Stub for sparse matrix type used in some SMPL variants."""
    def __init__(self, *args, **kwargs):
        pass
