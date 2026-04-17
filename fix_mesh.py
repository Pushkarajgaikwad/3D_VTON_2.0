import pickle
import numpy as np

# A dummy class to catch the data without triggering NumPy's security
class ChumpyDummy:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)

def find_class(module, name):
    if 'chumpy' in module:
        return ChumpyDummy
    return pickle.Unpickler(None).find_class(module, name)

# Custom unpickler that uses the dummy class
class HeavyKiller(pickle.Unpickler):
    def find_class(self, module, name):
        if 'chumpy' in module: return ChumpyDummy
        return super().find_class(module, name)

with open('models/smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
    data = HeavyKiller(f, encoding='latin1').load()

# Extracting the raw values from the dummy objects
v = np.array(data['v_template']._v if hasattr(data['v_template'], '_v') else data['v_template'])
f = np.array(data['f'], dtype=np.int32)

np.savez('templates/tshirt/mesh.npz', vertices=v, faces=f)
print(f"✓ Success! Created mesh.npz with {v.shape[0]} vertices.")
