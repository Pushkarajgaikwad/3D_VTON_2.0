try:
    import patch_env
    import chumpy as ch
    import numpy as np
    # This line specifically triggers the 'object.__new__(Ch)' error if not fixed
    test_array = ch.array(np.zeros(5))
    print("✓ VERIFICATION SUCCESS: Basic Infrastructure (Chumpy) is now SAFE.")
except Exception as e:
    print(f"✗ VERIFICATION FAILED: {e}")
