import numpy
try:
    import cupy
except:
    print("WARNING: GPU acceleration is not available.")
    from . import cpu
else:
    from . import gpu