import numpy as np
import tensorly as tl
from ptucker import tucker


def orthogonal_slice_core(size, rng=np.random.normal, rng_args=None):
    if rng_args is None:
        rng_args = {}

    f = rng(size=size, **rng_args)
    # return tucker(f, size)[0]
    for m in range(len(size)):
        MmF = tl.unfold(f, m)
        _, c, V = np.linalg.svd(MmF, full_matrices=False)
        f = tl.fold(np.diag(c).dot(V), 0, size)
    return f


def core_ensemble(size, strength=1, rng=np.random.normal, rng_args=None):
    if rng_args is None:
        rng_args = {}
    f = rng(size=size, **rng_args)
    s = []
    for m in range(len(size)):
        MmF = tl.unfold(f, m)
        _, c, V = np.linalg.svd(MmF, full_matrices=False)
        s.append(np.min(np.abs(c)))
        f = tl.fold(np.diag(c).dot(V), m, size)

    return f * strength / min(s)


