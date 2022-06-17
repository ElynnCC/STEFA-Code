from tensorly.decomposition import tucker as old_tucker
from tensorly.tenalg import multi_mode_dot
import numpy as np
import tensorly as tl


def tucker(tensor, ranks,
           n_iter_max=100,
           init='svd',
           svd='numpy_svd',
           tol=10e-5,
           random_state=None,
           verbose=False):
    core, loading = old_tucker(tensor=tensor,
                               rank=ranks,
                               n_iter_max=n_iter_max,
                               init=init,
                               svd=svd,
                               tol=tol,
                               random_state=random_state,
                               verbose=verbose)

    dim = tl.ndim(tensor)
    sign = [np.sign(loading[k][0]) for k in range(dim)]
    loading = [loading[k] * sign[k] for k in range(dim)]
    core = multi_mode_dot(core, [np.diag(s) for s in sign])
    return core, np.array(loading)
