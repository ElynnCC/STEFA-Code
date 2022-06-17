import numpy as np
from ..sieve import get_basis, get_projection_matrix
from scipy.stats import ortho_group


def orthogonal_loading(dim, x, basis, order, basis_args=None, decay=0.5):
    if isinstance(basis, str):
        basis = get_basis(basis)

    assert callable(basis)

    if basis_args is None:
        basis_args = {}

    a = basis(x, order=order, **basis_args)
    xdim = (a.shape[1] - 1) // order
    scale = np.array([1, *np.tile(decay ** np.arange(order), xdim)])

    a = (a * scale).dot(np.random.normal(size=(a.shape[1], dim)))

    a, _ = np.linalg.qr(a)
    return a


def loading_residual(loading, rng=np.random.normal, rng_args=None):
    if rng_args is None:
        rng_args = {}
    gamma = rng(size=loading.shape, **rng_args)
    P = get_projection_matrix(loading)
    out = gamma - P.dot(gamma)
    std = np.sqrt(np.sum(out ** 2, 0))
    return out / std
