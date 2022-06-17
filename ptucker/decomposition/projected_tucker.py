from ..sieve import get_basis, get_projection_matrix
from . import tucker
from tensorly.tenalg import multi_mode_dot
import numpy as np
import tensorly as tl


def proj_tucker(tensor, ranks, x, basis,
                basis_args=None,
                n_iter_max=100,
                init='svd',
                svd='numpy_svd',
                tol=10e-5,
                random_state=None,
                verbose=False):
    d = len(tensor.shape)
    if isinstance(basis, list):
        basis = [get_basis(base) if isinstance(base, str) else base for base in basis]
    else:
        if isinstance(basis, str):
            basis = [get_basis(basis)] * d
        else:
            basis = [basis] * d

    assert isinstance(basis, list)
    for base in basis:
        if base:
            assert callable(base)

    if basis_args is None:
        basis_args = [{}] * d
    elif not isinstance(basis_args, list):
        basis_args = [basis_args] * d

    assert isinstance(basis_args, list)
    for basis_arg in basis_args:
        if basis_arg:
            assert isinstance(basis_arg, dict)

    modes = [k for k in range(d) if x[k] is not None]
    # n_modes = len(modes)
    basis_d = [np.linalg.qr(basis[k](x[k], **basis_args[k]))[0] if k in modes
             else None for k in range(d)]
    # basis = [basis(x[k], **basis_args) if k in modes
    #          else None for k in range(d)]
    projections = np.array([get_projection_matrix(basis_d[k]) if k in modes else None for k in range(d)])
    # projections = np.array([get_projection_matrix(np.linalg.qr(basis(x[k], **basis_args))[0]) if k in modes
    #                         else None for k in range(d)])
    projections_modes = projections[modes]

    factor, loadings = tucker(tensor=multi_mode_dot(tensor, projections_modes, modes),
                              ranks=ranks,
                              n_iter_max=n_iter_max,
                              init=init,
                              svd=svd,
                              tol=tol,
                              random_state=random_state,
                              verbose=verbose)

    coefs = [np.linalg.lstsq(basis_d[k], loadings[k], rcond=None)[0] if k in modes else None for k in range(d)]

    axis = {modes[k]: np.delete(modes, k) for k in range(len(modes))}
    axis_full = [np.delete(np.arange(d), k) for k in range(d)]
    y_partial = {k: tl.unfold(multi_mode_dot(tensor, projections[axis[k]], axis[k]), k) for k in modes}
    Q = {k: tl.unfold(multi_mode_dot(factor, loadings[axis_full[k]], axis_full[k]), k) for k in modes}
    # A = [y_partial[k].dot(Q[k].T).dot(np.linalg.inv(Q[k].dot(Q[k].T))) if k in modes
    #      else loadings[k] for k in range(d)]
    A = [np.linalg.lstsq(Q[k].T, y_partial[k].T, rcond=None)[0].T if k in modes
         else loadings[k] for k in range(d)]

    return factor, loadings, A, basis_d, coefs
