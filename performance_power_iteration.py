import tensorly as tl
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot
import numpy as np
from numpy import sqrt

import warnings


def check_random_state(seed):

    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('Seed should be None, int or np.random.RandomState')


def partial_tucker(tensor, modes, rank=None, n_iter_max=100, init='svd', tol=10e-5,
                   svd='numpy_svd', random_state=None, verbose=False, ranks=None, callback=None):

    if ranks is not None:
        message = "'ranks' is depreciated, please use 'rank' instead"
        warnings.warn(message, DeprecationWarning)
        rank = ranks

    if rank is None:
        message = "No value given for 'rank'. The decomposition will preserve the original size."
        warnings.warn(message, Warning)
        rank = [tl.shape(tensor)[mode] for mode in modes]
    elif isinstance(rank, int):
        message = "Given only one int for 'rank' intead of a list of {} modes. Using this rank for all modes.".format(len(modes))
        warnings.warn(message, Warning)
        rank = [rank for _ in modes]

    try:
        svd_fun = tl.SVD_FUNS[svd]
    except KeyError:
        message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                svd, tl.get_backend(), tl.SVD_FUNS)
        raise ValueError(message)

    # SVD init
    if init == 'svd':
        factors = []
        for index, mode in enumerate(modes):
            eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank[index])
            factors.append(eigenvecs)
    else:
        rng = check_random_state(random_state)
        core = tl.tensor(rng.random_sample(rank), **tl.context(tensor))
        factors = [tl.tensor(rng.random_sample((tl.shape(tensor)[mode], rank[index])), **tl.context(tensor)) for (index, mode) in enumerate(modes)]

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if callable(callback):
        callback(0, multi_mode_dot(tensor, factors, modes=modes, transpose=True), factors)

    for iteration in range(n_iter_max):

        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(tensor, factors, modes=modes, skip=index, transpose=True)
            eigenvecs, _, _ = svd_fun(unfold(core_approximation, mode), n_eigenvecs=rank[index])
            factors[index] = eigenvecs

        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

        # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
        rec_error = sqrt(abs(norm_tensor**2 - tl.norm(core, 2)**2)) / norm_tensor
        rec_errors.append(rec_error)

        if callable(callback):
            callback(iteration + 1, core, factors)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return core, factors


def tucker(tensor, rank=None, ranks=None, n_iter_max=100, init='svd',
           svd='numpy_svd', tol=10e-5, random_state=None, verbose=False, callback=None):
    modes = list(range(tl.ndim(tensor)))
    return partial_tucker(tensor, modes, rank=rank, ranks=ranks, n_iter_max=n_iter_max, init=init,
                          svd=svd, tol=tol, random_state=random_state, verbose=verbose, callback=callback)


if __name__ == "__main__":

    from ptucker.random import core_ensemble, orthogonal_loading
    from ptucker.metric import schatten
    from ptucker.sieve import get_basis, get_projection_matrix
    import matplotlib.pyplot as plt
    from tensorly.tenalg import mode_dot

    F_shape = [3, 3, 3]
    S_shape = [200, 200, 200]
    X_dims = [2, 2, 2]
    snr = 0.8
    eta = 0.0
    order = 10
    F = core_ensemble(F_shape, np.min(S_shape) ** snr)
    X = [np.random.uniform(size=[S_shape[k], X_dims[k]]) for k in range(3)]
    G = [orthogonal_loading(F_shape[k], X[k], basis='legendre_basis', order=order, decay=0.8) for k in range(3)]
    Gamma = [eta * np.random.normal(size=G[k].shape) for k in range(3)]
    A = G
    A = [np.linalg.qr(G[k] + Gamma[k])[0] for k in range(3)]

    P = [get_projection_matrix(np.linalg.qr(get_basis('legendre_basis')(X[k], order=order))[0]) for k in range(3)]

    # print(np.linalg.eigvals(P[0]))
    S0 = multi_mode_dot(F, A)
    E = np.random.normal(size=S0.shape)
    S = S0 + E

    PS = multi_mode_dot(S, P)

    errors = []


    def callback(iteration, core, factors):
        global errors
        # F_error = tl.norm(F - core) ** 2 / tl.norm(F) ** 2
        A_error = np.mean([schatten(A[k], factors[k]) for k in range(3)])
        Y_error = tl.norm(multi_mode_dot(core, factors) - S0) ** 2 / tl.norm(S0) ** 2

        # axis = [np.delete(np.arange(3), k) for k in range(3)]
        # y_partial = [tl.unfold(multi_mode_dot(S, [factors[kk].T for kk in axis[k]], axis[k]), k) for k in range(3)]
        # Q = [tl.unfold(core, k) for k in range(3)]
        # Ahat = [y_partial[k].dot(Q[k].T).dot(np.linalg.inv(Q[k].dot(Q[k].T))) for k in range(3)]
        # Y_error = tl.norm(multi_mode_dot(core, Ahat) - S0) ** 2 / tl.norm(F) ** 2
        print(f"Iter {iteration}: Y_error: {Y_error:.4f}, A_error: {A_error:.2f}.")
        errors.append([Y_error, A_error])


    tol = 0
    iter = 10


    factor, loadings = tucker(PS, F_shape, n_iter_max=iter, tol=tol, callback=callback)

    errors = np.array(errors)
    plt.plot(np.arange(iter+1), errors[:, 0], label="pTucker: Y error")
    plt.plot(np.arange(iter+1), errors[:, 1], label="pTucker: G error")
    errors = []
    tucker(S, F_shape, n_iter_max=iter, tol=tol, callback=callback)
    errors = np.array(errors)
    plt.plot(np.arange(iter+1), errors[:, 0], label="Tucker: Y error")
    plt.plot(np.arange(iter+1), errors[:, 1], label="Tucker: G error")
    plt.xlabel("power iteration")
    plt.ylabel('Error')
    plt.legend()
    plt.title(f"SNR: alpha = {snr}, order(J) = {order}")
    plt.savefig(f'snr_{snr}_order_{order}.png', dpi=100)
    # plt.show()




