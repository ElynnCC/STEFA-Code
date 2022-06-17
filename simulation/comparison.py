import numpy as np
from pandas import to_pickle
import json

from tensorly.tenalg import multi_mode_dot
from multiprocessing import Pool

from ptucker import tucker, proj_tucker
from ptucker.random import orthogonal_loading, core_ensemble, loading_residual
from ptucker.metric import schatten


def repeat(func, repeats, name, kwargs=None, cores=3):
    def task(i):
        return func(**kwargs)

    result = np.array([func(**kwargs) for _ in range(repeats)])
    # pool = Pool(cores)
    # out= pool.map(task, range(repeats))
    # pool.close()
    # result = np.array(out)
    to_pickle(result, 'output/' + name + '.pkl')
    with open('output/' + name + '.config', 'w+') as f:
        json.dump(kwargs, f)
    return result


def simulation_compare(S_shape, F_shape, X_dims, snr, sieve_basis, gen_order, fit_order, q, eta=0, fit_with_gamma=False):
    F = core_ensemble(F_shape, strength=np.min(S_shape) ** snr)
    X = [np.random.uniform(size=[S_shape[k], X_dims[k]]) for k in range(3)]
    G = [orthogonal_loading(F_shape[k], X[k], sieve_basis, order=gen_order, decay=0.8) for k in range(3)]
    Gamma = [eta * loading_residual(G[k]) for k in range(3)]
    A = [np.linalg.qr(G[k] + Gamma[k])[0] for k in range(3)]
    S0 = multi_mode_dot(F, A)
    E = np.random.normal(size=S0.shape)
    S = S0 + E

    # _, G_truth = tucker(multi_mode_dot(F, G), F_shape)

    # F_truth, loadings_truth = tucker(S0, F_shape)

    F_tucker, loadings_tucker = tucker(S, F_shape)
    Y_tucker = multi_mode_dot(F_tucker, loadings_tucker)

    F_ptucker, G_ptucker, A_ptucker, _, _ = proj_tucker(S, F_shape, X, sieve_basis, {'order': fit_order})

    if not fit_with_gamma:
        A_ptucker = G_ptucker
    #     Y_ptucker = multi_mode_dot(F_ptucker, A_ptucker)
    # else:
    Y_ptucker = multi_mode_dot(F_ptucker, A_ptucker)

    A_error_tucker = [schatten(A[k], loadings_tucker[k], q) for k in range(3)]
    A_error_ptucker = [schatten(A[k], A_ptucker[k], q) for k in range(3)]
    # F_error_tucker = np.linalg.norm(F_truth - F_tucker) / np.linalg.norm(F_truth)
    # F_error_ptucker = np.linalg.norm(F_truth - F_ptucker) / np.linalg.norm(F_truth)

    G1_error = np.sum((G_ptucker[0] * np.sign(G_ptucker[0][0]) - G[0] * np.sign(G[0][0])) ** 2, axis=0) / np.sum(G[0] ** 2, axis=0)
    Y_error_tucker = np.linalg.norm(S0 - Y_tucker) / np.linalg.norm(S0)
    Y_error_ptucker = np.linalg.norm(S0 - Y_ptucker) / np.linalg.norm(S0)

    return [*A_error_tucker, Y_error_tucker, *A_error_ptucker, Y_error_ptucker, *G1_error]






