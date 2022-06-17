import numpy as np
from scipy.special import legendre
from scipy.interpolate._bspl import evaluate_all_bspl
from functools import wraps


def broadcast(f):
    @wraps(f)
    def inner(x, order, param=None):
        ones = np.ones([len(x), 1])

        if param is None:
            param = {}
        if isinstance(param, dict) and x.ndim > 1:
            param = [param] * x.shape[1]

        if isinstance(order, int) and x.ndim > 1:
            order = [order] * x.shape[1]

        if x.ndim == 1:
            return np.hstack([ones, f(x, order, **param)])
        else:
            basis_by_col = [f(x[:, k], order[k], **param[k]) for k in range(x.shape[1])]
            return np.hstack([ones, * basis_by_col])
    return inner

@broadcast
def vanilla_polynomial(x, order):
    return np.array([x ** k for k in range(1, order + 1)]).T


# def vanilla_polynomial(x, order, output_list=False):
#     if x.ndim == 1:
#         if output_list:
#             return [np.ones([len(x), 1]), [x ** k for k in range(1, order + 1)]]
#         else:
#             return np.hstack([x ** k for k in range(order + 1)])
#     else:
#         colbasis = [x[:, col:col + 1] ** np.arange(1, order + 1) for col in range(x.shape[1])]
#         if output_list:
#             return [np.ones([x.shape[0], 1]), *colbasis]
#         else:
#             return np.hstack([np.ones([x.shape[0], 1]), *colbasis])


@broadcast
def legendre_basis(x, order):
    return np.array([legendre(k)(2 * x - 1) for k in range(1, order + 1)]).T

# def legendre_basis(x, order):
#     n, xdim = x.shape
#     return np.hstack([np.ones(shape=(n, 1)),
#                       *[np.hstack([legendre(k)(2 * x[:, col:col + 1] - 1) for k in np.arange(1, order + 1)]) for col in
#                         range(xdim)]])

@broadcast
def bspline(x, order, knots):
    t = sorted(knots)
    t = [0] * order + t + [1] * order
    t = np.array(t, dtype="double")
    m = len(knots) + order - 1

    index = np.searchsorted(t, x, side="right") - 1
    index = np.minimum(index, m - 1)

    out = np.zeros((len(x), m))
    for i in range(len(x)):
        out[i, index[i] - order: index[i] + 1] = evaluate_all_bspl(t, order, x[i], index[i])
    return out[:, :-1]

# def bspline(x, order, knots):
#     n, xdim = x.shape
#
#     if isinstance(order, int):
#         order = [order] * xdim
#
#     if not isinstance(knots[0], list) and not isinstance(knots[0], np.ndarray):
#         knots = [knots] * xdim
#
#     basis = [_bspline(x[:, k], order[k], knots[k]) for k in range(xdim)]
#     return np.hstack(basis)

#
# def _bspline(x, order, knots):
#     t = sorted(knots)
#     a, b = t[0], t[-1]
#     t = [a] * order + t + [b] * order
#     t = np.array(t, dtype="double")
#     m = len(knots) + order - 1
#
#     index = np.searchsorted(t, x, side="right") - 1
#     index = np.minimum(index, m - 1)
#
#     out = np.zeros((len(x), m))
#     for i in range(len(x)):
#         out[i, index[i] - order: index[i] + 1] = evaluate_all_bspl(t, order, x[i], index[i])
#
#     return out


''' 
Example: 
x = np.random.uniform(0, 1, size=(100, 2)) # 100 units dim = 2

basis = bspline(x, 
                order=[3, 2],
                param=[{"knots": np.linspace(0, 1, num=4)},
                       {"knots": np.linspace(0, 1, num=5)}])

import matplotlib.pyplot as plt

order = np.argsort(x, axis=0)
x_sort0 = x[order[:, 0], 0]
basis_sort0 = basis[order[:, 0], :6]
plt.plot(x_sort0, basis_sort0)
plt.show()
'''


@broadcast
def sinusoidal_basis(x, order):
    return np.array([np.sin(np.pi * x * k) for k in range(1, order + 1)]).T


@broadcast
def wavelet_haar(x, order):
    basis = []
    for k in range(1, order + 1):
        per_basis = np.zeros(shape=(len(x), 2 ** (k - 1)))
        index = np.minimum(np.floor(x * 2 ** k).astype(np.int), 2 ** k - 1)
        block = index // 2
        value = 2 * (index % 2) - 1
        per_basis[range(len(x)), block] = value
        basis.append(per_basis)
    return np.hstack(basis)
