import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from ptucker import proj_tucker
from ptucker.random import core_ensemble, orthogonal_loading, loading_residual
from tensorly.tenalg import multi_mode_dot
from ptucker.sieve import get_basis


I1, I2, I3 = 200, 200, 200
R1, R2, R3 = 3, 3, 3
D1, D2, D3 = 2, 2, 2

S_shape = (I1, I2, I3)
F_shape = (R2, R2, R3)
X_dims = (D1, D2, D3)

sigma = 1
decay = 0.8
order = 3
eta = 0
alpha = 0.3

F = core_ensemble(F_shape, np.min(S_shape) ** alpha)
X = [np.random.uniform(size=(S_shape[k], X_dims[k])) for k in range(3)]
G = [orthogonal_loading(F_shape[k], X[k], basis='legendre_basis', order=order, decay=decay) for k in range(3)]
Gamma = [eta * loading_residual(G[k]) for k in range(3)]
A = [np.linalg.qr(G[k] + Gamma[k])[0] for k in range(3)]
S0 = multi_mode_dot(F, A)
S = S0 + sigma * np.random.normal(size=S0.shape)

factor, loadings, Ahat, _, _ = proj_tucker(S, F_shape, X, 'legendre_basis', {'order': order})

legendre = get_basis("legendre_basis")

basis0 = legendre(X[0], order=order)
coef0 = np.linalg.lstsq(basis0, G[0][:, 0:1])[0]
coef0 *= np.sign(coef0[0])

x1 = np.outer(np.linspace(0, 1, 101), np.ones(101))
x2 = x1.copy().T

xx = np.array([x1.flatten(), x2.flatten()]).T
z0 = legendre(xx, order=order).dot(coef0).reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z0, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('additive_g11.png', dpi=200)
plt.close()

coef = np.linalg.lstsq(basis0, loadings[0][:, 0:1])[0]
coef *= np.sign(coef[0])

z = legendre(xx, order=order).dot(coef).reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('additive_g11_hat.png', dpi=200)
plt.close()



