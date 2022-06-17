import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from ptucker import proj_tucker
from ptucker.random import core_ensemble, orthogonal_loading, loading_residual
from tensorly.tenalg import multi_mode_dot
from ptucker.sieve import get_basis, get_projection_matrix


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
fitorder = 3

F = core_ensemble(F_shape, np.min(S_shape) ** alpha)
X = [np.random.uniform(size=(S_shape[k], X_dims[k])) for k in range(3)]
legendre = get_basis("legendre_basis")

coef_comp = [np.random.normal(size=(order + 1, X_dims[k], F_shape[k])) * (decay ** np.arange(order + 1))[:, None, None] for k in range(3)]
basis = [[legendre(X[k][:, d:d+1], order=order) for d in range(X_dims[k])] for k in range(3)]
G_comp = [[[basis[k][d].dot(coef_comp[k][:, d, r]) for d in range(X_dims[k])] for r in range(F_shape[k])]for k in range(3)]
G = [np.array([np.product(G_comp[k][r], axis=0) for r in range(F_shape[k])]).T for k in range(3)]
G = [np.linalg.qr(G[k])[0] for k in range(3)]


Gamma = [eta * loading_residual(G[k]) for k in range(3)]
A = [np.linalg.qr(G[k] + Gamma[k])[0] for k in range(3)]
S0 = multi_mode_dot(F, A)
S = S0 + sigma * np.random.normal(size=S0.shape)

factor, loadings, Ahat, _, _ = proj_tucker(S, F_shape, X, 'legendre_basis', {'order': fitorder})



x1 = np.outer(np.linspace(0, 1, 101), np.ones(101))
x2 = x1.copy().T

xx = np.array([x1.flatten(), x2.flatten()]).T

xx_basis0 = [legendre(xx[:, d:d+1], order=order) for d in range(X_dims[0])]

z_comp = [[xx_basis0[d].dot(coef_comp[0][:, d, r]) for d in range(X_dims[0])] for r in range(F_shape[0])]

z0 = np.array([np.product(z_comp[r], axis=0) for r in range(F_shape[0])]).T
print(z0.shape)
z0 = np.linalg.qr(z0)[0][:, 0]

# z0 = legendre(xx[:, 0:1], order=order).dot(coef_comp[0][:, 0, 0]) * legendre(xx[:, 1:2], order=order).dot(coef_comp[0][:, 1, 0])
z0 = z0.reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z0 * np.sign(z0[0, 0]), cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('multiple_g11.png', dpi=200)
plt.close()

basis = legendre(X[0], order=fitorder)

coef = np.linalg.lstsq(basis, loadings[0][:, 0:1])[0]
# coef *= np.sign(coef[0])

z = legendre(xx, order=fitorder).dot(coef).reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, z * np.sign(z[0, 0]), cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('mutiple_g11_hat.png', dpi=200)
plt.close()

xx_basis = legendre(xx, order=fitorder)
Pz0 = xx_basis.dot(np.linalg.inv(xx_basis.T.dot(xx_basis)).dot(xx_basis.T.dot(z0.flatten()))).reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, Pz0 * np.sign(Pz0[0, 0]), cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('mutiple_g11_p.png', dpi=200)
plt.close()

coefs = np.linalg.lstsq(basis, loadings[0])[0]
g_hat = xx_basis.dot(coefs)

transform = np.linalg.lstsq(g_hat, z0.flatten())[0]
linear_z = g_hat.dot(transform).reshape(101, 101)

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, linear_z * np.sign(linear_z[0, 0]), cmap='viridis', edgecolor='none', rstride=1, cstride=1)
plt.tight_layout()
plt.savefig('mutiple_g11_hat_linear.png', dpi=200)
plt.close()



