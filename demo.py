import numpy as np
from ptucker import proj_tucker, tucker
from ptucker.random import core_ensemble, orthogonal_loading, loading_residual
from ptucker.metric import schatten
from tensorly.tenalg import multi_mode_dot
from ptucker.sieve import get_basis, get_projection_matrix


I1, I2, I3 = 200, 210, 220
R1, R2, R3 = 5, 5, 5
D1, D2, D3 = 2, 2, 2

S_shape = (I1, I2, I3)
F_shape = (R1, R2, R3)
X_dims = (D1, D2, D3)

sigma = 1
decay = 0.8
order = 10
eta = 0.1

# Generate F
F = core_ensemble(F_shape, np.min(S_shape) ** 0.5)

# Generate X1, X2, X3
X = [np.random.uniform(size=(S_shape[k], X_dims[k])) for k in range(3)]

# Generate G(X)
G = [orthogonal_loading(F_shape[k], X[k], basis='legendre_basis', order=order, decay=decay) for k in range(3)]

# Generate Gamma
Gamma = [eta * loading_residual(G[k]) for k in range(3)]

# Assemble
A = [np.linalg.qr(G[k] + Gamma[k])[0] for k in range(3)]


S00 = multi_mode_dot(F, G)
S0 = multi_mode_dot(F, A)
S = multi_mode_dot(F, A) + sigma * np.random.normal(size=S0.shape)

# Solve the problem without Gamma -- easy
factor00, loadings00 = tucker(S00, F_shape)
print("Tucker without Gamma")
print([schatten(loadings00[k], G[k]) for k in range(3)])

# Projected Tucker
factor0, loadings0= tucker(S, F_shape)
print("Tucker G error")
print([schatten(loadings0[k], G[k]) for k in range(3)])
print("Tucker A error")
print([schatten(loadings0[k], A[k]) for k in range(3)])
print("Tucker Y error")
print(np.linalg.norm(multi_mode_dot(factor0, loadings0) - S0))

# Projected Tucker
factor, loadings, Ahat, basis, coefs = proj_tucker(S, F_shape, [None, X[1], X[2]], 'legendre_basis', {'order': order})
print("pTucker G error")
print([schatten(loadings[k], G[k]) for k in range(3)])
print("pTucker A error")
print([schatten(Ahat[k], A[k]) for k in range(3)])
print("Tucker Y error")
print(np.linalg.norm(multi_mode_dot(factor, Ahat) - S0))

print(basis)
print(coefs)
print(basis[1].dot(coefs[1]) - loadings[1])
