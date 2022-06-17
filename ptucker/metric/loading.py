import numpy as np


def schatten(A, B, q=2):
    A = np.linalg.qr(A)[0]
    B = np.linalg.qr(B)[0]
    _, s, _ = np.linalg.svd(A.T.dot(B))
    s[np.isclose(s, 1.0)] = 1.0
    nuc = np.sqrt(1 - s ** 2)
    return np.linalg.norm(nuc, q)
