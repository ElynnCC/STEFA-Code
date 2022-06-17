import numpy as np


def get_projection_matrix(basis):
    return basis.dot(np.linalg.inv(basis.T.dot(basis))).dot(basis.T)
