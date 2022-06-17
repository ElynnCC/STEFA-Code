__all__ = ['vanilla_polynomial', 'legendre_basis', 'get_basis', 'get_projection_matrix']

from .basis import vanilla_polynomial, legendre_basis, bspline, sinusoidal_basis, wavelet_haar
from .utils import get_projection_matrix

SIEVE_DICT = {'vanilla_polynomial': vanilla_polynomial,
              'legendre_basis': legendre_basis,
              'bspline': bspline,
              'sinusoidal_basis': sinusoidal_basis,
              'wavelet_haar': wavelet_haar}


def get_basis(name):
    return SIEVE_DICT[name]
