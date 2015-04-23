import numpy as np
from numpy.linalg import lapack_lite

def check_for_duplicated_electrodes(elec_pos):
    """
    **Parameters**
    elec_pos : np.array
    **Returns**
    has_duplicated_elec : Boolean
    """
    unique_elec_pos = np.vstack({tuple(row) for row in elec_pos})
    has_duplicated_elec = unique_elec_pos.shape == elec_pos.shape
    return has_duplicated_elec

def faster_inverse(A): #Taken from http://stackoverflow.com/a/11999063/603292
    b = np.identity(A.shape[1], dtype=A.dtype)
    n_eq = A.shape[0]
    n_rhs = A.shape[1]
    pivots = np.zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = np.zeros(n_eq, np.intc)
        results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise LinAlgError('Singular matrix')
        return b
    return lapack_inverse(A)
