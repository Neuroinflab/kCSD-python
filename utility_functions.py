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

def calc_error(k_pot, pots, lambd, index_generator):
    '''Useful for Cross validation - when done in parallel'''
    err = 0
    for idx_train, idx_test in index_generator:
        B_train = k_pot[np.ix_(idx_train, idx_train)]
        V_train = pots[idx_train]
        V_test = pots[idx_test]
        I = np.identity(len(idx_train))
        B_new = np.matrix(B_train) + (lambd*I)
        beta_new = np.dot(np.matrix(B_new).I, np.matrix(V_train))
        #beta_new = np.dot(faster_inverse(B_new), np.matrix(V_train))
        B_test = k_pot[np.ix_(idx_test, idx_train)]
        V_est = np.zeros((len(idx_test),1))
        for ii in range(len(idx_train)):
            V_est += beta_new[ii,0] * B_test[:, ii]
        err += np.linalg.norm(V_est-V_test)
    return err
