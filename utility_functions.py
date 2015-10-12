import numpy as np
from numpy.linalg import LinAlgError

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

def calc_error(k_pot, pots, lambd, index_generator):
    """
    Useful for Cross validation error calculations
    """
    err = 0
    for idx_train, idx_test in index_generator:
        B_train = k_pot[np.ix_(idx_train, idx_train)]
        V_train = pots[idx_train]
        V_test = pots[idx_test]
        I_matrix = np.identity(len(idx_train))
        B_new = np.matrix(B_train) + (lambd*I_matrix)
        try:                                                                                                                                                                     
            beta_new = np.dot(np.matrix(B_new).I, np.matrix(V_train))
            B_test = k_pot[np.ix_(idx_test, idx_train)]
            V_est = np.zeros((len(idx_test), pots.shape[1]))
            for ii in range(len(idx_train)):
                for tt in range(pots.shape[1]):
                    V_est[:, tt] += beta_new[ii, tt] * B_test[:, ii]
            err += np.linalg.norm(V_est-V_test)
        except LinAlgError:
            print 'Encoutered Singular Matrix Error: try changing ele_pos'
            err = 10000. #singluar matrix errors!
    return err
