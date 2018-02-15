# -*- coding: utf-8 -*-
"""
Unit tests for the kCSD methods
This was written by :
Chaitanya Chintaluri,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
from kcsd import generate as utils
from kcsd import csd_profile as CSD
from kcsd import KCSD1D, KCSD2D, MoIKCSD, KCSD3D


class KCSD1D_TestCase(unittest.TestCase):
    def setUp(self):
        dim = 1
        self.num_ele, self.ele_pos = utils.generate_electrodes(dim=dim)
        self.csd_profile = CSD.gauss_1d_mono
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile, dim=dim)
        pots = utils.calculate_potential(self.csd_at, self.csd, self.ele_pos, h=1.)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD1D'
        self.test_params = {'h': 1., 'sigma':1.}


    def test_kcsd1d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD1D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_x)
        rms = np.linalg.norm(np.array(vals[0, :]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS between trueCSD and estimate > 0.5')

    def test_valid_inputs(self):
        self.test_method = 'KCSD1D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd1d_estimate)
        self.test_method = 'KCSD1D'
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd1d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd1d_estimate, cv_params)


class KCSD2D_TestCase(unittest.TestCase):
    def setUp(self):
        dim = 2
        self.num_ele, self.ele_pos = utils.generate_electrodes(ele_lim=[0.05, 0.95], ele_res=9, dim=dim)
        self.csd_profile = CSD.gauss_2d_large
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile, dim=dim)
        pots = utils.calculate_potential(self.csd_at, self.csd, self.ele_pos, h=10., sigma=0.3)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD2D'
        self.test_params = {'gdx': 0.25, 'gdy': 0.25, 'R_init': 0.08,
                            'h': 10., 'xmin': 0., 'xmax': 1.,
                            'ymin': 0., 'ymax': 1., 'sigma':0.3}


    def test_kcsd2d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD2D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_pos)
        print(true_csd.shape, vals.shape) # Meh here!
        rms = np.linalg.norm(np.array(vals[:, :, 0]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS ' + str(rms) +
                        'between trueCSD and estimate > 0.5')

    def test_moi_estimate(self):
        result = MoIKCSD(self.ele_pos, self.pots, n_src_init=500,
                         MoI_iters=20, sigma_S=0.3, **self.test_params)
        result.cross_validate(Rs=np.array((0.41, 0.42)))
        vals = result.values()
        true_csd = self.csd_profile(result.estm_pos)
        # print(true_csd.shape, vals[0,:]) # Meh here!
        rms = np.linalg.norm(np.array(vals[:, :, 0]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 2.5, msg='RMS ' + str(rms) +
                        'between trueCSD and estimate > 2.5')

    def test_valid_inputs(self):
        self.test_method = 'KCSD2D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd2d_estimate)
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd2d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd2d_estimate, cv_params)


class KCSD3D_TestCase(unittest.TestCase):
    def setUp(self):
        dim = 3
        self.num_ele, self.ele_pos = utils.generate_electrodes(ele_lim=[0.15, 0.85], ele_res=5, dim=dim)
        self.csd_profile = CSD.gauss_3d_large
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile, dim=dim)
        pots = utils.calculate_potential(self.csd_at, self.csd, self.ele_pos, h=1., sigma=0.3)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD3D'
        self.test_params = {'gdx': 0.05, 'gdy': 0.05, 'gdz': 0.05,
                            'lambd': 5.10896977451e-19, 'src_type': 'step',
                            'R_init': 0.31, 'xmin': 0., 'xmax': 1., 'ymin': 0.,
                            'ymax': 1., 'zmin': 0., 'zmax': 1.}


    def test_kcsd3d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD3D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_pos)
        print(true_csd.shape, vals.shape) # Meh here!
        rms = np.linalg.norm(np.array(vals[:, :, :, 0]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS ' + str(rms) +
                        'between trueCSD and estimate > 0.5')

    def test_valid_inputs(self):
        self.test_method = 'KCSD3D'
        self.test_params = {'src_type': 22}
        self.assertRaises(KeyError, self.test_kcsd3d_estimate)
        self.test_params = {'InvalidKwarg': 21}
        self.assertRaises(TypeError, self.test_kcsd3d_estimate)
        cv_params = {'InvalidCVArg': np.array((0.1, 0.25, 0.5))}
        self.assertRaises(TypeError, self.test_kcsd3d_estimate, cv_params)

if __name__ == '__main__':
    unittest.main()
