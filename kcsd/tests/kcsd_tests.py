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
from kcsd import ValidateKCSD1D, ValidateKCSD2D, ValidateKCSD3D
from kcsd import csd_profile as CSD
from kcsd import KCSD1D, KCSD2D, MoIKCSD, KCSD3D, oKCSD2D, oKCSD3D


class KCSD1D_TestCase(unittest.TestCase):
    def setUp(self):
        dim = 1
        utils = ValidateKCSD1D(csd_seed=42)
        self.ele_pos = utils.generate_electrodes(total_ele=10,
                                                 ele_lims=[0.1, 0.9])
        self.csd_profile = CSD.gauss_1d_mono
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile,
                                                   csd_seed=42)
        pots = utils.calculate_potential(self.csd, self.csd_at, self.ele_pos,
                                         h=1., sigma=0.3)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD1D'
        self.test_params = {'h': 1., 'sigma': 0.3, 'R_init': 0.2,
                            'n_src_init': 100, 'xmin': 0., 'xmax': 1.,}

    def test_kcsd1d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD1D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_x, 42)
        rms = np.linalg.norm(np.array(vals[:, 0]) - true_csd)
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
        utils = ValidateKCSD2D(csd_seed=43)
        self.ele_pos = utils.generate_electrodes(total_ele=49,
                                                 ele_lims=[0.1, 0.9])
        self.csd_profile = CSD.gauss_2d_large
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile,
                                                   csd_seed=43)
        pots = utils.calculate_potential(self.csd, self.csd_at, self.ele_pos,
                                         h=10., sigma=0.3)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD2D'
        self.test_params = {'gdx': 0.25, 'gdy': 0.25, 'R_init': 0.3,
                            'h': 10., 'xmin': 0., 'xmax': 1.,
                            'ymin': 0., 'ymax': 1., 'sigma': 0.3}

    def test_kcsd2d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD2D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_pos, 43)
#        print(true_csd.shape, vals.shape)  # Meh here!
        rms = np.linalg.norm(np.array(vals[:, :, 0]) - true_csd)
        rms /= np.linalg.norm(true_csd)
        self.assertLess(rms, 0.5, msg='RMS ' + str(rms) +
                        'between trueCSD and estimate > 0.5')

    def test_moi_estimate(self):
        result = MoIKCSD(self.ele_pos, self.pots, n_src_init=200,
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
        utils = ValidateKCSD3D(csd_seed=44)
        self.ele_pos = utils.generate_electrodes(total_ele=64,
                                                 ele_lims=[0.1, 0.9])
        self.csd_profile = CSD.gauss_3d_large
        self.csd_at, self.csd = utils.generate_csd(self.csd_profile,
                                                   csd_seed=44)
        pots = utils.calculate_potential(self.csd, self.csd_at, self.ele_pos,
                                         h=50., sigma=1.)
        self.pots = np.reshape(pots, (-1, 1))
        self.test_method = 'KCSD3D'
        self.test_params = {'gdx': 0.2, 'gdy': 0.2, 'gdz': 0.2,
                            'src_type': 'gauss',
                            'R_init': 0.31, 'xmin': 0., 'xmax': 1., 'ymin': 0.,
                            'ymax': 1., 'zmin': 0., 'zmax': 1.,
                            'n_src_init': 100}

    def test_kcsd3d_estimate(self, cv_params={}):
        self.test_params.update(cv_params)
        result = KCSD3D(self.ele_pos, self.pots,
                        **self.test_params)
        result.cross_validate()
        vals = result.values()
        true_csd = self.csd_profile(result.estm_pos, 44)
        print(true_csd.shape, vals.shape)  # Meh here!
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

class oKCSD2D_TestCase(unittest.TestCase):
    def test_2D(self):
        ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [1.2, 1.2]])
        pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1]])
        own_est = np.array([[1,2,3,4,5,6,7,8,9,10], [1,2,3,1,2,2,5,1,5,1]])
        k = oKCSD2D(ele_pos, pots, own_src = own_src, own_est = own_est)
        k.cross_validate()
#        k.L_curve()
        print('oKCSD2D test done')

    def test_2D_no_est(self):
        ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [1.2, 1.2]])
        pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1]])
        k = oKCSD2D(ele_pos, pots, own_src = own_src)
        print('own_est overwritten with own_src: ', (k.own_src == k.own_est).all())
#        k.L_curve()
        print('oKCSD2D test done')

    def test_2D_wrong_param(self):
        ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [1.2, 1.2]])
        pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
        self.assertRaises(KeyError, oKCSD2D, ele_pos, pots)

class oKCSD3D_TestCase(unittest.TestCase):
    def test_3D(self):
        ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                            (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),(0.5, 0.5, 0.5)])
        pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1], [1,1,1,1,1,5,3,4,2,5]])
        own_est = own_src+5
        k = oKCSD3D(ele_pos, pots, own_src = own_src, own_est = own_est)
        k.cross_validate()
#        k.L_curve()
        print('oKCSD3D test done')

    def test_3D_no_est(self):
        ele_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                            (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),(0.5, 0.5, 0.5)])
        pots = np.array([[-0.5], [0], [-0.5], [0], [0], [0.2], [0], [0], [1]])
        own_src = np.array([[1,2,3,4,5,6,7,8,9,10], [0,0,1,1,2,2,1,1,1,1], [1,1,1,1,1,5,3,4,2,5]])
        k = oKCSD3D(ele_pos, pots, own_src = own_src)
        print('own_est overwritten with own_src: ',(k.own_src == k.own_est).all())
#        k.L_curve()
        print('oKCSD3D test done')

    def test_3D_wrong_param(self):
        ele_pos = np.array([[-0.2, -0.2], [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [1.2, 1.2]])
        pots = np.array([[-1], [-1], [-1], [0], [0], [1], [-1.5]])
        self.assertRaises(KeyError, oKCSD3D, ele_pos, pots)


if __name__ == '__main__':
    unittest.main()