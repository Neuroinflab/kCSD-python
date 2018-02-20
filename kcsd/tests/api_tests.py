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
from kcsd import KCSD1D, KCSD2D, KCSD3D


available_1d = ['KCSD1D']
available_2d = ['KCSD2D', 'MoIKCSD']
available_3d = ['KCSD3D']
kernel_methods = ['KCSD1D', 'KCSD2D', 'KCSD3D', 'MoIKCSD']


class LFP_TestCase(unittest.TestCase):
    def default_setting(self, dim, csd_instance):
        num_ele, ele_pos = utils.generate_electrodes(dim=dim)
        csd_at, csd = utils.generate_csd(csd_instance, dim=dim)
        lfp = utils.calculate_potential(csd_at, csd, ele_pos, h=1)        
        return ele_pos, lfp
    
    def test_lfp1d_electrodes(self):
        ele_pos, lfp = self.default_setting(1, CSD.gauss_1d_dipole)
        self.assertEqual(ele_pos.shape[1], 1)
        self.assertEqual(ele_pos.shape[0], len(lfp))

    def test_lfp2d_electrodes(self):
        ele_pos, lfp = self.default_setting(2, CSD.gauss_2d_large)
        self.assertEqual(ele_pos.shape[1], 2)
        self.assertEqual(ele_pos.shape[0], len(lfp))

    def test_lfp3d_electrodes(self):
        ele_pos, lfp = self.default_setting(3, CSD.gauss_3d_small)
        self.assertEqual(ele_pos.shape[1], 3)
        self.assertEqual(ele_pos.shape[0], len(lfp))

        
# class CSD1D_TestCase(unittest.TestCase):
#     def setUp(self):
#         self.num_ele, self.ele_pos = utils.generate_electrodes(dim=1)
#         self.lfp = utils.generate_lfp(CSD.gauss_1d_dipole, self.ele_pos)
#         self.csd_method = csd.estimate_csd

#         self.params = {}  # Input dictionaries for each method
#         self.params['KCSD1D'] = {'h': 50., 'Rs': np.array((0.1, 0.25, 0.5))}

#     def test_validate_inputs(self):
#         self.assertRaises(TypeError, self.csd_method, lfp=[[1], [2], [3]])
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           coords=self.ele_pos * pq.mm)
#         # inconsistent number of electrodes
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           coords=[1, 2, 3, 4] * pq.mm, method='StandardCSD')
#         # bad method name
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           method='InvalidMethodName')
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           method='KCSD2D')
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           method='KCSD3D')

#     def test_inputs_standardcsd(self):
#         method = 'StandardCSD'
#         result = self.csd_method(self.lfp, method=method)
#         self.assertEqual(result.t_start, 0.0 * pq.s)
#         self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#         self.assertEqual(len(result.times), 1)

#     def test_inputs_deltasplineicsd(self):
#         methods = ['DeltaiCSD', 'SplineiCSD']
#         for method in methods:
#             self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                               method=method)
#             result = self.csd_method(self.lfp, method=method,
#                                      **self.params[method])
#             self.assertEqual(result.t_start, 0.0 * pq.s)
#             self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#             self.assertEqual(len(result.times), 1)

#     def test_inputs_stepicsd(self):
#         method = 'StepiCSD'
#         self.assertRaises(ValueError, self.csd_method, lfp=self.lfp,
#                           method=method)
#         self.assertRaises(AssertionError, self.csd_method, lfp=self.lfp,
#                           method=method, **self.params[method])
#         self.params['StepiCSD'].update({'h': np.ones(5) * 100E-6 * pq.m})
#         result = self.csd_method(self.lfp, method=method,
#                                  **self.params[method])
#         self.assertEqual(result.t_start, 0.0 * pq.s)
#         self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#         self.assertEqual(len(result.times), 1)

#     def test_inuts_kcsd(self):
#         method = 'KCSD1D'
#         result = self.csd_method(self.lfp, method=method,
#                                  **self.params[method])
#         self.assertEqual(result.t_start, 0.0 * pq.s)
#         self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#         self.assertEqual(len(result.times), 1)


# class CSD2D_TestCase(unittest.TestCase):
#     def setUp(self):
#         xx_ele, yy_ele = utils.generate_electrodes(dim=2)
#         self.lfp = csd.generate_lfp(utils.large_source_2D, xx_ele, yy_ele)
#         self.params = {}  # Input dictionaries for each method
#         self.params['KCSD2D'] = {'sigma': 1., 'Rs': np.array((0.1, 0.25, 0.5))}

#     def test_kcsd2d_init(self):
#         method = 'KCSD2D'
#         result = csd.estimate_csd(lfp=self.lfp, method=method,
#                                   **self.params[method])
#         self.assertEqual(result.t_start, 0.0 * pq.s)
#         self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#         self.assertEqual(len(result.times), 1)


# class CSD3D_TestCase(unittest.TestCase):
#     def setUp(self):
#         xx_ele, yy_ele, zz_ele = utils.generate_electrodes(dim=3)
#         self.lfp = csd.generate_lfp(utils.gauss_3d_dipole,
#                                     xx_ele, yy_ele, zz_ele)
#         self.params = {}
#         self.params['KCSD3D'] = {'gdx': 0.1, 'gdy': 0.1, 'gdz': 0.1,
#                                  'src_type': 'step',
#                                  'Rs': np.array((0.1, 0.25, 0.5))}

#     def test_kcsd2d_init(self):
#         method = 'KCSD3D'
#         result = csd.estimate_csd(lfp=self.lfp, method=method,
#                                   **self.params[method])
#         self.assertEqual(result.t_start, 0.0 * pq.s)
#         self.assertEqual(result.sampling_rate, 1000 * pq.Hz)
#         self.assertEqual(len(result.times), 1)


if __name__ == '__main__':
    unittest.main()
