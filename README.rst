Kernel Current Source Density
=============================

This is 1.1+ version of kCSD inverse method proposed in

J. Potworowski, W. Jakuczun, S. Łęski, D. K. Wójcik
"Kernel Current Source Density Method"
Neural Computation 24 (2012), 541–575

For citation policy see the end of this file.


Code status
-----------

ipynb_tests branch

.. image:: https://travis-ci.org/Neuroinflab/kCSD-python.png?branch=ipynb_tests
   :target: https://travis-ci.org/Neuroinflab/kCSD-python

.. image:: https://coveralls.io/repos/github/Neuroinflab/kCSD-python/badge.png?branch=ipynb_tests
   :target: https://coveralls.io/github/Neuroinflab/kCSD-python?branch=ipynb_tests

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/Neuroinflab/kCSD-python/ipynb_tests?filepath=tutorials%2Ftutorial_kcsd.ipynb

error_maps branch

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/m-kowalska/kCSD-python/error_maps?filepath=tutorials%2Fkcsd_tutorial.ipynb
	    
Earlier Stable versions
-----------------------
Please see git tags for earlier versions

- v1.0 corresponds to the version with the test cases written inside tests folder
- v1.1 corresponds to the elephant python library version - no tests here


License
-------
`Modified BSD License`_

.. _Modified BSD License: https://opensource.org/licenses/BSD-3-Clause


Installation
------------

pip install .

or

python setup.py install

(for development / purposes)

pip install -e .

or

python setup.py develop


Requirements
------------

- python 2.7 / 3.4 / 3.5
- numpy 1.10
- scipy 0.17


Additional Packages
-------------------

- scikit-monaco 0.2 (for monte carlo type integration for 3D case)
- matplotlib 0.99 (for visualization and plotting)


Status
------

- KCSD1D (Beta)
- KCSD2D (Beta)
- KCSD3D (Beta)
- MoIKCSD (Beta)


Usage
-----

from kcsd import KCSD1D, KCSD2D, KCSD3D, MoIKCSD

from kcsd import generate as utils

from kcsd import csd_profile as CSD


For data acquired from experiments, please use Elephant_ instead.

.. _Elephant: http://neuralensemble.org/elephant/


Contact
-------
Prof. Daniel K. Wojcik

d.wojcik[at]nencki[dot]gov[dot]pl



Citation policy
---------------
If you use this software in published research please cite the following work

- KCSD1D - [1, 2]
- KCSD2D - [1, 3]
- KCSD3D - [1, 4]
- MoIkCSD - [1, 3, 5]

1) Potworowski, J., Jakuczun, W., Łęski, S. & Wójcik, D. (2012) 'Kernel current source density method.' Neural Comput 24(2), 541-575.

2) Pettersen, K. H., Devor, A., Ulbert, I., Dale, A. M. & Einevoll, G. T. (2006) 'Current-source density estimation based on inversion of electrostatic forward solution: effects of finite extent of neuronal activity and conductivity discontinuities.' J Neurosci Methods 154(1-2), 116-133.

3) Łęski, S., Pettersen, K. H., Tunstall, B., Einevoll, G. T., Gigg, J. & Wójcik, D. K. (2011) 'Inverse Current Source Density method in two dimensions: Inferring neural activation from multielectrode recordings.' Neuroinformatics 9(4), 401-425.

4) Łęski, S., Wójcik, D. K., Tereszczuk, J., Świejkowski, D. A., Kublik, E. & Wróbel, A. (2007) 'Inverse current-source density method in 3D: reconstruction fidelity, boundary effects, and influence of distant sources.' Neuroinformatics 5(4), 207-222.

5) Ness, T. V., Chintaluri, C., Potworowski, J., Łeski, S., Głabska, H., Wójcik, D. K. & Einevoll, G. T. (2015) 'Modelling and Analysis of Electrical Potentials Recorded in Microelectrode Arrays (MEAs).' Neuroinformatics 13(4), 403-426.





