This is 1.0 version of kCSD inverse method proposed in

J. Potworowski, W. Jakuczun, S. Łęski, D. K. Wójcik
"Kernel Current Source Density Method"
Neural Computation 24 (2012), 541–575

For citation policy see the end of this file.

License
-------
Modified BSD License

Contact
-------
Prof. Daniel K. Wojcik
d.wojcik[at]nencki[dot]gov[dot]pl

Status
------
KCSD1D + tests(rc)
KCSD2D + tests(rc)
KCSD3D + tests(rc)
MoIKCSD (for MEA's with Saline)

Requirements
------------
python 2.7
numpy 1.10
scipy 0.17
matplotlib 1.5 (Only For tests and visualization)

Additional Packages - Only of 3D for newer basis functions only.
-------------------
scikit-monaco 0.2
joblib

Try
---
In file test_kCSD1D.py, test_kCSD2D.py and test_kCSD3D.py play with 
1) csd_seed to change True CSD profile
2) total_ele to change the number of electrodes in the plane of electrodes, n evenly spaced in 1D, n^2 in 2D and n^3 in 3D volume
3) in main_loop change ele_lims to change the position of the electrodes 
4) in do_kcsd function play with params for regularization 

Legend
------
CSD.py - base class of CSD
KCSD.py - base class for kernel CSD methods.
KCSD1D.py - relevant KCD1D reconstruction file (Inherits from KCSD.py)
KCSD2D.py - relevant KCD2D reconstruction file (Inherits from KCSD.py)
KCSD3D.py - relevant KCD3D reconstruction file (inherits from KCSD.py)
MoIKCSD.py - relevant KCSD2D which includes the method of images - models saline conductivity

basis_functions.py - necessary functions that are used as basis sources
utility_functions.py - necessary generic functions 

tests/test_kCSD1D.py - file generates TrueCSD, potentials in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py

tests/test_kCSD2D.py - file generates TrueCSD, potentials in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py

tests/test_kCSD3D.py - file generates TrueCSD, potentials in a volume, and its kCSD reconstruction
	     - use with relevant seed - for small sources, large sources, monopoles, dipole sources
	     - illustrates the basic API of KCSD3D.py

tests/csd_profile.py - used by test_kCSD1D.py, test_kCSD2D.py, test_kCSD3D.py for CSD profiles.


Citation policy
---------------
If you use this software in published research please cite the following work
- kCSD1D - [1, 2]
- kCSD2D - [1, 3]
- kCSD3D - [1, 4]
- MoIkCSD - [1, 3, 5]

[1] Potworowski, J., Jakuczun, W., Łęski, S. & Wójcik, D. (2012) 'Kernel current source density method.' Neural Comput 24(2), 541-575.

[2] Pettersen, K. H., Devor, A., Ulbert, I., Dale, A. M. & Einevoll, G. T. (2006) 'Current-source density estimation based on inversion of electrostatic forward solution: effects of finite extent of neuronal activity and conductivity discontinuities.' J Neurosci Methods 154(1-2), 116-133.

[3] Łęski, S., Pettersen, K. H., Tunstall, B., Einevoll, G. T., Gigg, J. & Wójcik, D. K. (2011) 'Inverse Current Source Density method in two dimensions: Inferring neural activation from multielectrode recordings.' Neuroinformatics 9(4), 401-425.

[4] Łęski, S., Wójcik, D. K., Tereszczuk, J., Świejkowski, D. A., Kublik, E. & Wróbel, A. (2007) 'Inverse current-source density method in 3D: reconstruction fidelity, boundary effects, and influence of distant sources.' Neuroinformatics 5(4), 207-222.

[5] Ness, T. V., Chintaluri, C., Potworowski, J., Łeski, S., Głabska, H., Wójcik, D. K. & Einevoll, G. T. (2015) 'Modelling and Analysis of Electrical Potentials Recorded in Microelectrode Arrays (MEAs).' Neuroinformatics 13(4), 403-426.





