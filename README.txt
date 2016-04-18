This is 1.0 version of kCSD inverse method.
See Jan Potworowski et.al. 2012 kernel Current Source Density

License
-------
???


Contact
-------
Prof. Daniel K. Wojcik
d.wojcik[at]nencki[dot]gov[dot]pl

Status
------
KCSD1D + tests - rc
KCSD2D + tests - rc
KCSD3D + tests - rc

Requirements
------------
python 2.7
numpy 1.10
scipy 0.17

Additional Packages
-------------------
scikit-monaco 0.2
joblib
multiprocessing

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
KCSD2D.py - relevant KCD2D reconstruction file (Includes Cross validation for R and lambd)
KCSD3D.py - relevant KCD3D reconstruction file (inherits from KCSD2D.py)
KCSD1D.py - relevant KCD1D reconstruction file (inherits from KCSD2D.py)

basis_functions.py - necessary functions that are used as basis sources
utility_functions.py - necessary generic functions 

./tests/test_kCSD2D.py - file generates, TrueCSD, potentails in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py

./tests/test_kCSD3D.py - file generates, TrueCSD, potentails in a volume, and its kCSD reconstruction
	     - use with relevant seed - for small sources, large sources, monopoles, dipole sources
	     - illustrates the basic API of KCSD3D.py

./tests/test_kCSD1D.py - file generates, TrueCSD, potentails in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py

./tests/csd_profile.py - used by test_kCSD1D.py, test_kCSD2D.py, test_kCSD3D.py for CSD profiles.
