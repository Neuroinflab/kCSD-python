
2017_12_20
1) Tagged the previous stable version as v1.0
2) Tagged the version on elephant as v1.1
3) Merged elephant version which is Py2/3 compatible into master
4) Changed the folder structure a bit to accomodate for the new dev
   -corelib has the KCSD.py file which includes all the necessary classes
   -tests is where the unit tests will eventually go
   -validation is where the tests from the previous edition go (poor naming choice by me, in retrospect)
               these are being re-written in a saner 'class'y way now.
   -tutorials is where ther ipython notebooks will eventually supposed to go.
   

NOTES FROM v1.0 which is no longer applicable. Eventually should all move to tutorials.
=====================================================================
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

========================================================================


2016_04_18
Preparing for release
Moved functions from KCSD1D_Helpers etc into something more sane.
Using **kwargs instead of params to pass agruments
Saner method names.

Comments for future
1) Possibility to use errfunc instead of gaussian

2016_03_07
Pushing KCSD1D + tests to beta phase

Comments for future
0) Write unit tests and facilitate regression tests for future.
1) Write test_kCSDx.py as a class
2) The gaussian source normalization, see guass_rescale_xD function.
3) Better uniform way to generate csd_profiles, across 3 dims. For legacy reasons, left as it is for now.
4) Better names for functions and variables? (k.update_b_interp_pot, and k_interp_cross etc)

2016_02_09
Pushing KCSD3D + tests to beta phase

2015_10_12
REASON for such absurd merging from tags is as follows
1) intended to not inclue the test_kcsd2d cases
2.1) someone asked as to how they can trust the method
2.2) someone asked if the default values in kcsd2d method have any meaning - which they dont
3) in two separate instances, gave the code to the respective people hence multiple versions
4) unable to keep up with multiple versions, now including this in trunk.
