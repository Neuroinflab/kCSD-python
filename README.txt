This is ALPHA version of kCSD method.
See Jan Potworowski et.al. 2012 kernel Current Source Density

Status:

KCSD2D + tests : BETA
KCSD3D + tests : BETA

Legend:

CSD.py - base class of CSD
KCSD2D.py - relevant KCD2D reconstruction file (Includes Cross validation for R and lambd)
KCSD3D.py - relevant KCD3D reconstruction file (inherits from KCSD2D.py)

KCSD2D_Helpers.py - relevant KCD2D Helper functions
KCSD3D_Helpers.py - relevant KCD3D Helper functions

utility_functions.py - necessary generic functions 

./tests/test_kCSD2D.py - file generates, TrueCSD, potentails in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py

./tests/test_kCSD3D.py - file generates, TrueCSD, potentails in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources, large sources, monopoles, dipole sources
	     - illustrates the basic API of KCSD3D.py

./tests/csd_profile.py - used by test_kCSD2D.py, test_kCSD3D.py for CSD profiles.

Try:

In file test_kCSD2D.py and test_kCSD3D.py play with 
1) csd_seed to change True CSD profile
2) total_ele to change the number of electrodes in the plane of electrodes n^2 in 2D and n^3 in 3D
3) in main_loop change ele_lims to change the position of the electrodes 
4) in do_kcsd function play with params for regularization 

NOTES:

2016_02_09
Pushing KCSD3D + tests to beta phase

2015_10_12
REASON for such absurd merging from tags is as follows
1) intended to not inclue the test_kcsd2d cases
2.1) someone asked as to how they can trust the method
2.2) someone asked if the default values in kcsd2d method have any meaning - which they dont
3) in two separate instances, gave the code to the respective people hence multiple versions
4) unable to keep up with multiple versions, now including this in trunk.
