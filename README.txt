This is ALPHA version of kCSD method.
See Jan Potworowski et.al. 2012 kernel Current Source Density

branched test cases from tag 0.1.1c version (see NOTES)

Status:

KCSD2D + tests : BETA
KCSD3D : ALPHA

Legend:

KCSD2D.py - relevant KCD2D reconstruction file (Includes Cross validation for R and lambd)
CSD.py - base class of CSD
KCSD2D_Helpers.py - relevant KCD2D Helper functions
utility_functions.py - necessary generic functions 

./tests/test_kCSD2D.py - file generates, TrueCSD, potentails in a plane, and its kCSD reconstruction
	     - use with relevant seed - for small sources and large sources.
	     - illustrates the basic API of KCSD2D.py
./tests/csd_profile.py - used by test_kCSD2D.py for CSD profiles.

Try:

In file test_kCSD2D.py play with 
1) csd_seed to change True CSD profile
2) total_ele to change the number of electrodes in the plane of electrodes
3) in main_loop change ele_lims to change the position of the electrodes 
4) in do_kcsd function play with params

NOTES:

2015_10_12
REASON for such absurd merging from tags is as follows
1) intended to not inclue the test_kcsd2d cases
2.1) someone asked as to how they can trust the method
2.2) someone asked if the default values in kcsd2d method have any meaning - which they dont
3) in two separate instances, gave the code to the respective people hence multiple versions
4) unable to keep up with multiple versions, now including this in trunk.
