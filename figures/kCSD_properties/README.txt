Instructions for the figures from kCSD-python, reliable current source density estimation with quality control.

~~~~~~~~~~~~~~~~~~~~~~~
Figure 1 - Schematic

name: figure1.png

~~~~~~~~~~~~~~~~~~~~~~~
Figure 2 - L-curve method for regularization

You will need to run L_curve_simulation.py first.

figure_LC.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 3 - Error propagation map

error_propagation.py
colorblind_friendly.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 4 - Reliability map 2D

reliability_map_2d.py

Download kCSD_properties.zip file from here: http://bit.ly/kCSD-supplementary
and unzip the data
kCSD_properties/error_maps_2D/point_error_large_100_all_ele.npy
kCSD_properties/error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 5 - Reliability map; Use case in a 2D dipolar large source

kCSD_with_reliability_map_2D.py

Download kCSD_properties.zip file from here http://bit.ly/kCSD-supplementary
and unzip the data
kCSD_properties/error_maps_2D/point_error_large_100_all_ele.npy
kCSD_properties/error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 6 - Average Error (Diff) when broken electrode and loss in reconstruction quality

You will need to run tutorial3.py first or download kCSD_properties.zip file from here
http://bit.ly/kCSD-supplementary

tutorial_broken_electrodes_diff_err.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 7 - L-Curve and CV landscape

You will need to run L_curve_simulation.py first.

figure_LCandCV.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 8 - Basic features tutorial

You will need to run tutorial3.py first or download kCSD_properties.zip file from here
http://bit.ly/kCSD-supplementary

tutorial_basic.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 9 - Noisy electrodes tutorial

tutorial_noisy_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 10 - Broken electrodes tutorial

Download kCSD_properties.zip file first from
http://bit.ly/kCSD-supplementary
(generated from tweaking tutorial3.py)

tutorial_broken_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 11 - Error propagation maps for 1D

pots_propagation.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 12 - 3D source reconstruction

tutorial_basic_3d.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 13 - sKCSD example

You will need to install LFPy package first:
pip install lfpy

skcsd_and_l_curve_complex_morphology.py
