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

error_maps_2D/point_error_large_100_all_ele.npy
error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 5 - Reliability map; Use case in a 2D dipolar large source

kCSD_with_reliability_map_2D.py

error_maps_2D/point_error_large_100_all_ele.npy
error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 6 - Average Error (Diff) when broken electrode and loss in reconstruction quality

You will need to run tutorial3.py first or download files from here
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0

tutorial_broken_electrodes_diff_err.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 7 - L-Curve and CV landscape

You will need to run L_curve_simulation.py first.

figure_LCandCV.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 8 - Basic features tutorial

You will need to run tutorial3.py first or download files from here
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0

tutorial_basic.py

~~~~~~~~~~~~~~~~~~~~~~~

Figure 9 - Noisy electrodes tutorial

tutorial_noisy_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 10 - Broken electrodes tutorial

Download first from
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0
(generated from tweaking tutorial3.py)

tutorial_broken_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 11 - Error propagation maps for 1D

pots_propagation.py

~~~~~~~~~~~~~~~~~~~~~~~~~~
Figure 13-Figure supplement 1 - Eigensurces 7-12 ('start=6', 'stop=12'),
Figure 13-Figure supplement 2 - Eigensurces 13-18 ('start=12', 'stop=18'),
Figure 13-Figure supplement 3 - Eigensurces 19-24 ('start=18', 'stop=24'),
Figure 13-Figure supplement 4 - Eigensurces 25-30 ('start=24', 'stop=30'),
Figure 13-Figure supplement 5 - Eigensurces 31-36 ('start=30', 'stop=36'),
Figure 13-Figure supplement 6 - Eigensurces 37-42 ('start=36', 'stop=42'),
Figure 13-Figure supplement 7 - Eigensurces 43-48 ('start=42', 'stop=48')

All supplementary figures to Figure 13 were created using different
'start' and 'stop'parameters at:
npx/figure_traub_eigensources.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 12 - 3D source reconstruction

tutorial_basic_3d.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 13 - sKCSD example

skcsd_and_l_curve_complex_morphology.py
