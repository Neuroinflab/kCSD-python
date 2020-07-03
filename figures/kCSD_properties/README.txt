Instructions for the figures from kCSD-revisited paper.

~~~~~~~~~~~~~~~~~~~~~~~
Figure 1 - Schematic

name: figure1.png

~~~~~~~~~~~~~~~~~~~~~~~
Figure 2 - 1D spectral properties of kCSD method

figure_eigensources_M_1D.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 3 - Noise-free Electrode / Basis source placement

figure_Tbasis.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 4 - Noisy electrodes / Basis source placement

figure_Tbasis_noise.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 5 - L-curve method for regularization

You will need to run L_curve_simulation.py first.

figure_LC.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 6 - L-curve versus Cross-validation

You will need to run L_curve_simulation.py first.

figure_LCandCVperformance.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 7 - Error propagation map

error_propagation.py
colorblind_friendly.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 8 - Reliability map

reliability_map_2d.py

error_maps_2D/point_error_large_100_all_ele.npy
error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 9 - Reliability map; Use case in a 2D dipolar large source

kCSD_with_reliability_map_2D.py

error_maps_2D/point_error_large_100_all_ele.npy
error_maps_2D/point_error_small_100_all_ele.npy

~~~~~~~~~~~~~~~~~~~~~~~
Figure 10 - Average Error (Diff) when broken electrode and loss in reconstruction quality

You will need to run tutorial3.py first or download files from here
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0

tutorial_broken_electrodes_diff_err.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 11 - Simulated cortical recordings in Traubs's model

You will need to download files from:
https://repod.pon.edu.pl/dataset/thalamocortical-network/resource/6add09e1-bfe4-4082-b990-24b469756886

npx/traub_data_kcsd_column_figure.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 12 - LFP and CSD as a function of time - Traub's model

You will need to download files from:
https://repod.pon.edu.pl/dataset/thalamocortical-network/resource/6add09e1-bfe4-4082-b990-24b469756886

npx/figure_traub_timespace.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 13 - Six first eigensources for a single bank of a Neuropixels probe

You will need to download files from:
https://repod.pon.edu.pl/dataset/thalamocortical-network/resource/6add09e1-bfe4-4082-b990-24b469756886

npx/figure_traub_eigensources.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 14 - LFP and CSD space profiles - experimental data from a single bank of a Neuropixels probe

You will need to download files from:

npx/kCSD2D_reconstruction_from_npx.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 15 - LFP and CSD as a function of time - experimental data

You will need to download files from:

npx/kCSD2D_reconstruction_from_npx.py

~~~~~~~~~~~~~~~~~~~~~~~~
Figure 16 - L-Curve and CV landscape

You will need to run L_curve_simulation.py first.

figure_LCandCV.py

~~~~~~~~~~~~~~~~~~~~~~~
Figure 17 - Schematic - location of Neuropixels bank 0

name: figure17.png

=====================
Supplementary Figures
=====================

~~~~~~~~~~~~~~~~~~~~~~~~~~
Figure 2-Figure supplement 1 - spectral properties of kCSD method for simple 2D case

figure_eigensources_M_2D.py

~~~~~~~~~~~~~~~~~~~~~~~~~~
Figure 7-Figure supplement 1 - Error propagation maps for 1D

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

================
Appendix Figures
================

~~~~~~~~~~~~~~~~~~~~~~~
Appendix 1 Figure 1 - Basic features tutorial

You will need to run tutorial3.py first or download files from here
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0

tutorial_basic.py

~~~~~~~~~~~~~~~~~~~~~~~
Appendix 1 Figure 2 - Noisy electrodes tutorial

tutorial_noisy_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~~
Appendix 1 Figure 3 - Broken electrodes tutorial

Download first from
https://www.dropbox.com/sh/6kykj4d3dx3fp5s/AAACtN49VCbAHA9otOfNXbnOa?dl=0
(generated from tweaking tutorial3.py)

tutorial_broken_electrodes.py

~~~~~~~~~~~~~~~~~~~~~~~~~~
Appendix 1 Figure 4 - 3D source reconstruction

tutorial_basic_3d.py

~~~~~~~~~~~~~~~~~~~~~~~~~~
Appendix 1 Figure 5 - sKCSD example

skcsd_and_l_curve_complex_morphology.py
