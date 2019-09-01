Tutorials
---------

kcsd-python library comes with two extensive tutorials (at this
moment, more soon). These can be tested using Jupyter notebook on your
desktop after installing the kCSD-python package. Alternatively, they
can be played around with without any installation on Binder in a
web-browser. Note that in this case, the results from a browser are
not saved, and cannot be retreived.

Basic Features
~~~~~~~~~~~~~~

This tutorial showcases, the use of the kCSD-python package in the
case of the simplest 2D square grid of electrodes. It defines a region
where can place true current sources, probe it with a grid or
electrodes, based on these measurements and using the proposed method,
estimate the CSD. Since we also know the true sources, we can now
estimate the error in the estimation method as well. To this
preliminary setup, we add noise to the measurements at the electrodes,
and repeat the procedure. We also explore the situation when certain
number of the electrode are broken, and how that would effect the
reconstructions.

The tutorial is provided with Binder access, click the button below to
interact with it in a browser, no installation necessary.

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/Neuroinflab/kCSD-python/master?filepath=tutorials%2Ftutorial_basic.ipynb

	    
For a non-interactive version of this tutorial, see
`Tutorail1<https://github.com/Neuroinflab/kCSD-python/blob/master/tutorials/tutorial_basic.ipynb>`_.

	    
	    
Advanced Features
~~~~~~~~~~~~~~~~~

This tutorial showcases many more features of the kcsd-python package,
especially the cases when the electrodes are distributed in 1D
(laminar probe like), 2D (Neuropixel like), 3D (Utah array like) or on
a microelectrode arrays (MEA, slice with saline). It is provided with
buttons and easy clickable interface to enable the user to navigate
the otherwise dense details. In some ways, it extends on the Basic
Features tutorial and expands it to the other dimensions. NOTE, the
user might have to 'Reset this tutorial' before using to enable the
buttons (Kernel>Restart & Run All).

The tutorial is provided with Binder access, click the button below to
interact with it in a browser, no installation necessary.

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/Neuroinflab/kCSD-python/master?filepath=tutorials%2Ftutorial_advanced.ipynb


For a non-inteactive version of this tutorial, see
`Tutorail2<https://github.com/Neuroinflab/kCSD-python/blob/master/tutorials/tutorial_advanced.ipynb>`_.


More Tutorials
~~~~~~~~~~~~~~

There are many more ways to test this method some of these can be made
into tutorials for easier understanding. While we are working on this,
they will appear here once they reach production level. Please
watch-out this space in the near future. We also take requests for
tutorials.

Do you wish there was a tutorial for your setup? Can you provide us
with the electrode positions and some sample recordings under open
license? If so, we might be able to tailor-make a tutorial for you!
Please get in touch. See Contacts.
