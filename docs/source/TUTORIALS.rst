Tutorials
---------

kcsd-python library comes with three extensive tutorials. These
tutorials can be explored online or installed on your desktop computer
depending on your usage. To simple test it out try the online version
first, and if required install the python package.

**Online version**

To play around with the library without any installation, you can run
the tutorials on Google collaboratory in a web-browser. **Note:** the
results from a browser are not saved and cannot be retreived. If you
wish to have the results stored somewhere, please use the desktop
version.

**Desktop version**

This requires you to install the kCSD-package and additionally to
install jupyter notebook on your desktop.

..  code-block:: bash

		 pip install kcsd
		 pip install jupyter notebook




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

The tutorial is provided with Google collaboratory, click the button below to
interact with it in a browser, no installation necessary.

`Basic tutorial <https://colab.research.google.com/drive/1M7fCR5iZ9c7SAZWWoq9WLfFpk7pCaufd?usp=sharing>`_

..
      .. image:: https://mybinder.org/badge.svg
	 :target: https://mybinder.org/v2/gh/Neuroinflab/kCSD-python/master?filepath=tutorials%2Ftutorial_basic.ipynb

	    
For a non-interactive static version of this tutorial, see
`Tutorial1 <https://github.com/Neuroinflab/kCSD-python/blob/master/tutorials/tutorial_basic.ipynb>`_.

	    
	    
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

The tutorial is provided with Google collaboratory, click the button below to
interact with it in a browser, no installation necessary.

`Advanced tutorial <https://colab.research.google.com/drive/1gIuBJ2XzOGmgnRuxKguHevcYoE6eY_o1?usp=sharing>`_

..
   .. image:: https://mybinder.org/badge.svg
      :target: https://mybinder.org/v2/gh/Neuroinflab/kCSD-python/master?filepath=tutorials%2Ftutorial_advanced.ipynb


For a non-interactive static version of this tutorial, see
`Tutorial2 <https://github.com/Neuroinflab/kCSD-python/blob/master/tutorials/tutorial_advanced.ipynb>`_.

sKCSD Tutorial
~~~~~~~~~~~~~~

This tutorial showcases the possibility to reconstruct the current sources
of a single neuron provided the morphology of the said neuron is known.
This methods has been described extensively here: https://doi.org/10.7554/eLife.29384


`sKCSD tutorial <https://colab.research.google.com/drive/1tjOvC5-OTteiGT_f-MBQ3hqN7P3i1P8e?usp=sharing>`_


For a non-interactive static version of this tutorial, see
`sKCSD Tutorial <https://github.com/Neuroinflab/kCSD-python/blob/master/tutorials/skcsd_tutorial.ipynb>`_


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
