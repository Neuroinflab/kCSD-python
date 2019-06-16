Documentation
------------
.. toctree::
   :maxdepth: 1
   :caption: Contents:

KCSD Methods
~~~~~~~~~~~~

.. automodule:: kcsd.KCSD
   :members:
   :inherited-members:

.. autoclass:: kcsd.sKCSD
   :members:
   :inherited-members:      
      
Spectral Structure
~~~~~~~~~~~~~~~~~~

.. autoclass:: kcsd.validation.ValidateKCSD.SpectralStructure
   :members:
   :inherited-members:            
      
Validation
~~~~~~~~~~

.. autoclass:: kcsd.validation.ValidateKCSD.ValidateKCSD
   :members:
   :inherited-members:            
   
.. autoclass:: kcsd.validation.ValidateKCSD.ValidateKCSD1D
   :members:
   :inherited-members:      
		
.. autoclass:: kcsd.validation.ValidateKCSD.ValidateKCSD2D
   :members:
   :inherited-members:
      
.. autoclass:: kcsd.validation.ValidateKCSD.ValidateKCSD3D
   :members:
   :inherited-members:
      
Reliability Maps
~~~~~~~~~~~~~~~~

.. automodule:: kcsd.validation.VisibilityMap
   :members:
   :inherited-members:
      
Basis functions
~~~~~~~~~~~~~~~

1-Dimensional
+++++++++++++

.. autofunction:: kcsd.basis_functions.gauss_1D

.. autofunction:: kcsd.basis_functions.gauss_lim_1D

.. autofunction:: kcsd.basis_functions.step_1D		  

2-Dimensional
+++++++++++++

.. autofunction:: kcsd.basis_functions.gauss_2D

.. autofunction:: kcsd.basis_functions.gauss_lim_2D
		  
.. autofunction:: kcsd.basis_functions.step_2D

3-Dimensional
+++++++++++++
		  
.. autofunction:: kcsd.basis_functions.gauss_3D

.. autofunction:: kcsd.basis_functions.gauss_lim_3D

.. autofunction:: kcsd.basis_functions.step_3D		  


CSD Test Sources
~~~~~~~~~~~~~~~~

Variable (Seedable)
+++++++++++++++++++

.. autofunction:: kcsd.validation.csd_profile.gauss_1d_dipole

.. autofunction:: kcsd.validation.csd_profile.gauss_1d_mono

.. autofunction:: kcsd.validation.csd_profile.gauss_2d_small

.. autofunction:: kcsd.validation.csd_profile.gauss_2d_large

.. autofunction:: kcsd.validation.csd_profile.gauss_3d_small

.. autofunction:: kcsd.validation.csd_profile.gauss_3d_large

Fixed (Non seedable)
++++++++++++++++++++
		  
.. autofunction:: kcsd.validation.csd_profile.gauss_1d_dipole_f

.. autofunction:: kcsd.validation.csd_profile.gauss_2d_large_f

.. autofunction:: kcsd.validation.csd_profile.gauss_2d_small_f

.. autofunction:: kcsd.validation.csd_profile.gauss_3d_mono1_f

.. autofunction:: kcsd.validation.csd_profile.gauss_3d_mono2_f

.. autofunction:: kcsd.validation.csd_profile.gauss_3d_mono3_f		  
