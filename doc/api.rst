API Reference
=============

This section provides comprehensive documentation for all public APIs.

Core Functions
--------------

.. autofunction:: kim_convergence.run_length_control

UCL Methods
-----------

.. automodule:: kim_convergence.ucl
   :members:

Statistical Functions
---------------------

.. automodule:: kim_convergence.stats
   :members:

Time Series Functions
---------------------

.. automodule:: kim_convergence.timeseries
   :members:

Utility Functions
-----------------

batch
~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.batch

outlier_test
~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.outlier_test

Scaler classes
~~~~~~~~~~~~~~~~

.. autoclass:: kim_convergence.MinMaxScale
   :members:

.. autoclass:: kim_convergence.TranslateScale
   :members:

.. autoclass:: kim_convergence.StandardScale
   :members:

.. autoclass:: kim_convergence.RobustScale
   :members:

.. autoclass:: kim_convergence.MaxAbsScale
   :members:

Convenience functions
~~~~~~~~~~~~~~~~~~~~~

minmax_scale
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.minmax_scale

translate_scale
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.translate_scale

standard_scale
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.standard_scale

robust_scale
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.robust_scale

maxabs_scale
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.maxabs_scale

validate_split
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.validate_split

train_test_split
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kim_convergence.train_test_split

Error Classes
-------------

CRError
~~~~~~~

.. autoexception:: kim_convergence.err.CRError

CRSampleSizeError
~~~~~~~~~~~~~~~~~

.. autoexception:: kim_convergence.err.CRSampleSizeError
