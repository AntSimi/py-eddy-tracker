=====================
How do I get set up ?
=====================

Source are available on github https://github.com/AntSimi/py-eddy-tracker

Use python3.
To avoid problems with installation, use of the virtualenv Python virtual environment is recommended or conda.

Then use pip to install all dependencies (numpy, scipy, matplotlib, netCDF4, ...), e.g.:

.. code-block:: bash

    pip install numpy scipy netCDF4 matplotlib opencv-python pyyaml pint polygon3


Then run the following to install the eddy tracker:

.. code-block:: bash

    python setup.py install

Several executables are available in your PATH:

.. code-block:: bash

    GridFiltering # Allow to apply a high frequency filter on a NetCDF grid
    EddyId # Provide identification of eddies for one grid
    EddySubSetter # Allow to apply sub setting on eddies dataset
    EddyTracking # Allow to track Identification dataset