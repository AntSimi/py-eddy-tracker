========
Tracking
========

Requirements
************

Before tracking, you will need to run identification on every time step of the period (period of your study).

**Advice** : Before tracking, displaying some identification files. You will learn a lot

Default method
**************

To run a tracking just create an yaml file with minimal specification (*FILES_PATTERN* and *SAVE_DIR*).
You will run tracking separately between Cyclonic eddies and Anticyclonic eddies.

Example of conf.yaml

.. code-block:: yaml

    PATHS:
      # Files produces with EddyIdentification
      FILES_PATTERN: MY_IDENTIFICATION_PATH/Anticyclonic*.nc
      SAVE_DIR: MY_OUTPUT_PATH

    # Number of consecutive timesteps with missing detection allowed
    VIRTUAL_LENGTH_MAX: 3
    # Minimal number of timesteps to considered as a long trajectory
    TRACK_DURATION_MIN: 10

To run:

.. code-block:: bash

    EddyTracking conf.yaml -v DEBUG

It will use the default tracker:

- No travel longer than 125 km between two observations
- Amplitude and speed radius must be close to the previous observation
- In case of several candidates only the closest is kept


It will produce 4 files by run:

- A file of correspondences which will contain all the information to merge all identifications file
- A file which will contain all the observations which are alone
- A file which will contain all the short tracks which are shorter than **TRACK_DURATION_MIN**
- A file which will contain all the long tracks which are longer than **TRACK_DURATION_MIN**

Use Python module
*****************

An example of tracking with the Python module is available in the gallery:
:ref:`sphx_glr_python_module_08_tracking_manipulation_pet_run_a_tracking.py`

Choose a tracker
****************

With yaml you could also select another tracker:

.. code-block:: yaml

    PATHS:
      # Files produced with EddyIdentification
      FILES_PATTERN: MY/IDENTIFICATION_PATH/Anticyclonic*.nc
      SAVE_DIR: MY_OUTPUT_PATH

    # Number of consecutive timesteps with missing detection allowed
    VIRTUAL_LENGTH_MAX: 3
    # Minimal number of timesteps to considered as a long trajectory
    TRACK_DURATION_MIN: 10

    CLASS:
        # Give the module to import,
        # must be available when you do "import module" in python
        MODULE: py_eddy_tracker.featured_tracking.old_tracker_reference
        # Give class name which must be inherit from
        # py_eddy_tracker.observations.observation.EddiesObservations
        CLASS: CheltonTracker

This tracker is like the one described in CHELTON11[https://doi.org/10.1016/j.pocean.2011.01.002].
Code is here :meth:`py_eddy_tracker.featured_tracking.old_tracker_reference`
