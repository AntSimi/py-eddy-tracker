========
Tracking
========

To run a tracking just create an yaml file with minimal specification (*FILES_PATTERN* and *SAVE_DIR*).

Example of yaml

.. code-block:: yaml

    PATHS:
      # Files produces with EddyIdentification
      FILES_PATTERN: MY/IDENTIFICATION_PATH/Anticyclonic*.nc
      SAVE_DIR: MY_OUTPUT_PATH

    # Number of timestep for missing detection
    VIRTUAL_LENGTH_MAX: 3
    # Minimal time to consider as a full track
    TRACK_DURATION_MIN: 10

To run:

.. code-block:: bash

    EddyTracking conf.yaml -v DEBUG


It will produce 4 files by run:

- A file of correspondances which will contains all the information to merge all identifications file
- A file which will contains all the observations which are alone
- A file which will contains all the short track which are shorter than TRACK_DURATION_MIN
- A file which will contains all the long track which are longer than TRACK_DURATION_MIN