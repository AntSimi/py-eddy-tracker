Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en>`_
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------
Changed
^^^^^^^

- Now time allows second precision (instead of daily precision) in storage on uint32 from 01/01/1950 to 01/01/2086
  New identifications are produced with this type, old files could still be loaded.
  If you use old identifications for tracking use the `--unraw` option to unpack old times and store data with the new format.
- Now amplitude is stored with .1 mm of precision (instead of 1 mm), same advice as for time.

Fixed
^^^^^

- Fix bug in convolution(filter), lowest rows was replace by zeros in convolution computation.
  Important impact for tiny kernel

Added
^^^^^

[3.5.0] - 2021-06-22
--------------------

Fixed
^^^^^
- GridCollection get_next_time_step & get_previous_time_step needed more files to work in the dataset list.
  The loop needed explicitly self.dataset[i+-1] even when i==0, therefore indice went out of range

[3.4.0] - 2021-03-29
--------------------
Changed
^^^^^^^
- `TrackEddiesObservations.filled_by_interpolation` method stop to normalize longitude, to continue to have same
  beahviour you must call before `TrackEddiesObservations.normalize_longitude`

Fixed
^^^^^
- Use `safe_load` for yaml load
- repr of EddiesObservation when the collection is empty (time attribute empty array)
- display_timeline and event_timeline can now use colors according to 'y' values.
- event_timeline now plot all merging event in one plot, instead of one plot per merging. Same for splitting. (avoid bad legend)

Added
^^^^^
- Identification file could be load in memory before to be read with netcdf library to get speed up in case of slow disk
- Add a filter option in EddyId to be able to remove fine scale (like noise) with same filter order than high scale
  filter
- Add **EddyQuickCompare** to have few figures about several datasets in comparison based on match function
- Color and text field for contour in **EddyAnim** could be choose
- Save EddyAnim in mp4
- Add method to get eddy contour which enclosed obs defined with (x,y) coordinates
- Add **EddyNetworkSubSetter** to subset network which need special tool and operation after subset
- Network:
    - Add method to find relatives segments
    - Add method to get cloase network in an other atlas
- Management of time cube data for advection

[3.3.0] - 2020-12-03
--------------------
Added
^^^^^
- Add an implementation of visvalingam algorithm to simplify polygons with low modification
- Add method to found close tracks in an other atlas
- Allow to give a x reference when we display grid to be able to change xlim
- Add option to EddyId to select data index like `--indexs time=5 depth=2`
- Add a method to merge several indexs type for eddy obs
- Get dataset variable like attribute, and lifetime/age are available for all observations
- Add **EddyInfos** application to get general information about eddies dataset
- Add method to inspect contour rejection (which are not in eddies)
- Grid interp could be "nearest" or "bilinear"

Changed
^^^^^^^
- Now to have object informations in plot label used python ```format``` style, several key are available :

    - "t0"
    - "t1"
    - "nb_obs"
    - "nb_tracks" (only for tracked eddies)

[3.2.0] - 2020-09-16
--------------------

[3.1.0] - 2020-06-25
--------------------
