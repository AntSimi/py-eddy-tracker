# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add an implementation of visvalingam algorithm to simplify polygons with low modification
- Add method to found close tracks in an other atlas
- Allow to give a x reference when we display grid to be able to change xlim
- Add option to EddyId to select data index like `--indexs time=5 depth=2`
- Add a method to merge several indexs type for eddy obs
- Acces at dataset variable like attribute, and lifetime/age are available for all observations
- Add **EddyInfos** application to get general information about eddies dataset
- Add method to inspect contour rejection (which are not in eddies)
- Grid interp could be "nearest" or "bilinear" 
### Changed
- Now to have object informations in plot label used python ```fomat``` style, several key are available : 
    - "t0"
    - "t1"
    - "nb_obs"
    - "nb_tracks" (only for tracked eddies)
### Removed


## [3.2.0] - 2020-09-16

## [3.1.0] - 2020-06-25
