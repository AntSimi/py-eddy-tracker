from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.data import get_path
from netCDF4 import Dataset

a = EddiesObservations.load_file(get_path("Anticyclonic_20190223.nc"))
c = EddiesObservations.load_file(get_path("Cyclonic_20190223.nc"))


def test_merge():
    new = a.merge(c)
    assert len(new) == len(a) + len(c)


# def test_write():
#     with Dataset
