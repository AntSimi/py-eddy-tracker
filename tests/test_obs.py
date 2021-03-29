import zarr

from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.observations.observation import EddiesObservations

a_filename, c_filename = (
    get_demo_path("Anticyclonic_20190223.nc"),
    get_demo_path("Cyclonic_20190223.nc"),
)
a = EddiesObservations.load_file(a_filename)
a_raw = EddiesObservations.load_file(a_filename, raw_data=True)
memory_store = zarr.group()
# Dataset was raw loaded from netcdf and save in zarr
a_raw.to_zarr(memory_store, chunck_size=100000)
# We load zarr data without raw option
a_zarr = EddiesObservations.load_from_zarr(memory_store)
c = EddiesObservations.load_file(c_filename)


def test_merge():
    new = a.merge(c)
    assert len(new) == len(a) + len(c)


def test_zarr_raw():
    assert a == a_zarr


def test_index():
    a_nc_subset = EddiesObservations.load_file(
        a_filename, indexs=dict(obs=slice(500, 1000))
    )
    a_zarr_subset = EddiesObservations.load_from_zarr(
        memory_store, indexs=dict(obs=slice(500, 1000)), buffer_size=50
    )
    assert a_nc_subset == a_zarr_subset
