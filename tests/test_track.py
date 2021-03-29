import zarr
from netCDF4 import Dataset

from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.tracking import Correspondances

filename = get_demo_path("Anticyclonic_20190223.nc")
a0 = EddiesObservations.load_file(filename)
a1 = a0.copy()


def test_area_tracking_parameter():
    delta = 0.2
    # All eddies will be shift of delta in longitude and latitude
    for k in (
        "lon",
        "lon_max",
        "contour_lon_s",
        "contour_lon_e",
        "lat",
        "lat_max",
        "contour_lat_s",
        "contour_lat_e",
    ):
        a1[k][:] -= delta
    a1.time[:] += 1
    # wrote in memory a0 and a1
    h0, h1 = zarr.group(), zarr.group()
    a0.to_zarr(h0), a1.to_zarr(h1)
    cmin = 0.5
    class_kw = dict(cmin=cmin)
    c = Correspondances(datasets=(h0, h1), class_method=AreaTracker, class_kw=class_kw)
    c.track()
    c.prepare_merging()
    # We have now an eddy object
    eddies_tracked = c.merge(raw_data=False)
    cost = eddies_tracked.cost_association
    m = cost < 1
    assert cost[m].max() <= (1 - cmin)

    # Try to save netcdf
    with Dataset("tata", mode="w", diskless=True) as h:
        c.to_netcdf(h)
        c_reloaded = Correspondances.from_netcdf(h)
        assert class_kw == c_reloaded.class_kw

    # test access to the lifetime (item)
    eddies_tracked["lifetime"]
