from datetime import datetime

from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import RegularGridDataset

g = RegularGridDataset(
    get_demo_path("dt_med_allsat_phy_l4_20160515_20190101.nc"), "longitude", "latitude"
)


def test_id():
    g.add_uv("adt")
    a, c = g.eddy_identification("adt", "u", "v", datetime(2019, 2, 23))
    assert len(a) == 36
    assert len(c) == 36
