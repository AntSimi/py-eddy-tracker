from matplotlib.path import Path
from numpy import arange, array, isnan, ma, nan, ones, zeros
from pytest import approx

from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import RegularGridDataset

G = RegularGridDataset(get_demo_path("mask_1_60.nc"), "lon", "lat")
X = 0.025
contour = Path(
    (
        (-X, 0),
        (X, 0),
        (X, X),
        (-X, X),
        (-X, 0),
    )
)


# contour
def test_contour_lon():
    assert (contour.lon == (-X, X, X, -X, -X)).all()


def test_contour_lat():
    assert (contour.lat == (0, 0, X, X, 0)).all()


def test_contour_mean():
    assert (contour.mean_coordinates == (0, X / 2)).all()


def test_contour_fit_circle():
    x, y, r, err = contour.fit_circle()
    assert x == approx(0)
    assert y == approx(X / 2)
    assert r == approx(3108, rel=1e-1)
    assert err == approx(49.1, rel=1e-1)


def test_pixels_in():
    i, j = contour.pixels_in(G)
    assert (i == (21599, 0, 1)).all()
    assert (j == (5401, 5401, 5401)).all()


def test_contour_grid_slice():
    assert contour.bbox_slice == ((21598, 4), (5400, 5404))


# grid
def test_bounds():
    x0, x1, y0, y1 = G.bounds
    assert x0 == -1 / 120.0 and x1 == 360 - 1 / 120
    assert y0 == approx(-90 - 1 / 120.0) and y1 == approx(90 - 1 / 120)


def test_interp():
    # Fake grid
    g = RegularGridDataset.with_array(
        coordinates=("x", "y"),
        datas=dict(
            z=ma.array(((0, 1), (2, 3)), dtype="f4"),
            x=array((0, 20)),
            y=array((0, 10)),
        ),
        centered=True,
    )
    x0, y0 = array((10,)), array((5,))
    x1, y1 = array((15,)), array((5,))
    # outside but usable with nearest
    x2, y2 = array((25,)), array((5,))
    # Outside for any interpolation
    x3, y3 = array((25,)), array((16,))
    x4, y4 = array((55,)), array((25,))
    # Interp nearest
    assert g.interp("z", x0, y0, method="nearest") == 0
    assert g.interp("z", x1, y1, method="nearest") == 2
    assert isnan(g.interp("z", x4, y4, method="nearest"))
    assert g.interp("z", x2, y2, method="nearest") == 2
    assert isnan(g.interp("z", x3, y3, method="nearest"))

    # Interp bilinear
    assert g.interp("z", x0, y0) == 1.5
    assert g.interp("z", x1, y1) == 2
    assert isnan(g.interp("z", x2, y2))


def test_convolution():
    """
    Add some dummy check on convolution filter
    """
    # Fake grid
    z = ma.array(
        arange(12).reshape((-1, 1)) * arange(10).reshape((1, -1)),
        mask=zeros((12, 10), dtype="bool"),
        dtype="f4",
    )
    g = RegularGridDataset.with_array(
        coordinates=("x", "y"),
        datas=dict(
            z=z,
            x=arange(0, 6, 0.5),
            y=arange(0, 5, 0.5),
        ),
        centered=True,
    )

    def kernel_func(lat):
        return ones((3, 3))

    # After transpose we must get same result
    d = g.convolve_filter_with_dynamic_kernel("z", kernel_func)
    assert (d.T[:9, :9] == d[:9, :9]).all()
    # We mask one value and check convolution result
    z.mask[2, 2] = True
    d = g.convolve_filter_with_dynamic_kernel("z", kernel_func)
    assert d[1, 1] == z[:3, :3].sum() / 8
    # Add nan and check only nearest value is contaminate
    z[2, 2] = nan
    d = g.convolve_filter_with_dynamic_kernel("z", kernel_func)
    assert not isnan(d[0, 0])
    assert isnan(d[1:4, 1:4]).all()
