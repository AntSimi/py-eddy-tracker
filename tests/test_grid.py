from matplotlib.path import Path
from pytest import approx

from py_eddy_tracker.data import get_path
from py_eddy_tracker.dataset.grid import RegularGridDataset

G = RegularGridDataset(get_path("mask_1_60.nc"), "lon", "lat")
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
