import numpy as np
from pytest import approx

from py_eddy_tracker.poly import (
    convex,
    fit_circle,
    get_convex_hull,
    poly_area_vertice,
    visvalingam,
    get_pixel_in_regular,
)
from py_eddy_tracker.generic import bbox_indice_regular

# Vertices for next test
V = np.array(((2, 2, 3, 3, 2), (-10, -9, -9, -10, -10)))
V_concave = np.array(((2, 2, 2.5, 3, 3, 2), (-10, -9, -9.5, -9, -10, -10)))


def test_poly_area():
    assert 1 == poly_area_vertice(V.T)


def test_fit_circle():
    x0, y0, r, err = fit_circle(*V)
    assert x0 == approx(2.5, rel=1e-10)
    assert y0 == approx(-9.5, rel=1e-10)
    assert r == approx(2**0.5 / 2, rel=1e-10)
    assert err == approx((1 - 2 / np.pi) * 100, rel=1e-10)


def test_convex():
    assert convex(*V) is True
    assert convex(*V[::-1]) is True
    assert convex(*V_concave) is False
    assert convex(*V_concave[::-1]) is False


def test_convex_hull():
    assert convex(*get_convex_hull(*V_concave)) is True


def test_visvalingam():
    x = np.array([1, 2, 3, 4, 5, 6.75, 6, 1])
    y = np.array([-0.5, -1.5, -1, -1.75, -1, -1, -0.5, -0.5])
    x_target = [1, 2, 3, 4, 6, 1]
    y_target = [-0.5, -1.5, -1, -1.75, -0.5, -0.5]
    x_, y_ = visvalingam(x, y, 6)
    assert (x_target == x_).all()
    assert (y_target == y_).all()
    x_, y_ = visvalingam(x[:-1], y[:-1], 6)
    assert (x_target == x_).all()
    assert (y_target == y_).all()
    x_, y_ = visvalingam(np.roll(x, 2), np.roll(y, 2), 6)
    assert (x_target[:-1] == x_[1:]).all()
    assert (y_target[:-1] == y_[1:]).all()


def test_pixel_in_contour():
    xmin, xmax, ymin, ymax = -1, 31, -2, 11
    v = np.array([
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin],
            [xmin, ymin],
            [xmin, ymax],
        ])
    xstep = ystep = 7.5
    # Global grid
    for x0_ in [-10, 0]:
        x0, y0 = (x0_, x0_ + 360), (-10, 20)    
        x_c, y_c = np.arange(*x0, xstep), np.arange(*y0, ystep)
        (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(v, x0, y0, xstep, ystep, 1, True, int(360/5))
        i, j = get_pixel_in_regular(v, x_c, y_c, x_start, x_stop, y_start, y_stop)
        assert (x_c[i] < xmax).all(), f"{x0=}"
        assert (x_c[i] > xmin).all(), f"{x0=}"

    # Regional grid
    # contour over two bounds
    x0, y0 = (20, 370), (-10, 20)    
    x_c, y_c = np.arange(*x0, xstep), np.arange(*y0, ystep)
    (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(v, x0, y0, xstep, ystep, 1, False, int(360/5))
    i, j = get_pixel_in_regular(v, x_c, y_c, x_start, x_stop, y_start, y_stop)
    assert (x_c[i] < xmax).all(), f"{x0=}"
    assert (x_c[i] > xmin).all(), f"{x0=}"

    # first case grid fully cover contour, and not in second case
    for x0_ in [-2, -.5]:
        x0, y0 = (x0_, 100), (-10, 20)    
        x_c, y_c = np.arange(*x0, xstep), np.arange(*y0, ystep)
        (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(v, x0, y0, xstep, ystep, 1, False, int(360/5))
        i, j = get_pixel_in_regular(v, x_c, y_c, x_start, x_stop, y_start, y_stop)
        assert (x_c[i] < xmax).all(), f"{x0=}"
        assert (x_c[i] > xmin).all(), f"{x0=}"
        assert (y_c[j] < ymax).all(), f"{y0=}"
        assert (y_c[j] > ymin).all(), f"{y0=}"