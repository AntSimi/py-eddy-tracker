from numpy import array, pi
from pytest import approx

from py_eddy_tracker.poly import (
    convex,
    fit_circle,
    get_convex_hull,
    poly_area_vertice,
    visvalingam,
)

# Vertices for next test
V = array(((2, 2, 3, 3, 2), (-10, -9, -9, -10, -10)))
V_concave = array(((2, 2, 2.5, 3, 3, 2), (-10, -9, -9.5, -9, -10, -10)))


def test_poly_area():
    assert 1 == poly_area_vertice(V.T)


def test_fit_circle():
    x0, y0, r, err = fit_circle(*V)
    assert x0 == approx(2.5, rel=1e-10)
    assert y0 == approx(-9.5, rel=1e-10)
    assert r == approx(2 ** 0.5 / 2, rel=1e-10)
    assert err == approx((1 - 2 / pi) * 100, rel=1e-10)


def test_convex():
    assert convex(*V) is True
    assert convex(*V[::-1]) is True
    assert convex(*V_concave) is False
    assert convex(*V_concave[::-1]) is False


def test_convex_hull():
    assert convex(*get_convex_hull(*V_concave)) is True


def test_visvalingam():
    x = array([1, 2, 3, 4, 5, 6.75, 6, 1])
    y = array([-0.5, -1.5, -1, -1.75, -1, -1, -0.5, -0.5])
    x_, y_ = visvalingam(x, y, 6)
    assert ([1, 2, 3, 4, 6, 1] == x_).all()
    assert ([-0.5, -1.5, -1, -1.75, -0.5, -0.5] == y_).all()
