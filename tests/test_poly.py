from py_eddy_tracker.poly import poly_area_vertice, fit_circle, convex, get_convex_hull
from numpy import array, pi
from pytest import approx

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
