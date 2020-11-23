from numpy import arange, nan, zeros

from py_eddy_tracker.generic import simplify


def test_simplify():
    x = arange(10, dtype="f4")
    y = zeros(10, dtype="f4")
    # Will jump one value on two
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 5
    x_, y_ = simplify(x, y, precision=0.99)
    assert x_.shape[0] == 10
    # check nan management
    x[4] = nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 6
    x[3] = nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 6
    x[:4] = nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 3
    x[:] = nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 0
