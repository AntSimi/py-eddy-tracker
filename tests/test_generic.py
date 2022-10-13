from numpy import arange, array, nan, ones, zeros

from py_eddy_tracker.generic import cumsum_by_track, simplify, wrap_longitude


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


def test_cumsum_by_track():
    a = ones(10, dtype="i4") * 2
    track = array([1, 1, 2, 2, 2, 2, 44, 44, 44, 48])
    assert (cumsum_by_track(a, track) == [2, 4, 2, 4, 6, 8, 2, 4, 6, 2]).all()


def test_wrapping():
    y = x = arange(-5,5, dtype='f4')
    x_, _ = wrap_longitude(x, y, ref=-10)
    assert (x_ == x).all()
    x_, _ = wrap_longitude(x, y, ref=1)
    assert x.size == x_.size
    assert (x_[6:] == x[6:]).all()
    assert (x_[:6] == x[:6] + 360).all()
    x_, _ = wrap_longitude(x, y, ref=1, cut=True)
    assert x.size + 3 == x_.size
    assert (x_[6 + 3:] == x[6:]).all()
    assert (x_[:7] == x[:7] + 360).all()

    # FIXME Need evolution in wrap_longitude
    # x %= 360
    # x_, _ = wrap_longitude(x, y, ref=-10, cut=True)
    # assert x.size == x_.size
