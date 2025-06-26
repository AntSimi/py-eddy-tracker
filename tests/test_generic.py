import numpy as np

from py_eddy_tracker.generic import cumsum_by_track, simplify, wrap_longitude, bbox_indice_regular


def test_simplify():
    x = np.arange(10, dtype="f4")
    y = np.zeros(10, dtype="f4")
    # Will jump one value on two
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 5
    x_, y_ = simplify(x, y, precision=0.99)
    assert x_.shape[0] == 10
    # check nan management
    x[4] = np.nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 6
    x[3] = np.nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 6
    x[:4] = np.nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 3
    x[:] = np.nan
    x_, y_ = simplify(x, y, precision=1)
    assert x_.shape[0] == 0


def test_cumsum_by_track():
    a = np.ones(10, dtype="i4") * 2
    track = np.array([1, 1, 2, 2, 2, 2, 44, 44, 44, 48])
    assert (cumsum_by_track(a, track) == [2, 4, 2, 4, 6, 8, 2, 4, 6, 2]).all()


def test_wrapping():
    y = x = np.arange(-5, 5, dtype="f4")
    x_, _ = wrap_longitude(x, y, ref=-10)
    assert (x_ == x).all()
    x_, _ = wrap_longitude(x, y, ref=1)
    assert x.size == x_.size
    assert (x_[6:] == x[6:]).all()
    assert (x_[:6] == x[:6] + 360).all()
    x_, _ = wrap_longitude(x, y, ref=1, cut=True)
    assert x.size + 3 == x_.size
    assert (x_[6 + 3 :] == x[6:]).all()
    assert (x_[:7] == x[:7] + 360).all()

    # FIXME Need evolution in wrap_longitude
    # x %= 360
    # x_, _ = wrap_longitude(x, y, ref=-10, cut=True)
    # assert x.size == x_.size


def test_bbox_fit_contour():
    v= np.array([
        [200.6, 10.5],
        [203.4, 10.4],
        [204.4, 9.6],
        [198.4, 9.3],
        [200.6, 10.5],
    ])
    xc0, yc0 = v.min(axis=0)
    xc1, yc1 = v.max(axis=0)    
    for step in [1, .5, .2, .133333, .125,.1,.07, .025]:
        x_c = np.arange(0, 360, step)
        y_c = np.arange(-80, 80, step)
        xstep, ystep = x_c[1] - x_c[0], y_c[1] - y_c[0]
        x0, y0 = x_c - xstep / 2.0, y_c - ystep / 2.0
        nb_x = x_c.shape[0]
        (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(v, x0, y0, xstep, ystep, 0, True, nb_x)
        i_x = np.where((x_c >= xc0) * (x_c <= xc1))[0]
        i_y = np.where((y_c >= yc0) * (y_c <= yc1))[0]
        (x_start_raw, x_stop_raw), (y_start_raw, y_stop_raw) = (i_x.min(), i_x.max()), (i_y.min(), i_y.max())
        assert x_start == x_start_raw
        assert x_stop == x_stop_raw + 1
        assert y_start == y_start_raw
        assert y_stop == y_stop_raw + 1