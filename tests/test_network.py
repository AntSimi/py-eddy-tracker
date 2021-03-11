from py_eddy_tracker.observations.network import Network


def test_group_translate():
    translate = Network.group_translator(5, ((0, 1), (0, 2), (1, 3)))
    assert (translate == [3, 3, 3, 3, 4]).all()

    translate = Network.group_translator(5, ((1, 3), (0, 1), (0, 2)))
    assert (translate == [3, 3, 3, 3, 4]).all()

    translate = Network.group_translator(8, ((1, 3), (2, 3), (2, 4), (5, 6), (4, 5)))
    assert (translate == [0, 6, 6, 6, 6, 6, 6, 7]).all()

    translate = Network.group_translator(6, ((0, 1), (0, 2), (1, 3), (4, 5)))
    assert (translate == [3, 3, 3, 3, 5, 5]).all()
