from resmda import reservoir_simulator


def test_all_dir():
    assert set(reservoir_simulator.__all__) == set(dir(reservoir_simulator))
