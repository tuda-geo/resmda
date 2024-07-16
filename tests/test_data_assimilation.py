from resmda import data_assimilation


def test_all_dir():
    assert set(data_assimilation.__all__) == set(dir(data_assimilation))
