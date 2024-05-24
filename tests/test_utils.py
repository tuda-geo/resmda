import numpy as np

from resmda import utils


def test_random():
    assert isinstance(utils.rng(), np.random.Generator)
    assert isinstance(utils.rng(11), np.random.Generator)
    rng = np.random.default_rng()
    assert rng == utils.rng(rng)
