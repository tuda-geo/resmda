# Copyright 2024 Dieter Werthmüller, Gabriel Serrao Seabra
#
# This file is part of resmda.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

from datetime import datetime

import numpy as np
from scooby import Report as ScoobyReport

try:
    from resmda.version import version as __version__
except ImportError:
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')


__all__ = ['Report', 'rng']


def __dir__():
    return __all__


def rng(random=None):
    """Module-wide Random Number Generator.

    Mainly meant for internal use.


    Parameters
    ----------
    random : {None, int,  np.random.Generator}, default: None
        - If ``None`` it returns a :func:`numpy.random.default_rng()` instance
          instantiated on a module level.
        - If ``int``, it returns a newly created
          :func:`numpy.random.default_rng()`` instance, instantiated with
          ``int`` as seed.
        - If it is already a :class:`numpy.random.Generator` instance, it
          simply returns it.


    Returns
    -------
    rng : random number generator
        A :class:`numpy.random.Generator` instance.

    """
    if isinstance(random, int):
        return np.random.default_rng(random)
    elif isinstance(random, np.random.Generator):
        return random
    else:
        if not hasattr(rng, '_rng'):
            rng._rng = np.random.default_rng()
        return rng._rng


class Report(ScoobyReport):
    """Print a Scooby report; see ``scooby.Report()`` for info."""

    def __init__(self, **kwargs):
        """Initiate a scooby.Report instance."""
        kwargs = {'ncol': 3, **kwargs}
        kwargs['core'] = ['numpy', 'scipy', 'numba', 'resmda']
        kwargs['optional'] = ['matplotlib', 'IPython']
        super().__init__(**kwargs)
