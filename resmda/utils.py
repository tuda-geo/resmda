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

import numpy as np
from scooby import Report as ScoobyReport

__all__ = ['Report', ]


def __dir__():
    return __all__


# Instantiate a random number generator
RANDOM_NUMBER_GENERATOR = np.random.default_rng()


def rng(random=None):
    if isinstance(random, int):
        return np.random.default_rng(int)
    elif isinstance(random, np.random._generator.Generator):
        return random
    else:
        return RANDOM_NUMBER_GENERATOR


class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``resmda``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``, and
    ``matplotlib``. It also shows MKL information, if available.

    All modules provided in ``add_pckg`` are also shown.


    Parameters
    ----------
    add_pckg : {package, str}, default: None
        Package or list of packages to add to output information (must be
        imported beforehand or provided as string).

    ncol : int, default: 3
        Number of package-columns in html table (no effect in text-version).

    text_width : int, default: 80
        The text width for non-HTML display modes

    sort : bool, default: False
        Sort the packages when the report is shown

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'resmda']

        # Optional packages.
        optional = ['matplotlib', 'IPython']

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)
