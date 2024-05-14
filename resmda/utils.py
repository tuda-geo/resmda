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

# Instantiate a random number generator
RANDOM_NUMBER_GENERATOR = np.random.default_rng()


def rng(random=None):
    if isinstance(random, int):
        return np.random.default_rng(int)
    elif isinstance(random, np.random._generator.Generator):
        return random
    else:
        return RANDOM_NUMBER_GENERATOR
