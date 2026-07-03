# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.inference.processor import Processor
from anemoi.inference.types import State


class ComputeTvPlugin(Processor):
    """Replace temperature ``T`` with virtual temperature ``Tv`` on model levels.

    For every temperature field named ``t_<level>`` for which a matching specific
    humidity field ``q_<level>`` exists, the temperature is replaced in place by the
    virtual temperature::

        Tv = T * (1 + 0.61 * q)

    Example
    -------
    ```yaml
    post_processors:
        - compute_tv
    ```
    """

    def process(self, state: State) -> State:
        """Replace T with Tv = T*(1 + 0.61*q) for each model level.

        Parameters
        ----------
        state : State
            The state dictionary.

        Returns
        -------
        State
            The state dictionary with temperature fields replaced by virtual
            temperature where a matching humidity field is available.
        """
        state = state.copy()
        fields = state["fields"].copy()
        for key in list(fields):
            if key.startswith("t_"):
                lev = key[2:]
                q_key = f"q_{lev}"
                if q_key in fields:
                    fields[key] = fields[key] * (1.0 + 0.61 * fields[q_key])
        state["fields"] = fields
        return state
