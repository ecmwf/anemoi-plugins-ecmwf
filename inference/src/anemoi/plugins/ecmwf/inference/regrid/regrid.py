# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.geo as ekg
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.post_processors.earthkit_state import unwrap_state
from anemoi.inference.post_processors.earthkit_state import wrap_state
from anemoi.inference.processor import Processor
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


class RegridPreprocessor(Processor):
    """Regrid an input fieldlist.

    Can only be used when working with grib, from any of the earthkit data sources.
    i.e. mars, cds, opendata, grib files, etc.
    """

    def __init__(
        self, context: Context, metadata: Metadata, *, grid: str | list[float], area: str | list[float] | None = None
    ) -> None:
        """Initialise the Regridding processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        metadata : Metadata
            The metadata associated with the dataset this processor is handling.
        grid : str | list[float]
            The target grid for regridding.
        area : str | list[float] | None, optional
            The target area for regridding, by default None
        """
        super().__init__(context, metadata=metadata)
        self._grid = grid
        self._area = area

    def process(self, state: State) -> State:  # type: ignore
        """Process the fields by regridding them to the specified grid and area.

        Parameters
        ----------
        state : State
            The state containing the fields to process.

        Returns
        -------
        State
            The updated state with regridded fields.
        """
        state["fields"] = ekg.regrid(state["fields"], {"grid": self._grid, "area": self._area})
        return state


class RegridPostprocessor(Processor):
    """Regrid an output state"""

    def __init__(
        self, context: Context, metadata: Metadata, *, grid: str | list[float], area: str | list[float] | None = None
    ) -> None:
        """Initialise the Regridding processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        metadata : Metadata
            The metadata associated with the dataset this processor is handling.
        grid : str | list[float]
            The target grid for regridding.
        area : str | list[float] | None, optional
            The target area for regridding, by default None
        """
        super().__init__(context, metadata=metadata)
        self._grid = grid
        self._area = area

    def process(self, state: State) -> State:  # type: ignore
        """Process the fields by regridding them to the specified grid and area.

        Parameters
        ----------
        state : State
            The state containing the fields to process.

        Returns
        -------
        State
            The updated state with regridded fields.
        """
        fieldlist = wrap_state(state)
        fieldlist = ekg.regrid(fieldlist, {"grid": self._grid, "area": self._area})

        state = unwrap_state(fieldlist, state, namer=self.metadata.default_namer())
        return state
