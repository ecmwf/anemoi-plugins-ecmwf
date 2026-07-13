# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.inference.context import Context
from anemoi.inference.inputs.fdb import FDBInput
from anemoi.inference.metadata import Metadata

from .pre_processor import FDBPlusPreProcessor

LOG = logging.getLogger(__name__)


class FDBPlusInput(FDBInput):
    """Enhanced FDB input that handles optimised vorticity/divergence to u/v conversion and regridding of other fields."""

    trace_name = "fdb+"

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        grid: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the FDBPlusInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        metadata : Metadata
            The metadata associated with the input.
        grid : str | None, optional
            The target grid for the input. If not specified, the grid from the metadata will be used.
        """
        super().__init__(context, metadata=metadata, **kwargs)
        target_grid = grid or metadata.grid
        if target_grid is None:
            raise ValueError("`grid` must be specified in metadata or init for FDBPlusInput")
        self.pre_processors.insert(
            0,
            FDBPlusPreProcessor(
                context=context,
                metadata=metadata,
                target_grid=target_grid,
            ),
        )
