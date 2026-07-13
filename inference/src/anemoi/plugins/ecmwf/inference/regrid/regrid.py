# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.processor import Processor
from anemoi.inference.types import State
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec

from .named import KNOWN_GRIDS
from .named import NamedRegrid

LOG = logging.getLogger(__name__)
CHECKPOINT_SENTINEL = "checkpoint"


def _open_coord_files(grid: dict[str, str]) -> dict[str, list[float]]:
    """Open the coordinate files for the specified grid."""
    coords = {}
    for coord_name, coord_path in grid.items():
        coords[coord_name] = np.load(coord_path).tolist()
    return coords


class RegridPreprocessor(Processor):
    """Regrid an input fieldlist.

    Can only be used when working with grib, from any of the earthkit data sources.
    i.e. mars, cds, opendata, grib files, etc.
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        grid: GridSpec | dict[str, str] | None = None,
        area: str | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Initialise the Regridding processor.

        Parameters
        ----------
        context : Context
            The context in which the processor operates.
        metadata : Metadata
            The metadata associated with the dataset this processor is handling.
        grid : str | list[float] | tuple[float, ...] | dict[str, list[float]] | dict[str, str]
            The target grid for regridding. Can be a grid string (e.g. "O32"),
            a list/tuple of increments, a named grid (e.g. "meps"),
            a dict of coordinate file paths, or a dict of coordinate lists.
        area : str | list[float] | tuple[float, ...] | None, optional
            The target area for regridding, by default None
        """
        super().__init__(context, metadata=metadata)

        if isinstance(grid, dict):
            values = list(grid.values())
            all_str = all(isinstance(v, str) for v in values)
            all_list = all(isinstance(v, list) for v in values)
            if all_str:
                resolved_grid = _open_coord_files(grid)  # type: ignore
            elif all_list:
                resolved_grid = grid  # type: ignore
            else:
                raise ValueError(
                    "Grid dict values must be all strings (file paths) or all lists (coordinates), "
                    f"got mixed types: {[type(v).__name__ for v in values]}"
                )

        elif isinstance(grid, str):
            if grid.lower() in KNOWN_GRIDS:
                named_regrid = NamedRegrid(grid)
                resolved_grid = named_regrid.gridspec["grid"]

            elif grid.lower().startswith(CHECKPOINT_SENTINEL):
                coord_path = grid.lstrip(f"{CHECKPOINT_SENTINEL}:")
                if (
                    f"{coord_path}/latitudes" not in self.metadata.supporting_arrays
                    or f"{coord_path}/longitudes" not in self.metadata.supporting_arrays
                ):
                    raise ValueError(
                        f"Checkpoint grid specified but metadata does not contain '{coord_path}/latitudes' and '{coord_path}/longitudes' supporting arrays. "
                        f"Available supporting arrays: {list(self.metadata.supporting_arrays.keys())}"
                    )
                ckpt_lat = self.metadata.supporting_arrays[f"{coord_path}/latitudes"].tolist()
                ckpt_lon = self.metadata.supporting_arrays[f"{coord_path}/longitudes"].tolist()
                resolved_grid = {"latitudes": ckpt_lat, "longitudes": ckpt_lon}
            else:
                resolved_grid = grid
        elif grid is None:
            resolved_grid = metadata.grid
        else:
            resolved_grid = grid

        self._regrid = MIRRegrid(grid=resolved_grid, area=area)

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
        state["fields"] = self._regrid.forward(state["fields"])
        state["latitudes"] = next(iter(state["fields"])).metadata().geography.latitudes()
        state["longitudes"] = next(iter(state["fields"])).metadata().geography.longitudes()
        return state

    def __repr__(self) -> str:
        return f"RegridPreprocessor({self._regrid!r})"
