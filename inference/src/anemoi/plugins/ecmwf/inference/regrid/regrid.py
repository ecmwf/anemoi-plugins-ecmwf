# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import logging
from typing import TypeVar

import earthkit.data as ekd
import numpy as np
import tqdm.auto as tqdm
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.processor import Processor
from anemoi.inference.types import State

from .named import KNOWN_GRIDS
from .named import NamedRegrid

LOG = logging.getLogger(__name__)
CHECKPOINT_SENTINEL = "checkpoint"

GridSpec = str | list[float] | tuple[float, ...] | dict[str, list[float]]
Field = TypeVar("Field", ekd.Field, ekd.FieldList)


def _mir_regrid(fields: "ekd.Field", grid: GridSpec, area: str | list[float] | None) -> "ekd.FieldList":
    import io

    import mir

    job_args = {"grid": grid}
    if area:
        job_args["area"] = area
    job = mir.Job(**job_args, edition=2)  # type: ignore

    input_buffer = io.BytesIO()
    output_buffer = io.BytesIO()
    fields.to_target("file", input_buffer)

    job.execute(mir.GribMemoryInput(input_buffer.getvalue()), output_buffer)  # type: ignore
    return ekd.from_source("memory", output_buffer.getvalue())[0]  # type: ignore


def regrid(
    fields: Field,
    grid: GridSpec,
    area: str | list[float] | tuple[float, ...] | None,
    *,
    verbose: bool = True,
) -> Field:
    """Regrid a list of fields to a specified grid and area.

    TO BE REPLACED WITH EARTHKIT-REGRID
    """
    if isinstance(grid, (list, tuple)):
        grid = "/".join(map(str, grid))
    if isinstance(area, (list, tuple)):
        area = "/".join(map(str, area))

    if isinstance(grid, (int, float)):
        grid = f"{grid}/{grid}"

    single_field = False
    if isinstance(fields, ekd.Field):
        field_list = [fields]
        single_field = True
    else:
        field_list = list(fields)  # type: ignore[reportArgumentType]

    gridding_summary = f"grid: {str(grid)!r}, area: {str(area)!r}"
    LOG.info(f"Starting regridding of {len(field_list)} fields to {gridding_summary}.")

    results = list(
        tqdm.tqdm(
            map(functools.partial(_mir_regrid, grid=grid, area=area), field_list),
            total=len(field_list),
            desc="Regridding fields",
            disable=not verbose,
        )
    )
    result_fieldlist = ekd.FieldList.from_fields(results)

    return result_fieldlist[0] if single_field else result_fieldlist  # type: ignore[reportReturnType]


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
        grid: GridSpec | dict[str, str],
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
        else:
            resolved_grid = grid

        self._grid = resolved_grid
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
        state["fields"] = regrid(state["fields"], self._grid, self._area)
        return state
