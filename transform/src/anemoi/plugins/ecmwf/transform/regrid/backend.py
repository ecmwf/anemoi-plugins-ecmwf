# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import logging

import earthkit.data as ekd

LOG = logging.getLogger(__name__)


GridSpec = str | list[float] | tuple[float, ...] | dict[str, list[float]]


def _make_mir_grid(grid: GridSpec):
    """Build a ``mir.Grid`` from a grid specification."""
    import mir

    if isinstance(grid, str):
        return mir.Grid(grid=grid.upper())
    elif isinstance(grid, (list, tuple)):
        return mir.Grid(grid=list(grid))
    elif isinstance(grid, dict):
        return mir.Grid(**grid)
    raise ValueError(f"Unsupported grid specification: {grid}")


def grid_repr(grid: GridSpec) -> str:
    """Return a human-readable representation of the grid specification."""
    if isinstance(grid, str):
        return grid.upper()
    if isinstance(grid, (list, tuple)):
        return str(list(grid)[:5] + ["..."] if len(grid) > 5 else grid)
    if isinstance(grid, dict):
        return str({k: f"list of len {len(v)}" for k, v in grid.items()})
    return repr(grid)


def mir_regrid(
    fields: ekd.FieldList,
    grid: GridSpec,
    area: str | list[float] | None = None,
    packing: str = "ccsds",
    accuracy: int = 16,
) -> ekd.FieldList:
    """Regrid fields to a target grid using MIR.

    Each field's values are passed to MIR through its array interface and the
    result is written back as a GRIB2 message via ``mir.PyGribOutput``.

    For unstructured lat/lon target grids, MIR leaves ``uuidOfHGrid`` all-zero
    on the output, so we stamp the real grid UID back on (see below).

    Parameters
    ----------
    fields : ekd.FieldList
        The input fields to regrid.
    grid : GridSpec
        The target grid specification (grid string, list/tuple of increments,
        or dict of coordinate lists).
    area : str or list of float or None, optional
        The target area specification.
    packing : str, optional
        GRIB packing type of the output.
    accuracy : int, optional
        GRIB bits per value of the output.

    Returns
    -------
    ekd.FieldList
        The regridded fields.
    """
    if len(fields) == 0:
        return fields

    LOG.info(
        f"Starting MIR regridding of {len(fields)} fields to grid: {grid_repr(grid)!r}, "
        f"area: {area!r}, packing: {packing!r}, accuracy: {accuracy!r}."
    )

    import mir

    mir_grid = _make_mir_grid(grid)

    job_args = {"grid": mir_grid.spec, "edition": 2, "packing": packing, "accuracy": accuracy}
    if area:
        job_args["area"] = area
    job = mir.Job(**job_args)

    out_fields = []
    for field in fields:
        input_buffer = io.BytesIO()
        field.to_target("file", input_buffer)
        input_buffer.seek(0)

        output_buffer = io.BytesIO()
        job.execute(mir.PyGribInput(input_buffer), mir.PyGribOutput(output_buffer))
        regridded = ekd.from_source("memory", output_buffer.getvalue())[0]

        input_buffer.close()
        output_buffer.close()

        out_fields.append(regridded)

    return ekd.FieldList.from_fields(out_fields)
