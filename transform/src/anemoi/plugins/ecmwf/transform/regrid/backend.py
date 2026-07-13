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
import numpy as np

LOG = logging.getLogger(__name__)


GridSpec = str | list[float] | tuple[float, ...] | dict[str, list[float]]

# GRIB keys copied from each source field onto the target-grid template
# when using the (fast) array-based regridding path. Order matters

TEMPLATE_OVERRIDE_KEYS: tuple[str, ...] = (
    "dataDate",
    "dataTime",
    "stepType",
    "stepRange",
    "typeOfLevel",
    "level",
    "paramId",
)


def normalise_grid(grid: GridSpec) -> str:
    """Normalise the grid specification to a string format.

    Parameters
    ----------
    grid : GridSpec
        The grid specification, which can be a string, list, tuple, or dict.

    Returns
    -------
    str
        The normalised grid specification as a string.
    """
    if isinstance(grid, (list, tuple)):
        return "/".join(map(str, grid))
    elif isinstance(grid, (int, float)):
        return f"{grid}/{grid}"
    elif isinstance(grid, dict):
        return "/".join(f"{k}={v}" for k, v in grid.items())
    else:
        return str(grid).upper()


def _make_job(grid: str, area: str | list[float] | None, packing: str, accuracy: int):
    import mir

    job_args = {"grid": grid}
    if area:
        job_args["area"] = area
    return mir.Job(**job_args, edition=2, packing=packing, accuracy=accuracy, truncation="auto")


def _mir_regrid_grib(
    fields: ekd.FieldList,
    grid: str,
    area: str | list[float] | None,
    packing: str,
    accuracy: int,
) -> ekd.FieldList:
    """Regrid via a full GRIB round-trip (slow, but preserves all metadata)."""
    job = _make_job(grid, area, packing, accuracy)
    input_buffer = io.BytesIO()
    output_buffer = io.BytesIO()
    fields.to_target("file", input_buffer)

    input_buffer.seek(0)
    job.execute(input_buffer, output_buffer)
    return ekd.from_source("memory", output_buffer.getvalue())


def _resolve_input_gridspec(field) -> dict | None:
    """Determine a MIR-compatible gridspec for a field's values.

    Returns ``None`` if the field has no gridspec (e.g. spectral fields),
    in which case it cannot go through the array interface.

    The gridspec derived from GRIB metadata contains an ``area`` rounded to
    6 decimals; at high resolutions (e.g. O1280) this rounding makes MIR
    crop grid rows, so its point count no longer matches the values array
    (``RawInput: values size equals iterator count`` assertion). For global
    fields (ecCodes computed key ``global``) the area is redundant, so it
    is dropped and the grid treated as global.
    """
    gridspec = field.metadata().gridspec
    if gridspec is None:
        return None
    gridspec = dict(gridspec)
    if field.metadata("global", default=0) == 1:
        gridspec.pop("area", None)
    return gridspec


def _mir_regrid_array(
    fields: ekd.FieldList,
    grid: str,
    area: str | list[float] | None,
    packing: str,
    accuracy: int,
) -> ekd.FieldList:
    """Regrid via MIR's array interface (fast).

    Field values are passed to MIR as numpy arrays, skipping the GRIB
    encode/decode round-trip of the (large) input-resolution messages.
    Output metadata is built from a target-grid GRIB template (obtained by
    regridding the first field through the GRIB path), overriding the
    per-field identity keys (``TEMPLATE_OVERRIDE_KEYS``).
    """
    import mir

    from ..utils import fields_to_numpy_parallel

    # Partition: fields without a gridspec (e.g. spectral) cannot use the
    # array interface and go through the GRIB round-trip instead.
    gridspecs = [_resolve_input_gridspec(f) for f in fields]
    grib_only = [i for i, gs in enumerate(gridspecs) if gs is None]

    if len(grib_only) == len(fields):
        return _mir_regrid_grib(fields, grid, area, packing, accuracy)

    if grib_only:
        LOG.info(
            f"{len(grib_only)} of {len(fields)} fields have no gridspec "
            "(e.g. spectral); regridding them via the GRIB round-trip."
        )
        grib_results = _mir_regrid_grib(
            ekd.FieldList.from_fields([fields[i] for i in grib_only]),
            grid,
            area,
            packing,
            accuracy,
        )

    # Target-grid template metadata from the first array-able field (one small
    # GRIB round-trip).
    # NOTE: this must run before any mir.ArrayInput is created: current mir-python
    # segfaults if an ArrayInput is constructed before MIR has executed once.
    first = next(i for i, gs in enumerate(gridspecs) if gs is not None)
    template = _mir_regrid_grib(fields[first : first + 1], grid, area, packing, accuracy)[0]
    template_md = template.metadata()

    job = _make_job(grid, area, packing, accuracy)
    output = mir.ArrayOutput()

    grib_iter = iter(grib_only and grib_results)
    out_fields = []
    in_fields = fields_to_numpy_parallel(fields)  # (nfields, npoints)
    for i, (field, input_gridspec) in enumerate(zip(fields, gridspecs)):
        if input_gridspec is None:
            out_fields.append(next(grib_iter))
            continue

        values = np.ascontiguousarray(in_fields[i], dtype=np.float64)
        job.execute(mir.ArrayInput(values, input_gridspec), output)

        overrides = {k: field.metadata(k, default=None) for k in TEMPLATE_OVERRIDE_KEYS}
        overrides = {k: v for k, v in overrides.items() if v is not None}
        out_fields.append(ekd.ArrayField(output.values(), template_md.override(overrides)))

    return ekd.FieldList.from_fields(out_fields)


def mir_regrid(
    fields: ekd.FieldList,
    grid: GridSpec,
    area: str | list[float] | None = None,
    packing: str = "ccsds",
    accuracy: int = 16,
    method: str = "grib",
) -> ekd.FieldList:
    """Regrid fields using the MIR library.

    Parameters
    ----------
    fields : ekd.FieldList
        The input fields to regrid.
    grid : GridSpec
        The target grid specification.
    area : str or list of float or None, optional
        The target area specification.
    packing : str, optional
        GRIB packing type of the output.
    accuracy : int, optional
        GRIB bits per value of the output.
    method : str, optional
        ``"grib"`` (default) round-trips everything through GRIB,
        preserving all metadata keys. ``"array"`` passes field values
        to MIR as numpy arrays, avoiding the GRIB round-trip of the
        input-resolution messages (faster when fields are already
        numpy-backed, e.g. ``ekd.ArrayField``); output metadata is
        rebuilt from a target-grid template.

    Returns
    -------
    ekd.FieldList
        The regridded fields.
    """
    if len(fields) == 0:
        return fields

    grid = normalise_grid(grid)

    LOG.info(
        f"Starting MIR regridding of {len(fields)} fields to grid: {grid!r}, area: {area!r}, "
        f"packing: {packing!r}, accuracy: {accuracy!r}, method: {method!r}."
    )

    if method == "array":
        return _mir_regrid_array(fields, grid, area, packing, accuracy)

    return _mir_regrid_grib(fields, grid, area, packing, accuracy)
