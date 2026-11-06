# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MIRRegrid as a pre-processor to a Lambert grid, then the mir template for it.

Run by ``test_preprocessor_lambert.py`` in a fresh interpreter with ``ECCODES_ECKIT_GEO=0``
(ecCodes reads it when it initialises), with one JSON argument:
``grid`` (the Lambert grid, a dict of GRIB keys), ``source`` (``O<N>`` or ``O<N>-grib1``, the ecCodes sample
``reduced_gg_pl_<N>_grib<edition>``), ``input_shape`` (``shapeOfTheEarth`` of a GRIB2 source or None),
``quantity`` (the values of the source: ``constant``, ``latitude`` or ``longitude`` of its points, longitudes
in [-180, 180)) and ``output`` (the ``.npz`` path the arrays are saved to). Prints a JSON report.

The steps are those of an anemoi-inference run:

1. ``MIRRegrid.forward`` regrids an O32 field (``ForwardTransformFilter`` puts the result in ``state["fields"]``).
   The source has two fields, which are regridded by filters given the grid with its keys in another order.
2. ``EkdInput.set_private_attributes`` records the grid and area of the regridded fields in ``state["_geography"]``.
3. The real ``TemplateManager`` (built with a stub owner that only has the checkpoint metadata and the typed
   variables) makes the template lookup in ``template()`` / ``load_template()`` and asks the ``mir`` provider for
   the template of the output state, whose points are those of the regridded field. The checkpoint area is a
   regional one, which the manager puts in the lookup as the regridded fields report no area.
"""

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import earthkit.data as ekd
import eccodes
import numpy as np
from anemoi.inference.grib.templates.manager import TemplateManager
from anemoi.inference.inputs.ekd import EkdInput
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid

#: The grid and the (regional, non-empty) area of the stub checkpoint: ``TemplateManager.load_template`` falls back to
#: the area when the regridded fields report none, and the grid is not the Lambert grid.
CHECKPOINT_GRID = "O96"
CHECKPOINT_AREA = [70.0, -20.0, 30.0, 40.0]

#: Grid keys read from the regridded field and from the template.
KEYS = (
    "gridType",
    "Nx",
    "Ny",
    "latitudeOfFirstGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "LaDInDegrees",
    "LoVInDegrees",
    "Latin1InDegrees",
    "Latin2InDegrees",
    "DxInMetres",
    "DyInMetres",
    "scanningMode",
    "shapeOfTheEarth",
    "scaleFactorOfRadiusOfSphericalEarth",
    "scaledValueOfRadiusOfSphericalEarth",
    "projectionCentreFlag",
    "resolutionAndComponentFlags",
    "latitudeOfSouthernPoleInDegrees",
    "longitudeOfSouthernPoleInDegrees",
    "numberOfDataPoints",
    "jScansPositively",
    "edition",
)


def source_fields(source: str, shape_of_the_earth: int | None, quantity: str) -> ekd.FieldList:
    """Two fields, constant or with the latitude or longitude of the points of the source as values."""
    grib1 = source.endswith("-grib1")
    handle = eccodes.codes_grib_new_from_samples(f"reduced_gg_pl_{source.split('-')[0][1:]}_grib{1 if grib1 else 2}")
    try:
        if shape_of_the_earth is not None and not grib1:
            eccodes.codes_set(handle, "shapeOfTheEarth", shape_of_the_earth)
        latitudes = eccodes.codes_get_array(handle, "latitudes")
        longitudes = eccodes.codes_get_array(handle, "longitudes")
        values = {
            "constant": np.full(latitudes.size, 300.0),
            "latitude": latitudes,
            "longitude": (longitudes + 180.0) % 360.0 - 180.0,
        }[quantity]
        if grib1:
            # GRIB1 has no ieee packing: 24 bits are about a metre for a coordinate
            eccodes.codes_set(handle, "bitsPerValue", 24)
        else:
            eccodes.codes_set(handle, "packingType", "grid_ieee")
        eccodes.codes_set_values(handle, values)
        message = eccodes.codes_get_message(handle)
        return ekd.from_source("memory", message + message)
    finally:
        eccodes.codes_release(handle)


def grid_keys(field: Any) -> dict[str, Any]:
    return {key: field.metadata(key, default=None) for key in KEYS}


def main(arguments: dict[str, Any]) -> dict[str, Any]:
    grid = arguments["grid"]
    source = source_fields(arguments["source"], arguments["input_shape"], arguments["quantity"])
    latlon = source[0].to_latlon(flatten=True)
    state: dict[str, Any] = {"latitudes": latlon["lat"], "longitudes": latlon["lon"]}

    # 1. the pre-processor: the same grid for the two fields, the keys in another order for the second
    reordered = dict(reversed(list(grid.items())))
    out = [
        MIRRegrid(grid=g, packing="ieee", accuracy=64).forward(source[i : i + 1])[0]
        for i, g in enumerate((grid, reordered))
    ]
    fields = ekd.FieldList.from_fields(out)

    # 2. anemoi-inference records the grid
    EkdInput.set_private_attributes(MagicMock(), state, fields)  # type: ignore[arg-type]
    geography = state["_geography"]

    # 3. the real TemplateManager (inference 0.14.0) makes the lookup in ``load_template`` and asks the ``mir``
    # provider. Only what its ``__init__`` and ``load_template`` read of the owner is stubbed: the checkpoint
    # metadata (a grid, and a regional area that is the fallback when the state has no area) and the typed variables.
    typed = SimpleNamespace(time_processing=None, grib_keys={"levtype": "sfc"})
    metadata = SimpleNamespace(
        typed_variables={"2t": typed},
        grid=CHECKPOINT_GRID,
        area=CHECKPOINT_AREA,
        number_of_grid_points=1234,
    )
    owner = SimpleNamespace(metadata=metadata, template_lookup=lambda name: {})
    manager = TemplateManager(owner, templates=["mir"])  # type: ignore[arg-type]
    provider = manager.templates_providers[0]
    assert isinstance(provider, MirTemplatesProvider)
    lookups: list[dict[str, Any]] = []
    provider_template = provider.template

    def recording_template(variable: str, lookup: dict[str, Any], **kwargs: Any) -> Any:
        lookups.append(lookup)
        return provider_template(variable, lookup, **kwargs)

    provider.template = recording_template  # type: ignore[method-assign]

    field_lat, field_lon = fields[0].grid_points()
    output_state = {"latitudes": field_lat, "longitudes": field_lon, "_geography": geography}
    template = manager.template("2t", output_state, metadata.typed_variables)
    lookup = lookups[0]
    template_lat, template_lon = template.grid_points()

    np.savez(
        arguments["output"],
        field_latitudes=field_lat,
        field_longitudes=field_lon,
        template_latitudes=template_lat,
        template_longitudes=template_lon,
        field_values=fields[0].values,
    )
    return {
        "geography": geography,
        "lookup": {key: lookup[key] for key in ("levtype", "grid", "area")},
        "field": grid_keys(fields[0]),
        "second_field": grid_keys(fields[1]),
        "template": grid_keys(template),
        "template_mars_grid": template.metadata().geography.mars_grid(),
        "nfields": len(fields),
    }


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
