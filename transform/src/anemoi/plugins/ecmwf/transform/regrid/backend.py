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
from typing import Any

import earthkit.data as ekd
import numpy as np
import pyproj
from earthkit.data.core.metadata import WrappedMetadata

LOG = logging.getLogger(__name__)


#: A grid specification: a grid name or string, increments, a dict of ``latitudes`` / ``longitudes`` lists,
#: or a dict of GRIB keys with scalar values (a Lambert grid, see :func:`canonical_lambert_grid`).
GridSpec = str | list[float] | tuple[float, ...] | dict[str, list[float] | float | int | str]


LAMBERT_GRID_TYPE = "lambert"

#: Keys of a Lambert grid, in the order of the canonical form (``gridType`` first).
_LAMBERT_KEYS = (
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
)

#: Keys that must be given.
_LAMBERT_REQUIRED = (
    "Nx",
    "Ny",
    "latitudeOfFirstGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "LaDInDegrees",
    "LoVInDegrees",
    "DxInMetres",
    "DyInMetres",
)

#: Keys with an integer value (the others are numbers in float).
_LAMBERT_INTEGER_KEYS = frozenset(
    {
        "Nx",
        "Ny",
        "scanningMode",
        "shapeOfTheEarth",
        "scaleFactorOfRadiusOfSphericalEarth",
        "scaledValueOfRadiusOfSphericalEarth",
        "projectionCentreFlag",
        "resolutionAndComponentFlags",
    }
)

#: GRIB keys that may be given (as in a GRIB message) but are not used: ``gridDefinitionTemplateNumber``
#: must be 30 (Lambert conformal), ``tablesVersion`` is ignored. Neither is part of the canonical form.
_LAMBERT_IGNORED_KEYS = ("gridDefinitionTemplateNumber", "tablesVersion")

#: ``shapeOfTheEarth`` a Lambert grid has if it does not give one (a sphere of radius 6371229 m).
DEFAULT_SHAPE_OF_THE_EARTH = 6

#: The ``shapeOfTheEarth`` that are spheres which need no further GRIB key, with their radius in metres.
#: ``shapeOfTheEarth=1`` (a sphere with the radius given by ``scaleFactorOfRadiusOfSphericalEarth`` and
#: ``scaledValueOfRadiusOfSphericalEarth``) is supported too. Ellipsoids are not.
SUPPORTED_SHAPES_OF_THE_EARTH: dict[int, float] = {0: 6367470.0, 6: 6371229.0, 8: 6371200.0}

#: Radius in metres of the sphere MIR 1.30.2 puts the points of a Lambert grid on, whatever the earth of the
#: input field (the header of the output only copies the earth of the input). The grid MIR is asked for is
#: compensated for it, see :func:`_mir_lambert_grid`.
MIR_LAMBERT_RADIUS = 6371229.0

#: ``scanningMode`` of a Lambert grid: first point south-west, rows running north (``jScansPositively=1``).
LAMBERT_SCANNING_MODE = 64


def is_lambert_grid(grid: Any) -> bool:
    """Whether ``grid`` is a Lambert grid: a dict of GRIB keys with ``gridType: lambert``.

    A dict of ``latitudes`` and ``longitudes`` lists is not.
    """
    return isinstance(grid, dict) and str(grid.get("gridType", "")).lower() == LAMBERT_GRID_TYPE


def _lambert_number(key: str, value: Any) -> int | float:
    """Return the value of a Lambert grid key as an int (for the integer keys) or a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Lambert grid: {key} must be a number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"Lambert grid: {key} must be finite, got {value!r}")
    if key in _LAMBERT_INTEGER_KEYS:
        if int(value) != value:
            raise ValueError(f"Lambert grid: {key} must be an integer, got {value!r}")
        return int(value)
    return float(value)


def canonical_lambert_grid(grid: dict[str, Any]) -> dict[str, int | float | str]:
    """Return the canonical form of a Lambert grid given as a dict of GRIB keys.

    The grid is a Lambert conformal conic grid on a sphere, tangent, scanning from the
    south-west corner (``scanningMode`` 64), and has keys as in a GRIB2 grid
    definition (template 3.30):

    - required: ``Nx``, ``Ny``, ``latitudeOfFirstGridPointInDegrees`` and
      ``longitudeOfFirstGridPointInDegrees`` (the south-west corner), ``LaDInDegrees``,
      ``LoVInDegrees``, ``DxInMetres``, ``DyInMetres``;
    - optional: ``Latin1InDegrees`` and ``Latin2InDegrees`` (default ``LaDInDegrees``, and
      must equal it), ``scanningMode`` (default and only value 64), ``shapeOfTheEarth`` (default 6;
      0, 1, 6 and 8 are supported, 1 needs ``scaleFactorOfRadiusOfSphericalEarth`` and
      ``scaledValueOfRadiusOfSphericalEarth``, the other shapes none), ``projectionCentreFlag``
      (default and only value 0), ``resolutionAndComponentFlags`` (default 48),
      ``latitudeOfSouthernPoleInDegrees`` and ``longitudeOfSouthernPoleInDegrees`` (default -90 and 0,
      the only values);
    - accepted and ignored: ``gridDefinitionTemplateNumber`` (must be 30) and ``tablesVersion``.

    The canonical form has every key but the radius keys of a shape other than 1, with its
    default if it was not given, in a fixed order, so equal grids have equal ``str``.
    Integer keys are ``int``, the others ``float``.

    Parameters
    ----------
    grid : dict[str, Any]
        A dict with ``gridType: lambert`` and the keys above.

    Returns
    -------
    dict[str, int | float | str]
        The canonical grid.

    Raises
    ------
    ValueError
        If ``grid`` is not a Lambert grid, has an unknown key, misses a required key, has a value
        that is not a number or out of range, is secant (``Latin1InDegrees`` or ``Latin2InDegrees`` other
        than ``LaDInDegrees``), or asks for a ``scanningMode``, ``shapeOfTheEarth``, ``projectionCentreFlag`` or
        southern pole that is not supported.
    """
    if not is_lambert_grid(grid):
        raise ValueError(f"Not a Lambert grid, expected a dict with gridType={LAMBERT_GRID_TYPE!r}: {grid!r}")

    known = (*_LAMBERT_KEYS, *_LAMBERT_IGNORED_KEYS)
    unknown = [key for key in grid if key not in known]
    if unknown:
        raise ValueError(f"Lambert grid: unknown key(s) {unknown}, expected keys: {list(_LAMBERT_KEYS[1:])}")
    missing = [key for key in _LAMBERT_REQUIRED if key not in grid]
    if missing:
        raise ValueError(f"Lambert grid: missing required key(s) {missing}")

    if "gridDefinitionTemplateNumber" in grid and grid["gridDefinitionTemplateNumber"] != 30:
        raise ValueError(
            f"Lambert grid: gridDefinitionTemplateNumber must be 30 (Lambert conformal), "
            f"got {grid['gridDefinitionTemplateNumber']!r}"
        )

    lad = _lambert_number("LaDInDegrees", grid["LaDInDegrees"])
    defaults: dict[str, int | float] = {
        "Latin1InDegrees": lad,
        "Latin2InDegrees": lad,
        "scanningMode": LAMBERT_SCANNING_MODE,
        "shapeOfTheEarth": DEFAULT_SHAPE_OF_THE_EARTH,
        "projectionCentreFlag": 0,
        "resolutionAndComponentFlags": 48,
        "latitudeOfSouthernPoleInDegrees": -90.0,
        "longitudeOfSouthernPoleInDegrees": 0.0,
    }
    values = {key: _lambert_number(key, grid[key]) if key in grid else defaults.get(key) for key in _LAMBERT_KEYS[1:]}

    if values["Nx"] < 1 or values["Ny"] < 1:
        raise ValueError(f"Lambert grid: Nx and Ny must be at least 1, got {values['Nx']} and {values['Ny']}")
    if values["DxInMetres"] <= 0 or values["DyInMetres"] <= 0:
        raise ValueError(
            f"Lambert grid: DxInMetres and DyInMetres must be positive, "
            f"got {values['DxInMetres']} and {values['DyInMetres']}"
        )
    if not 0 < abs(lad) < 90:
        raise ValueError(f"Lambert grid: LaDInDegrees must be between -90 and 90, and not 0, got {lad}")
    if not -90 <= values["latitudeOfFirstGridPointInDegrees"] <= 90:
        raise ValueError(
            f"Lambert grid: latitudeOfFirstGridPointInDegrees must be between -90 and 90, "
            f"got {values['latitudeOfFirstGridPointInDegrees']}"
        )

    parallels = (values["Latin1InDegrees"], values["Latin2InDegrees"])
    if any(parallel != lad for parallel in parallels):
        raise ValueError(
            f"Lambert grid: only tangent grids are supported, Latin1InDegrees and Latin2InDegrees "
            f"(default LaDInDegrees={lad:g}) must equal LaDInDegrees, got {parallels[0]:g} and {parallels[1]:g}"
        )
    if values["scanningMode"] != LAMBERT_SCANNING_MODE:
        raise ValueError(
            f"Lambert grid: only scanningMode={LAMBERT_SCANNING_MODE} (from the south-west corner, rows running north) "
            f"is supported, got {values['scanningMode']}"
        )
    if values["projectionCentreFlag"] != 0:
        raise ValueError(
            f"Lambert grid: only projectionCentreFlag=0 is supported, got {values['projectionCentreFlag']}"
        )
    if not 0 <= values["resolutionAndComponentFlags"] <= 255:
        raise ValueError(
            f"Lambert grid: resolutionAndComponentFlags must be between 0 and 255, "
            f"got {values['resolutionAndComponentFlags']}"
        )
    if (values["latitudeOfSouthernPoleInDegrees"], values["longitudeOfSouthernPoleInDegrees"]) != (-90.0, 0.0):
        raise ValueError(
            "Lambert grid: only the southern pole at (-90, 0) is supported (not rotated), got "
            f"({values['latitudeOfSouthernPoleInDegrees']}, {values['longitudeOfSouthernPoleInDegrees']})"
        )

    shape = values["shapeOfTheEarth"]
    radius_keys = ("scaleFactorOfRadiusOfSphericalEarth", "scaledValueOfRadiusOfSphericalEarth")
    if shape == 1:
        absent = [key for key in radius_keys if values[key] is None]
        if absent:
            raise ValueError(f"Lambert grid: shapeOfTheEarth=1 needs {absent}")
        if values["scaledValueOfRadiusOfSphericalEarth"] <= 0:
            raise ValueError("Lambert grid: scaledValueOfRadiusOfSphericalEarth must be positive")
    elif shape in SUPPORTED_SHAPES_OF_THE_EARTH:
        given = [key for key in radius_keys if key in grid]
        if given:
            raise ValueError(f"Lambert grid: {given} only go with shapeOfTheEarth=1, got shapeOfTheEarth={shape}")
    else:
        raise ValueError(
            f"Lambert grid: shapeOfTheEarth={shape} is not supported, expected one of "
            f"{[0, 1, 6, 8]} (spheres; 1 with scaleFactorOfRadiusOfSphericalEarth and "
            "scaledValueOfRadiusOfSphericalEarth, 0, 6 and 8 of radius "
            f"{list(SUPPORTED_SHAPES_OF_THE_EARTH.values())} m for 0, 6, 8)"
        )

    canonical: dict[str, int | float | str] = {"gridType": LAMBERT_GRID_TYPE}
    canonical.update({key: value for key, value in values.items() if value is not None})
    return canonical


def lambert_radius(grid: dict[str, Any]) -> float:
    """Return the radius in metres of the earth sphere of a Lambert grid (see :func:`canonical_lambert_grid`)."""
    keys = canonical_lambert_grid(grid)
    shape = keys["shapeOfTheEarth"]
    if shape == 1:
        return float(keys["scaledValueOfRadiusOfSphericalEarth"]) / 10.0 ** keys["scaleFactorOfRadiusOfSphericalEarth"]
    return SUPPORTED_SHAPES_OF_THE_EARTH[shape]


def lambert_projection(lad: float, lov: float, radius: float) -> pyproj.Proj:
    """Spherical, tangent Lambert conformal conic projection (pyproj): the standard parallels are ``lad``.

    Called as ``projection(longitude, latitude)`` it returns the ``x, y`` coordinates in metres, and with
    ``inverse=True`` as ``projection(x, y, inverse=True)`` the longitudes and latitudes in degrees. Both
    accept numpy arrays. The longitudes are in [-180, 180].
    """
    return pyproj.Proj(proj="lcc", lat_0=lad, lat_1=lad, lat_2=lad, lon_0=lov, R=radius)


def lambert_points(grid: dict[str, Any], radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes (degrees, flat) of the points of a Lambert grid, by spherical LCC maths (pyproj).

    The first point of the grid is the first point of the result (the south-west corner), the
    columns run east and the rows north, row by row, as in the output of :func:`mir_regrid`.

    Parameters
    ----------
    grid : dict[str, Any]
        A Lambert grid, see :func:`canonical_lambert_grid`.
    radius : float, optional
        Radius of the earth sphere in metres, by default that of the grid (see :func:`lambert_radius`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The latitudes and the longitudes, each of ``Nx * Ny`` points.
    """
    keys = canonical_lambert_grid(grid)
    projection = lambert_projection(
        keys["LaDInDegrees"], keys["LoVInDegrees"], lambert_radius(keys) if radius is None else radius
    )
    nx, ny = keys["Nx"], keys["Ny"]
    x0, y0 = projection(keys["longitudeOfFirstGridPointInDegrees"], keys["latitudeOfFirstGridPointInDegrees"])
    x = x0 + keys["DxInMetres"] * np.arange(nx)[None, :] + np.zeros((ny, 1))
    y = y0 + keys["DyInMetres"] * np.arange(ny)[:, None] + np.zeros((1, nx))
    longitudes, latitudes = projection(x, y, inverse=True)
    return latitudes.ravel(), longitudes.ravel()


def _mir_lambert_grid(grid: dict[str, Any], with_lad: bool) -> str:
    """Return the MIR typed-grid string that makes MIR interpolate onto the points of a Lambert grid.

    MIR 1.30.2 only builds a Lambert grid from a typed-grid string whose first point is the top-left corner,
    the rows running south, and it places the points on a sphere of radius ``MIR_LAMBERT_RADIUS``, whatever
    the earth of the grid or of the input field. Both are compensated for: the corner given is the
    north-west one (from the south-west corner, by spherical LCC maths at the radius of the grid) and the
    steps are those of the grid scaled by ``MIR_LAMBERT_RADIUS / radius``, so that the points MIR
    interpolates at are the points of the grid. Floats are given with their full ``repr`` precision (4 decimals
    would be 5 m off).

    Parameters
    ----------
    grid : dict[str, Any]
        A canonical Lambert grid.
    with_lad : bool
        Whether to add ``writeLaDInDegrees=1``, which GRIB2 needs (otherwise LaD is written as 0) and
        GRIB1 does not take.

    Returns
    -------
    str
        The typed-grid string for ``mir.Job``.
    """
    radius = lambert_radius(grid)
    projection = lambert_projection(grid["LaDInDegrees"], grid["LoVInDegrees"], radius)
    x0, y0 = projection(grid["longitudeOfFirstGridPointInDegrees"], grid["latitudeOfFirstGridPointInDegrees"])
    longitude, latitude = (float(v) for v in projection(x0, y0 + (grid["Ny"] - 1) * grid["DyInMetres"], inverse=True))

    scale = MIR_LAMBERT_RADIUS / radius
    lad = grid["LaDInDegrees"]
    items = [
        "gridType=lambert",
        f"LaDInDegrees={lad!r}",
        f"LoVInDegrees={grid['LoVInDegrees']!r}",
        f"Latin1InDegrees={lad!r}",
        f"Latin2InDegrees={lad!r}",
        f"Ni={grid['Nx']}",
        f"Nj={grid['Ny']}",
        f"grid={grid['DxInMetres'] * scale!r}/{grid['DyInMetres'] * scale!r}",
        f"latitudeOfFirstGridPointInDegrees={latitude!r}",
        f"longitudeOfFirstGridPointInDegrees={longitude!r}",
        "writeLonPositive=1",
    ]
    if with_lad:
        items.append("writeLaDInDegrees=1")
    return ";".join(items)


def _lambert_input(message: bytes) -> tuple[bytes, int]:
    """Return the GRIB ``message`` to hand to MIR for a Lambert grid, and its edition.

    MIR gives its output the earth of the input. A GRIB2 message gets ``shapeOfTheEarth=6``
    (R=6371229 m, the sphere MIR places the points on, see ``MIR_LAMBERT_RADIUS``) on a copy, so
    the compensation of :func:`_mir_lambert_grid` stays right if MIR one day honours the earth of its input.
    A GRIB1 message is returned as it is. ``message`` is not modified.
    """
    handle = ekd.from_source("memory", message)[0].handle
    edition = handle.get("edition")
    if edition != 2 or handle.get("shapeOfTheEarth", default=None) is None:
        return message, edition
    handle.set("shapeOfTheEarth", 6)
    return handle.get_buffer(), edition


def _lambert_output(message: bytes, grid: dict[str, Any]) -> bytes:
    """Return a copy of a Lambert GRIB2 ``message`` written by MIR, with the header of ``grid``.

    MIR writes the grid it was given by :func:`_mir_lambert_grid`: first point top-left, rows running south
    (``jScansPositively=0``), on the sphere of ``MIR_LAMBERT_RADIUS``. ecCodes (``ECCODES_ECKIT_GEO=0``)
    places the points of such a grid north of the first point, so its coordinates do not match the
    values. The message is rewritten: the rows of the values are reversed, and the keys of ``grid`` are set,
    so the first point is the south-west corner as given, ``scanningMode`` is 64 and the earth is the
    sphere of the grid. The points ecCodes decodes are then those of the grid.

    A bitmap (missing values) is reversed with the values, and the packing type is kept.

    Parameters
    ----------
    message : bytes
        The GRIB2 message MIR wrote for the Lambert grid.
    grid : dict[str, Any]
        The canonical Lambert grid it was made for.

    Returns
    -------
    bytes
        The rewritten GRIB2 message.

    Raises
    ------
    ValueError
        If the message is scanned with ``alternativeRowScanning``, ``jPointsAreConsecutive`` or
        ``iScansNegatively``.
    RuntimeError
        If the message is not as MIR 1.30.2 writes it for this grid (``jScansPositively=1`` or another size):
        the compensation then has to be checked against that MIR.
    """
    import eccodes

    handle = eccodes.codes_new_from_message(message)
    try:
        if eccodes.codes_get(handle, "jScansPositively") == 1:
            raise RuntimeError("MIR wrote a Lambert grid with jScansPositively=1, which the plugin does not expect")
        for key in ("alternativeRowScanning", "jPointsAreConsecutive", "iScansNegatively"):
            if eccodes.codes_is_defined(handle, key) and eccodes.codes_get(handle, key):
                raise ValueError(f"Lambert grid with {key}=1 cannot be rewritten to south-to-north scanning")
        ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
        if (ni, nj) != (grid["Nx"], grid["Ny"]):
            raise RuntimeError(f"MIR wrote a {ni}x{nj} Lambert grid, expected {grid['Nx']}x{grid['Ny']}")

        # The values are read with ecCodes directly: with a bitmap, the missing points are the ``missingValue``
        # sentinel, which must be reversed with the other values and be written back as the sentinel (earthkit
        # turns it into NaN, which ecCodes would encode as a value). The bitmap itself is kept (or kept absent).
        has_bitmap = bool(eccodes.codes_get(handle, "bitmapPresent"))
        values = eccodes.codes_get_values(handle).reshape(nj, ni)[::-1].ravel()
        if has_bitmap:
            eccodes.codes_set(handle, "missingValue", eccodes.codes_get(handle, "missingValue"))
            eccodes.codes_set(handle, "bitmapPresent", 1)
        eccodes.codes_set_values(handle, values)

        eccodes.codes_set(handle, "shapeOfTheEarth", grid["shapeOfTheEarth"])
        if grid["shapeOfTheEarth"] == 1:
            eccodes.codes_set(
                handle, "scaleFactorOfRadiusOfSphericalEarth", grid["scaleFactorOfRadiusOfSphericalEarth"]
            )
            eccodes.codes_set(
                handle, "scaledValueOfRadiusOfSphericalEarth", grid["scaledValueOfRadiusOfSphericalEarth"]
            )
        eccodes.codes_set(handle, "latitudeOfFirstGridPointInDegrees", grid["latitudeOfFirstGridPointInDegrees"])
        eccodes.codes_set(
            handle, "longitudeOfFirstGridPointInDegrees", grid["longitudeOfFirstGridPointInDegrees"] % 360.0
        )
        for key in (
            "resolutionAndComponentFlags",
            "LaDInDegrees",
            "LoVInDegrees",
            "DxInMetres",
            "DyInMetres",
            "projectionCentreFlag",
            "scanningMode",
            "Latin1InDegrees",
            "Latin2InDegrees",
            "latitudeOfSouthernPoleInDegrees",
            "longitudeOfSouthernPoleInDegrees",
        ):
            eccodes.codes_set(handle, key, grid[key])
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


class _LambertGeography:
    """Geography of a field on a Lambert grid, with ``mars_grid`` and ``mars_area`` overridden.

    The geography ecCodes builds for the field is only read (lazily) for other
    attributes. A Lambert field has no ``mars_grid`` / ``mars_area`` of its own
    (``None`` or a partial area), so the Lambert grid and the requested area are reported.
    """

    def __init__(self, metadata: "_LambertMetadata") -> None:
        self._metadata = metadata

    def mars_grid(self) -> dict[str, int | float | str]:
        return dict(self._metadata.lambert_grid)

    def mars_area(self) -> list[float] | None:
        return self._metadata.lambert_area

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._metadata.metadata.geography, name)


class _LambertMetadata(WrappedMetadata):
    """Metadata of a field regridded to a Lambert grid, reporting the grid and area requested.

    Everything but ``geography.mars_grid()`` and ``geography.mars_area()`` is
    taken from the wrapped metadata.
    """

    def __init__(
        self,
        metadata: Any,
        lambert_grid: dict[str, int | float | str] | None = None,
        lambert_area: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(metadata, **kwargs)
        self.lambert_grid = lambert_grid or {}
        self.lambert_area = lambert_area

    @property
    def geography(self) -> _LambertGeography:
        return _LambertGeography(self)

    def _clone(self, metadata: Any) -> "_LambertMetadata":
        r = _LambertMetadata(metadata, self.lambert_grid, self.lambert_area)
        r._update(self)
        return r

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["lambert_grid"] = self.lambert_grid
        state["lambert_area"] = self.lambert_area
        return state

    def __setstate__(self, state: dict) -> None:
        self.lambert_grid = state.pop("lambert_grid")
        self.lambert_area = state.pop("lambert_area")
        super().__setstate__(state)


def _with_lambert_geography(field: ekd.Field, lambert_grid: dict[str, Any], area: Any) -> ekd.Field:
    """Return ``field`` reporting ``lambert_grid`` and ``area`` as its MARS grid and area."""
    lambert_area = (list(area) if isinstance(area, (list, tuple)) else area) or None
    metadata = _LambertMetadata(field._metadata, lambert_grid, lambert_area, owner=field)
    return field.clone(metadata=metadata)


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


def check_lambert_area(grid: Any, area: Any) -> None:
    """Raise ``ValueError`` if ``area`` is given for a Lambert grid: its keys define the domain and MIR cannot crop it."""
    if is_lambert_grid(grid) and area is not None and len(area) > 0:
        raise ValueError(
            f"An area ({area!r}) cannot be combined with a Lambert grid: the keys of the grid define its domain "
            "and MIR cannot crop it"
        )


def grid_repr(grid: GridSpec) -> str:
    """Return a human-readable representation of the grid specification."""
    if is_lambert_grid(grid):
        return str(canonical_lambert_grid(grid))  # type: ignore[arg-type]
    if isinstance(grid, str):
        return grid.upper()
    if isinstance(grid, (list, tuple)):
        return str(list(grid)[:5] + ["..."] if len(grid) > 5 else grid)
    if isinstance(grid, dict):
        return str(
            {k: f"list of len {len(v)}" if isinstance(v, (list, tuple, np.ndarray)) else v for k, v in grid.items()}
        )
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

    Parameters
    ----------
    fields : ekd.FieldList
        The input fields to regrid.
    grid : GridSpec
        The target grid specification (grid string, list/tuple of increments,
        dict of coordinate lists, or a Lambert grid). A dict with
        ``gridType: lambert`` is a Lambert grid given as GRIB keys (see
        :func:`canonical_lambert_grid`, e.g. ``Nx``, ``Ny``, the south-west corner as first
        point, ``LaDInDegrees``, ``LoVInDegrees``, ``DxInMetres``, ``DyInMetres``, ``shapeOfTheEarth``).
        MIR is given the north-west corner and the steps that put its points on the
        points of that grid whatever its earth, and the header of the output is written from the keys
        (first point, scanning from south to north, earth, ...); the input fields are not
        modified. The output fields report the canonical Lambert grid as their MARS grid and
        ``None`` as their MARS area, see ``ekd.Field.metadata().geography``. GRIB1 input fields
        work as well.
    area : str or list of float or None, optional
        The target area specification. Not with a Lambert grid, whose keys define the domain
        and which MIR cannot crop: a ``ValueError`` is raised.
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

    lambert = canonical_lambert_grid(grid) if is_lambert_grid(grid) else None  # type: ignore[arg-type]
    check_lambert_area(grid, area)
    job_args: dict[str, Any] = {
        "edition": 2,
        "packing": packing,
        "accuracy": accuracy,
    }
    if area:
        job_args["area"] = area

    job = None
    lambert_jobs: dict[bool, Any] = {}
    grid_uid = None
    if lambert is None:
        mir_grid = _make_mir_grid(grid)
        mir_spec = mir_grid.spec
        # Unstructured lat/lon grids carry a UID in their spec; regular grids do
        # not. MIR fails to write this UID into the output's ``uuidOfHGrid``,
        # leaving it all-zero, which later makes the grid unresolvable
        # (eckit-geo raises ``GridUnknownError``). We stamp it back on below.
        # TODO: remove once MIR populates uuidOfHGrid for unstructured lat/lon grids.
        grid_uid = mir_spec.get("uid") if isinstance(mir_spec, dict) else None
        job = mir.Job(grid=mir_spec, **job_args)

    out_fields = []
    for field in fields:
        input_buffer = io.BytesIO()
        field.to_target("file", input_buffer)
        if lambert is not None:
            message, edition = _lambert_input(input_buffer.getvalue())
            input_buffer = io.BytesIO(message)
            # The Lambert string differs by input edition: GRIB1 does not take writeLaDInDegrees
            if (edition == 2) not in lambert_jobs:
                lambert_jobs[edition == 2] = mir.Job(grid=_mir_lambert_grid(lambert, edition == 2), **job_args)
            job = lambert_jobs[edition == 2]
        input_buffer.seek(0)

        output_buffer = io.BytesIO()
        job.execute(mir.PyGribInput(input_buffer), mir.PyGribOutput(output_buffer))
        message = output_buffer.getvalue()
        if lambert is not None:
            message = _lambert_output(message, lambert)
        regridded = ekd.from_source("memory", message)[0]

        input_buffer.close()
        output_buffer.close()

        if grid_uid:
            regridded.handle.set("uuidOfHGrid", grid_uid)

        if lambert is not None:
            # no area is possible with a Lambert grid: mars_area() is None
            regridded = _with_lambert_geography(regridded, lambert, None)

        out_fields.append(regridded)

    return ekd.FieldList.from_fields(out_fields)
