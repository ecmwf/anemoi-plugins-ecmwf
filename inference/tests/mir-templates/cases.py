# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The case table of the real MIR template tests, and the helpers that check a case.

A case is a template lookup, optionally an output state, and what the template
must be. Adding a regridding case is adding one entry to ``CASES``.

Nothing here imports MIR at module level: the states that need MIR are built
lazily, when a test asks for them.

Template lookups and the way a caller describes the output
----------------------------------------------------------
A cutout joins a regular limited area model (LAM) grid, the first source, with
the points of a global Gaussian grid that lie outside the LAM box, the second
source. The output state is the LAM points followed by those global points.

The template lookup only describes one source at a time, so what a caller
passes depends on how the output is produced:

- no extraction: the global grid and a global area, describing only the global source
- extraction of the LAM source: a bare numeric increment (anemoi-inference
  collapses ``[d, d]`` to ``d``) and the LAM area
- extraction of the global source: the global grid, and the extent of the
  points it has left

Whenever the lookup does not describe the points of the output, the template
has to be an unstructured grid built from the state's coordinates.
"""

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
from anemoi.plugins.ecmwf.transform.regrid.backend import canonical_lambert_grid

State = dict[str, Any]

#: The MEPS Lambert grid (2.5 km, 849 x 969 points) as a dict of GRIB keys: Lambert conformal, tangent
#: at 63.3 N, on a sphere of radius 6371000 m (``shapeOfTheEarth=1``), scanning from the south-west corner.
MEPS_LAMBERT_GRID: dict[str, Any] = {
    "gridType": "lambert",
    "Nx": 849,
    "Ny": 969,
    "latitudeOfFirstGridPointInDegrees": 51.633816,
    "longitudeOfFirstGridPointInDegrees": 1.60491,
    "LaDInDegrees": 63.3,
    "LoVInDegrees": 15.0,
    "Latin1InDegrees": 63.3,
    "Latin2InDegrees": 63.3,
    "DxInMetres": 2500,
    "DyInMetres": 2500,
    "scanningMode": 64,
    "shapeOfTheEarth": 1,
    "scaleFactorOfRadiusOfSphericalEarth": 0,
    "scaledValueOfRadiusOfSphericalEarth": 6371000,
    "projectionCentreFlag": 0,
    "resolutionAndComponentFlags": 48,
    "latitudeOfSouthernPoleInDegrees": -90.0,
    "longitudeOfSouthernPoleInDegrees": 0.0,
}
MEPS_NPOINTS = 849 * 969
#: Size (``Nx``, ``Ny``) of ``LAMBERT_GRID``, a small and fast grid with the geometry of MEPS: 25 km steps.
LAMBERT_NX, LAMBERT_NY = 60, 70
LAMBERT_GRID: dict[str, Any] = {
    **MEPS_LAMBERT_GRID,
    "Nx": LAMBERT_NX,
    "Ny": LAMBERT_NY,
    "DxInMetres": 25000,
    "DyInMetres": 25000,
}
NULL_UID = "00000000000000000000000000000000"


@dataclass(frozen=True)
class TemplateCase:
    """One template lookup, and what the template built from it must be.

    Attributes
    ----------
    id : str
        The pytest id of the case.
    lookup : dict | Callable[[], dict]
        The template lookup, or what builds it lazily if it needs MIR. Use ``lookup_of`` to get it.
    state : Callable[[], dict] | None
        Builds the output state (with ``latitudes`` and ``longitudes``) lazily, if the case has one.
    grid_type : str | None
        The expected ``gridType``. For ``unstructured_grid`` with a state, the
        ``uuidOfHGrid`` must be that of MIR's grid of the state coordinates.
    npoints : int | None
        The expected ``numberOfDataPoints``, for cases without a state.
        With a state it always is the size of the state.
    mars_grid : Any
        The expected ``mars_grid()`` of the template, not checked if None.
    mars_area : Any
        The expected ``mars_area()`` of the template, not checked if None.
    header : dict[str, Any] | None
        GRIB keys of the template with their expected values (the Lambert grid keys), not checked if None.
    coordinates : bool
        Whether the template coordinates must be those of the state. Not for a Lambert template: the
        points ecCodes reports for it need ``ECCODES_ECKIT_GEO=0`` (see ``test_preprocessor_lambert.py``).
    exception : type[Exception] | None
        The exception the template must raise.
    match : str | None
        What the exception message must match.
    returns_none : bool
        Whether the template must be None.
    variable : str
        The variable to get the template for.
    """

    id: str
    lookup: dict[str, Any] | Callable[[], dict[str, Any]] = field(default_factory=dict)
    state: Callable[[], State] | None = None
    grid_type: str | None = None
    npoints: int | None = None
    mars_grid: Any = None
    mars_area: Any = None
    header: dict[str, Any] | None = None
    coordinates: bool = True
    exception: type[Exception] | None = None
    match: str | None = None
    returns_none: bool = False
    variable: str = "lsm"


# Helpers to check a case


def lookup_of(case: TemplateCase) -> dict[str, Any]:
    """A fresh copy of the lookup of a case: the cases are shared, so a test must not modify them."""
    lookup = case.lookup() if callable(case.lookup) else case.lookup
    return dict(lookup)


def flat_coordinates(state: State) -> tuple[np.ndarray, np.ndarray]:
    """The latitudes and longitudes of a state, flattened."""
    return np.asarray(state["latitudes"], dtype=float).ravel(), np.asarray(state["longitudes"], dtype=float).ravel()


def assert_coordinates(template, state: State) -> None:
    """The template points are the state points, in order (longitudes modulo 360)."""
    latitudes, longitudes = flat_coordinates(state)
    latlon = template.to_latlon(flatten=True)
    np.testing.assert_allclose(latlon["lat"], latitudes, atol=1e-4, rtol=0)
    np.testing.assert_allclose(np.mod(latlon["lon"], 360.0), np.mod(longitudes, 360.0), atol=1e-4, rtol=0)


def state_uid(state: State) -> str:
    """The ``uuidOfHGrid`` MIR gives to the grid of the state coordinates."""
    import mir

    latitudes, longitudes = flat_coordinates(state)
    return mir.Grid(latitudes=list(latitudes), longitudes=list(longitudes)).uid


def assert_unstructured(template, state: State) -> None:
    assert template.metadata("gridType") == "unstructured_grid"
    assert template.metadata("numberOfDataPoints") == len(flat_coordinates(state)[0])
    assert template.metadata("uuidOfHGrid") == state_uid(state)


# Lazy state builders


def state_of_grid(**grid: Any) -> Callable[[], State]:
    """The points of a MIR structured grid (``mir.Grid(**grid)``), flat."""

    def build() -> State:
        import mir

        latitudes, longitudes = mir.Grid(**grid).to_latlons()
        return {"latitudes": np.asarray(latitudes), "longitudes": np.asarray(longitudes)}

    return build


def state_of_grid_2d(**grid: Any) -> Callable[[], State]:
    """The points of a regular MIR grid, as meshgrid-shaped (2D) coordinates."""

    def build() -> State:
        flat = state_of_grid(**grid)()
        lats = np.unique(np.round(flat["latitudes"], 6))[::-1]
        lons = np.unique(np.round(flat["longitudes"], 6))
        assert lats.size * lons.size == len(flat["latitudes"])
        lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
        assert lat2d.ndim == 2
        return {"latitudes": lat2d, "longitudes": lon2d}

    return build


def state_of_zeros(**grid: Any) -> Callable[[], State]:
    """A state with as many points as a MIR grid, all at (0, 0)."""

    def build() -> State:
        import mir

        npoints = mir.Grid(**grid).size()
        return {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}

    return build


def random_state(shape: tuple[int, ...] = (300,), seed: int = 0) -> Callable[[], State]:
    """Scattered points over the north-west of Europe."""

    def build() -> State:
        rng = np.random.default_rng(seed)
        if len(shape) == 1:
            return {"latitudes": np.sort(rng.uniform(40.0, 60.0, shape)), "longitudes": rng.uniform(-10.0, 10.0, shape)}
        return {"latitudes": rng.uniform(40, 60, shape), "longitudes": rng.uniform(-10, 10, shape)}

    return build


def fixed_state(latitudes: list[float], longitudes: list[float]) -> Callable[[], State]:
    def build() -> State:
        return {"latitudes": latitudes, "longitudes": longitudes}

    return build


# Cutout layouts


@dataclass(frozen=True)
class Layout:
    """A cutout of a regular LAM into a global Gaussian grid."""

    grid: str  # global grid, e.g. "O32"
    increment: float  # LAM grid increment, in degrees
    area: tuple[float, float, float, float]  # LAM north, west, south, east

    @property
    def lam_area(self) -> list[float]:
        """The LAM area as a lookup would have it, longitudes in [0, 360]."""
        north, west, south, east = self.area
        return [north, west % 360.0, south, east % 360.0]

    @property
    def lam_grid(self) -> str:
        return f"{self.increment}/{self.increment}"


LAYOUTS = {
    "mid_latitude": Layout("O32", 1.0, (60.0, 0.0, 30.0, 40.0)),
    "half_degree": Layout("O48", 0.5, (55.0, 5.0, 35.0, 25.0)),
    "crosses_meridian": Layout("O32", 1.0, (60.0, -15.0, 40.0, 15.0)),
    "crosses_dateline": Layout("O32", 1.0, (50.0, 170.0, 30.0, 190.0)),
}


@functools.cache
def cutout_states(layout_name: str) -> dict[str, State]:
    """The states of the full cutout, and of its LAM and global parts (cached, they are costly)."""
    layout = LAYOUTS[layout_name]
    north, west, south, east = layout.area
    d = layout.increment

    nlat = round((north - south) / d) + 1
    nlon = round((east - west) / d) + 1
    lam_lat, lam_lon = np.meshgrid(north - d * np.arange(nlat), west + d * np.arange(nlon), indexing="ij")
    lam_lat, lam_lon = lam_lat.ravel(), lam_lon.ravel()

    glob_lat, glob_lon = (np.asarray(x) for x in state_of_grid(grid=layout.grid)().values())
    tolerance = 1e-6
    inside = (
        (glob_lat <= north + tolerance)
        & (glob_lat >= south - tolerance)
        & (np.mod(glob_lon - west, 360.0) <= (east - west) + tolerance)
    )

    lam = {"latitudes": lam_lat, "longitudes": lam_lon}
    glob = {"latitudes": glob_lat[~inside], "longitudes": glob_lon[~inside]}
    full = {key: np.concatenate([lam[key], glob[key]]) for key in lam}

    # The box must cut points out of the global grid, and leave some in it
    assert 0 < inside.sum() < len(glob_lat)
    assert len(full["latitudes"]) == len(lam["latitudes"]) + len(glob["latitudes"])
    return {"full": full, "lam": lam, "global": glob}


def cutout_state(layout_name: str, part: str) -> Callable[[], State]:
    return lambda: cutout_states(layout_name)[part]


def extent_area(state: State) -> list[float]:
    """The extent of a set of points, as an area with longitudes in [0, 360]."""
    latitudes = state["latitudes"]
    longitudes = np.mod(state["longitudes"], 360.0)
    return [float(latitudes.max()), float(longitudes.min()), float(latitudes.min()), float(longitudes.max())]


def cutout_lookup_with_global_extent(layout_name: str) -> dict[str, Any]:
    """The lookup of the global part of a cutout: the global grid, and the extent of the points left."""
    area = extent_area(cutout_states(layout_name)["global"])
    return {"levtype": "sfc", "grid": LAYOUTS[layout_name].grid, "area": area}


def _cutout_cases() -> list[TemplateCase]:
    cases = []
    for name, layout in LAYOUTS.items():
        full, lam, glob = (cutout_state(name, part) for part in ("full", "lam", "global"))
        unstructured = "unstructured_grid"

        # The lookup describes the global source only, the output has the LAM points too
        for variable, levtype in [("2t", "sfc"), ("t", "pl")]:
            lookup = {"levtype": levtype, "grid": layout.grid, "area": [90, 0.0, -90, 360]}
            cases.append(
                TemplateCase(f"{name}-full-global-grid-area-{variable}", lookup, full, unstructured, variable=variable)
            )
        cases.append(
            TemplateCase(
                f"{name}-full-global-grid", {"levtype": "sfc", "grid": layout.grid}, full, unstructured, variable="2t"
            )
        )

        # The lookup describes the LAM source only, the output has the global points too
        lookup = {"levtype": "sfc", "grid": layout.increment, "area": layout.lam_area}
        cases.append(TemplateCase(f"{name}-full-lam-grid", lookup, full, unstructured, variable="2t"))

        # A bare increment (or d/d) and the LAM area are enough for a regular lat/lon grid
        for label, grid in [("bare-increment", layout.increment), ("explicit-grid", layout.lam_grid)]:
            lookup = {"levtype": "sfc", "grid": grid, "area": layout.lam_area}
            cases.append(TemplateCase(f"{name}-lam-{label}", lookup, lam, "regular_ll", variable="2t"))

        # The global points outside the LAM are not a Gaussian grid, the template follows the state
        lookup = {"levtype": "sfc", "grid": layout.grid}
        cases.append(TemplateCase(f"{name}-global-no-area", lookup, glob, unstructured, variable="2t"))
        lookup = functools.partial(cutout_lookup_with_global_extent, name)
        cases.append(TemplateCase(f"{name}-global-extent-area", lookup, glob, unstructured, variable="2t"))
    return cases


# The old matrix of structured grids, and of grids and areas, without a state.
# Each lookup is built here, nothing is shared (and so nothing is mutated) between cases.

GRIDS = {
    # lookup grid: (mars_grid, number of points)
    "O48": ("O48", 10944),
    "O96": ("O96", 40320),
    "N320": ("N320", 542080),
    "O1280": ("O1280", 6599680),
    "1.0/1.0": ([1.0, 1.0], 181 * 360),
    "0.5/0.5": ([0.5, 0.5], 361 * 720),
    "0.25/0.25": ([0.25, 0.25], 721 * 1440),
}

AREAS = {
    # label: (lookup area, mars_area)
    "10/-0.5/0.5/11": ("10/-0.5/0.5/11", [10.0, 359.5, 0.5, 11.0]),
    "2/4/2/4": ("2/4/2/4", [2.0, 4.0, 2.0, 4.0]),
    "[2,4,2,4]": ([2, 4, 2, 4], [2.0, 4.0, 2.0, 4.0]),
}


def _grid_cases() -> list[TemplateCase]:
    cases = []
    for levtype in ["pl", "sfc"]:
        for grid, (mars_grid, npoints) in GRIDS.items():
            cases.append(
                TemplateCase(f"{levtype}-{grid}", {"levtype": levtype, "grid": grid}, None, None, npoints, mars_grid)
            )
        cases.append(
            TemplateCase(f"{levtype}-[0.5,0.5]", {"levtype": levtype, "grid": [0.5, 0.5]}, None, None, None, [0.5, 0.5])
        )

        # The old matrix: each grid with each area, the area as a string or as a list
        for grid, grid_label in [("0.5/0.5", "0.5/0.5"), ("0.25/0.25", "0.25/0.25"), ([0.5, 0.5], "[0.5,0.5]")]:
            mars_grid = GRIDS[grid][0] if isinstance(grid, str) else grid
            for area_label, (area, mars_area) in AREAS.items():
                cases.append(
                    TemplateCase(
                        f"{levtype}-{grid_label}-area-{area_label}",
                        {"levtype": levtype, "grid": grid, "area": area},
                        None,
                        None,
                        None,
                        mars_grid,
                        mars_area,
                    )
                )
    return cases


# Output states on structured grids: the template keeps that grid.
# (id, lookup, arguments of the mir.Grid of the state, gridType)
STRUCTURED = [
    ("O48", {"grid": "O48"}, {"grid": "O48"}, "reduced_gg"),
    ("o48-lowercase", {"grid": "o48"}, {"grid": "O48"}, "reduced_gg"),
    ("N320", {"grid": "N320"}, {"grid": "N320"}, "reduced_gg"),
    ("n320-lowercase", {"grid": "n320"}, {"grid": "N320"}, "reduced_gg"),
    ("1.0/1.0", {"grid": "1.0/1.0"}, {"grid": "1.0/1.0"}, "regular_ll"),
    ("1.0-bare", {"grid": 1.0}, {"grid": "1.0/1.0"}, "regular_ll"),
    ("[1.0,1.0]", {"grid": [1.0, 1.0]}, {"grid": [1.0, 1.0]}, "regular_ll"),
    (
        "0.5/0.5-area",
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        "regular_ll",
    ),
    (
        "0.5-bare-area",
        {"grid": 0.5, "area": [70, -20, 20, 30]},
        {"grid": "0.5/0.5", "area": [70, -20, 20, 30]},
        "regular_ll",
    ),
]

# Regular grids as meshgrid-shaped coordinates, which behave like flat ones.
# (id, lookup, arguments of the mir.Grid of the state, mars_area)
TWO_DIMENSIONAL = [
    ("1.0/1.0", {"grid": "1.0/1.0"}, {"grid": "1.0/1.0"}, [90.0, 0.0, -90.0, 359.0]),
    ("1.0-bare", {"grid": 1.0}, {"grid": "1.0/1.0"}, [90.0, 0.0, -90.0, 359.0]),
    (
        "0.5/0.5-area",
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        {"grid": "0.5/0.5", "area": "70/-20/20/30"},
        [70.0, 340.0, 20.0, 30.0],
    ),
]


def _structured_cases() -> list[TemplateCase]:
    cases = [
        TemplateCase(
            f"state-{label}",
            {"levtype": "sfc", **lookup},
            state_of_grid(**mir_grid),
            grid_type,
            mars_grid=None,
        )
        for label, lookup, mir_grid, grid_type in STRUCTURED
    ]
    cases += [
        TemplateCase(
            f"state-2d-{label}",
            {"levtype": "sfc", **lookup},
            state_of_grid_2d(**mir_grid),
            "regular_ll",
            mars_area=mars_area,
        )
        for label, lookup, mir_grid, mars_area in TWO_DIMENSIONAL
    ]
    cases += [
        # The grid of the lookup is that of the state: the template is unchanged. The state points are not the grid's
        TemplateCase(
            "state-O96-unchanged",
            {"levtype": "sfc", "grid": "O96"},
            state_of_zeros(grid="O96"),
            "reduced_gg",
            mars_grid="O96",
            coordinates=False,
        ),
        TemplateCase(
            "state-0.5/0.5-area-unchanged",
            {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/0/0/10"},
            state_of_grid(grid="0.5/0.5", area="10/0/0/10"),
            "regular_ll",
            mars_area=[10.0, 0.0, 0.0, 10.0],
        ),
    ]
    return cases


# Output states that the lookup does not describe: the template is an unstructured grid of the state

COORDINATE_GRID = {"latitudes": [50.0, 51.0, 52.0], "longitudes": [0.0, 1.0, 2.0]}
UNSTRUCTURED = "unstructured_grid"


def _unstructured_cases() -> list[TemplateCase]:
    state = random_state()
    cases = [
        TemplateCase("no-grid-uses-state", {"levtype": "sfc", "grid": None}, state, UNSTRUCTURED),
        TemplateCase("missing-grid-key-uses-state", {"levtype": "sfc"}, state, UNSTRUCTURED),
        TemplateCase("mismatched-grid-sfc", {"levtype": "sfc", "grid": "O96"}, state, UNSTRUCTURED),
        TemplateCase("mismatched-grid-pl", {"levtype": "pl", "grid": "O96"}, state, UNSTRUCTURED),
        # A cutout checkpoint: the lookup has the LAM source's grid and area
        TemplateCase(
            "mismatched-grid-and-area",
            {"levtype": "sfc", "grid": "0.5/0.5", "area": "10/0/0/10"},
            state,
            UNSTRUCTURED,
        ),
        TemplateCase(
            "2d-state-no-grid-is-flattened",
            {"levtype": "sfc", "grid": None},
            random_state((10, 20), seed=1),
            UNSTRUCTURED,
        ),
    ]

    # A lookup grid given as coordinates is not compared: the state coordinates are used
    lookup_state = fixed_state([10.0, 11.0, 12.0], [20.0, 21.0, 22.0])
    for label, grid, area in [
        ("coordinates", COORDINATE_GRID, None),
        ("coordinates-area", COORDINATE_GRID, [51, 0, 49, 1]),
        ("named", "meps", None),
        ("named-area", "meps", [60, 10, 59, 11]),
    ]:
        lookup = {"levtype": "sfc", "grid": grid, "area": area}
        cases.append(TemplateCase(f"{label}-lookup-uses-state", lookup, lookup_state, UNSTRUCTURED))
    return cases


def _failing_cases() -> list[TemplateCase]:
    state = random_state()
    return [
        TemplateCase("no-grid-no-state", {"levtype": "sfc", "grid": None}, None, returns_none=True),
        # A grid MIR cannot build is an error, not a reason to use the state coordinates
        TemplateCase(
            "unbuildable-grid-with-state",
            {"levtype": "sfc", "grid": "not-a-grid"},
            state,
            exception=RuntimeError,
            match="cannot build grid without 'type'",
        ),
        TemplateCase(
            "unbuildable-grid-without-state",
            {"levtype": "sfc", "grid": "not-a-grid"},
            None,
            exception=RuntimeError,
            match="cannot build grid without 'type'",
        ),
        # Coordinates of different lengths: MIR refuses them when it regrids
        TemplateCase(
            "unbuildable-coordinates-with-area-without-state",
            {"levtype": "sfc", "grid": {"latitudes": [1.0], "longitudes": []}, "area": "1/0/0/1"},
            None,
            exception=RuntimeError,
            match=re.escape("lat_.size() == lon_.size()"),
        ),
    ]


def _state_of_zeros(npoints: int) -> Callable[[], State]:
    """A state with ``npoints`` points, all at (0, 0)."""
    return lambda: {"latitudes": np.zeros(npoints), "longitudes": np.zeros(npoints)}


def lambert_header(grid: dict[str, Any]) -> dict[str, Any]:
    """The keys a Lambert template must have: those of the grid, scanning from south to north, GRIB2."""
    keys = {key: value for key, value in canonical_lambert_grid(grid).items() if key != "gridType"}
    return {**keys, "jScansPositively": 1, "edition": 2}


def _lambert_cases() -> list[TemplateCase]:
    # A Lambert grid is a dict of GRIB keys, not coordinates: no mir.Grid can be built from it, its size is Nx * Ny
    cases = []
    for label, grid in [("lambert", LAMBERT_GRID), ("meps-lambert", MEPS_LAMBERT_GRID)]:
        lookup = {"levtype": "sfc", "grid": grid}
        npoints = grid["Nx"] * grid["Ny"]
        expected = {
            "grid_type": "lambert",
            "mars_grid": canonical_lambert_grid(grid),
            "header": lambert_header(grid),
        }
        cases += [
            TemplateCase(label, lookup, None, npoints=npoints, **expected),
            # The output has the size of the Lambert grid: the template is that grid (the state points are not its)
            TemplateCase(
                f"{label}-state-same-size",
                lookup,
                _state_of_zeros(npoints),
                coordinates=False,
                **expected,
            ),
            # Another size: the template follows the state
            TemplateCase(f"{label}-state-other-size", lookup, random_state(), grid_type="unstructured_grid"),
        ]

    # The lookup has the (regional) area of the checkpoint: it is ignored, a Lambert grid defines its domain
    cases.append(
        TemplateCase(
            "lambert-lookup-area-ignored",
            {"levtype": "sfc", "grid": LAMBERT_GRID, "area": [70.0, -20.0, 30.0, 40.0]},
            None,
            grid_type="lambert",
            npoints=LAMBERT_NX * LAMBERT_NY,
            mars_grid=canonical_lambert_grid(LAMBERT_GRID),
            header=lambert_header(LAMBERT_GRID),
        )
    )

    # The same grid with the keys in another order, and with the ignored GRIB keys of a message
    shuffled = {
        **dict(reversed(list(LAMBERT_GRID.items()))),
        "gridDefinitionTemplateNumber": 30,
        "tablesVersion": 32,
    }
    cases.append(
        TemplateCase(
            "lambert-shuffled-keys",
            {"levtype": "sfc", "grid": shuffled},
            None,
            grid_type="lambert",
            npoints=LAMBERT_NX * LAMBERT_NY,
            mars_grid=canonical_lambert_grid(LAMBERT_GRID),
            header=lambert_header(LAMBERT_GRID),
        )
    )
    return cases


CASES = [
    *_grid_cases(),
    *_structured_cases(),
    *_unstructured_cases(),
    *_cutout_cases(),
    *_failing_cases(),
    *_lambert_cases(),
]
