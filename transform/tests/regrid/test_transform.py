# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the regrid plugin logic that need no MIR.

Covers the ``MIRRegrid`` filter (construction, named-grid resolution,
``__repr__``, argument plumbing in ``forward()`` with ``mir_regrid`` stubbed),
the Lambert grid given as GRIB keys (validation, geometry, the grid asked of MIR,
the rewrite of the message MIR writes, the geography of the output), the pure helpers
``grid_repr`` and ``GridSpec``, and the named grids (``KNOWN_GRIDS``, ``NamedRegrid``).
Real MIR runs are in ``test_mir.py``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid import backend
from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec
from anemoi.plugins.ecmwf.transform.regrid.backend import _with_lambert_geography
from anemoi.plugins.ecmwf.transform.regrid.backend import canonical_lambert_grid
from anemoi.plugins.ecmwf.transform.regrid.backend import grid_repr
from anemoi.plugins.ecmwf.transform.regrid.backend import is_lambert_grid
from anemoi.plugins.ecmwf.transform.regrid.backend import lambert_points
from anemoi.plugins.ecmwf.transform.regrid.backend import lambert_radius
from anemoi.plugins.ecmwf.transform.regrid.backend import mir_regrid
from anemoi.plugins.ecmwf.transform.regrid.named import KNOWN_GRIDS
from anemoi.plugins.ecmwf.transform.regrid.named import NamedRegrid

from .cases import EARTHS
from .cases import LAMBERT_GRID
from .cases import MEPS_LAMBERT_GRID
from .cases import distance_m
from .cases import lambert_grid
from .cases import pyproj_points
from .cases import signed_longitude

#: The module of the ``MIRRegrid`` filter (the package re-exports the class under the module's name).
FILTER_MODULE = "anemoi.plugins.ecmwf.transform.regrid.regrid"

#: The smallest Lambert grid: only the required keys.
MINIMAL_LAMBERT_GRID = {
    key: MEPS_LAMBERT_GRID[key]
    for key in (
        "gridType",
        "Nx",
        "Ny",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "LaDInDegrees",
        "LoVInDegrees",
        "DxInMetres",
        "DyInMetres",
    )
}


class TestMIRRegridConstruction:
    """Tests for ``MIRRegrid`` construction and stored configuration."""

    def test_string_grid_stored(self):
        """A string grid that is not a named grid is kept unchanged."""
        # "O16" is a valid MIR grid string, not a named grid.
        assert "o16" not in KNOWN_GRIDS
        assert MIRRegrid(grid="O16").grid == "O16"

    def test_area_stored(self):
        """The grid and area are kept unchanged."""
        r = MIRRegrid(grid="O1280", area=[40, -25, -40, 55])

        assert r.grid == "O1280"
        assert r._area == [40, -25, -40, 55]

    def test_lambert_grid_stored_canonical(self):
        """A Lambert grid is stored in its canonical form (the patch with ints and floats as in GRIB)."""
        grid = MIRRegrid(grid=MEPS_LAMBERT_GRID).grid

        assert grid == canonical_lambert_grid(MEPS_LAMBERT_GRID)
        assert grid["DxInMetres"] == 2500.0
        assert is_lambert_grid(grid)

    def test_lambert_grid_is_not_a_named_grid_or_coordinates(self):
        """A dict with gridType lambert is not resolved or treated as a coordinate dict."""
        assert "latitudes" not in MIRRegrid(grid=MEPS_LAMBERT_GRID).grid

    def test_minimal_lambert_grid_gets_the_defaults(self):
        grid = MIRRegrid(grid=MINIMAL_LAMBERT_GRID).grid

        assert grid["Latin1InDegrees"] == grid["Latin2InDegrees"] == 63.3
        assert grid["scanningMode"] == 64
        assert grid["shapeOfTheEarth"] == 6
        assert "scaledValueOfRadiusOfSphericalEarth" not in grid

    @pytest.mark.parametrize("area", [[50, 0, 40, 10], "50/0/40/10"])
    def test_area_with_a_lambert_grid_raises_at_construction(self, area):
        """The keys define the domain of a Lambert grid and MIR cannot crop it."""
        with pytest.raises(ValueError, match="cannot be combined with a Lambert grid"):
            MIRRegrid(grid=MEPS_LAMBERT_GRID, area=area)

    @pytest.mark.parametrize("area", [None, [], ()])
    def test_no_area_with_a_lambert_grid(self, area):
        MIRRegrid(grid=MEPS_LAMBERT_GRID, area=area)

    def test_invalid_lambert_grid_raises_at_construction(self):
        with pytest.raises(ValueError, match="unknown key"):
            MIRRegrid(grid={**MEPS_LAMBERT_GRID, "Bogus": 1})

    def test_named_grid_resolved_to_coord_dict(self):
        """A known grid name is resolved to a lat/lon coordinate dict."""
        r = MIRRegrid(grid="meps")

        assert isinstance(r.grid, dict)
        assert set(r.grid) == {"latitudes", "longitudes"}
        assert len(r.grid["latitudes"]) == len(r.grid["longitudes"])
        assert len(r.grid["latitudes"]) > 0

    def test_named_grid_case_insensitive(self):
        """Named grids are matched case-insensitively."""
        r_lower = MIRRegrid(grid="meps")
        r_upper = MIRRegrid(grid="MEPS")

        assert r_upper.grid["latitudes"] == r_lower.grid["latitudes"]
        assert r_upper.grid["longitudes"] == r_lower.grid["longitudes"]

    def test_named_grid_matches_namedregrid(self):
        """MIRRegrid resolves the same coordinates as NamedRegrid directly."""
        r = MIRRegrid(grid="meps")
        named = NamedRegrid("meps")

        assert r.grid["latitudes"] == named.latitudes
        assert r.grid["longitudes"] == named.longitudes

    @pytest.mark.parametrize("grid", [[1.0, 1.0], {"latitudes": [0.0], "longitudes": [0.0]}])
    def test_non_string_grid_passed_through(self, grid):
        """List and coordinate dict grids are kept unchanged."""
        assert MIRRegrid(grid=grid).grid == grid


class TestMIRRegridRepr:
    """Tests for ``MIRRegrid.__repr__()``."""

    def test_repr_string_grid(self):
        """repr shows grid and area."""
        r = MIRRegrid(grid="O32", area=[90, 0, -90, 360])

        assert "O32" in repr(r)
        assert "90" in repr(r)

    def test_repr_no_area(self):
        """repr shows None area."""
        r = MIRRegrid(grid="N320")

        assert "N320" in repr(r)
        assert "None" in repr(r)

    def test_repr_full(self):
        """repr shows the grid and the full area."""
        r = MIRRegrid(grid="O1280", area=[40, -25, -40, 55])

        assert repr(r) == "MIRRegrid(grid='O1280', area=[40, -25, -40, 55])"

    def test_repr_lambert_grid(self):
        """repr shows the canonical Lambert grid, not a summary of lists."""
        r = MIRRegrid(grid=MEPS_LAMBERT_GRID)

        assert repr(r) == f"MIRRegrid(grid={str(canonical_lambert_grid(MEPS_LAMBERT_GRID))!r}, area=None)"
        assert "'Nx': 849" in repr(r)


class TestMIRRegridForward:
    """Tests for ``MIRRegrid.forward()`` with ``mir_regrid`` stubbed."""

    def test_forward_passes_arguments_to_mir_regrid(self, monkeypatch):
        """forward() passes fields, grid, area, packing and accuracy on."""
        module = sys.modules[FILTER_MODULE]
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        sentinel = object()

        def fake_mir_regrid(*args: Any, **kwargs: Any) -> object:
            calls.append((args, kwargs))
            return sentinel

        monkeypatch.setattr(module, "mir_regrid", fake_mir_regrid)
        fields = object()
        area = [90, 0, 0, 180]

        result = MIRRegrid(grid="O32", area=area, packing="simple", accuracy=12).forward(fields)

        assert result is sentinel
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == (fields, "O32", area)
        assert kwargs == {"packing": "simple", "accuracy": 12}

    def test_forward_lambert_grid_canonical(self, monkeypatch):
        """forward() passes the canonical Lambert grid to mir_regrid."""
        module = sys.modules[FILTER_MODULE]
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(module, "mir_regrid", lambda *args, **kwargs: calls.append((args, kwargs)))
        fields = object()

        MIRRegrid(grid=MEPS_LAMBERT_GRID).forward(fields)

        assert calls == [
            ((fields, canonical_lambert_grid(MEPS_LAMBERT_GRID), None), {"packing": "ccsds", "accuracy": 16})
        ]

    @staticmethod
    def _forward_with_env(monkeypatch, caplog, grid: GridSpec, eckit_geo: str | None) -> list[str]:
        """Run ``forward`` with ``mir_regrid`` stubbed, return the warnings."""
        module = sys.modules[FILTER_MODULE]
        monkeypatch.setattr(module, "mir_regrid", lambda *args, **kwargs: None)
        if eckit_geo is None:
            monkeypatch.delenv("ECCODES_ECKIT_GEO", raising=False)
        else:
            monkeypatch.setenv("ECCODES_ECKIT_GEO", eckit_geo)

        with caplog.at_level(logging.WARNING, logger=module.LOG.name):
            MIRRegrid(grid=grid).forward("fields")

        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_lambert_grid_warns_with_eckit_geo(self, monkeypatch, caplog):
        """A Lambert grid with ECCODES_ECKIT_GEO=1 logs a warning naming the workaround."""
        warnings = self._forward_with_env(monkeypatch, caplog, MEPS_LAMBERT_GRID, "1")

        assert len(warnings) == 1
        assert "Lambert" in warnings[0]
        assert "ECCODES_ECKIT_GEO=1" in warnings[0]
        assert "ECCODES_ECKIT_GEO=0" in warnings[0]

    @pytest.mark.parametrize("eckit_geo", ["0", None])
    def test_lambert_grid_no_warning_without_eckit_geo(self, monkeypatch, caplog, eckit_geo):
        """No warning when ECCODES_ECKIT_GEO is 0 or unset."""
        assert self._forward_with_env(monkeypatch, caplog, MEPS_LAMBERT_GRID, eckit_geo) == []

    @pytest.mark.parametrize("grid", ["O32", {"latitudes": [1.0], "longitudes": [2.0]}], ids=["string", "coordinates"])
    def test_other_grid_no_warning_with_eckit_geo(self, monkeypatch, caplog, grid):
        """Other grids do not warn with ECCODES_ECKIT_GEO=1."""
        assert self._forward_with_env(monkeypatch, caplog, grid, "1") == []

    def test_forward_default_arguments(self, monkeypatch):
        """forward() passes no area and the default packing and accuracy."""
        module = sys.modules[FILTER_MODULE]
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_mir_regrid(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(module, "mir_regrid", fake_mir_regrid)
        fields = object()

        MIRRegrid(grid="O32").forward(fields)

        assert calls == [((fields, "O32", None), {"packing": "ccsds", "accuracy": 16})]


class TestIsLambertGrid:
    """Lambert grids are told apart from the other dicts."""

    @pytest.mark.parametrize(
        "grid", [MEPS_LAMBERT_GRID, MINIMAL_LAMBERT_GRID, {"gridType": "LAMBERT"}, {"gridType": "lambert"}]
    )
    def test_lambert(self, grid):
        assert is_lambert_grid(grid)

    @pytest.mark.parametrize(
        "grid",
        [
            "O32",
            "gridType=lambert",
            [1.0, 1.0],
            {"latitudes": [0.0], "longitudes": [0.0]},
            {"gridType": "regular_ll"},
            {},
            None,
        ],
    )
    def test_other_grids(self, grid):
        assert not is_lambert_grid(grid)

    def test_a_typed_grid_string_is_no_longer_special(self):
        """The typed-grid string interface is gone: it is a plain string for MIR (and fails there)."""
        for name in ("canonical_typed_grid", "is_typed_grid", "_TYPED_GRID_KEYS", "typed_grid_shape_of_the_earth"):
            assert not hasattr(backend, name)


class TestCanonicalLambertGrid:
    """Validation and canonical form of a Lambert grid given as GRIB keys."""

    def test_the_patch_is_valid_and_complete(self):
        grid = canonical_lambert_grid(MEPS_LAMBERT_GRID)

        assert list(grid) == list(MEPS_LAMBERT_GRID)
        for key, value in MEPS_LAMBERT_GRID.items():
            assert grid[key] == value, key

    def test_numbers_are_ints_and_floats_as_in_grib(self):
        grid = canonical_lambert_grid({**MEPS_LAMBERT_GRID, "Nx": 849.0, "DxInMetres": 2500, "LoVInDegrees": 15})

        assert type(grid["Nx"]) is int
        assert type(grid["DxInMetres"]) is float
        assert type(grid["LoVInDegrees"]) is float

    def test_the_order_of_the_keys_does_not_matter(self):
        """Equal grids have the same ``str`` (anemoi-inference compares ``str`` of the grid of each field)."""
        shuffled = dict(reversed(list(MEPS_LAMBERT_GRID.items())))
        assert list(shuffled) != list(MEPS_LAMBERT_GRID)

        assert str(canonical_lambert_grid(shuffled)) == str(canonical_lambert_grid(MEPS_LAMBERT_GRID))

    def test_the_defaults(self):
        grid = canonical_lambert_grid(MINIMAL_LAMBERT_GRID)

        assert grid["Latin1InDegrees"] == grid["Latin2InDegrees"] == grid["LaDInDegrees"]
        assert grid["scanningMode"] == 64
        assert grid["shapeOfTheEarth"] == 6
        assert grid["projectionCentreFlag"] == 0
        assert grid["resolutionAndComponentFlags"] == 48
        assert (grid["latitudeOfSouthernPoleInDegrees"], grid["longitudeOfSouthernPoleInDegrees"]) == (-90.0, 0.0)
        assert "scaleFactorOfRadiusOfSphericalEarth" not in grid
        assert "scaledValueOfRadiusOfSphericalEarth" not in grid

    def test_the_input_is_not_modified(self):
        grid = dict(MINIMAL_LAMBERT_GRID)

        canonical_lambert_grid(grid)

        assert grid == MINIMAL_LAMBERT_GRID

    def test_grib_keys_that_are_ignored(self):
        """``gridDefinitionTemplateNumber`` 30 and ``tablesVersion`` of a GRIB message are accepted and dropped."""
        grid = canonical_lambert_grid({**MEPS_LAMBERT_GRID, "gridDefinitionTemplateNumber": 30, "tablesVersion": 32})

        assert grid == canonical_lambert_grid(MEPS_LAMBERT_GRID)

    def test_another_template_number_raises(self):
        with pytest.raises(ValueError, match="gridDefinitionTemplateNumber must be 30"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "gridDefinitionTemplateNumber": 20})

    @pytest.mark.parametrize(
        "key",
        [
            "Nx",
            "Ny",
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "LaDInDegrees",
            "LoVInDegrees",
            "DxInMetres",
            "DyInMetres",
        ],
    )
    def test_missing_required_key_raises(self, key):
        grid = {k: v for k, v in MEPS_LAMBERT_GRID.items() if k != key}

        with pytest.raises(ValueError, match=rf"missing required key\(s\) \['{key}'\]"):
            canonical_lambert_grid(grid)

    @pytest.mark.parametrize("key", ["Ni", "Bogus", "latitudes", "radius", "grid"])
    def test_unknown_key_raises(self, key):
        with pytest.raises(ValueError, match=f"unknown key.*{key}"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, key: 1})

    def test_not_a_lambert_grid_raises(self):
        with pytest.raises(ValueError, match="Not a Lambert grid"):
            canonical_lambert_grid({"latitudes": [1.0], "longitudes": [2.0]})

    @pytest.mark.parametrize("value", ["849", None, True, [849], float("nan"), float("inf")])
    def test_not_a_number_raises(self, value):
        with pytest.raises(ValueError, match="Nx must be"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "Nx": value})

    def test_a_non_integer_count_raises(self):
        with pytest.raises(ValueError, match="Nx must be an integer"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "Nx": 849.5})

    @pytest.mark.parametrize(("key", "value"), [("Nx", 0), ("Ny", 0), ("DxInMetres", 0), ("DyInMetres", -2500)])
    def test_size_and_steps_must_be_positive(self, key, value):
        with pytest.raises(ValueError, match="at least 1|must be positive"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, key: value})

    @pytest.mark.parametrize("lad", [0, 90, -90, 100])
    def test_lad_range(self, lad):
        with pytest.raises(ValueError, match="LaDInDegrees must be between"):
            canonical_lambert_grid({**MINIMAL_LAMBERT_GRID, "LaDInDegrees": lad})

    @pytest.mark.parametrize(
        "parallels",
        [
            {"Latin1InDegrees": 50.0, "Latin2InDegrees": 58.0},
            {"Latin1InDegrees": 50.0},
            {"Latin2InDegrees": 50.0},
            {"Latin1InDegrees": 50.0, "Latin2InDegrees": 50.0},
        ],
    )
    def test_secant_grids_raise(self, parallels):
        """Only tangent grids are supported: the parallels (default LaD) must be LaD."""
        grid = {**MINIMAL_LAMBERT_GRID, **parallels}

        with pytest.raises(ValueError, match="only tangent grids are supported"):
            canonical_lambert_grid(grid)
        with pytest.raises(ValueError, match="only tangent grids are supported"):
            MIRRegrid(grid=grid)
        with pytest.raises(ValueError, match="only tangent grids are supported"):
            mir_regrid(ekd.SimpleFieldList([None]), grid)

    def test_the_parallels_follow_lad(self):
        grid = canonical_lambert_grid({**MINIMAL_LAMBERT_GRID, "LaDInDegrees": 45, "Latin1InDegrees": 45})

        assert grid["Latin1InDegrees"] == grid["Latin2InDegrees"] == 45.0

    @pytest.mark.parametrize("mode", [0, 128, 192, 65])
    def test_other_scanning_modes_raise(self, mode):
        """Only south-west first, rows north: ecCodes places the points of other scanning modes wrongly."""
        with pytest.raises(ValueError, match="only scanningMode=64"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "scanningMode": mode})

    def test_projection_centre_flag(self):
        with pytest.raises(ValueError, match="projectionCentreFlag=0"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "projectionCentreFlag": 128})

    @pytest.mark.parametrize(
        "pole", [{"latitudeOfSouthernPoleInDegrees": -60.0}, {"longitudeOfSouthernPoleInDegrees": 10.0}]
    )
    def test_rotated_pole_raises(self, pole):
        with pytest.raises(ValueError, match="southern pole at"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, **pole})

    @pytest.mark.parametrize("flags", [-1, 256])
    def test_resolution_and_component_flags_range(self, flags):
        with pytest.raises(ValueError, match="resolutionAndComponentFlags must be between"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "resolutionAndComponentFlags": flags})

    @pytest.mark.parametrize("label", EARTHS)
    def test_supported_earths(self, label):
        earth, radius = EARTHS[label]
        grid = canonical_lambert_grid({**MINIMAL_LAMBERT_GRID, **earth})

        assert grid["shapeOfTheEarth"] == earth["shapeOfTheEarth"]
        assert lambert_radius(grid) == radius
        assert ("scaledValueOfRadiusOfSphericalEarth" in grid) == (earth["shapeOfTheEarth"] == 1)

    @pytest.mark.parametrize("shape", [2, 3, 4, 5, 7, 9, 10, 255])
    def test_unsupported_shape_raises(self, shape):
        """Ellipsoids and the shapes that need axes are refused."""
        with pytest.raises(ValueError, match=f"shapeOfTheEarth={shape} is not supported"):
            canonical_lambert_grid({**MINIMAL_LAMBERT_GRID, "shapeOfTheEarth": shape})

    @pytest.mark.parametrize("missing", ["scaleFactorOfRadiusOfSphericalEarth", "scaledValueOfRadiusOfSphericalEarth"])
    def test_shape_1_needs_the_radius_keys(self, missing):
        grid = {k: v for k, v in MEPS_LAMBERT_GRID.items() if k != missing}

        with pytest.raises(ValueError, match=f"shapeOfTheEarth=1 needs \\['{missing}'\\]"):
            canonical_lambert_grid(grid)

    def test_radius_keys_need_shape_1(self):
        with pytest.raises(ValueError, match="only go with shapeOfTheEarth=1"):
            canonical_lambert_grid({**MEPS_LAMBERT_GRID, "shapeOfTheEarth": 6})

    def test_scaled_radius(self):
        earth, radius = EARTHS["shape-1-scaled"]

        assert lambert_radius({**MINIMAL_LAMBERT_GRID, **earth}) == radius == 6370000.0


class TestLambertPoints:
    """``lambert_points``: spherical LCC (pyproj), the south-west corner first and the rows running north."""

    @pytest.mark.parametrize("label", EARTHS)
    def test_points_equal_pyproj(self, label):
        grid = lambert_grid(**EARTHS[label][0])
        latitudes, longitudes = pyproj_points(grid)

        points_latitudes, points_longitudes = lambert_points(grid)

        np.testing.assert_allclose(points_latitudes, latitudes, atol=1e-9, rtol=0)
        np.testing.assert_allclose(signed_longitude(points_longitudes - longitudes), 0.0, atol=1e-9, rtol=0)

    def test_the_meps_corners(self):
        """The four corners of the MEPS grid, as computed by the hand-written LCC maths this replaced."""
        latitudes, longitudes = lambert_points(MEPS_LAMBERT_GRID)

        assert latitudes.shape == longitudes.shape == (849 * 969,)
        corners = {  # index: (latitude, longitude), south-west, south-east, north-west, north-east
            0: (51.633816, 1.60491),
            848: (51.119236870119316, 31.82800969334004),
            -849: (72.28764077435638, -13.150142048117967),
            -1: (71.24633296236979, 49.41830433393738),
        }
        for index, (latitude, longitude) in corners.items():
            assert latitudes[index] == pytest.approx(latitude, abs=1e-9)
            assert longitudes[index] == pytest.approx(longitude, abs=1e-9)

    def test_the_first_point_is_the_given_one_and_the_rows_run_north(self):
        latitudes, longitudes = lambert_points(LAMBERT_GRID)

        assert latitudes[0] == pytest.approx(LAMBERT_GRID["latitudeOfFirstGridPointInDegrees"], abs=1e-9)
        assert longitudes[0] == pytest.approx(LAMBERT_GRID["longitudeOfFirstGridPointInDegrees"], abs=1e-9)
        assert latitudes[-1] > latitudes[0]
        assert latitudes[1] != latitudes[0]  # a Lambert row is not along a parallel
        assert longitudes[1] > longitudes[0]

    def test_the_radius_can_be_given(self):
        default = lambert_points(LAMBERT_GRID)[0]
        other = lambert_points(LAMBERT_GRID, 6371229.0)[0]

        assert not np.array_equal(default, other)
        assert np.abs(default - other).max() < 1e-2


class TestMirLambertGrid:
    """``_mir_lambert_grid``: the grid asked of MIR compensates for what MIR does, by LCC maths (no MIR)."""

    @staticmethod
    def parse(grid: str) -> dict[str, str]:
        return dict(item.split("=", 1) for item in grid.split(";"))

    def test_the_meps_grid(self):
        """The north-west corner from the south-west corner at R=6371000, the steps scaled by 6371229/6371000."""
        keys = self.parse(backend._mir_lambert_grid(canonical_lambert_grid(MEPS_LAMBERT_GRID), True))

        assert keys["gridType"] == "lambert"
        assert (keys["LaDInDegrees"], keys["LoVInDegrees"]) == ("63.3", "15.0")
        assert (keys["Latin1InDegrees"], keys["Latin2InDegrees"]) == ("63.3", "63.3")
        assert (keys["Ni"], keys["Nj"]) == ("849", "969")
        assert keys["writeLonPositive"] == "1"
        assert keys["writeLaDInDegrees"] == "1"
        dx, dy = (float(v) for v in keys["grid"].split("/"))
        assert dx == dy == pytest.approx(2500.0 * 6371229.0 / 6371000.0, rel=1e-14)
        assert float(keys["latitudeOfFirstGridPointInDegrees"]) == pytest.approx(72.28764077435638, abs=1e-9)
        assert float(keys["longitudeOfFirstGridPointInDegrees"]) == pytest.approx(-13.150142048117967, abs=1e-9)

    def test_no_lad_key_without_with_lad(self):
        """GRIB1 input does not take ``writeLaDInDegrees``."""
        grid = canonical_lambert_grid(MEPS_LAMBERT_GRID)

        assert "writeLaDInDegrees" not in self.parse(backend._mir_lambert_grid(grid, False))
        assert backend._mir_lambert_grid(grid, True) == backend._mir_lambert_grid(grid, False) + ";writeLaDInDegrees=1"

    def test_a_grid_on_the_sphere_of_mir_needs_no_scaling(self):
        keys = self.parse(backend._mir_lambert_grid(canonical_lambert_grid(lambert_grid(**EARTHS["shape-6"][0])), True))

        assert keys["grid"] == "25000.0/25000.0"

    @pytest.mark.parametrize("label", EARTHS)
    def test_mir_places_the_points_of_the_grid(self, label):
        """MIR projects the first point it is given and steps on a sphere of R=6371229: these are the points of the grid.

        MIR's construction is simulated (it is verified with real MIR in ``test_mir.py``); the points are
        compared with the exact points of the grid on its own earth, the error is far below a millimetre.
        """
        grid = canonical_lambert_grid(lambert_grid(Nx=20, Ny=25, **EARTHS[label][0]))
        keys = self.parse(backend._mir_lambert_grid(grid, True))
        mir = backend.lambert_projection(
            float(keys["LaDInDegrees"]), float(keys["LoVInDegrees"]), backend.MIR_LAMBERT_RADIUS
        )
        dx, dy = (float(v) for v in keys["grid"].split("/"))
        x0, y0 = mir(
            float(keys["longitudeOfFirstGridPointInDegrees"]), float(keys["latitudeOfFirstGridPointInDegrees"])
        )
        x = x0 + dx * np.arange(20)[None, :] + np.zeros((25, 1))
        y = y0 - dy * np.arange(25)[:, None] + np.zeros((1, 20))  # the rows run south
        mir_longitudes, mir_latitudes = mir(x, y, inverse=True)

        latitudes, longitudes = lambert_points(grid)

        error = distance_m(mir_latitudes[::-1].ravel(), mir_longitudes[::-1].ravel(), latitudes, longitudes)
        assert error.max() < 1e-3

    def test_a_wrong_radius_would_not_be_compensated(self):
        """Control of the test above: the points of the uncompensated grid are off by 100 m."""
        grid = canonical_lambert_grid(MEPS_LAMBERT_GRID)
        keys = self.parse(backend._mir_lambert_grid(grid, True))
        mir = backend.lambert_projection(63.3, 15.0, backend.MIR_LAMBERT_RADIUS)
        x0, y0 = mir(
            float(keys["longitudeOfFirstGridPointInDegrees"]), float(keys["latitudeOfFirstGridPointInDegrees"])
        )
        # the last point of the last row, with the unscaled step
        longitude, latitude = mir(x0 + 2500.0 * 848, y0 - 2500.0 * 968, inverse=True)

        exact_latitudes, exact_longitudes = lambert_points(grid)
        assert distance_m(latitude, longitude, exact_latitudes[848], exact_longitudes[848]) > 100.0


NI, NJ = 10, 8
#: ``missingValue`` of the test messages with a bitmap.
MISSING_VALUE = 9999.0
#: The grid the messages of ``lambert_message`` are made for.
SMALL_LAMBERT_GRID = lambert_grid(Nx=NI, Ny=NJ)
#: Columns of the values of the message, with the rows reversed (what the output has).
FLIPPED = np.arange(1, NI * NJ + 1).reshape(NJ, NI)[::-1].ravel()


def lambert_message(
    shape: int = 6, packing: str | None = None, missing: list[int] | str | None = None, **keys: Any
) -> bytes:
    """A GRIB2 Lambert message as MIR writes it for ``SMALL_LAMBERT_GRID`` (made with ecCodes, without MIR).

    The values are 1 to 80 in the order of the message, the first point is the top-left corner and the rows run south.

    Parameters
    ----------
    shape : int
        ``shapeOfTheEarth``.
    packing : str, optional
        ``packingType`` (``grid_simple``, ``grid_ccsds``, ``grid_ieee``), the sample's by default.
    missing : list[int] or ``"all"``, optional
        Indices of the missing values (with a bitmap), or ``"all"``.
    **keys
        Further keys to set.

    Returns
    -------
    bytes
        The encoded GRIB2 message.
    """
    import eccodes

    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(handle, "gridDefinitionTemplateNumber", 30)
        for key, value in {
            "shapeOfTheEarth": shape,
            "Ni": NI,
            "Nj": NJ,
            "DxInMetres": 25000,
            "DyInMetres": 25000,
            "LaDInDegrees": 63.3,
            "LoVInDegrees": 15.0,
            "Latin1InDegrees": 63.3,
            "Latin2InDegrees": 63.3,
            "latitudeOfFirstGridPointInDegrees": 66.0,
            "longitudeOfFirstGridPointInDegrees": 346.0,
            "jScansPositively": 0,
            **keys,
        }.items():
            eccodes.codes_set(handle, key, value)
        if packing is not None:
            eccodes.codes_set(handle, "packingType", packing)
        values = np.arange(1, NI * NJ + 1, dtype=float)
        if missing is not None:
            eccodes.codes_set(handle, "missingValue", MISSING_VALUE)
            eccodes.codes_set(handle, "bitmapPresent", 1)
            values[slice(None) if missing == "all" else missing] = MISSING_VALUE
        eccodes.codes_set_values(handle, values)
        return eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)


#: Decodes a message on stdin in a process with ``ECCODES_ECKIT_GEO=0`` (without the regrid package, which sets it).
_DECODE_CODE = """
import json, sys
import eccodes

handle = eccodes.codes_new_from_message(sys.stdin.buffer.read())
print(json.dumps({
    "latitudes": eccodes.codes_get_array(handle, "latitudes").tolist(),
    "longitudes": eccodes.codes_get_array(handle, "longitudes").tolist(),
}))
"""


def decoded_points(message: bytes) -> tuple[np.ndarray, np.ndarray]:
    """The points ecCodes reports for a message (``ECCODES_ECKIT_GEO=0`` is needed for a Lambert grid)."""
    proc = subprocess.run(
        [sys.executable, "-c", _DECODE_CODE],
        input=message,
        capture_output=True,
        env={**os.environ, "ECCODES_ECKIT_GEO": "0"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    report = json.loads(proc.stdout)
    return np.array(report["latitudes"]), np.array(report["longitudes"])


class TestLambertOutput:
    """``_lambert_output`` rewrites a MIR Lambert message to the header of the grid (no MIR needed)."""

    @staticmethod
    def rewrite(message: bytes, grid: dict[str, Any] = SMALL_LAMBERT_GRID) -> Any:
        return ekd.from_source("memory", backend._lambert_output(message, canonical_lambert_grid(grid)))[0]

    def test_rows_are_reversed(self):
        out = self.rewrite(lambert_message())

        np.testing.assert_array_equal(out.values, FLIPPED)

    @pytest.mark.parametrize("label", EARTHS)
    def test_the_header_is_the_grid(self, label):
        """The header keys are those of the grid, whatever MIR wrote (here another earth and first point)."""
        grid = lambert_grid(Nx=NI, Ny=NJ, resolutionAndComponentFlags=56, DxInMetres=25001.5, **EARTHS[label][0])

        out = self.rewrite(lambert_message(shape=0), grid)

        for key, value in canonical_lambert_grid(grid).items():
            assert out.metadata(key) == pytest.approx(value, abs=1e-9), key
        assert out.metadata("jScansPositively") == 1
        assert out.metadata("iScansNegatively") == 0
        assert out.metadata("gridType") == "lambert"

    def test_the_first_point_is_given_positive(self):
        grid = lambert_grid(Nx=NI, Ny=NJ, longitudeOfFirstGridPointInDegrees=-15.0)

        out = self.rewrite(lambert_message(), grid)

        assert out.metadata("longitudeOfFirstGridPointInDegrees") == pytest.approx(345.0)

    @pytest.mark.parametrize("label", EARTHS)
    def test_ecc_points_are_the_points_of_the_grid(self, label):
        """The points ecCodes decodes from the rewritten message are the exact points of the grid, south to north."""
        grid = lambert_grid(Nx=NI, Ny=NJ, **EARTHS[label][0])

        latitudes, longitudes = decoded_points(backend._lambert_output(lambert_message(), canonical_lambert_grid(grid)))

        expected_latitudes, expected_longitudes = pyproj_points(grid)
        np.testing.assert_allclose(latitudes, expected_latitudes, atol=1e-6, rtol=0)
        np.testing.assert_allclose(signed_longitude(longitudes - expected_longitudes), 0.0, atol=1e-6, rtol=0)

    def test_the_rest_of_the_message_is_unchanged(self):
        before = ekd.from_source("memory", lambert_message())[0]
        out = self.rewrite(lambert_message())

        for key in ("gridType", "Ni", "Nj", "paramId", "dataDate", "edition", "numberOfDataPoints"):
            assert out.metadata(key) == before.metadata(key), key
        for key in ("iScansNegatively", "alternativeRowScanning", "jPointsAreConsecutive"):
            assert out.metadata(key) == before.metadata(key), key

    @pytest.mark.parametrize("packing", ["grid_simple", "grid_ccsds", "grid_ieee"])
    def test_missing_values_are_reversed_with_the_values(self, packing):
        """The bitmap is kept: the same number of missing points, at the flipped positions, other values unchanged."""
        missing = [3, 40, 79]
        before = ekd.from_source("memory", lambert_message(packing=packing, missing=missing))[0]
        assert before.metadata("numberOfMissing") == 3

        out = self.rewrite(lambert_message(packing=packing, missing=missing))

        expected = before.values.reshape(NJ, NI)[::-1].ravel()
        assert out.metadata("packingType") == packing
        assert out.metadata("bitmapPresent") == 1
        assert out.metadata("numberOfMissing") == 3
        assert np.flatnonzero(np.isnan(out.values)).tolist() == [9, 30, 73]
        np.testing.assert_array_equal(np.isnan(out.values), np.isnan(expected))
        np.testing.assert_allclose(out.values[~np.isnan(expected)], expected[~np.isnan(expected)])
        assert out.metadata("jScansPositively") == 1

    @pytest.mark.parametrize("packing", ["grid_simple", "grid_ccsds", "grid_ieee"])
    def test_all_missing_stays_all_missing(self, packing):
        out = self.rewrite(lambert_message(packing=packing, missing="all"))

        assert out.metadata("packingType") == packing
        assert out.metadata("numberOfMissing") == NI * NJ
        assert np.isnan(out.values).all()
        assert out.metadata("jScansPositively") == 1

    @pytest.mark.parametrize("packing", ["grid_simple", "grid_ccsds", "grid_ieee"])
    def test_no_bitmap_stays_without_a_bitmap(self, packing):
        out = self.rewrite(lambert_message(packing=packing))

        assert out.metadata("bitmapPresent") == 0
        assert out.metadata("packingType") == packing
        np.testing.assert_allclose(out.values, FLIPPED)

    @pytest.mark.parametrize("key", ["alternativeRowScanning", "jPointsAreConsecutive", "iScansNegatively"])
    def test_other_scanning_is_refused(self, key):
        with pytest.raises(ValueError, match=f"{key}=1 cannot be rewritten"):
            self.rewrite(lambert_message(**{key: 1}))

    def test_a_message_scanning_north_is_refused(self):
        """MIR 1.30.2 writes ``jScansPositively=0``: another MIR needs the compensation checked."""
        with pytest.raises(RuntimeError, match="jScansPositively=1"):
            self.rewrite(lambert_message(jScansPositively=1))

    def test_a_message_of_another_size_is_refused(self):
        with pytest.raises(RuntimeError, match="expected 12x8"):
            self.rewrite(lambert_message(), lambert_grid(Nx=12, Ny=NJ))


class TestMirRegridLambert:
    """``mir_regrid`` hands the compensated string to ``mir.Job`` with a stubbed ``mir`` module."""

    class _Stop(Exception):
        """Raised by the fake job to end ``mir_regrid`` once the job is built."""

    @staticmethod
    def _install_fake_mir(monkeypatch, jobs: list[dict[str, Any]], inputs: list[bytes] | None = None) -> None:
        class Job:
            def __init__(self, **kwargs: Any) -> None:
                jobs.append(kwargs)

            def execute(self, *args: Any) -> None:
                raise TestMirRegridLambert._Stop

        def grib_input(buffer: io.BytesIO) -> None:
            if inputs is not None:
                inputs.append(buffer.getvalue())

        fake = SimpleNamespace(Job=Job, PyGribInput=grib_input, PyGribOutput=lambda *a: None, Grid=None)
        monkeypatch.setitem(sys.modules, "mir", fake)

    @staticmethod
    def _grib1_fields() -> Any:
        import eccodes

        handle = eccodes.codes_grib_new_from_samples("reduced_gg_pl_32_grib1")
        try:
            return ekd.from_source("memory", eccodes.codes_get_message(handle))
        finally:
            eccodes.codes_release(handle)

    def test_the_compensated_string_reaches_job(self, monkeypatch, grib_fieldlist):
        """The string is the one of ``_mir_lambert_grid``; ``_make_mir_grid`` is not used."""
        jobs: list[dict[str, Any]] = []
        self._install_fake_mir(monkeypatch, jobs)

        def fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("_make_mir_grid must not be used for a Lambert grid")

        monkeypatch.setattr(backend, "_make_mir_grid", fail)

        with pytest.raises(self._Stop):
            mir_regrid(grib_fieldlist(grid="O32"), MEPS_LAMBERT_GRID, packing="simple", accuracy=12)

        expected = backend._mir_lambert_grid(canonical_lambert_grid(MEPS_LAMBERT_GRID), True)
        assert jobs == [{"grid": expected, "edition": 2, "packing": "simple", "accuracy": 12}]
        assert "writeLaDInDegrees=1" in expected

    def test_grib1_input_gets_a_string_without_lad(self, monkeypatch):
        """MIR fails writing LaD for GRIB1 input: no ``writeLaDInDegrees``, and the input is not changed."""
        jobs: list[dict[str, Any]] = []
        inputs: list[bytes] = []
        self._install_fake_mir(monkeypatch, jobs, inputs)
        fields = self._grib1_fields()

        with pytest.raises(self._Stop):
            mir_regrid(fields, MEPS_LAMBERT_GRID)

        assert jobs[0]["grid"] == backend._mir_lambert_grid(canonical_lambert_grid(MEPS_LAMBERT_GRID), False)
        assert "writeLaDInDegrees" not in jobs[0]["grid"]
        assert jobs[0]["edition"] == 2
        assert inputs == [fields[0].message()]

    def test_input_earth_is_shape_6_on_a_copy(self, monkeypatch, grib_fieldlist):
        """MIR is given a copy of the input with the sphere it places the points on, so the scaling stays right."""
        jobs: list[dict[str, Any]] = []
        inputs: list[bytes] = []
        self._install_fake_mir(monkeypatch, jobs, inputs)
        fields = ekd.FieldList.from_fields([grib_fieldlist(grid="O32")[0].clone(metadata={"shapeOfTheEarth": 5})])
        message = fields[0].message()

        with pytest.raises(self._Stop):
            mir_regrid(fields, MEPS_LAMBERT_GRID)

        assert ekd.from_source("memory", inputs[0])[0].metadata("shapeOfTheEarth") == 6
        assert fields[0].message() == message
        assert fields[0].metadata("shapeOfTheEarth") == 5

    def test_invalid_grid_raises_before_mir(self, monkeypatch, grib_fieldlist):
        jobs: list[dict[str, Any]] = []
        self._install_fake_mir(monkeypatch, jobs)

        with pytest.raises(ValueError, match="shapeOfTheEarth=5 is not supported"):
            mir_regrid(grib_fieldlist(grid="O32"), {**MEPS_LAMBERT_GRID, "shapeOfTheEarth": 5})

        assert jobs == []

    @pytest.mark.parametrize("area", [[50, 0, 40, 10], "50/0/40/10"])
    def test_area_raises_before_mir(self, monkeypatch, grib_fieldlist, area):
        """MIR cannot crop a Lambert grid, whose keys define the domain: an area is refused, no job is made."""
        jobs: list[dict[str, Any]] = []
        self._install_fake_mir(monkeypatch, jobs)

        with pytest.raises(ValueError, match="cannot be combined with a Lambert grid"):
            mir_regrid(grib_fieldlist(grid="O32"), LAMBERT_GRID, area=area)

        assert jobs == []

    @pytest.mark.parametrize("area", [None, [], ()])
    def test_no_area_is_accepted(self, monkeypatch, grib_fieldlist, area):
        jobs: list[dict[str, Any]] = []
        self._install_fake_mir(monkeypatch, jobs)

        with pytest.raises(self._Stop):
            mir_regrid(grib_fieldlist(grid="O32"), LAMBERT_GRID, area=area)

        assert "area" not in jobs[0]

    @staticmethod
    def _regrid_with_stubbed_mir(
        monkeypatch, fields: Any, message: bytes, grid: dict[str, Any], jobs: list | None = None
    ):
        class Job:
            def __init__(self, **kwargs: Any) -> None:
                if jobs is not None:
                    jobs.append(kwargs)

            def execute(self, input_buffer: Any, output_buffer: Any) -> None:
                output_buffer.write(message)

        fake = SimpleNamespace(
            Job=Job, PyGribInput=lambda buffer: buffer, PyGribOutput=lambda buffer: buffer, Grid=None
        )
        monkeypatch.setitem(sys.modules, "mir", fake)
        return mir_regrid(fields, grid)

    def test_mir_regrid_rewrites_what_mir_writes(self, monkeypatch, grib_fieldlist):
        """``mir_regrid`` gives the rewritten message (stubbed MIR writing a ``jScansPositively=0`` message)."""
        out = self._regrid_with_stubbed_mir(
            monkeypatch, grib_fieldlist(grid="O32"), lambert_message(), SMALL_LAMBERT_GRID
        )[0]

        assert out.metadata("jScansPositively") == 1
        assert out.metadata("scanningMode") == 64
        assert out.metadata("shapeOfTheEarth") == 1
        np.testing.assert_array_equal(out.values, FLIPPED)
        assert out.metadata().geography.mars_grid() == canonical_lambert_grid(SMALL_LAMBERT_GRID)

    def test_one_job_per_edition(self, monkeypatch, grib_fieldlist):
        """Fields of both GRIB editions get their own job, each created once."""
        jobs: list[dict[str, Any]] = []
        fields = ekd.FieldList.from_fields(
            [grib_fieldlist(grid="O32")[0], self._grib1_fields()[0], grib_fieldlist(grid="O32")[0]]
        )

        out = self._regrid_with_stubbed_mir(monkeypatch, fields, lambert_message(), SMALL_LAMBERT_GRID, jobs)

        assert len(out) == 3
        assert [("writeLaDInDegrees=1" in job["grid"]) for job in jobs] == [True, False]


class TestLambertGeography:
    """The geography override of ``mir_regrid`` outputs on Lambert grids, on a real field (without MIR)."""

    GRID = canonical_lambert_grid(LAMBERT_GRID)

    def test_mars_grid_and_area_overridden(self, grib_fieldlist):
        """``mars_grid()`` and ``mars_area()`` give the Lambert grid and the area."""
        field = grib_fieldlist(grid="O32")[0]

        out = _with_lambert_geography(field, self.GRID, (60, -10, 45, 20))

        assert out.metadata().geography.mars_grid() == self.GRID
        assert out.metadata().geography.mars_area() == [60, -10, 45, 20]
        assert out.mars_grid == self.GRID
        assert out.mars_area == [60, -10, 45, 20]

    def test_mars_grid_is_a_copy(self, grib_fieldlist):
        out = _with_lambert_geography(grib_fieldlist(grid="O32")[0], self.GRID, None)

        out.metadata().geography.mars_grid()["Nx"] = 1

        assert out.metadata().geography.mars_grid() == self.GRID

    @pytest.mark.parametrize("area", [None, [], ()])
    def test_no_area_is_none(self, grib_fieldlist, area):
        """Without an area ``mars_area()`` is None."""
        field = grib_fieldlist(grid="O32")[0]

        out = _with_lambert_geography(field, self.GRID, area)

        assert out.metadata().geography.mars_area() is None

    def test_everything_else_unchanged(self, grib_fieldlist):
        """Values, other metadata, the geography and the encoded message are those of the field."""
        field = grib_fieldlist(grid="O32")[0]

        out = _with_lambert_geography(field, self.GRID, None)

        np.testing.assert_array_equal(out.values, field.values)
        assert out.metadata("paramId") == field.metadata("paramId")
        assert out.metadata("gridType") == field.metadata("gridType")
        np.testing.assert_array_equal(out.grid_points()[0], field.grid_points()[0])
        assert out.metadata().geography.shape() == field.metadata().geography.shape()
        assert out.message() == field.message()

    def test_clone_and_write_keep_the_override(self, grib_fieldlist):
        """A clone with new values keeps the override, and the field can be written as GRIB."""
        field = grib_fieldlist(grid="O32")[0]
        out = _with_lambert_geography(field, self.GRID, None)

        clone = out.clone(values=out.values * 2)
        buffer = io.BytesIO()
        clone.to_target("file", buffer)

        assert clone.metadata().geography.mars_grid() == self.GRID
        np.testing.assert_allclose(clone.values, 2 * field.values)
        assert ekd.from_source("memory", buffer.getvalue())[0].metadata("paramId") == field.metadata("paramId")

    def test_pickle_keeps_the_override(self, grib_fieldlist):
        """The metadata pickles with the override."""
        metadata = _with_lambert_geography(grib_fieldlist(grid="O32")[0], self.GRID, [1, 2, 3, 4])._metadata

        restored = pickle.loads(pickle.dumps(metadata))

        assert restored.geography.mars_grid() == self.GRID
        assert restored.geography.mars_area() == [1, 2, 3, 4]

    def test_equal_grids_in_different_key_order_have_the_same_str_on_every_field(self, monkeypatch, grib_fieldlist):
        """anemoi-inference compares ``str(mars_grid())`` of the fields: one grid, whatever the order of the keys.

        The grids reach the fields through ``MIRRegrid``, which gives the canonical form.
        """
        shuffled = dict(reversed(list(MEPS_LAMBERT_GRID.items())))
        fields = grib_fieldlist(grid="O32", nfields=2)

        grids = []
        for grid, field in zip((MEPS_LAMBERT_GRID, shuffled), fields):
            grids.append(
                _with_lambert_geography(field, MIRRegrid(grid=grid).grid, None).metadata().geography.mars_grid()
            )

        assert grids[0] == grids[1]
        assert len({str(grid) for grid in grids}) == 1


class TestGridRepr:
    """Tests for ``grid_repr`` which renders grid specs for logging."""

    def test_string_grid_uppercased(self):
        """A string grid specification is uppercased."""
        assert grid_repr("n320") == "N320"
        assert grid_repr("O32") == "O32"

    def test_lambert_grid_shows_the_keys(self):
        """A Lambert grid is shown with its (scalar) values, in the canonical form."""
        result = grid_repr(MINIMAL_LAMBERT_GRID)

        assert result == str(canonical_lambert_grid(MINIMAL_LAMBERT_GRID))
        assert "'Nx': 849" in result
        assert "list of len" not in result

    def test_short_list_grid_shown_in_full(self):
        """A short list grid is shown as-is."""
        assert grid_repr([0.25, 0.25]) == "[0.25, 0.25]"

    def test_long_list_grid_truncated(self):
        """A long list grid is truncated with an ellipsis."""
        result = grid_repr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        assert "..." in result

    def test_dict_grid_summarised(self):
        """A coordinate dict grid is summarised by coordinate-list lengths."""
        result = grid_repr({"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]})

        assert "latitudes" in result
        assert "list of len 2" in result

    def test_scalar_dict_grid_is_shown(self):
        """A dict with scalar values is not summarised with ``len`` (which fails for numbers)."""
        assert grid_repr({"grid": "O32", "n": 3}) == str({"grid": "O32", "n": 3})


class TestGridSpecType:
    """Tests for the GridSpec type alias."""

    def test_string_is_valid(self):
        """Strings are valid GridSpec values."""
        grid: GridSpec = "O32"

        assert isinstance(grid, str)

    def test_list_is_valid(self):
        """Lists of floats are valid GridSpec values."""
        grid: GridSpec = [0.25, 0.25]

        assert isinstance(grid, list)

    def test_dict_is_valid(self):
        """Dicts with list values are valid GridSpec values."""
        grid: GridSpec = {"latitudes": [1.0], "longitudes": [2.0]}

        assert isinstance(grid, dict)

    def test_scalar_dict_is_valid(self):
        """Dicts with scalar values (a Lambert grid) are valid GridSpec values."""
        grid: GridSpec = MEPS_LAMBERT_GRID

        assert isinstance(grid, dict)


class TestKnownGrids:
    """Tests for ``KNOWN_GRIDS``."""

    def test_known_grids_is_list(self):
        """KNOWN_GRIDS is populated from package resources."""
        assert isinstance(KNOWN_GRIDS, list)

    def test_meps_in_known_grids(self):
        """The 'meps' grid should be available as a named grid."""
        assert "meps" in KNOWN_GRIDS

    def test_no_non_grid_entries(self):
        """KNOWN_GRIDS should not contain non-grid entries like __init__.py or .gitignore."""
        for name in KNOWN_GRIDS:
            assert not name.startswith(".")
            assert not name.startswith("_")
            assert not name.endswith(".py")


class TestNamedRegrid:
    """Tests for ``NamedRegrid``."""

    def test_unknown_grid_raises(self):
        """Instantiating with an unknown grid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown grid name"):
            NamedRegrid("nonexistent_grid_name")

    def test_case_insensitive(self):
        """Grid names are case-insensitive."""
        grid = NamedRegrid("MEPS")

        assert grid.name == "meps"
        assert len(grid.latitudes) > 0

    def test_meps_latitudes(self):
        """Latitudes for the meps grid are loaded as a list of floats."""
        lats = NamedRegrid("meps").latitudes

        assert isinstance(lats, list)
        assert len(lats) > 0
        assert all(isinstance(v, float) for v in lats)
        assert all(-90 <= v <= 90 for v in lats)

    def test_meps_longitudes(self):
        """Longitudes for the meps grid are loaded as a list of floats."""
        lons = NamedRegrid("meps").longitudes

        assert isinstance(lons, list)
        assert len(lons) > 0
        assert all(isinstance(v, float) for v in lons)

    def test_meps_gridspec(self):
        """gridspec returns a dict with 'grid' containing latitudes and longitudes."""
        spec = NamedRegrid("meps").gridspec

        assert "grid" in spec
        assert "latitudes" in spec["grid"]
        assert "longitudes" in spec["grid"]
        assert len(spec["grid"]["latitudes"]) == len(spec["grid"]["longitudes"])

    def test_latitudes_and_longitudes_same_length(self):
        """Latitudes and longitudes arrays have the same length."""
        grid = NamedRegrid("meps")

        assert len(grid.latitudes) == len(grid.longitudes)
