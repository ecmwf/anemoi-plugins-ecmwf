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
the pure helpers ``grid_repr`` and ``GridSpec``, and the named grids
(``KNOWN_GRIDS``, ``NamedRegrid``).
Real MIR runs are in ``test_mir.py``.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.backend import GridSpec
from anemoi.plugins.ecmwf.transform.regrid.backend import grid_repr
from anemoi.plugins.ecmwf.transform.regrid.named import KNOWN_GRIDS
from anemoi.plugins.ecmwf.transform.regrid.named import NamedRegrid

#: The module of the ``MIRRegrid`` filter (the package re-exports the class under the module's name).
FILTER_MODULE = "anemoi.plugins.ecmwf.transform.regrid.regrid"


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


class TestGridRepr:
    """Tests for ``grid_repr`` which renders grid specs for logging."""

    def test_string_grid_uppercased(self):
        """A string grid specification is uppercased."""
        assert grid_repr("n320") == "N320"
        assert grid_repr("O32") == "O32"

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
