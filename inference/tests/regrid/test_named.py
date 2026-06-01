# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from anemoi.plugins.ecmwf.inference.regrid.named import KNOWN_GRIDS
from anemoi.plugins.ecmwf.inference.regrid.named import NamedRegrid


class TestKnownGrids:
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
        grid = NamedRegrid("meps")
        lats = grid.latitudes
        assert isinstance(lats, list)
        assert len(lats) > 0
        assert all(isinstance(v, float) for v in lats)
        assert all(-90 <= v <= 90 for v in lats)

    def test_meps_longitudes(self):
        """Longitudes for the meps grid are loaded as a list of floats."""
        grid = NamedRegrid("meps")
        lons = grid.longitudes
        assert isinstance(lons, list)
        assert len(lons) > 0
        assert all(isinstance(v, float) for v in lons)

    def test_meps_gridspec(self):
        """gridspec returns a dict with 'grid' containing latitudes and longitudes."""
        grid = NamedRegrid("meps")
        spec = grid.gridspec
        assert "grid" in spec
        assert "latitudes" in spec["grid"]
        assert "longitudes" in spec["grid"]
        assert len(spec["grid"]["latitudes"]) == len(spec["grid"]["longitudes"])

    def test_latitudes_and_longitudes_same_length(self):
        """Latitudes and longitudes arrays have the same length."""
        grid = NamedRegrid("meps")
        assert len(grid.latitudes) == len(grid.longitudes)
