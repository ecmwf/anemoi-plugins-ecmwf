# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the dynamics plugins: helper functions, RegressionPerturbation,
and SubtractTendency.

The helper tests (haversine, Gaspari-Cohn) are pure unit tests with no
external data dependencies.  The plugin tests use synthetic GRIB files
created in ``tmp_path`` so they run anywhere.
"""

from typing import cast
from unittest.mock import MagicMock

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.inference.context import Context

from anemoi.plugins.ecmwf.inference.dynamics.regression_perturbation import (
    RegressionPerturbationPlugin,
    _gaspari_cohn,
    _haversine,
)

# ---- unit tests for helpers ----


class TestHaversine:
    def test_zero_distance(self):
        assert _haversine(0, 0, 0, 0) == pytest.approx(0.0, abs=1e-10)

    def test_quarter_circumference(self):
        # 90 degrees along a meridian ≈ 10007 km
        d = _haversine(0, 0, 0, 90)
        assert 9900 < d < 10100

    def test_symmetry(self):
        d1 = _haversine(10, 20, 30, 40)
        d2 = _haversine(30, 40, 10, 20)
        np.testing.assert_allclose(d1, d2, rtol=1e-10)

    def test_antipodal(self):
        # Opposite sides of the globe ≈ half circumference ≈ 20015 km
        d = _haversine(0, 0, 180, 0)
        assert 19900 < d < 20200


class TestGaspariCohn:
    def test_max_at_center(self):
        lat = np.linspace(-90, 90, 500)
        lon = np.linspace(0, 360, 500)
        cov = _gaspari_cohn(lat, lon, 250, 250, 2000.0)
        assert cov[250] == pytest.approx(1.0)

    def test_zero_far_away(self):
        lat = np.linspace(-90, 90, 500)
        lon = np.linspace(0, 360, 500)
        cov = _gaspari_cohn(lat, lon, 250, 250, 100.0)
        assert cov[0] == 0.0
        assert cov[-1] == 0.0

    def test_no_negatives(self):
        lat = np.linspace(-90, 90, 1000)
        lon = np.linspace(0, 360, 1000)
        cov = _gaspari_cohn(lat, lon, 500, 500, 3000.0)
        assert np.all(cov >= 0.0)

    def test_bounded_by_one(self):
        lat = np.linspace(-90, 90, 1000)
        lon = np.linspace(0, 360, 1000)
        cov = _gaspari_cohn(lat, lon, 500, 500, 5000.0)
        assert np.all(cov <= 1.0)

    def test_monotonic_decrease(self):
        """Values should decrease (or stay equal) as distance from center grows."""
        n = 200
        lat = np.full(n, 0.0)
        lon = np.linspace(0, 180, n)
        cov = _gaspari_cohn(lat, lon, 0, 0, 5000.0)
        # Non-increasing from center outward
        for i in range(1, n):
            assert cov[i] <= cov[i - 1] + 1e-12

def _make_regression_plugin(mocker, **overrides):
    """Create a RegressionPerturbationPlugin with a mocked context."""
    context = cast(Context, mocker)
    context.checkpoint = MagicMock()
    context.checkpoint.grid = "O96"
    defaults = dict(
        context=context,
        season="JAS",
        data_path="/nonexistent/",
        data_grid="N320",
        ylat=15.0,
        xlon=320.0,
        xlev=5,
        locrad=1000.0,
        amp=-1.0,
        param_pl=["z", "t"],
        level_pl=[500, 850],
        param_sfc=["msl"],
        method="add",
    )
    defaults.update(overrides)
    return RegressionPerturbationPlugin(**defaults)


class TestRegressionPerturbationInit:
    """Test constructor validation and field filter setup."""

    def test_invalid_season_raises(self):
        with pytest.raises(ValueError, match="Invalid season"):
            _make_regression_plugin(MagicMock(), season="XYZ")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid method"):
            _make_regression_plugin(MagicMock(), method="invalid")

    def test_valid_construction(self):
        plugin = _make_regression_plugin(MagicMock())
        assert plugin._season == "JAS"
        assert plugin._amp == -1.0
        assert plugin._method == "add"

    def test_field_filters_built_correctly(self):
        plugin = _make_regression_plugin(
            MagicMock(),
            param_pl=["z", "t"],
            level_pl=[500, 850],
            param_sfc=["msl"],
        )
        # 2 params × 2 levels + 1 sfc = 5 filter entries
        assert len(plugin._fields) == 5
        assert {"shortName": "z", "level": 500} in plugin._fields
        assert {"shortName": "t", "level": 850} in plugin._fields
        assert {"shortName": "msl"} in plugin._fields

    def test_default_rescale(self):
        plugin = _make_regression_plugin(MagicMock())
        assert plugin._rescale == 1.0

    def test_custom_rescale(self):
        plugin = _make_regression_plugin(MagicMock(), rescale=20.0)
        assert plugin._rescale == 20.0

    def test_target_grid_uses_data_grid(self):
        plugin = _make_regression_plugin(MagicMock(), data_grid="O96")
        assert plugin._target_grid == "O96"

    def test_target_grid_different_value(self):
        plugin = _make_regression_plugin(MagicMock(), data_grid="N640")
        assert plugin._target_grid == "N640"


class TestRegressionPerturbationApply:
    """Test _apply_perturbation and process with injected perturbation maps."""

    @pytest.fixture
    def plugin_with_map(self):
        """Create a plugin with a manually injected perturbation map."""
        plugin = _make_regression_plugin(MagicMock())
        n = 50
        # Inject a fake perturbation map (bypass _compute_regression)
        fake_map = {
            ("z", 500): np.ones(n, dtype=np.float64) * 10.0,
            ("z", 850): np.ones(n, dtype=np.float64) * 20.0,
            ("t", 500): np.ones(n, dtype=np.float64) * 5.0,
            ("t", 850): np.ones(n, dtype=np.float64) * 3.0,
            ("msl", None): np.ones(n, dtype=np.float64) * 100.0,
        }
        # Write directly to the cached property backing store
        plugin.__dict__["_perturbation_map"] = fake_map
        return plugin, n

    def test_apply_perturbation_add(self, plugin_with_map):
        plugin, n = plugin_with_map
        data = np.full(n, 1000.0)
        field = ekd.ArrayField(data, {"shortName": "z", "level": 500})
        result = plugin._apply_perturbation(field)
        expected = 1000.0 + 10.0 * plugin._rescale
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_apply_perturbation_surface(self, plugin_with_map):
        plugin, n = plugin_with_map
        data = np.full(n, 500.0)
        field = ekd.ArrayField(data, {"shortName": "msl"})
        result = plugin._apply_perturbation(field)
        expected = 500.0 + 100.0 * plugin._rescale
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_apply_perturbation_unmatched_field_unchanged(self, plugin_with_map):
        plugin, n = plugin_with_map
        data = np.full(n, 42.0)
        field = ekd.ArrayField(data, {"shortName": "unknown_param", "level": 999})
        result = plugin._apply_perturbation(field)
        np.testing.assert_array_equal(result.to_numpy(), 42.0)

    def test_apply_with_rescale(self):
        plugin = _make_regression_plugin(MagicMock(), rescale=5.0)
        n = 30
        plugin.__dict__["_perturbation_map"] = {
            ("z", 500): np.ones(n, dtype=np.float64) * 2.0,
        }
        data = np.full(n, 100.0)
        field = ekd.ArrayField(data, {"shortName": "z", "level": 500})
        result = plugin._apply_perturbation(field)
        # 100 + 2*5 = 110
        np.testing.assert_allclose(result.to_numpy(), 110.0)

    def test_repr_contains_key_info(self):
        plugin = _make_regression_plugin(MagicMock())
        r = repr(plugin)
        assert "RegressionPerturbationPlugin" in r
        assert "JAS" in r
        assert "15.0" in r
        assert "add" in r


class TestRegressionGlobFiles:
    """Test _glob_data_files with synthetic directory structures."""

    def test_glob_finds_season_files(self, tmp_path):
        # Create files matching JAS months (07, 08, 09)
        for name in ["20200715_pl_n320.grib", "20200810_pl_n320.grib", "20200905_sfc_n320.grib"]:
            (tmp_path / name).touch()
        # And a non-matching file
        (tmp_path / "20201215_pl_n320.grib").touch()

        plugin = _make_regression_plugin(
            MagicMock(),
            data_path=str(tmp_path) + "/",
            file_glob_suffix="_n320.grib",
        )
        files = plugin._glob_data_files()
        assert len(files) == 3
        assert all("12" not in f for f in files)

    def test_glob_raises_if_no_files(self, tmp_path):
        plugin = _make_regression_plugin(
            MagicMock(),
            data_path=str(tmp_path) + "/",
            file_glob_suffix="_n320.grib",
        )
        with pytest.raises(FileNotFoundError, match="No data files found"):
            plugin._glob_data_files()
