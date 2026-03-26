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

import eccodes
import numpy as np
import pytest
import torch
from anemoi.inference.context import Context
from anemoi.plugins.ecmwf.inference.dynamics.subtract_tendency import SubtractTendencyPlugin

# ---- SubtractTendency tests with synthetic GRIB ----


def _write_grib_fields(path, param_names, values_list, *, levels=None) -> None:
    """Write a minimal GRIB file with given fields using eccodes.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    param_names : list[str]
        Short names for each field.
    values_list : list[np.ndarray]
        1-D value arrays, one per field.
    levels : list[int] | None
        If provided, each field gets the corresponding level.
    """
    with open(str(path), "wb") as f:
        for i, (param, vals) in enumerate(zip(param_names, values_list)):
            sample_id = eccodes.codes_grib_new_from_samples("reduced_gg_pl_32_grib2")
            eccodes.codes_set(sample_id, "shortName", param)
            eccodes.codes_set(sample_id, "date", 20200101)
            if levels is not None:
                eccodes.codes_set(sample_id, "level", levels[i])
                eccodes.codes_set(sample_id, "typeOfLevel", "isobaricInhPa")
            else:
                eccodes.codes_set(sample_id, "typeOfLevel", "surface")
            eccodes.codes_set_values(sample_id, vals.astype(np.float64))
            eccodes.codes_write(sample_id, f)
            eccodes.codes_release(sample_id)


class TestSubtractTendency:
    """Test SubtractTendencyPlugin with synthetic tendency GRIB files."""

    @pytest.fixture
    def tendency_files(self, tmp_path):
        """Create synthetic PL and SFC tendency files."""
        n_gridpoints = 6114  # smallest valid reduced Gaussian grid (N32)
        param_pl = ["t", "u"]
        level_pl = [500, 850]
        param_sfc = ["msl"]

        # PL: 2 params × 2 levels = 4 fields
        pl_params = []
        pl_values = []
        pl_levels = []
        for p in param_pl:
            for lev in level_pl:
                pl_params.append(p)
                pl_levels.append(lev)
                pl_values.append(np.random.randn(n_gridpoints).astype(np.float32))

        pl_path = tmp_path / "tend_pl.grib"
        _write_grib_fields(pl_path, pl_params, pl_values, levels=pl_levels)

        # SFC: 1 field
        sfc_values = [np.random.randn(n_gridpoints).astype(np.float32)]
        sfc_path = tmp_path / "tend_sfc.grib"
        _write_grib_fields(sfc_path, param_sfc, sfc_values)

        return {
            "pl_path": str(pl_path),
            "sfc_path": str(sfc_path),
            "param_pl": param_pl,
            "level_pl": level_pl,
            "param_sfc": param_sfc,
            "n_gridpoints": n_gridpoints,
            "pl_values": {f"{p}_{lev}": v for p, lev, v in zip(pl_params, pl_levels, pl_values)},
            "sfc_values": {p: v for p, v in zip(param_sfc, sfc_values)},
        }

    def _make_plugin(self, tendency_files):
        context = cast(Context, MagicMock())
        return SubtractTendencyPlugin(
            context=context,
            tend_pl_path=tendency_files["pl_path"],
            tend_sfc_path=tendency_files["sfc_path"],
            param_pl=tendency_files["param_pl"],
            level_pl=tendency_files["level_pl"],
            param_sfc=tendency_files["param_sfc"],
        )

    def test_loads_correct_number_of_fields(self, tendency_files):
        plugin = self._make_plugin(tendency_files)
        expected = len(tendency_files["param_pl"]) * len(tendency_files["level_pl"]) + len(tendency_files["param_sfc"])
        assert len(plugin._tendency_np) == expected

    def test_tendency_keys(self, tendency_files):
        plugin = self._make_plugin(tendency_files)
        for p in tendency_files["param_pl"]:
            for lev in tendency_files["level_pl"]:
                assert f"{p}_{lev}" in plugin._tendency_np
        for p in tendency_files["param_sfc"]:
            assert p in plugin._tendency_np

    def test_process_subtracts_correctly(self, tendency_files):
        plugin = self._make_plugin(tendency_files)
        n = tendency_files["n_gridpoints"]

        # Build a fake state with torch tensors
        fields = {}
        originals = {}
        for name in plugin._tendency_np:
            orig = torch.randn(n, dtype=torch.float32)
            originals[name] = orig.clone()
            fields[name] = orig

        state = {"fields": fields, "date": "2020-01-01", "step": 6}
        new_state = plugin.process(state)

        for name in plugin._tendency_np:
            expected = originals[name] - torch.from_numpy(plugin._tendency_np[name])
            torch.testing.assert_close(new_state["fields"][name], expected, rtol=1e-5, atol=1e-6)

    def test_process_leaves_unknown_fields_unchanged(self, tendency_files):
        plugin = self._make_plugin(tendency_files)
        n = tendency_files["n_gridpoints"]

        extra = torch.randn(n, dtype=torch.float32)
        state = {
            "fields": {"unknown_var": extra.clone()},
            "date": "2020-01-01",
            "step": 6,
        }
        new_state = plugin.process(state)
        torch.testing.assert_close(new_state["fields"]["unknown_var"], extra)

    def test_process_returns_new_state_dict(self, tendency_files):
        """process() returns a new top-level dict (shallow copy)."""
        plugin = self._make_plugin(tendency_files)
        n = tendency_files["n_gridpoints"]

        fields = {}
        for name in plugin._tendency_np:
            fields[name] = torch.randn(n, dtype=torch.float32)

        state = {"fields": fields, "date": "2020-01-01", "step": 6}
        new_state = plugin.process(state)

        # Returned dict should be a different object
        assert new_state is not state
        # But fields dict is shared (shallow copy), which is standard for processors
        assert new_state["fields"] is state["fields"]

    def test_repr(self, tendency_files):
        plugin = self._make_plugin(tendency_files)
        r = repr(plugin)
        assert "SubtractTendencyPlugin" in r
        assert "tend_pl.grib" in r
        assert "tend_sfc.grib" in r
