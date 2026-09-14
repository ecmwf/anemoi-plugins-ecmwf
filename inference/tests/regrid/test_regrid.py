# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import State
from anemoi.plugins.ecmwf.inference.regrid.regrid import RegridPreprocessor
from anemoi.plugins.ecmwf.inference.regrid.regrid import _open_coord_files

# --- _open_coord_files ---


class TestOpenCoordFiles:
    def test_loads_numpy_files(self, tmp_path):
        """Coordinate files are loaded from numpy arrays and returned as lists."""
        lats = np.array([10.0, 20.0, 30.0])
        lons = np.array([100.0, 110.0, 120.0])
        np.save(tmp_path / "latitudes.npy", lats)
        np.save(tmp_path / "longitudes.npy", lons)

        grid = {
            "latitudes": str(tmp_path / "latitudes.npy"),
            "longitudes": str(tmp_path / "longitudes.npy"),
        }
        result = _open_coord_files(grid)

        assert isinstance(result, dict)
        assert list(result.keys()) == ["latitudes", "longitudes"]
        np.testing.assert_array_almost_equal(result["latitudes"], lats.tolist())
        np.testing.assert_array_almost_equal(result["longitudes"], lons.tolist())

    def test_returns_plain_lists(self, tmp_path):
        """Values are plain Python lists, not numpy arrays."""
        np.save(tmp_path / "lats.npy", np.array([1.0, 2.0]))
        result = _open_coord_files({"lats": str(tmp_path / "lats.npy")})
        assert isinstance(result["lats"], list)

    def test_missing_file_raises(self, tmp_path):
        """A missing coordinate file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _open_coord_files({"latitudes": str(tmp_path / "nonexistent.npy")})


# --- RegridPreprocessor ---


class TestRegridPreprocessor:
    def _make_processor(self, mocker, grid, area=None):
        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        with patch("anemoi.plugins.ecmwf.inference.regrid.regrid.MIRRegrid"):
            return RegridPreprocessor(context=context, metadata=metadata, grid=grid, area=area)

    def test_string_grid(self, mocker):
        """A plain string grid is stored in the MIRRegrid instance."""
        proc = self._make_processor(mocker, "O32")
        assert proc._regrid is not None

    def test_list_grid(self, mocker):
        """A list grid is passed to MIRRegrid."""
        proc = self._make_processor(mocker, [0.25, 0.25])
        assert proc._regrid is not None

    def test_area_stored(self, mocker):
        """Area parameter is passed to MIRRegrid."""
        proc = self._make_processor(mocker, "O32", area=[90, -180, -90, 180])
        assert proc._regrid is not None

    def test_dict_grid_with_file_paths(self, mocker, tmp_path):
        """A dict grid with string values triggers coordinate file loading."""
        lats = np.array([10.0, 20.0])
        lons = np.array([100.0, 110.0])
        np.save(tmp_path / "lats.npy", lats)
        np.save(tmp_path / "lons.npy", lons)

        grid = {
            "latitudes": str(tmp_path / "lats.npy"),
            "longitudes": str(tmp_path / "lons.npy"),
        }

        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        with patch("anemoi.plugins.ecmwf.inference.regrid.regrid.MIRRegrid") as mock_mir:
            RegridPreprocessor(context=context, metadata=metadata, grid=grid)
            call_kwargs = mock_mir.call_args.kwargs
            np.testing.assert_array_almost_equal(call_kwargs["grid"]["latitudes"], lats.tolist())
            np.testing.assert_array_almost_equal(call_kwargs["grid"]["longitudes"], lons.tolist())

    def test_dict_grid_with_float_lists(self, mocker):
        """A dict grid with float list values is stored as-is (no file loading)."""
        grid = {"latitudes": [10.0, 20.0], "longitudes": [100.0, 110.0]}

        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        with patch("anemoi.plugins.ecmwf.inference.regrid.regrid.MIRRegrid") as mock_mir:
            RegridPreprocessor(context=context, metadata=metadata, grid=grid)
            call_kwargs = mock_mir.call_args.kwargs
            assert call_kwargs["grid"] == grid

    def test_dict_grid_mixed_types_raises(self, mocker, tmp_path):
        """A dict grid with mixed value types raises ValueError."""
        np.save(tmp_path / "lats.npy", np.array([10.0, 20.0]))
        grid = {
            "latitudes": str(tmp_path / "lats.npy"),
            "longitudes": [100.0, 110.0],
        }
        with pytest.raises(ValueError, match="mixed types"):
            context = cast(Context, mocker.MagicMock())
            metadata = cast(Metadata, mocker.MagicMock())
            RegridPreprocessor(context=context, metadata=metadata, grid=grid)

    def test_named_grid_case_insensitive(self, mocker):
        """Named grid lookup is case-insensitive."""
        mock_named = mocker.MagicMock()
        mock_named.gridspec = {"grid": {"latitudes": [1.0], "longitudes": [2.0]}}

        mocker.patch(
            "anemoi.plugins.ecmwf.inference.regrid.regrid.KNOWN_GRIDS",
            ["test_grid"],
        )
        mocker.patch(
            "anemoi.plugins.ecmwf.inference.regrid.regrid.NamedRegrid",
            return_value=mock_named,
        )

        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        with patch("anemoi.plugins.ecmwf.inference.regrid.regrid.MIRRegrid") as mock_mir:
            RegridPreprocessor(context=context, metadata=metadata, grid="TEST_GRID")
            call_kwargs = mock_mir.call_args.kwargs
            assert call_kwargs["grid"] == {"latitudes": [1.0], "longitudes": [2.0]}

    def test_named_grid(self, mocker):
        """A known named grid is resolved to latitudes/longitudes from package resources."""
        mock_named = mocker.MagicMock()
        mock_named.gridspec = {"grid": {"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]}}

        mocker.patch(
            "anemoi.plugins.ecmwf.inference.regrid.regrid.KNOWN_GRIDS",
            ["test_grid"],
        )
        mocker.patch(
            "anemoi.plugins.ecmwf.inference.regrid.regrid.NamedRegrid",
            return_value=mock_named,
        )

        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        with patch("anemoi.plugins.ecmwf.inference.regrid.regrid.MIRRegrid") as mock_mir:
            RegridPreprocessor(context=context, metadata=metadata, grid="test_grid")
            call_kwargs = mock_mir.call_args.kwargs
            assert call_kwargs["grid"] == {
                "latitudes": [1.0, 2.0],
                "longitudes": [3.0, 4.0],
            }

    def _make_mock_regridded(self):
        """Create a mock regridded fieldlist with proper iteration support."""
        mock_field = MagicMock()
        mock_field.metadata.return_value.geography.latitudes.return_value = [1.0, 2.0]
        mock_field.metadata.return_value.geography.longitudes.return_value = [3.0, 4.0]
        mock_regridded = MagicMock()
        mock_regridded.__iter__ = MagicMock(side_effect=lambda: iter([mock_field]))
        return mock_regridded

    def test_process_calls_forward(self, mocker):
        """process() calls MIRRegrid.forward with the state fields."""
        proc = self._make_processor(mocker, "O32", area="global")

        mock_fields = MagicMock()
        mock_regridded = self._make_mock_regridded()
        proc._regrid.forward = MagicMock(return_value=mock_regridded)

        state: State = {"fields": mock_fields}  # type: ignore
        result = proc.process(state)

        proc._regrid.forward.assert_called_once_with(mock_fields)
        assert result["fields"] is mock_regridded

    def test_process_preserves_other_state_keys(self, mocker):
        """process() only modifies 'fields'; other state keys are preserved."""
        proc = self._make_processor(mocker, "O32")

        mock_regridded = self._make_mock_regridded()
        proc._regrid.forward = MagicMock(return_value=mock_regridded)

        state: State = {"fields": MagicMock(), "date": "2024-01-01", "step": 6}  # type: ignore
        result = proc.process(state)

        assert result["date"] == "2024-01-01"
        assert result["step"] == 6
