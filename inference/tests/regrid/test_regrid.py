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
from anemoi.plugins.ecmwf.inference.regrid.regrid import regrid

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


# --- regrid function ---


class TestRegridFunction:
    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid._mir_regrid")
    def test_list_grid_converted_to_string(self, mock_mir_regrid):
        """A list grid specification is joined into a slash-separated string."""
        mock_field = MagicMock()
        mock_mir_regrid.return_value = mock_field

        mock_fieldlist = MagicMock()
        mock_fieldlist.__iter__ = MagicMock(return_value=iter([mock_field]))
        mock_fieldlist.__len__ = MagicMock(return_value=1)

        with patch("earthkit.data.FieldList.from_fields") as mock_from_fields:
            mock_from_fields.return_value = MagicMock()
            regrid(mock_fieldlist, [0.25, 0.25], None)

        mock_mir_regrid.assert_called_once()
        call_args = mock_mir_regrid.call_args
        assert call_args[0][1] == "0.25/0.25"
        assert call_args[0][2] is None

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid._mir_regrid")
    def test_tuple_grid_converted_to_string(self, mock_mir_regrid):
        """A tuple grid specification is also joined into a slash-separated string."""
        mock_field = MagicMock()
        mock_mir_regrid.return_value = mock_field

        mock_fieldlist = MagicMock()
        mock_fieldlist.__iter__ = MagicMock(return_value=iter([mock_field]))
        mock_fieldlist.__len__ = MagicMock(return_value=1)

        with patch("earthkit.data.FieldList.from_fields") as mock_from_fields:
            mock_from_fields.return_value = MagicMock()
            regrid(mock_fieldlist, (0.5, 0.5), None)

        call_args = mock_mir_regrid.call_args
        assert call_args[0][1] == "0.5/0.5"

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid._mir_regrid")
    def test_area_list_converted_to_string(self, mock_mir_regrid):
        """A list area specification is joined into a slash-separated string."""
        mock_field = MagicMock()
        mock_mir_regrid.return_value = mock_field

        mock_fieldlist = MagicMock()
        mock_fieldlist.__iter__ = MagicMock(return_value=iter([mock_field]))
        mock_fieldlist.__len__ = MagicMock(return_value=1)

        with patch("earthkit.data.FieldList.from_fields") as mock_from_fields:
            mock_from_fields.return_value = MagicMock()
            regrid(mock_fieldlist, "O32", [90, -180, -90, 180])

        call_args = mock_mir_regrid.call_args
        assert call_args[0][1] == "O32"
        assert call_args[0][2] == "90/-180/-90/180"

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid._mir_regrid")
    def test_string_grid_passed_through(self, mock_mir_regrid):
        """A string grid specification is passed through unchanged."""
        mock_field = MagicMock()
        mock_mir_regrid.return_value = mock_field

        mock_fieldlist = MagicMock()
        mock_fieldlist.__iter__ = MagicMock(return_value=iter([mock_field]))
        mock_fieldlist.__len__ = MagicMock(return_value=1)

        with patch("earthkit.data.FieldList.from_fields") as mock_from_fields:
            mock_from_fields.return_value = MagicMock()
            regrid(mock_fieldlist, "N320", None)

        call_args = mock_mir_regrid.call_args
        assert call_args[0][1] == "N320"

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid._mir_regrid")
    def test_regrids_all_fields(self, mock_mir_regrid):
        """Each field in the fieldlist is regridded individually."""
        fields = [MagicMock(), MagicMock(), MagicMock()]
        mock_mir_regrid.side_effect = fields

        mock_fieldlist = MagicMock()
        mock_fieldlist.__iter__ = MagicMock(return_value=iter(fields))
        mock_fieldlist.__len__ = MagicMock(return_value=3)

        with patch("earthkit.data.FieldList.from_fields") as mock_from_fields:
            mock_from_fields.return_value = MagicMock()
            regrid(mock_fieldlist, "O32", None)

        assert mock_mir_regrid.call_count == 3


# --- RegridPreprocessor ---


class TestRegridPreprocessor:
    def _make_processor(self, mocker, grid, area=None):
        context = cast(Context, mocker.MagicMock())
        metadata = cast(Metadata, mocker.MagicMock())
        return RegridPreprocessor(context=context, metadata=metadata, grid=grid, area=area)

    def test_string_grid(self, mocker):
        """A plain string grid is stored as-is."""
        proc = self._make_processor(mocker, "O32")
        assert proc._grid == "O32"
        assert proc._area is None

    def test_list_grid(self, mocker):
        """A list grid is stored as-is (conversion happens in regrid())."""
        proc = self._make_processor(mocker, [0.25, 0.25])
        assert proc._grid == [0.25, 0.25]

    def test_area_stored(self, mocker):
        """Area parameter is stored correctly."""
        proc = self._make_processor(mocker, "O32", area=[90, -180, -90, 180])
        assert proc._area == [90, -180, -90, 180]

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
        proc = self._make_processor(mocker, grid)

        assert isinstance(proc._grid, dict)
        np.testing.assert_array_almost_equal(proc._grid["latitudes"], lats.tolist())
        np.testing.assert_array_almost_equal(proc._grid["longitudes"], lons.tolist())

    def test_dict_grid_with_float_lists(self, mocker):
        """A dict grid with float list values is stored as-is (no file loading)."""
        grid = {"latitudes": [10.0, 20.0], "longitudes": [100.0, 110.0]}
        proc = self._make_processor(mocker, grid)
        assert proc._grid == grid

    def test_dict_grid_mixed_types_raises(self, mocker, tmp_path):
        """A dict grid with mixed value types raises ValueError."""
        np.save(tmp_path / "lats.npy", np.array([10.0, 20.0]))
        grid = {
            "latitudes": str(tmp_path / "lats.npy"),
            "longitudes": [100.0, 110.0],
        }
        with pytest.raises(ValueError, match="mixed types"):
            self._make_processor(mocker, grid)

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

        proc = self._make_processor(mocker, "TEST_GRID")
        assert proc._grid == {"latitudes": [1.0], "longitudes": [2.0]}

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

        proc = self._make_processor(mocker, "test_grid")
        assert proc._grid == {"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]}

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid.regrid")
    def test_process_calls_regrid(self, mock_regrid, mocker):
        """process() calls regrid with the state fields, grid, and area."""
        proc = self._make_processor(mocker, "O32", area="global")

        mock_fields = MagicMock()
        mock_regridded = MagicMock()
        mock_regrid.return_value = mock_regridded

        state: State = {"fields": mock_fields}  # type: ignore
        result = proc.process(state)

        mock_regrid.assert_called_once_with(mock_fields, "O32", "global")
        assert result["fields"] is mock_regridded

    @patch("anemoi.plugins.ecmwf.inference.regrid.regrid.regrid")
    def test_process_preserves_other_state_keys(self, mock_regrid, mocker):
        """process() only modifies 'fields'; other state keys are preserved."""
        proc = self._make_processor(mocker, "O32")
        mock_regrid.return_value = MagicMock()

        state: State = {"fields": MagicMock(), "date": "2024-01-01", "step": 6}  # type: ignore
        result = proc.process(state)

        assert result["date"] == "2024-01-01"
        assert result["step"] == 6
