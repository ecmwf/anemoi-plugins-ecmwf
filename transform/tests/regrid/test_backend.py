# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.plugins.ecmwf.transform.regrid.backend import grid_repr


class TestGridRepr:
    """Tests for grid_repr which renders grid specs for logging."""

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
        """A dict grid is summarised by coordinate-list lengths."""
        result = grid_repr({"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]})
        assert "latitudes" in result
        assert "list of len 2" in result
