# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.plugins.ecmwf.transform.regrid.backend import normalise_grid


class TestNormaliseGrid:
    """Tests for normalise_grid which converts grid specs to string format."""

    def test_list_grid_converted_to_string(self):
        """A list grid specification is joined into a slash-separated string."""
        assert normalise_grid([0.25, 0.25]) == "0.25/0.25"

    def test_tuple_grid_converted_to_string(self):
        """A tuple grid specification is also joined into a slash-separated string."""
        assert normalise_grid((0.5, 0.5)) == "0.5/0.5"

    def test_string_grid_uppercased(self):
        """A string grid specification is uppercased."""
        assert normalise_grid("n320") == "N320"
        assert normalise_grid("O32") == "O32"

    def test_int_float_grid_duplicated(self):
        """A single int/float grid is duplicated as 'v/v'."""
        assert normalise_grid(1.0) == "1.0/1.0"
        assert normalise_grid(0.25) == "0.25/0.25"

    def test_dict_grid_converted(self):
        """A dict grid is converted to key=value slash-separated string."""
        result = normalise_grid({"latitudes": [1.0], "longitudes": [2.0]})
        assert "latitudes=" in result
        assert "longitudes=" in result
