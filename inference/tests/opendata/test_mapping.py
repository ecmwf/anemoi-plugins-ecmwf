# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from unittest.mock import MagicMock
from unittest.mock import patch

import earthkit.data as ekd
from anemoi.plugins.ecmwf.inference.opendata.opendata import INVERSE_MAPPINGS
from anemoi.plugins.ecmwf.inference.opendata.opendata import MAPPINGS
from anemoi.plugins.ecmwf.inference.opendata.opendata import _expand_request
from anemoi.plugins.ecmwf.inference.opendata.opendata import _rename_params
from anemoi.plugins.ecmwf.inference.opendata.opendata import retrieve


def test_mappings_loaded():
    """Test that MAPPINGS are loaded from YAML."""
    assert isinstance(MAPPINGS, dict)
    assert len(MAPPINGS) > 0
    # Check soil level mappings exist
    assert "soil_lvl1" in MAPPINGS
    assert "soil_lvl2" in MAPPINGS
    assert "soil_lvl3" in MAPPINGS


def test_soil_mappings_structure():
    """Test that soil mappings have expected structure."""
    # Check stl1 mapping
    assert "stl1" in MAPPINGS["soil_lvl1"]
    assert MAPPINGS["soil_lvl1"]["stl1"]["param"] == "sot"
    assert MAPPINGS["soil_lvl1"]["stl1"]["levelist"] == 1
    assert "inverse" in MAPPINGS["soil_lvl1"]["stl1"]

    # Check swvl1 mapping
    assert "swvl1" in MAPPINGS["soil_lvl1"]
    assert MAPPINGS["soil_lvl1"]["swvl1"]["param"] == "vsw"
    assert MAPPINGS["soil_lvl1"]["swvl1"]["levelist"] == 1


def test_inverse_mappings_built():
    """Test that inverse mappings are built correctly."""
    assert isinstance(INVERSE_MAPPINGS, list)
    assert len(INVERSE_MAPPINGS) > 0

    # Find stl1 in inverse mappings
    stl1_mappings = [inv for inv in INVERSE_MAPPINGS if inv.true_param == "stl1"]
    assert len(stl1_mappings) > 0

    # Check structure
    inv = stl1_mappings[0]
    assert inv.true_param == "stl1"
    assert inv.id_pattern == "{param}{levelist}"
    assert inv.expected_pattern == "{param}{levelist}"
    assert inv.request_attrs["param"] == "sot"


def test_expand_request_with_soil_params():
    """Test _expand_request expands soil parameters correctly."""
    request = {
        "date": "20250101",
        "time": "00",
        "step": "0",
        "levtype": "sfc",
        "param": ["2t", "stl1", "swvl1"],
    }

    expanded = _expand_request(request)

    # Should have 2 requests: one for soil_lvl1 (stl1 and swvl1), one for remaining params
    assert len(expanded) == 2

    # First request should be for soil params
    soil_request = expanded[0]
    assert "sot" in soil_request["param"] or "vsw" in soil_request["param"]
    assert soil_request["levelist"] == 1 or soil_request["levelist"] == [1]
    assert "levtype" not in soil_request  # Should be popped

    # Second request should have remaining params
    regular_request = expanded[1]
    assert "2t" in regular_request["param"]
    assert "stl1" not in regular_request["param"]
    assert "swvl1" not in regular_request["param"]


def test_expand_request_with_multiple_soil_levels():
    """Test _expand_request with multiple soil levels."""
    request = {
        "date": "20250101",
        "time": "00",
        "step": "0",
        "levtype": "sfc",
        "param": ["stl1", "stl2", "stl3"],
    }

    expanded = _expand_request(request)

    # Should have 4 requests: one for each soil level, one for remaining params
    assert len(expanded) == 4


def test_expand_request_no_mapped_params():
    """Test _expand_request with no mapped parameters."""
    request = {
        "date": "20250101",
        "time": "00",
        "step": "0",
        "levtype": "sfc",
        "param": ["2t", "msl", "10u"],
    }

    expanded = _expand_request(request)

    # Should have only 1 request (original)
    assert len(expanded) == 1
    assert expanded[0]["param"] == ["2t", "msl", "10u"]


def test_expand_request_preserves_original():
    """Test that _expand_request doesn't modify the original request."""
    original_request = {
        "date": "20250101",
        "time": "00",
        "step": "0",
        "levtype": "sfc",
        "param": ["2t", "stl1"],
    }
    request_copy = {k: v.copy() if isinstance(v, list) else v for k, v in original_request.items()}

    _expand_request(original_request)

    # Original request should be unchanged
    assert original_request == request_copy


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.shortname_to_paramid")
def test_rename_params_stl(mock_shortname_to_paramid):
    """Test renaming soil temperature data."""
    # Setup
    mock_shortname_to_paramid.return_value = 139  # Example paramId for stl1

    # Create mock field with metadata
    mock_metadata_dict = {"param": "sot", "levelist": "1"}
    mock_namespace_result = MagicMock()
    mock_namespace_result.__iter__ = lambda self: iter(mock_metadata_dict.items())

    mock_metadata = MagicMock()
    mock_metadata.as_namespace.return_value = mock_metadata_dict
    mock_override_result = MagicMock()
    mock_metadata.override.return_value = mock_override_result

    mock_field = MagicMock()
    mock_field.metadata.return_value = mock_metadata

    mock_fieldlist = [mock_field]

    # Execute
    _ = _rename_params(mock_fieldlist)

    # Verify
    mock_shortname_to_paramid.assert_called_once_with("stl1")
    mock_metadata.override.assert_called_once_with(paramId=139)


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.shortname_to_paramid")
def test_rename_params_swvl(mock_shortname_to_paramid):
    """Test renaming soil moisture data."""
    # Setup
    mock_shortname_to_paramid.return_value = 39  # Example paramId for swvl1

    mock_metadata = MagicMock()
    mock_metadata.as_namespace.return_value = {"param": "vsw", "levelist": "1"}
    mock_override_result = MagicMock()
    mock_metadata.override.return_value = mock_override_result

    mock_field = MagicMock()
    mock_field.metadata.return_value = mock_metadata

    mock_fieldlist = [mock_field]

    # Execute
    _ = _rename_params(mock_fieldlist)

    # Verify
    mock_shortname_to_paramid.assert_called_once_with("swvl1")


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.shortname_to_paramid")
def test_rename_params_multiple_fields(mock_shortname_to_paramid):
    """Test renaming multiple soil fields."""
    # Setup
    mock_shortname_to_paramid.side_effect = [139, 170, 39]  # paramIds for stl1, stl2, swvl1

    # Field 1: sot level 1
    mock_metadata1 = MagicMock()
    mock_metadata1.as_namespace.return_value = {"param": "sot", "levelist": "1"}
    mock_metadata1.override.return_value = MagicMock()
    mock_field1 = MagicMock()
    mock_field1.metadata.return_value = mock_metadata1

    # Field 2: sot level 2
    mock_metadata2 = MagicMock()
    mock_metadata2.as_namespace.return_value = {"param": "sot", "levelist": "2"}
    mock_metadata2.override.return_value = MagicMock()
    mock_field2 = MagicMock()
    mock_field2.metadata.return_value = mock_metadata2

    # Field 3: vsw level 1
    mock_metadata3 = MagicMock()
    mock_metadata3.as_namespace.return_value = {"param": "vsw", "levelist": "1"}
    mock_metadata3.override.return_value = MagicMock()
    mock_field3 = MagicMock()
    mock_field3.metadata.return_value = mock_metadata3

    mock_fieldlist = [mock_field1, mock_field2, mock_field3]

    # Execute
    _ = _rename_params(mock_fieldlist)

    # Verify
    assert mock_shortname_to_paramid.call_count == 3
    assert mock_shortname_to_paramid.call_args_list[0][0][0] == "stl1"
    assert mock_shortname_to_paramid.call_args_list[1][0][0] == "stl2"
    assert mock_shortname_to_paramid.call_args_list[2][0][0] == "swvl1"


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata._rename_params")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekr.regrid")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.from_source")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.SimpleFieldList")
def test_retrieve_with_soil_variables(mock_simple_fieldlist, mock_from_source, mock_regrid, mock_rename_params):
    """Test retrieve function when soil variables are included."""
    # Setup
    mock_result_list = MagicMock()
    mock_simple_fieldlist.return_value = mock_result_list

    mock_fieldlist = MagicMock(spec=ekd.FieldList)
    mock_from_source.return_value = mock_fieldlist
    mock_regrid.return_value = mock_fieldlist
    mock_rename_params.return_value = mock_fieldlist

    requests = [
        {
            "date": "20250101",
            "time": "00",
            "step": "0",
            "levtype": "sfc",
            "param": ["2t", "stl1", "swvl1"],
        }
    ]

    # Execute
    _ = retrieve(requests, grid="1.0/1.0", area=None)

    # Verify that from_source was called for expanded requests
    assert mock_from_source.call_count == 2  # One for soil_lvl1, one for regular

    # Verify rename_params was called once at the end
    mock_rename_params.assert_called_once()


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata._rename_params")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekr.regrid")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.from_source")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.SimpleFieldList")
def test_retrieve_only_soil_variables(mock_simple_fieldlist, mock_from_source, mock_regrid, mock_rename_params):
    """Test retrieve function with only soil variables."""
    # Setup
    mock_result_list = MagicMock()
    mock_simple_fieldlist.return_value = mock_result_list

    mock_fieldlist = MagicMock(spec=ekd.FieldList)
    mock_from_source.return_value = mock_fieldlist
    mock_regrid.return_value = mock_fieldlist
    mock_rename_params.return_value = mock_fieldlist

    requests = [
        {
            "date": "20250101",
            "time": "00",
            "step": "0",
            "levtype": "sfc",
            "param": ["stl1", "stl2", "swvl1"],
        }
    ]

    # Execute
    _ = retrieve(requests, grid="1.0/1.0", area=None)

    # Verify from_source was called for each soil level
    # Note: stl1 and swvl1 are in soil_lvl1, stl2 is in soil_lvl2, stl3 would be in soil_lvl3
    # But we only have stl1, stl2, swvl1, so we get 2 groups + 1 empty remaining = 3 calls
    assert mock_from_source.call_count == 3

    # Verify rename_params was called once at the end
    mock_rename_params.assert_called_once()


@patch("anemoi.plugins.ecmwf.inference.opendata.opendata._rename_params")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekr.regrid")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.from_source")
@patch("anemoi.plugins.ecmwf.inference.opendata.opendata.ekd.SimpleFieldList")
def test_retrieve_no_soil_variables(mock_simple_fieldlist, mock_from_source, mock_regrid, mock_rename_params):
    """Test retrieve function without soil variables."""
    # Setup
    mock_result_list = MagicMock()
    mock_simple_fieldlist.return_value = mock_result_list

    mock_regular_fieldlist = MagicMock(spec=ekd.FieldList)
    mock_from_source.return_value = mock_regular_fieldlist
    mock_regrid.return_value = mock_regular_fieldlist
    mock_rename_params.return_value = mock_regular_fieldlist

    requests = [
        {
            "date": "20250101",
            "time": "00",
            "step": "0",
            "levtype": "sfc",
            "param": ["2t", "msl", "10u"],
        }
    ]

    # Execute
    _ = retrieve(requests, grid="1.0/1.0", area=None)

    # Verify only one call (no soil retrieval, just regular params)
    assert mock_from_source.call_count == 1
    call_args = mock_from_source.call_args[0]
    request_arg = call_args[1]
    assert request_arg["param"] == ["2t", "msl", "10u"]

    # Verify rename_params was still called
    mock_rename_params.assert_called_once()
