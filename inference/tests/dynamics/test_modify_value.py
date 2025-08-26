# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from typing import cast

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.inference.context import Context
from anemoi.plugins.ecmwf.inference.dynamics.modify_value import ModifyValuePlugin
from pytest_mock import MockerFixture


def make_field(param: str) -> ekd.ArrayField:
    return ekd.ArrayField(
        np.random.random((100,)),
        {
            "shortName": param,
        },
    )


@pytest.fixture
def mock_fields() -> ekd.FieldList:
    params = ["2t", "msl"]
    return ekd.FieldList.from_fields(map(make_field, params))


@pytest.mark.parametrize("value", [1.0, -5.5, 0.0])
def test_modify_value(mocker: MockerFixture, mock_fields: ekd.FieldList, value):
    # mock the context to return the mask when load_supporting_array is called
    context = cast(Context, mocker.MagicMock())
    field_specifier = [{"shortName": "2t"}]
    processor = ModifyValuePlugin(context=context, fields=field_specifier, value=value, method="add")

    # check that assignment works as expected
    new_state = processor.process(fields=mock_fields)
    assert isinstance(new_state, ekd.FieldList)
    assert len(new_state) == len(mock_fields)

    original_2t = mock_fields.sel(shortName="2t")[0].to_numpy()
    modified_2t = new_state.sel(shortName="2t")[0].to_numpy()
    np.testing.assert_array_equal(modified_2t, original_2t + value)

    # check that other fields are unchanged
    original_msl = mock_fields.sel(shortName="msl")[0].to_numpy()
    modified_msl = new_state.sel(shortName="msl")[0].to_numpy()
    np.testing.assert_array_equal(modified_msl, original_msl)


@pytest.mark.parametrize("value", [1.0, -5.5, 0.0])
def test_modify_value_with_numpy_file(mocker: MockerFixture, mock_fields: ekd.FieldList, tmp_path, value):
    # Create a temporary numpy file with a single value
    numpy_file = tmp_path / "value.npy"
    np.save(numpy_file, np.array([value]))

    # mock the context to return the mask when load_supporting_array is called
    context = cast(Context, mocker.MagicMock())
    field_specifier = [{"shortName": "2t"}]
    processor = ModifyValuePlugin(context=context, fields=field_specifier, value=str(numpy_file), method="add")

    # check that assignment works as expected
    new_state = processor.process(fields=mock_fields)
    assert isinstance(new_state, ekd.FieldList)
    assert len(new_state) == len(mock_fields)

    original_2t = mock_fields.sel(shortName="2t")[0].to_numpy()
    modified_2t = new_state.sel(shortName="2t")[0].to_numpy()
    np.testing.assert_array_equal(modified_2t, original_2t + value)

    # check that other fields are unchanged
    original_msl = mock_fields.sel(shortName="msl")[0].to_numpy()
    modified_msl = new_state.sel(shortName="msl")[0].to_numpy()
    np.testing.assert_array_equal(modified_msl, original_msl)
