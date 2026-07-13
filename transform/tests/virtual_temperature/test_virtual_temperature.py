# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.plugins.ecmwf.transform.virtual_temperature import VirtualTemperature

NPOINTS = 20


def make_field(param: str, values: np.ndarray, **metadata) -> ekd.ArrayField:
    return ekd.ArrayField(values, {"param": param, **metadata})


@pytest.fixture
def mock_fields() -> ekd.FieldList:
    """A field list containing specific humidity, temperature and an unrelated field."""
    return ekd.FieldList.from_fields(
        [
            make_field("q", np.random.uniform(0.0, 0.02, size=NPOINTS)),
            make_field("t", np.random.uniform(220.0, 310.0, size=NPOINTS)),
            make_field("msl", np.random.uniform(500.0, 1500.0, size=NPOINTS)),
        ]
    )


def test_virtual_temperature_computes_vtmp(mock_fields: ekd.FieldList):
    q = mock_fields.sel(param="q")[0].to_numpy()
    t = mock_fields.sel(param="t")[0].to_numpy()

    result = VirtualTemperature().forward(mock_fields)

    vtmp = result.sel(param="vtmp")[0].to_numpy()
    np.testing.assert_allclose(vtmp, (1.0 + 0.61 * q) * t)


def test_virtual_temperature_returns_inputs(mock_fields: ekd.FieldList):
    result = VirtualTemperature().forward(mock_fields)

    # the new field is added alongside the (unchanged) inputs
    params = sorted(result.metadata("param"))
    assert params == ["msl", "q", "t", "vtmp"]

    np.testing.assert_array_equal(
        result.sel(param="q")[0].to_numpy(), mock_fields.sel(param="q")[0].to_numpy()
    )
    np.testing.assert_array_equal(
        result.sel(param="t")[0].to_numpy(), mock_fields.sel(param="t")[0].to_numpy()
    )
    np.testing.assert_array_equal(
        result.sel(param="msl")[0].to_numpy(), mock_fields.sel(param="msl")[0].to_numpy()
    )


def test_virtual_temperature_custom_names():
    q = np.random.uniform(0.0, 0.02, size=NPOINTS)
    t = np.random.uniform(220.0, 310.0, size=NPOINTS)
    fields = ekd.FieldList.from_fields(
        [make_field("spec_hum", q), make_field("temp", t)]
    )

    result = VirtualTemperature(
        virtual_temperature="virt_t",
        specific_humidity="spec_hum",
        temperature="temp",
    ).forward(fields)

    vtmp = result.sel(param="virt_t")[0].to_numpy()
    np.testing.assert_allclose(vtmp, (1.0 + 0.61 * q) * t)


def test_virtual_temperature_multiple_levels():
    """A vtmp field is produced for every level where both q and t are present."""
    data = {}
    fields = []
    for level in (850, 500):
        q = np.random.uniform(0.0, 0.02, size=NPOINTS)
        t = np.random.uniform(220.0, 310.0, size=NPOINTS)
        data[level] = (q, t)
        fields.append(make_field("q", q, levelist=level))
        fields.append(make_field("t", t, levelist=level))

    result = VirtualTemperature().forward(ekd.FieldList.from_fields(fields))

    vtmp_fields = result.sel(param="vtmp")
    assert len(vtmp_fields) == 2
    for field in vtmp_fields:
        level = field.metadata("levelist")
        q, t = data[level]
        np.testing.assert_allclose(field.to_numpy(), (1.0 + 0.61 * q) * t)


def test_virtual_temperature_unpaired_temperature_raises():
    """A temperature field without a matching humidity field is an error."""
    fields = ekd.FieldList.from_fields(
        [
            make_field("q", np.random.uniform(0.0, 0.02, size=NPOINTS), levelist=850),
            make_field("t", np.random.uniform(220.0, 310.0, size=NPOINTS), levelist=850),
            # temperature at 500 hPa has no matching humidity
            make_field("t", np.random.uniform(220.0, 310.0, size=NPOINTS), levelist=500),
        ]
    )

    with pytest.raises(ValueError, match="Missing component"):
        VirtualTemperature().forward(fields)
