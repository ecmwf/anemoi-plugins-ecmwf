# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from anemoi.plugins.ecmwf.inference.multio.multio_output import UserDefinedMetadata


def test_ensemble_validation():
    """Test that ValueError is raised when ensemble size does not match."""
    with pytest.raises(ValueError, match="numberOfForecastsInEnsemble must be an integer if number is provided"):
        UserDefinedMetadata(
            **{"stream": "oper", "type": "fc", "class": "od", "expver": "1", "model": "test_model", "number": 1}
        )
