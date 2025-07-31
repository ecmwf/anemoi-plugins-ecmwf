# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import earthkit.data as ekd
from anemoi.inference.processor import Processor
from anemoi.transform.filters.orog_to_z import Orography

LOG = logging.getLogger(__name__)


class InferenceOrography(Orography):
    """Inference Orography Filter.

    Patches the data request to replace `z` with `orog` if `z` is present.
    """

    def patch_data_request(self, data_request: Any) -> Any:
        param = data_request.get("param")
        if param is None:
            return data_request

        if self.z in param and (data_request.get("levtype", "") == "pl" or data_request.get("levelist", [])):
            data_request["param"] = [self.orog if p == self.z else p for p in param]
        return data_request


class OrographyProcessor(Processor):
    """A processor that applies the InferenceOrography filter to the given fields."""

    def __init__(self, context: Any, **kwargs: Any) -> None:
        """Initialize the OrographyProcessor.

        Parameters
        ----------
        context : object
            The context in which the filter is being used.
        **kwargs : dict
            Additional keyword arguments to pass to the filter.
        """
        super().__init__(context)
        self.filter = InferenceOrography(**kwargs)

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Process the given fields using the InferenceOrography filter.

        Parameters
        ----------
        fields : ekd.FieldList
            The fields to be processed.

        Returns
        -------
        ekd.FieldList
            The processed fields.
        """
        return self.filter.forward(fields)

    def patch_data_request(self, data_request: Any) -> Any:
        """Patch the data request using the filter.

        Parameters
        ----------
        data_request : object
            The data request to be patched.

        Returns
        -------
        object
            The patched data request.
        """
        return self.filter.patch_data_request(data_request)
