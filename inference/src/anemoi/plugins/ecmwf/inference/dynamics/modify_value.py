# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Literal

import earthkit.data as ekd
import numpy as np
from anemoi.inference.processor import Processor

from ._operate_on_fields import apply_function_to_fields

VALID_METHODS = Literal["add", "subtract", "multiply", "divide", "replace"]
METHOD_FUNCTIONS = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "replace": lambda x, y: np.full_like(x, y),
}


class ModifyValuePlugin(Processor):
    """Modify the value of a field based on a specified method and value.

    Example
    --------
    ```yaml
    pre_processors:
        - modify_value:
            value: 2
            fields:
                - {"level": 850, "shortName": "t"}
            method: "add"
    ```
    """

    def __init__(self, context, fields: list[dict[str, Any]], value: float | str, method: VALID_METHODS = "add"):
        super().__init__(context)
        self.method = method
        if isinstance(value, str):
            value = np.load(value)
        self.value = value
        self.fields = fields

    def _modify_value(self, field: ekd.Field) -> ekd.Field:
        """Modify the value of the field based on the specified method and value."""
        if self.method not in METHOD_FUNCTIONS:
            raise ValueError(f"Invalid method: {self.method}. Valid methods are: {', '.join(METHOD_FUNCTIONS.keys())}")

        data = field.to_numpy()
        data = METHOD_FUNCTIONS[self.method](data, self.value)

        return ekd.ArrayField(data, field.metadata())  # type: ignore

    def process(self, fields: ekd.FieldList) -> ekd.FieldList:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Process the fields and apply the value modification."""
        return apply_function_to_fields(
            self._modify_value,
            fields,
            filter=self.fields,
        )
