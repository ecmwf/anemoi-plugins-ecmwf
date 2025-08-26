# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Callable

import earthkit.data as ekd


def _metadata_matches(metadata: dict[str, Any], filter_criteria: dict[str, Any]) -> bool:
    """Check if metadata matches the filter criteria."""
    return all(key in metadata and metadata[key] == val for key, val in filter_criteria.items())


def filter_matches(metadata: dict[str, Any], filter_criteria: list[dict[str, Any]]) -> bool:
    """Check if metadata matches any of the filter criteria."""
    return any(_metadata_matches(metadata, criteria) for criteria in filter_criteria)


def apply_function_to_fields(
    func: Callable[[ekd.Field], ekd.Field], fields: ekd.FieldList, filter: list[dict[str, Any]]
) -> ekd.FieldList:
    """Apply a function to fields in a FieldList based on metadata filter criteria."""
    result = []
    for field in fields:
        metadata = dict(field.metadata())
        if filter_matches(metadata, filter or []):
            field = func(field)  # type: ignore
        result.append(field)
    return ekd.SimpleFieldList(result)
