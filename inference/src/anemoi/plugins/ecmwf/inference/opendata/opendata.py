# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib.resources
import logging
from dataclasses import dataclass
from string import Formatter
from typing import Any

import earthkit.data as ekd
import yaml
from anemoi.inference.context import Context
from anemoi.inference.inputs.mars import MarsInput
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.utils.grib import shortname_to_paramid

from ..regrid import regrid as ekr
from .geopotential_height import OrographyProcessor

LOG = logging.getLogger(__name__)

# Constants for mapping configuration
POP_SENTINEL = "%POP%"  # Sentinel value to indicate key should be removed from request
RESERVED_MAPPING_KEYS = {"param", "inverse"}  # Keys with special meaning in mappings

MAPPINGS: dict[str, dict[str, dict[str, str]]] = yaml.safe_load(
    importlib.resources.files("anemoi.plugins.ecmwf.inference.opendata").joinpath("mappings.yaml").read_text()
)


def _merge_request_value(existing: Any, new_value: Any) -> list:
    """Merge a new value into an existing request field.

    Parameters
    ----------
    existing : Any
        The existing value in the request (can be a list or scalar).
    new_value : Any
        The new value to merge.

    Returns
    -------
    list
        A list containing the merged unique values.
    """
    if isinstance(existing, list):
        return list(set(existing) | {new_value})
    return list({existing, new_value})


def _apply_mapping_to_request(
    base_request: dict[str, Any],
    params_to_map: list[str],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply parameter mappings to create a new request.

    Parameters
    ----------
    base_request : dict[str, Any]
        The base request to apply mappings to.
    params_to_map : list[str]
        List of parameter names to map.
    mapping : dict[str, dict[str, Any]]
        The mapping configuration for these parameters.

    Returns
    -------
    dict[str, Any]
        A new request with mappings applied.
    """
    new_request = base_request.copy()
    new_request["param"] = [mapping[p]["param"] for p in params_to_map]

    for param in params_to_map:
        for key, value in mapping[param].items():
            if key in RESERVED_MAPPING_KEYS:
                continue
            elif value == POP_SENTINEL:
                new_request.pop(key, None)
            elif key in new_request:
                new_request[key] = _merge_request_value(new_request[key], value)
            else:
                new_request[key] = value

    return new_request


def _expand_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand request based on mappings.

    Parameters
    ----------
    request : dict[str, Any]
        The original request.

    Returns
    -------
    list[dict[str, Any]]
        The expanded requests.
    """
    expanded_requests = []
    request = request.copy()

    for group, mapping in MAPPINGS.items():
        if any(param in mapping for param in request.get("param", [])):
            params_to_map = [p for p in request["param"] if p in mapping]
            request["param"] = [p for p in request["param"] if p not in params_to_map]  # Remove mapped params

            new_request = _apply_mapping_to_request(request, params_to_map, mapping)
            expanded_requests.append(new_request)

    expanded_requests.append(request)
    return expanded_requests


@dataclass
class InvertedMapping:
    """Mapping for converting retrieved parameters back to expected names.

    Attributes
    ----------
    true_param : str
        Target parameter name (e.g., "stl1").
    id_pattern : str
        Pattern to match against actual field metadata (e.g., "{param}{levelist}").
    expected_pattern : str
        Expected pattern after substitution with mapping attributes.
    request_attrs : dict
        Additional attributes from the mapping configuration.
    """

    true_param: str
    id_pattern: str
    expected_pattern: str
    request_attrs: dict

    def matches(self, field_metadata: dict) -> bool:
        """Check if this mapping applies to the given field metadata.

        Parameters
        ----------
        field_metadata : dict
            The metadata from the field to check.

        Returns
        -------
        bool
            True if this mapping matches the field, False otherwise.
        """
        # Extract all format keys from the id pattern
        required_keys = [key[1] for key in Formatter().parse(self.id_pattern) if key[1] is not None]

        # Check if all required keys are present in the metadata
        if not all(key in field_metadata for key in required_keys):
            return False

        # Build expected metadata by applying mapping attributes
        expected_metadata = field_metadata.copy()
        expected_metadata.update(self.request_attrs)

        # Check if patterns match after substitution
        actual = self.id_pattern.format(**field_metadata)
        expected = self.expected_pattern.format(**expected_metadata)
        return actual == expected


def _build_inverse_mappings() -> list[InvertedMapping]:
    """Build inverse mappings from MAPPINGS configuration.

    Returns
    -------
    list[InvertedMapping]
        List of inverse mappings for parameter renaming.
    """
    inverse_mappings = []
    for group, mapping in MAPPINGS.items():
        for param, details in mapping.items():
            inverse = details.get("inverse")
            if inverse:
                expected = inverse
                if "==" in inverse:
                    inverse, expected = inverse.split("==")
                inverse_mappings.append(
                    InvertedMapping(
                        true_param=param,
                        id_pattern=inverse,
                        expected_pattern=expected,
                        request_attrs=details.copy(),
                    )
                )
    return inverse_mappings


# Build inverse mappings once at module load time
INVERSE_MAPPINGS = _build_inverse_mappings()


def _rename_params(fieldlist: ekd.FieldList) -> ekd.FieldList:
    """Rename params to match the expected format.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist with parameters to rename.

    Returns
    -------
    ekd.FieldList
        The fieldlist with renamed parameters.
    """
    for field in fieldlist:
        field_metadata = dict(field.metadata().as_namespace("mars"))

        for inv in INVERSE_MAPPINGS:
            if inv.matches(field_metadata):
                field._metadata = field.metadata().override(paramId=shortname_to_paramid(inv.true_param))  # type: ignore
                break

    return fieldlist


def retrieve(
    requests: list[dict[str, Any]],
    grid: str | list[float] | None,
    area: list[float] | None,
    patch: Any | None = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Retrieve data from ECMWF Opendata.

    Parameters
    ----------
    requests : list[dict[str, Any]]
        The list of requests to be retrieved.
    grid : Optional[Union[str, list[float]]]
        The grid for the retrieval.
    area : Optional[list[float]]
        The area for the retrieval.
    patch : Optional[Any], optional
        Optional patch for the request, by default None.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The retrieved data.
    """

    def _(r: DataRequest):
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    result = ekd.SimpleFieldList()
    expanded_requests = [req for r in requests for req in _expand_request(r)]

    for r in expanded_requests:
        r.update(kwargs)
        if r.get("class") in ("rd", "ea"):
            r["class"] = "od"

        if patch:
            r = patch(r)

        LOG.debug("%s", _(r))
        result += ekr.regrid(ekd.from_source("ecmwf-open-data", r), grid, area)  # type: ignore

    return _rename_params(result)  # type: ignore


class OpenDataInputPlugin(MarsInput):
    """Get input fields from ECMWF open-data."""

    trace_name = "opendata"

    def __init__(
        self,
        context: Context,
        **kwargs: Any,
    ) -> None:
        """Initialise the OpenDataInput.

        Parameters
        ----------
        context : Any
            The context in which the input is used.
        """
        rules_for_namer = [
            ({"levtype": "sol"}, "{param}"),
        ]
        kwargs.pop("namer", None)  # Ensure namer is not passed to MarsInput
        super().__init__(context, namer={"rules": rules_for_namer}, **kwargs)
        self.pre_processors.append(OrographyProcessor(context=context, orog="gh"))

        if self.context.use_grib_paramid:
            LOG.warning("`use_grib_paramid=True` is not supported for ECMWF Open Data and will be ignored.")

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve data for the given variables and dates.

        Parameters
        ----------
        variables : list[str]
            The list of variables to retrieve.
        dates : list[Any]
            The list of dates for which to retrieve the data.

        Returns
        -------
        Any
            The retrieved data.
        """

        requests = self.checkpoint.mars_requests(
            variables=variables,
            dates=dates,
            use_grib_paramid=False,
            type="fc",
        )

        if not requests:
            raise ValueError(f"No requests for {variables} ({dates})")

        kwargs = self.kwargs.copy()

        return retrieve(
            requests,
            self.checkpoint.grid,
            self.checkpoint.area,
            patch=self.patch_data_request,
            **kwargs,
        )
