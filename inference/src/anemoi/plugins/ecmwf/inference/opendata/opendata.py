# (C) Copyright 2025- ECMWF.
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
from anemoi.inference.context import Context
from anemoi.inference.inputs.mars import MarsInput
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date
from anemoi.utils.grib import shortname_to_paramid

from ..regrid import regrid as ekr
from .geopotential_height import OrographyProcessor

LOG = logging.getLogger(__name__)

SOIL_MAPPING = {"stl1": "sot", "stl2": "sot", "stl3": "sot", "swvl1": "vsw", "swvl2": "vsw", "swvl3": "vsw"}


def _retrieve_soil(request: dict, soil_params: list[str]) -> ekd.FieldList:
    """Retrieve soil data.

    Map the soil parameters to the correct ECMWF parameter IDs and levels.

    Parameters
    ----------
    request : dict
        Request for the soil data.
    soil_params : list[str]
        Parameters to be retrieved.

    Returns
    -------
    ekd.FieldList
        Soil data.
    """
    request = request.copy()

    request["param"] = list(SOIL_MAPPING[s] for s in soil_params)
    request["levelist"] = list({int(s[-1]) for s in soil_params})
    request.pop("levtype")

    soil_data = ekd.from_source("ecmwf-open-data", request)
    assert isinstance(soil_data, ekd.FieldList), "Expected a FieldList from the soil data request"
    return soil_data


def rename_soildata(soil_data: ekd.FieldList) -> ekd.FieldList:
    """Rename soil data param to match the expected format."""
    for field in soil_data:
        newname = {f"{v}{k[-1]}": k for k, v in SOIL_MAPPING.items()}[
            f"{field.metadata()['param']}{field.metadata()['level']}"
        ]
        field._metadata = field.metadata().override(paramId=shortname_to_paramid(newname))  # type: ignore

    return soil_data


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
    for r in requests:
        r.update(kwargs)
        if r.get("class") in ("rd", "ea"):
            r["class"] = "od"

        if patch:
            r = patch(r)

        if any(k in r["param"] for k in SOIL_MAPPING.keys()):
            requested_soil_variables = [k for k in SOIL_MAPPING.keys() if k in r["param"]]
            r["param"] = [p for p in r["param"] if p not in requested_soil_variables]
            result += rename_soildata(ekr.regrid(_retrieve_soil(r, requested_soil_variables), grid, area))

        LOG.debug("%s", _(r))
        result += ekr.regrid(ekd.from_source("ecmwf-open-data", r), grid, area)  # type: ignore

    return result


class OpenDataInputPlugin(MarsInput):
    """Get input fields from ECMWF open-data."""

    trace_name = "opendata"

    def __init__(
        self,
        context: Context,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenDataInput.

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
            use_grib_paramid=self.context.use_grib_paramid,
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
