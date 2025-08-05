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

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.inputs.mars import MarsInput
from anemoi.inference.inputs.mars import postproc
from anemoi.inference.types import DataRequest
from anemoi.inference.types import Date

LOG = logging.getLogger(__name__)


def retrieve(
    collection: str,
    requests: list[dict[str, Any]],
    grid: str | list[float] | None,
    area: list[float] | None,
    patch: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Retrieve data from Polytope.

    Parameters
    ----------
    collection : str
        The polytope collection to use for retrieval.
    requests : List[Dict[str, Any]]
        The list of requests to be retrieved.
    grid : Optional[Union[str, List[float]]]
        The grid for the retrieval.
    area : Optional[List[float]]
        The area for the retrieval.
    patch : Optional[Any], optional
        Optional patch for the request, by default None.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        The retrieved data.
    """
    import earthkit.data as ekd

    def _(r: DataRequest) -> str:
        mars = r.copy()
        for k, v in r.items():
            if isinstance(v, (list, tuple)):
                mars[k] = "/".join(str(x) for x in v)
            else:
                mars[k] = str(v)

        return ",".join(f"{k}={v}" for k, v in mars.items())

    pproc = postproc(grid, area)

    result = ekd.from_source("empty")
    for r in requests:
        if r.get("class") in ("rd", "ea"):
            r["class"] = "od"

        # ECMWF operational data has stream oper for 00 and 12 UTC and scda for 06 and 18 UTC

        if r.get("type") == "fc" and r.get("stream") == "oper" and r["time"] in ("0600", "1800"):
            r["stream"] = "scda"

        r.update(pproc)
        r.update(kwargs)

        if patch:
            r = patch(r)

        LOG.debug("%s", _(r))

        result += ekd.from_source("polytope", collection, r, stream=False)

    return result


@main_argument("collection")
class PolytopeInputPlugin(MarsInput):
    """Get input fields from Polytope."""

    trace_name = "polytope"

    def __init__(
        self,
        context: Context,
        collection: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the Polytope input plugin.

        Parameters
        ----------
        context : Context
            The context for the input plugin.
        collection : Optional[str], optional
            The collection to use for retrieval, by default 'ecmwf-mars'.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)
        self.collection = collection or "ecmwf-mars"

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve data for the given variables and dates.

        Parameters
        ----------
        variables : List[str]
            The list of variables to retrieve.
        dates : List[Any]
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
            patch_request=self.context.patch_data_request,
        )

        if not requests:
            raise ValueError(f"No requests for {variables} ({dates})")

        kwargs = self.kwargs.copy()
        kwargs.setdefault("expver", "0001")

        return retrieve(
            self.collection,
            requests,
            self.checkpoint.grid,
            self.checkpoint.area,
            self.patch,
            **kwargs,
        )
