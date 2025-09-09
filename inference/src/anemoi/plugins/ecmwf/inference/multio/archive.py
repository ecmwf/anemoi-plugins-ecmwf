# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
from collections import defaultdict
from typing import Any
from typing import TypedDict

from anemoi.inference.types import DataRequest


class Config(TypedDict):
    path: str
    extra: dict[str, Any] | None
    patch: dict[str, Any] | None
    indent: int | None


class ArchiveCollector:
    """Collects archive requests."""

    UNIQUE = {"date", "hdate", "time", "referenceDate", "type", "stream", "expver"}

    def __init__(self, config: Config) -> None:
        self.expect = 0
        self._config = config
        self._request = defaultdict(set)

    def add(self, field: dict[str, Any]) -> None:
        """Add a field to the archive request.

        Parameters
        ----------
        field : Dict[str,Any]
            The field dictionary.
        """
        self.expect += 1
        for k, v in field.items():
            self._request[k].add(str(v))
            if k in self.UNIQUE:
                if len(self._request[k]) > 1:
                    raise ValueError(f"Field {field} has different values for {k}: {self._request[k]}")

    @property
    def request(self) -> DataRequest:
        """Get the archive request."""
        return {k: sorted(v) for k, v in self._request.items()}

    def write(self, *, source: str, use_grib_paramid: bool = False) -> None:
        """Write the archive request to a file.

        Parameters
        ----------
        source : str
            The source where the data is saved to.
        use_grib_paramid : bool, optional
            Whether to use GRIB param ids, by default False.
        """
        path = self._config["path"]
        extra = self._config.get("extra") or {}
        patch = self._config.get("patch") or {}
        indent = self._config.get("indent", None)

        def _patch(r: DataRequest) -> DataRequest:
            if use_grib_paramid:
                param = r.get("param", [])
                if not isinstance(param, list):
                    param = [param]

                # Check if we're using param ids already
                try:
                    float(next(iter(param)))
                except ValueError:
                    from anemoi.utils.grib import shortname_to_paramid

                    r["param"] = [shortname_to_paramid(p) for p in param]

            for k, v in patch.items():
                if v is None:
                    r.pop(k, None)
                else:
                    r[k] = v

            return r

        with open(self._config["path"], "w") as f:
            requests = []

            assert path is not None, "Path is None"
            request: dict[str, Any] = dict(expect=self.expect)
            request["source"] = source
            request.update(_patch(self.request))
            request.update(extra)
            requests.append(request)

            json.dump(requests, f, indent=indent)
            f.write("\n")
