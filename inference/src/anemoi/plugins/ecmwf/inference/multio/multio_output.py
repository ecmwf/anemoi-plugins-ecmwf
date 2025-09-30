# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from datetime import timedelta
from typing import Any

import multio
import numpy as np
from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.output import Output
from anemoi.inference.post_processors.accumulate import Accumulate
from anemoi.inference.types import State
from anemoi.utils.grib import shortname_to_paramid
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from .archive import ArchiveCollector
from .archive import Config

CONVERT_PARAM_TO_PARAMID = True

LOG = logging.getLogger(__name__)


class UserDefinedMetadata(BaseModel):
    stream: str
    """Stream name, e.g. oper, enfo"""
    type: str
    """Type name, e.g. fc, an"""
    klass: str = Field(alias="class")
    """Class name, e.g. od, ai, ..."""
    expver: str | int
    """Experiment version, e.g. 0001"""
    model: str
    """Model name, e.g. aifs-single, ..."""
    number: int | None = None
    """Ensemble number, e.g. 0,1,2"""
    numberOfForecastsInEnsemble: int | None = Field(None, serialization_alias="misc-numberOfForecastsInEnsemble")
    """Number of ensembles in the forecast, e.g. 50"""
    generatingProcessIdentifier: int | None = Field(None, serialization_alias="misc-generatingProcessIdentifier")
    """Generating process identifier"""

    @field_validator("numberOfForecastsInEnsemble")
    def validate_number_of_forecasts(cls, numberOfForecastsInEnsemble, values):
        if isinstance(values.get("number"), int) and not isinstance(numberOfForecastsInEnsemble, int):
            raise ValueError("numberOfForecastsInEnsemble must be an integer if number is provided")
        return numberOfForecastsInEnsemble


class MultioMetadata(BaseModel):
    param: int
    """Param ID, e.g. 130"""
    levtype: str
    """Level type, e.g. sfc,pl,soil"""
    date: int
    """Reference date, e.g. 20220101"""
    time: int
    """Reference time, e.g. 1200"""
    step: int
    """Forecast step, e.g. 0,6,12,24"""
    grid: str
    """Grid name, e.g. n320, o96"""
    levelist: int | None = None
    """Level, e.g. 0,50,100"""

    timespan: int | None = None
    """Time span, e.g."""

    origin: str | None = None
    """Origin name, e.g. ecmf, ukmo"""
    packing: str | None = "ccsds"
    """Packing type, e.g. ccsds"""
    repres: str | None = None
    """Representation type"""

    @model_validator(mode="after")
    def set_repres(self):
        if self.repres is None:
            grid = self.grid.upper()
            if any(grid.startswith(prefix) for prefix in ["N", "O"]):
                self.repres = "gg"
            else:
                self.repres = "ll"
        return self


def _to_mars(metadata: MultioMetadata, user_metadata: UserDefinedMetadata) -> dict[str, Any]:
    """Convert MultioMetadata and UserDefinedMetadata to a MARS request dictionary for use with the ArchiveCollector."""
    mars_dict = {
        "levtype": metadata.levtype,
        "date": metadata.date,
        "time": metadata.time // 10000,
        "step": metadata.step,
        "param": metadata.param,
        "class": user_metadata.klass,
        "type": user_metadata.type,
        "stream": user_metadata.stream,
        "expver": str(user_metadata.expver),
    }
    if metadata.levelist is not None:
        mars_dict["levelist"] = metadata.levelist // 100
    if user_metadata.number is not None:
        mars_dict["number"] = user_metadata.number

    return mars_dict


class MultioOutputPlugin(Output):

    api_version = "1.0.0"
    schema = None

    source: str = "multio"
    _server: multio.Multio | None = None

    def __init__(
        self,
        context: Context,
        plan: str | dict | multio.plans.Client | multio.plans.Server,
        *,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        archive_requests: Config | None = None,
        initial_state_diagnostics_grib: str | None = None,
        **metadata: Any,
    ) -> None:
        super().__init__(
            context,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
        self._plan = plan
        self._archiver = ArchiveCollector(archive_requests) if archive_requests else None
        self._initial_state_diagnostics_grib = initial_state_diagnostics_grib

        try:
            self._user_defined_metadata = UserDefinedMetadata(**metadata)
        except TypeError as e:
            raise TypeError(f"Invalid metadata: {e}") from e

        dumped_plan = (
            self._plan.dump_yaml() if isinstance(self._plan, multio.plans.plans.MultioBaseModel) else self._plan
        )
        LOG.info("Using Multio plan:\n%s", dumped_plan)

    def open(self, state: State) -> None:
        if self._server is None:
            with multio.MultioPlan(self._plan):  # type: ignore
                self._server = multio.Multio()

        self._server.open_connections()
        self._server.write_parametrization(self._user_defined_metadata.model_dump(exclude_none=True, by_alias=True))

    def write_initial_state(self, state: State) -> None:
        """Write the initial step of the state.

        Parameters
        ----------
        state : State
            The state object.
        """

        state = state.copy()

        self.reference_date = state["date"]
        state.setdefault("step", timedelta(0))

        if self._initial_state_diagnostics_grib:
            self._copy_initial_state_diagnostics(state)

        return self.write_step(state)

    def _copy_initial_state_diagnostics(self, state: State) -> None:
        import earthkit.data as ekd

        ds = ekd.from_source("file", self._initial_state_diagnostics_grib)
        namer = self.context.checkpoint.default_namer()

        LOG.info(f"Copying step 0 diagnostic fields from {self._initial_state_diagnostics_grib} to output:")
        for field in ds:  # type: ignore
            name = namer(field, field.metadata())
            if name in state["fields"]:
                raise ValueError(f"Field {name!r} already exists in the initial state.")
            state["fields"][name] = field.to_numpy()
            LOG.info(f"+ {name}")

    def write_step(self, state: State) -> None:
        """Write a step of the state with multio."""
        if self._server is None:
            raise RuntimeError("Multio server is not open, call `.open()` first.")

        reference_date = self.reference_date or self.context.reference_date
        step = state["step"]

        shared_metadata = {
            "step": int(step.total_seconds() // 3600),
            "grid": str(self.context.checkpoint.grid).upper(),
            "date": int(reference_date.strftime("%Y%m%d")),  # type: ignore
            "time": int(reference_date.strftime("%H%M%S")),  # type: ignore
        }

        timespan = self.context.checkpoint.timestep.total_seconds() // 3600
        if any(isinstance(x, Accumulate) for x in self.context.create_post_processors()):  # type: ignore
            timespan = shared_metadata["step"]

        for param, field in state["fields"].items():
            variable = self.typed_variables[param]
            if variable.is_computed_forcing:
                continue

            param = variable.grib_keys.get("param", param)
            if CONVERT_PARAM_TO_PARAMID:
                param = shortname_to_paramid(param)

            levtype = variable.grib_keys.get("levtype")
            assert levtype is not None, f"levtype must be defined for variable {variable.name!r}"

            metadata = MultioMetadata(
                param=param,
                levtype=levtype,
                levelist=variable.level * 100 if not variable.is_surface_level else None,
                timespan=int(timespan) if variable.is_accumulation else None,
                **shared_metadata,
            )
            # Copy the field to ensure it is contiguous
            # Removes ValueError: ndarray is not C-contiguous
            # Replace NaNs with a missing value
            field = field.copy(order="C")
            missing_value = float(-999999.0)

            if np.isnan(field).any():
                field = np.nan_to_num(field, nan=missing_value)  # type: ignore

            self._server.write_field(
                {**metadata.model_dump(exclude_none=True, by_alias=True), "misc-missingValue": missing_value}, field
            )
            if self._archiver:
                self._archiver.add(_to_mars(metadata, self._user_defined_metadata))

        self._server.flush()

    def close(self) -> None:
        if self._server is None:
            raise RuntimeError("Multio server is not open to close, call `.open()` first.")

        self._server.close_connections()
        self._server = None

        if self._archiver:
            self._archiver.write(source=self.source, use_grib_paramid=self.context.use_grib_paramid)


def add_debug(locations: dict[int, str], plan: multio.plans.Plan) -> None:
    """Add debug print actions in place to a multio plan at specified locations.

    Parameters
    ----------
    locations : dict[int, str]
        A dictionary mapping action indices to debug prefixes.
    plan : multio.plans.Plan
        The multio plan to modify.
    """
    for index, prefix in sorted(locations.items(), reverse=True):
        plan.actions.insert(index, multio.plans.Print(stream="cout", prefix=prefix, only_fields=False))


@main_argument("path")
class MultioOutputGribPlugin(MultioOutputPlugin):
    """Multio output plugin for GRIB files.

    This plugin uses the multio library to write GRIB files.
    It is a subclass of the MultioOutputPlugin class.
    """

    def __init__(
        self,
        context: Context,
        path: str,
        append: bool = False,
        per_server: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """Multio Grib Output Plugin.

        Parameters
        ----------
        context : Context
            Model Runner
        path : str
            Path to write to
        append : bool
            Whether to append to the file or not
        per_server : bool
            Whether to write to a separate file per server or not
        debug : bool, optional
            Whether to enable debug output or not, default is False
        """
        self.source = path

        plan = multio.plans.Client(
            plans=[
                multio.plans.Plan(
                    name="output-to-file",
                    actions=[
                        multio.plans.EncodeMTG(geo_from_atlas=True),
                        multio.plans.Sink(
                            sinks=[
                                multio.plans.sinks.File(
                                    append=append,
                                    per_server=per_server,
                                    path=path,
                                )
                            ]
                        ),
                    ],
                )
            ]
        )
        if debug:
            add_debug({0: "MULTIO PRE-ENC DEBUG: ", 2: "MULTIO PST-ENC DEBUG: "}, plan.plans[0])

        super().__init__(context, plan=plan, **kwargs)


@main_argument("fdb_config")
class MultioOutputFDBPlugin(MultioOutputPlugin):
    """Multio output plugin to write to FDB.

    This plugin uses the multio library to write to FDB.
    It is a subclass of the MultioOutputPlugin class.
    """

    def __init__(self, context: Context, fdb_config: str, debug: bool = False, **kwargs: Any) -> None:
        """Multio FDB Output Plugin.

        Parameters
        ----------
        context : Context
            Model Runner
        fdb_config : str
            FDB Configuration file
        debug : bool, optional
            Whether to enable debug output or not, default is False
        """
        self.source = "fdb_config"

        plan = multio.plans.Client(
            plans=[
                multio.plans.Plan(
                    name="output-to-fdb",
                    actions=[
                        multio.plans.EncodeMTG(geo_from_atlas=True),
                        multio.plans.Sink(
                            sinks=[
                                multio.plans.sinks.FDB(
                                    config=fdb_config,
                                )
                            ]
                        ),
                    ],
                )
            ]
        )
        if debug:
            add_debug({0: "MULTIO PRE-ENC DEBUG: ", 2: "MULTIO PST-ENC DEBUG: "}, plan.plans[0])

        super().__init__(context, plan=plan, **kwargs)


@main_argument("plan")
class MultioOutputPlanPlugin(MultioOutputPlugin):
    """Multio output plugin to write with a plan."""

    def __init__(
        self, context: Context, plan: str | dict, *, sinks: list[multio.plans.sinks.SINKS] | None = None, **kwargs: Any
    ) -> None:
        """Multio FDB Output Plugin.

        Parameters
        ----------
        context : Context
            Model Runner
        plan : str | dict
            Multio Plan
        sinks : list[multio.plans.sinks.SINKS] | None
            List of sinks to use in the plan, will be appended to the end of the plan.
            If the plan contains sinks and sinks is not None, an exception is raised
            default is None
        """
        if sinks:
            realised_plan = (
                multio.plans.Client(**plan) if isinstance(plan, dict) else multio.plans.Client.from_yamlfile(plan)
            )
            if any(isinstance(action, multio.plans.sinks.SINKS) for p in realised_plan.plans for action in p.actions):
                raise ValueError("The plan already contains sinks, cannot add additional sinks.")

            for p in realised_plan.plans:
                p.actions.append(multio.plans.Sink(sinks=sinks))
            plan = realised_plan  # type: ignore

        super().__init__(context, plan=plan, **kwargs)


class MultioDisambiguousOutputPlugin(MultioOutputPlugin):
    """Provide a class to delegate to the correct MultioOutputPlugin subclass based on arguments."""

    def __new__(cls, *args, **kwargs):
        plugins = [
            ("plan", MultioOutputPlanPlugin),
            ("path", MultioOutputGribPlugin),
            ("fdb_config", MultioOutputFDBPlugin),
        ]

        selected_plugins = [(key, plugin) for key, plugin in plugins if key in kwargs]

        if len(selected_plugins) != 1:
            raise ValueError(
                "Must provide exactly one of 'plan', 'path', or 'fdb_config' keyword arguments to delegate to the correct MultioOutputPlugin subclass."
            )

        _, plugin = selected_plugins[0]
        return plugin(*args, **kwargs)
