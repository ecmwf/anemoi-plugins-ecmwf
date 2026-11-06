# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The logic of ``MirTemplatesProvider``, without MIR.

``MIRRegrid`` (as the provider module sees it) and ``MirTemplatesProvider._grid_size``
are stubbed, as is the reading of the base template, so what the provider
decides can be checked from what it asks of them. The templates MIR builds
are tested in ``test_mir.py``.
"""

import logging
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from anemoi.inference.grib.templates import create_template_provider
from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.inference.mir_templates import mir_templates as provider_module
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid
from anemoi.plugins.ecmwf.transform.regrid.backend import canonical_lambert_grid
from cases import LAMBERT_GRID
from cases import LAMBERT_NX
from cases import LAMBERT_NY
from cases import MEPS_LAMBERT_GRID
from cases import MEPS_NPOINTS

# The real ``_grid_size``, which the ``stubs`` fixture replaces
real_grid_size = MirTemplatesProvider._grid_size

STATE = {"latitudes": [50.0, 51.0, 52.0], "longitudes": [0.0, 1.0, 2.0]}
# What the provider builds a grid from, for that state
STATE_COORDINATES = {"latitudes": [50.0, 51.0, 52.0], "longitudes": [0.0, 1.0, 2.0]}
COORDINATE_GRID = {"latitudes": [1.0, 2.0], "longitudes": [3.0, 4.0]}


class FakeTemplate:
    """What ``MIRRegrid.forward`` returns: only the number of points is read."""

    def __init__(self, npoints: int) -> None:
        self.npoints = npoints

    def metadata(self, key: str) -> int:
        assert key == "numberOfDataPoints"
        return self.npoints


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub ``MIRRegrid`` and ``_grid_size`` in the provider module, and record how they are used.

    ``regrids``: every MIRRegrid built, in order, each with the keyword arguments it got in ``kwargs``.
    ``forwards``: the MIRRegrid whose ``forward`` was called, in order.
    ``sizes``: the arguments of every ``_grid_size`` call.
    ``grid_npoints``: the size ``_grid_size`` gives, and the number of points a forward gives to a lookup grid.
    ``forward_npoints``: if not None, the number of points of every forward.
    ``forward_error`` and ``size_error``: raised by forward and ``_grid_size`` if not None.
    """
    recorded = SimpleNamespace(
        regrids=[],
        forwards=[],
        sizes=[],
        base_lookups=[],
        grid_npoints=len(STATE["latitudes"]),
        forward_npoints=None,
        forward_error=None,
        size_error=None,
    )

    class RecordingRegrid(MIRRegrid):
        """The real constructor (it resolves named grids, and needs no MIR), and a forward that does not regrid."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.kwargs = kwargs
            recorded.regrids.append(self)

        def forward(self, fields: Any) -> list[FakeTemplate]:
            recorded.forwards.append(self)
            if recorded.forward_error is not None:
                raise recorded.forward_error
            if recorded.forward_npoints is not None:
                return [FakeTemplate(recorded.forward_npoints)]
            if isinstance(self.grid, dict) and "latitudes" in self.grid:
                return [FakeTemplate(len(self.grid["latitudes"]))]
            return [FakeTemplate(recorded.grid_npoints)]

    def grid_size(grid: Any, area: Any) -> int:
        recorded.sizes.append((grid, area))
        if recorded.size_error is not None:
            raise recorded.size_error
        return recorded.grid_npoints

    def base_template(variable: str, lookup: dict) -> bytes:
        recorded.base_lookups.append((variable, lookup))
        return b"base"

    monkeypatch.setattr(provider_module, "MIRRegrid", RecordingRegrid)
    monkeypatch.setattr(provider_module, "ekd", SimpleNamespace(from_source=lambda source, data: f"{source}:{data}"))
    monkeypatch.setattr(MirTemplatesProvider, "_grid_size", staticmethod(grid_size))
    recorded.base_template = base_template
    return recorded


@pytest.fixture
def stubbed_provider(provider: MirTemplatesProvider, stubs: SimpleNamespace) -> MirTemplatesProvider:
    provider._base_template_provider = SimpleNamespace(template=stubs.base_template)
    return provider


def test_registration():
    template_provider = create_template_provider(None, "mir")  # type: ignore
    message = "Template provider should be an instance of MirTemplatesProvider"

    assert isinstance(template_provider, MirTemplatesProvider), message


class TestCoordinates:
    """``_coordinates``: the usable latitudes and longitudes of a state, flat, or None."""

    @pytest.mark.parametrize(
        "state",
        [
            pytest.param(None, id="none"),
            pytest.param({}, id="empty"),
            pytest.param({"latitudes": [1.0]}, id="only-latitudes"),
            pytest.param({"longitudes": [1.0]}, id="only-longitudes"),
            pytest.param({"latitudes": None, "longitudes": [1.0]}, id="latitudes-none"),
            pytest.param({"latitudes": [1.0], "longitudes": None}, id="longitudes-none"),
            pytest.param({"latitudes": [], "longitudes": []}, id="size-zero"),
            pytest.param({"latitudes": np.zeros(0), "longitudes": np.zeros(0)}, id="size-zero-array"),
            pytest.param({"latitudes": [1.0, 2.0], "longitudes": [1.0]}, id="size-mismatch"),
        ],
    )
    def test_unusable_state(self, state: dict | None):
        assert MirTemplatesProvider._coordinates(state) is None

    def test_lists_become_float_arrays(self):
        latitudes, longitudes = MirTemplatesProvider._coordinates({"latitudes": [1, 2], "longitudes": [3, 4]})

        assert latitudes.dtype == longitudes.dtype == np.float64
        np.testing.assert_array_equal(latitudes, [1.0, 2.0])
        np.testing.assert_array_equal(longitudes, [3.0, 4.0])

    def test_two_dimensional_state_is_flattened(self):
        lat2d, lon2d = np.meshgrid([3.0, 2.0, 1.0], [10.0, 20.0], indexing="ij")

        latitudes, longitudes = MirTemplatesProvider._coordinates({"latitudes": lat2d, "longitudes": lon2d})

        assert latitudes.shape == longitudes.shape == (6,)
        np.testing.assert_array_equal(latitudes, lat2d.ravel())
        np.testing.assert_array_equal(longitudes, lon2d.ravel())


class TestLambertGridSize:
    """``_grid_size`` of a Lambert grid (a dict of GRIB keys) is ``Nx * Ny``, read from the keys, without MIR."""

    @pytest.fixture(autouse=True)
    def no_mir(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "mir", None)

    def test_size_is_nx_times_ny(self):
        assert real_grid_size(LAMBERT_GRID, None) == LAMBERT_NX * LAMBERT_NY

    def test_the_meps_grid(self):
        assert real_grid_size(MEPS_LAMBERT_GRID, None) == MEPS_NPOINTS == 822681

    def test_area_is_ignored(self):
        assert real_grid_size(LAMBERT_GRID, [60, 0, 30, 40]) == LAMBERT_NX * LAMBERT_NY

    def test_the_keys_of_a_grib_message_are_accepted(self):
        """``gridDefinitionTemplateNumber`` and ``tablesVersion`` are ignored."""
        grid = {**LAMBERT_GRID, "gridDefinitionTemplateNumber": 30, "tablesVersion": 32}

        assert real_grid_size(grid, None) == LAMBERT_NX * LAMBERT_NY

    @pytest.mark.parametrize("key", ["Nx", "Ny", "DxInMetres", "LaDInDegrees", "latitudeOfFirstGridPointInDegrees"])
    def test_missing_required_key_raises(self, key: str):
        with pytest.raises(ValueError, match="missing required key"):
            real_grid_size({k: v for k, v in LAMBERT_GRID.items() if k != key}, None)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown key"):
            real_grid_size({**LAMBERT_GRID, "Bogus": 1}, None)

    @pytest.mark.parametrize("value", [2, 3, 5, 7])
    def test_unsupported_earth_raises(self, value: int):
        """An earth shape that cannot be encoded correctly is refused when the lookup is sized."""
        with pytest.raises(ValueError, match="not supported"):
            real_grid_size({**LAMBERT_GRID, "shapeOfTheEarth": value}, None)

    @pytest.mark.parametrize(
        "parallels",
        [
            {"Latin1InDegrees": 50.0, "Latin2InDegrees": 58.0},
            {"Latin1InDegrees": 50.0},
            {"Latin2InDegrees": 60.0},
        ],
    )
    def test_secant_lambert_raises(self, parallels: dict):
        """MIR writes a tangent header for any parallels, so only tangent Lambert grids are accepted."""
        with pytest.raises(ValueError, match="only tangent grids are supported"):
            real_grid_size({**LAMBERT_GRID, **parallels}, None)

    @pytest.mark.parametrize("mode", [0, 128])
    def test_unsupported_scanning_mode_raises(self, mode: int):
        with pytest.raises(ValueError, match="only scanningMode=64"):
            real_grid_size({**LAMBERT_GRID, "scanningMode": mode}, None)


class TestNoGrid:
    @pytest.mark.parametrize(
        "state",
        [None, {}, {"latitudes": [1.0]}, {"latitudes": [], "longitudes": []}],
        ids=["no-state", "empty-state", "latitudes-only", "size-zero"],
    )
    @pytest.mark.parametrize("lookup", [{"levtype": "sfc"}, {"levtype": "sfc", "grid": None}], ids=["missing", "none"])
    def test_no_grid_and_no_usable_state_returns_none(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, lookup: dict, state: dict | None
    ):
        kwargs = {} if state is None else {"state": state}

        assert stubbed_provider.template("lsm", lookup, **kwargs) is None
        assert stubs.regrids == []
        assert stubs.base_lookups == []

    @pytest.mark.parametrize("lookup", [{"levtype": "sfc"}, {"levtype": "sfc", "grid": None}], ids=["missing", "none"])
    def test_no_grid_uses_the_state(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, lookup: dict):
        stubbed_provider.template("lsm", lookup, state=STATE)

        assert [regrid.kwargs for regrid in stubs.regrids] == [{"grid": STATE_COORDINATES}]
        assert stubs.forwards == stubs.regrids
        assert stubs.sizes == []

    def test_two_dimensional_state_is_flattened(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace):
        lat2d, lon2d = np.meshgrid([3.0, 2.0], [10.0, 20.0, 30.0], indexing="ij")

        stubbed_provider.template("lsm", {"levtype": "sfc"}, state={"latitudes": lat2d, "longitudes": lon2d})

        assert stubs.regrids[0].kwargs == {
            "grid": {"latitudes": lat2d.ravel().tolist(), "longitudes": lon2d.ravel().tolist()}
        }


class TestLookupToRegrid:
    @pytest.mark.parametrize(
        ("grid", "expected"),
        [
            (0.5, "0.5/0.5"),
            (1.0, "1.0/1.0"),
            (1, "1/1"),
            ("0.5/0.5", "0.5/0.5"),
            ([0.5, 0.5], [0.5, 0.5]),
            ("O96", "O96"),
        ],
    )
    def test_bare_number_grid_becomes_increment_pair(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, grid: Any, expected: Any
    ):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": grid})

        assert stubs.regrids[0].kwargs["grid"] == expected

    @pytest.mark.parametrize("with_state", [False, True])
    def test_grid_and_area_are_passed(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, with_state: bool
    ):
        lookup = {"levtype": "sfc", "grid": "O96", "area": [60, 0, 30, 40]}
        kwargs = {"state": STATE} if with_state else {}

        stubbed_provider.template("lsm", lookup, **kwargs)

        assert stubs.regrids[0].kwargs == {"grid": "O96", "area": [60, 0, 30, 40]}
        assert stubs.forwards == [stubs.regrids[0]]
        if with_state:
            assert stubs.sizes == [("O96", [60, 0, 30, 40])]
        else:
            assert stubs.sizes == []

    def test_missing_area_is_none(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=STATE)

        assert stubs.regrids[0].kwargs == {"grid": "O96", "area": None}
        assert stubs.sizes == [("O96", None)]

    def test_the_lookup_is_not_modified(self, stubbed_provider: MirTemplatesProvider):
        lookup = {"levtype": "sfc", "grid": "O96", "area": [60, 0, 30, 40]}
        expected = dict(lookup)

        stubbed_provider.template("lsm", lookup, state=STATE)

        assert lookup == expected

    def test_the_base_template_is_read_for_the_variable(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubbed_provider.template("2t", {"levtype": "sfc", "grid": "O96"})

        assert [variable for variable, _ in stubs.base_lookups] == ["2t"]

    def test_missing_base_template_raises(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(provider_module, "ekd", SimpleNamespace(from_source=lambda source, data: None))

        with pytest.raises(ValueError, match="Base template not found"):
            stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96"})

        assert stubs.regrids == []


class TestGridChoice:
    """With a state, the lookup grid is used if it has as many points as the state, else the state coordinates."""

    def test_no_state_uses_the_lookup_grid_without_size_check(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96"})

        assert [regrid.kwargs["grid"] for regrid in stubs.forwards] == ["O96"]
        assert stubs.sizes == []

    def test_size_match_uses_the_lookup_regrid(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace):
        stubs.grid_npoints = len(STATE["latitudes"])

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96", "area": "10/0/0/10"}, state=STATE)

        assert len(stubs.regrids) == 1
        assert stubs.forwards == stubs.regrids
        assert stubs.forwards[0].kwargs["grid"] == "O96"

    def test_size_mismatch_uses_the_state_coordinates(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubs.grid_npoints = 40320

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96", "area": "10/0/0/10"}, state=STATE)

        assert len(stubs.regrids) == 2
        assert stubs.regrids[0].kwargs["grid"] == "O96"
        # The state regrid has no area: the area is dropped with the lookup grid
        assert stubs.regrids[1].kwargs == {"grid": STATE_COORDINATES}
        assert stubs.forwards == [stubs.regrids[1]]

    @pytest.mark.parametrize("area", [None, [51, 0, 49, 1]], ids=["no-area", "area"])
    @pytest.mark.parametrize("grid", [COORDINATE_GRID, "meps"], ids=["coordinates", "named"])
    def test_coordinate_lookup_grid_skips_the_size_check(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, grid: Any, area: Any
    ):
        # The size would match, if it were asked
        stubs.grid_npoints = len(STATE["latitudes"])

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": grid, "area": area}, state=STATE)

        assert stubs.sizes == []
        assert len(stubs.regrids) == 2
        assert set(stubs.regrids[0].grid) == {"latitudes", "longitudes"}
        assert stubs.regrids[1].kwargs == {"grid": STATE_COORDINATES}
        assert stubs.forwards == [stubs.regrids[1]]

    def test_named_grid_without_state_is_the_lookup_regrid(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "meps"})

        assert stubs.sizes == []
        assert len(stubs.forwards) == 1
        assert set(stubs.forwards[0].grid) == {"latitudes", "longitudes"}

    @pytest.mark.parametrize("area", [None, [60, 0, 30, 40]], ids=["no-area", "area"])
    def test_lambert_grid_is_sized_and_not_taken_for_coordinates(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, area: Any
    ):
        """A dict lookup grid is only a coordinate set if it has ``latitudes``: a Lambert grid is sized and compared."""
        stubs.grid_npoints = len(STATE["latitudes"])

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID, "area": area}, state=STATE)

        # a Lambert grid defines its domain: the lookup area is dropped (MIR cannot crop it)
        assert stubs.sizes == [(canonical_lambert_grid(LAMBERT_GRID), None)]
        assert len(stubs.regrids) == 1
        assert stubs.regrids[0].kwargs == {"grid": LAMBERT_GRID, "area": None}
        assert stubs.forwards == stubs.regrids

    def test_lookup_area_of_a_lambert_grid_is_ignored_with_a_debug_message(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, caplog: pytest.LogCaptureFixture
    ):
        """The checkpoint's area, which ``TemplateManager`` puts in the lookup, is not given to MIR."""
        area = [60, 0, 30, 40]

        with caplog.at_level(logging.DEBUG, logger=provider_module.LOG.name):
            stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID, "area": area})

        assert stubs.regrids[0].kwargs == {"grid": LAMBERT_GRID, "area": None}
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Ignoring the lookup area" in m and str(area) in m for m in messages)

    def test_no_lookup_area_no_message(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.DEBUG, logger=provider_module.LOG.name):
            stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID})

        assert not [r for r in caplog.records if "Ignoring" in r.getMessage()]

    def test_lambert_grid_is_sized_without_mir_and_passed_on(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ):
        """A Lambert lookup is sized as ``Nx * Ny`` (``mir.Grid`` cannot build it) and given to MIRRegrid."""
        # The real _grid_size, not the stub, with mir blocked: it must not need MIR
        monkeypatch.setattr(MirTemplatesProvider, "_grid_size", staticmethod(real_grid_size))
        monkeypatch.setitem(sys.modules, "mir", None)
        stubs.grid_npoints = LAMBERT_NX * LAMBERT_NY
        state = {"latitudes": np.zeros(stubs.grid_npoints), "longitudes": np.zeros(stubs.grid_npoints)}

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID}, state=state)

        assert len(stubs.regrids) == 1
        assert stubs.regrids[0].kwargs == {"grid": LAMBERT_GRID, "area": None}
        assert stubs.regrids[0].grid == canonical_lambert_grid(LAMBERT_GRID)
        assert stubs.forwards == stubs.regrids

    def test_the_meps_grid_is_sized_without_mir(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(MirTemplatesProvider, "_grid_size", staticmethod(real_grid_size))
        monkeypatch.setitem(sys.modules, "mir", None)
        stubs.grid_npoints = MEPS_NPOINTS
        state = {"latitudes": np.zeros(MEPS_NPOINTS), "longitudes": np.zeros(MEPS_NPOINTS)}

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": MEPS_LAMBERT_GRID}, state=state)

        assert stubs.forwards == stubs.regrids
        assert len(stubs.regrids) == 1
        assert stubs.regrids[0].grid["Nx"] == 849

    def test_equal_lambert_grids_in_another_key_order_reach_the_regrid_the_same(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        """What anemoi-inference passes on is canonicalised by MIRRegrid, whatever the order of the keys."""
        shuffled = dict(reversed(list(LAMBERT_GRID.items())))

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID})
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": shuffled})

        assert str(stubs.regrids[0].grid) == str(stubs.regrids[1].grid)

    def test_lambert_grid_without_state_is_the_lookup_regrid(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID})

        assert [regrid.kwargs["grid"] for regrid in stubs.forwards] == [LAMBERT_GRID]
        assert stubs.sizes == []

    def test_lambert_size_mismatch_uses_the_state_coordinates(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(MirTemplatesProvider, "_grid_size", staticmethod(real_grid_size))
        monkeypatch.setitem(sys.modules, "mir", None)

        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": LAMBERT_GRID}, state=STATE)

        assert [regrid.kwargs["grid"] for regrid in stubs.regrids] == [LAMBERT_GRID, STATE_COORDINATES]
        assert stubs.forwards == [stubs.regrids[1]]

    def test_invalid_lambert_grid_raises_before_a_regrid(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        """The grid is validated when the lookup regrid is made, with a state or not."""
        for state in (None, STATE):
            kwargs = {} if state is None else {"state": state}
            with pytest.raises(ValueError, match="unknown key"):
                stubbed_provider.template("lsm", {"levtype": "sfc", "grid": {**LAMBERT_GRID, "Bogus": 1}}, **kwargs)

        assert stubs.forwards == []

    def test_upper_case_named_grid_is_resolved(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace):
        stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "MEPS"})

        assert stubs.sizes == []
        assert set(stubs.forwards[0].grid) == {"latitudes", "longitudes"}

    def test_unbuildable_grid_with_state_raises_without_regrid(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        """A grid MIR cannot build is an error, not a reason to use the state coordinates."""
        stubs.size_error = RuntimeError("cannot build the grid")

        with pytest.raises(RuntimeError, match="cannot build the grid"):
            stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "not-a-grid"}, state=STATE)

        assert stubs.forwards == []


class TestForward:
    @pytest.mark.parametrize(
        ("lookup", "state"),
        [
            pytest.param({"grid": "O96"}, None, id="no-state"),
            pytest.param({"grid": "O96"}, STATE, id="size-match"),
            pytest.param({"grid": "O48"}, STATE, id="size-mismatch"),
            pytest.param({"grid": "meps"}, STATE, id="named"),
            pytest.param({"grid": None}, STATE, id="no-grid"),
        ],
    )
    def test_one_forward_per_template(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, lookup: dict, state: dict | None
    ):
        stubs.grid_npoints = len(STATE["latitudes"]) if lookup["grid"] == "O96" else 99
        kwargs = {} if state is None else {"state": state}

        stubbed_provider.template("lsm", {"levtype": "sfc", **lookup}, **kwargs)

        assert len(stubs.forwards) == 1

    @pytest.mark.parametrize(
        ("lookup", "state"),
        [
            pytest.param({"grid": "O96"}, None, id="no-state"),
            pytest.param({"grid": "O96"}, STATE, id="size-match"),
            pytest.param({"grid": "O48"}, STATE, id="size-mismatch"),
            pytest.param({"grid": None}, STATE, id="no-grid"),
        ],
    )
    def test_error_propagates(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, lookup: dict, state: dict | None
    ):
        """An error from the regrid is not hidden by a fallback to another grid."""
        stubs.grid_npoints = len(STATE["latitudes"]) if lookup["grid"] == "O96" else 99
        stubs.forward_error = RuntimeError("MIR failed")
        kwargs = {} if state is None else {"state": state}

        with pytest.raises(RuntimeError, match="MIR failed"):
            stubbed_provider.template("lsm", {"levtype": "sfc", **lookup}, **kwargs)

        assert len(stubs.forwards) == 1

    def test_the_first_regridded_field_is_the_template(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        template = stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96"}, state=STATE)

        assert isinstance(template, FakeTemplate)

    @pytest.mark.parametrize("grid", ["O96", "O48", None])
    def test_wrong_point_count_raises(self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace, grid: Any):
        stubs.forward_npoints = len(STATE["latitudes"]) + 1

        with pytest.raises(ValueError, match="points"):
            stubbed_provider.template("lsm", {"levtype": "sfc", "grid": grid}, state=STATE)

    def test_point_count_is_not_checked_without_state(
        self, stubbed_provider: MirTemplatesProvider, stubs: SimpleNamespace
    ):
        stubs.forward_npoints = 12345

        template = stubbed_provider.template("lsm", {"levtype": "sfc", "grid": "O96"})

        assert template.metadata("numberOfDataPoints") == 12345
