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

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from anemoi.inference.grib.templates import create_template_provider

from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider
from anemoi.plugins.ecmwf.inference.mir_templates import mir_templates as provider_module
from anemoi.plugins.ecmwf.transform.regrid import MIRRegrid

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
