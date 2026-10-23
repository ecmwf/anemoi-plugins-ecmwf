# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import earthkit.data as ekd
import pytest
from anemoi.inference.runners import create_runner
from anemoi.inference.testing import fake_checkpoints
from anemoi.inference.testing.mock_checkpoint import MockRunConfiguration
from anemoi.plugins.ecmwf.inference.opendata.opendata import OpenDataInputPlugin

MODULE = "anemoi.plugins.ecmwf.inference.opendata.opendata"


def _requests() -> list[dict]:
    return [
        {"class": "od", "levtype": "sfc", "param": ["2t"], "date": "20250101", "time": "00", "step": 0},
        {
            "class": "od",
            "levtype": "pl",
            "levelist": [500],
            "param": ["z"],
            "date": "20250101",
            "time": "00",
            "step": 0,
        },
    ]


def _retrieve(mock_metadata, patches) -> list[dict]:
    """Build the opendata input and return the requests passed to ``ekd.from_source``."""
    config = MockRunConfiguration.load(
        (Path(__file__).parent / "configs/simple.yaml").absolute(),
        overrides=dict(
            input="mars",  # the runner's own input is irrelevant; the plugin is built directly below
            checkpoint=str((Path(__file__).parent.parent / "checkpoints/simple.ckpt").absolute()),
        ),
    )
    runner = create_runner(config)
    runner.pre_processors = defaultdict(list)
    runner.post_processors = defaultdict(list)

    plugin = OpenDataInputPlugin(runner, mock_metadata, variables=[], patches=patches)

    mock_metadata.mars_requests.return_value = _requests()
    mock_metadata.grid = None
    mock_metadata.area = None

    fieldlist = MagicMock(spec=ekd.FieldList)
    with (
        patch(f"{MODULE}._rename_params", return_value=fieldlist),
        patch(f"{MODULE}.MIRRegrid"),
        patch(f"{MODULE}.ekd.SimpleFieldList"),
        patch(f"{MODULE}.ekd.from_source") as from_source,
    ):
        plugin.retrieve(["2t", "z_500"], dates=[])

    assert mock_metadata.mars_requests.call_args.kwargs["type"] == "fc"
    for call in from_source.call_args_list:
        assert call.args[0] == "ecmwf-open-data"
    return [call.args[1] for call in from_source.call_args_list]


@fake_checkpoints
def test_patches_applied_to_all_requests(mock_metadata) -> None:
    patches = [({"class": "od"}, {"stream": "enfo", "type": "pf", "number": 3})]
    requests = _retrieve(mock_metadata, patches)

    assert len(requests) == 2
    for r in requests:
        assert r["stream"] == "enfo"
        assert r["type"] == "pf"
        assert r["number"] == 3


@fake_checkpoints
def test_patches_only_applied_to_matching_requests(mock_metadata) -> None:
    patches = [({"levtype": "sfc"}, {"stream": "enfo", "type": "cf"})]
    requests = _retrieve(mock_metadata, patches)

    by_levtype = {r["levtype"]: r for r in requests}
    assert by_levtype["sfc"]["stream"] == "enfo"
    assert by_levtype["sfc"]["type"] == "cf"
    assert "stream" not in by_levtype["pl"]
    assert "type" not in by_levtype["pl"]


@fake_checkpoints
def test_patches_applied_after_patch_data_request(mock_metadata) -> None:
    """Configured patches are applied last, so they override ``patch_data_request``."""
    patches = [({"class": "od"}, {"type": "cf"})]

    with patch(
        "anemoi.inference.inputs.mars.MarsInput.patch_data_request",
        side_effect=lambda r: {**r, "type": "fc", "expver": "0001"},
    ):
        requests = _retrieve(mock_metadata, patches)

    for r in requests:
        assert r["type"] == "cf"
        assert r["expver"] == "0001"


@fake_checkpoints
@pytest.mark.parametrize("patches", [None, []])
def test_no_patches_leaves_requests_unchanged(mock_metadata, patches) -> None:
    requests = _retrieve(mock_metadata, patches)

    assert sorted(r["levtype"] for r in requests) == ["pl", "sfc"]
    for r in requests:
        assert "stream" not in r
        assert "type" not in r
        assert "number" not in r
