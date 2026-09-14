# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from typing import Any

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.inputs.split import SplitInput
from anemoi.inference.metadata import Metadata

from .profiles import available_profiles
from .profiles import get_profile
from .profiles.schema import ThreddsEntry
from .profiles.schema import ThreddsProfile

LOG = logging.getLogger(__name__)


@main_argument("profile")
class ThreddsProfilesInput(SplitInput):
    """Handles Thredds profiles as an input."""

    def __init__(self, context: Context, metadata: Metadata, *, profile: str, **kwargs):
        """Initialise the ThreddsProfilesInput.

        Parameters
        ----------
        context : Context
            The context in which the input is used.
        metadata : Metadata
            The metadata associated with the input.
        profile: str
            The profile of the Thredds file / server.
            Must be one of the profiles defined in the Thredds configuration, or a file path to a Thredds Profile configuration file.
        """
        if profile not in available_profiles():
            if profile.endswith(".yaml"):
                self.profile = ThreddsProfile.from_yaml_file(profile)
            else:
                raise ValueError(
                    f"Profile '{profile}' not found in package '{__package__}'.\nAvailable profiles: {available_profiles()}"
                )
        else:
            self.profile = get_profile(profile)

        LOG.info(f"Using Thredds profile '{profile}' with {len(self.profile.entries)} entries.")
        splits = [self._resolve_entry(entry) for entry in self.profile.entries]
        LOG.debug(f"Resolved splits: {json.dumps(splits, indent=2)}")

        super().__init__(context, metadata, *splits, **kwargs)

    @staticmethod
    def _resolve_entry(entry: ThreddsEntry) -> dict[str, Any]:
        """Resolve a Thredds profile entry to an OpenDAPInput.

        Parameters
        ----------
        entry : ThreddsEntry
            The Thredds profile entry.

        Returns
        -------
        dict
            The resolved OpenDAPInput configuration ready for split input.
        """
        pre_processors = []
        if entry.transforms:
            for transform in entry.transforms:
                pre_processors.append(transform)

        if entry.rename:
            pre_processors.append({"rename": {"param": entry.rename}})

        params = [entry.rename.get(p, p) for p in entry.params] + [
            entry.rename.get(p, p) for p in (entry.derived_params or [])
        ]
        if entry.levels:
            params = [f"{param}_{level}" for param in params for level in entry.levels]

        return {
            "source": {
                "opendap": {
                    "url": str(entry.url),
                    "pre_processors": list(map(lambda x: {"forward_transform_filter": x}, pre_processors)),
                },
            },
            "variables": params,
        }
