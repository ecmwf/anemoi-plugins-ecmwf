# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ThreddsEntry(BaseModel):
    """A single entry in a THREDDS profile.

    Contains the URL to the THREDDS dataset, a list of variables to extract, an optional remapping of variable names, and optional transformations to apply to the data.
    """

    url: str
    "The URL to the THREDDS dataset."
    params: list[str]
    "A list of params to extract from the dataset."
    derived_params: list[str] | None = None
    "An optional list of the params derived from the transforms applied to the data."
    levels: list[int] | None = None
    "An optional list of levels to extract from the dataset for the params. MUST be used if the params are level-specific."
    rename: dict[str, str] = Field(default_factory=dict)
    "An optional mapping of param names to new names."
    transforms: list[dict[str, dict[str, Any] | Any]] = Field(default_factory=list)
    "A list of transformations to apply to the data, will be applied before the remapping and in order of the list."


class ThreddsProfile(BaseModel):
    """A profile detailing data from multiple THREDDS locations."""

    name: str
    "Name of the profile."
    entries: list[ThreddsEntry]
    "A list of THREDDS entries in the profile."

    @classmethod
    def from_yaml_file(cls, yaml_file: str | Path) -> "ThreddsProfile":
        """Create a ThreddsProfile from a YAML file."""
        import yaml

        with open(yaml_file, "r") as f:
            yaml_dict = yaml.safe_load(f)
        return cls(**yaml_dict)
