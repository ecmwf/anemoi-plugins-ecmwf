# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import cache
from importlib.resources import as_file
from importlib.resources import files

from .schema import ThreddsProfile


@cache
def available_profiles() -> list[str]:
    """Get a list of available THREDDS profiles."""
    profiles = []
    for f in files(__package__).iterdir():
        with as_file(f) as file:
            if file.suffix == ".yaml":
                profiles.append(file.stem)
    return profiles


@cache
def get_profile(name: str) -> ThreddsProfile:
    """Get a THREDDS profile by name."""
    with as_file(files(__package__) / f"{name}.yaml") as f:
        if not f.exists():
            raise FileNotFoundError(
                f"Profile '{name}' not found in package '{__package__}'.\nAvailable profiles: {available_profiles()}"
            )
        return ThreddsProfile.from_yaml_file(f)
