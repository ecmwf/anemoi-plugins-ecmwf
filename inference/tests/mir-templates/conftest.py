# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared fixtures for the mir-templates tests.

Nothing here imports MIR, so the provider-logic tests run without it.
"""

import pytest

from anemoi.plugins.ecmwf.inference.mir_templates import MirTemplatesProvider


class MockTemplateManager:
    """A stand-in for the template manager, which the provider only stores."""


@pytest.fixture
def provider() -> MirTemplatesProvider:
    return MirTemplatesProvider(MockTemplateManager())
