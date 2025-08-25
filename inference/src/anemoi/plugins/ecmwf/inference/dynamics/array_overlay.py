# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
from os import PathLike
from typing import Any
from typing import Literal

import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np
from anemoi.inference.processor import Processor

from ._operate_on_fields import apply_function_to_fields

LOG = logging.getLogger(__name__)

_PILLOW_AVAILABLE = False
try:
    from PIL import Image

    _PILLOW_AVAILABLE = True
except ImportError:
    Image = None


VALID_METHODS = Literal["add", "multiply", "replace"]


class ArrayOverlayPlugin(Processor):
    """Overlay an array/image to a field.

    Pads the overlay to the 1440x721 aspect ratio, and regrids from 0.25,0.25
    to the field grid.

    Usage
    ------
        Provide an array to overlay to the field.

        RGB channels are averaged and regridded to the field grid.

        The overlay is then rescaled and added to the field.

        If white is to be transparent, set invert=True.

    Example
    --------
    ```yaml
    pre_processors:
        - array_overlay:
            overlay: "https://get.ecmwf.int/repository/anemoi/assets/duck.jpg"
            fields:
                - {"level": 850, "shortName": "t"}
            rescale: 10
            method: "add"
            invert: true
    ```
    """

    def __init__(
        self,
        context,
        overlay: PathLike,
        fields: list[dict[str, Any]],
        rescale: float = 1,
        method: VALID_METHODS = "add",
        invert: bool = False,
    ):
        super().__init__(context)

        self._overlay = overlay
        self._fields = fields
        self._rescale = rescale
        self._method = method
        self._invert = invert

    @functools.cached_property
    def overlay(self) -> np.ndarray:
        """Load the overlay."""

        overlay = self._overlay
        if str(overlay).startswith("http"):
            from urllib.request import urlopen

            overlay = urlopen(str(overlay))

        if str(overlay).endswith(".npy"):
            overlay_loaded = np.load(overlay)
        else:
            if not _PILLOW_AVAILABLE:
                raise ImportError("Pillow is required for this pre-processor to work with images.")
            overlay_loaded = Image.open(overlay)  # pyright: ignore[reportOptionalMemberAccess]

        return np.array(overlay_loaded)

    def _pad_overlay_to_aspect_ratio(self, overlay: np.ndarray, target_aspect_ratio: float) -> np.ndarray:
        """Pad an overlay to a target aspect ratio."""
        height, width, _ = overlay.shape
        current_aspect_ratio = width / height

        if current_aspect_ratio > target_aspect_ratio:
            new_height = int(width / target_aspect_ratio)
            padding = ((new_height - height) // 2, 0)
        else:
            new_width = int(height * target_aspect_ratio)
            padding = (0, (new_width - width) // 2)

        new_size = (height + 2 * padding[0], width + 2 * padding[1], 3)
        new_overlay_array = np.full(new_size, 255 if self._invert else 0, dtype=np.uint8)
        new_overlay_array[padding[0] : padding[0] + height, padding[1] : padding[1] + width] = overlay

        return new_overlay_array

    def _overlay_to_array(self, overlay: np.ndarray, grid) -> np.ndarray:
        """Regrid an image to field gridspec, and average the RGB channels if exists."""

        if len(overlay.shape) == 3:
            overlay = np.mean(overlay, axis=-1)

        # TODO Harrison Cook: Remove when regrid can go from arbitrary to arbitrary grids.
        overlay = np.array(
            Image.fromarray(overlay).resize((1440, 721), Image.Resampling.BILINEAR)
        )  # pyright: ignore[reportOptionalMemberAccess]
        overlay = np.nan_to_num(overlay, nan=0)

        return ekr.interpolate(overlay, {"grid": (0.25, 0.25)}, {"grid": grid})

    @functools.lru_cache
    def prepare_overlay(self, grid):
        """Prepare the overlay for the field grid."""
        padded_overlay = self._pad_overlay_to_aspect_ratio(self.overlay, 1440 / 721)
        return self._overlay_to_array(padded_overlay, grid)

    def _apply_overlay(self, field: ekd.Field) -> ekd.Field:
        LOG.debug("ImageOverlay: Applying image overlay to %s.", field)
        data = field.to_numpy()

        image_array = self.prepare_overlay(self.checkpoint.grid) / 255.0
        rescaled_array = ((1 - image_array) if self._invert else image_array) * self._rescale

        if self._method == "add":
            data += rescaled_array
        elif self._method == "multiply":
            data *= rescaled_array
        elif self._method == "replace":
            data = np.where(rescaled_array > 0, rescaled_array, data)
        else:
            raise ValueError(f"Unknown method {self._method}.")

        return ekd.ArrayField(data, field.metadata())  # type: ignore

    def process(self, fields):  # type: ignore
        return apply_function_to_fields(
            self._apply_overlay,
            fields,
            filter=self._fields,
        )

    def __repr__(self):
        return f"ArrayOverlayPlugin(overlay='{self._overlay}', fields={list(map(dict, self._fields))}, rescale={self._rescale})"
