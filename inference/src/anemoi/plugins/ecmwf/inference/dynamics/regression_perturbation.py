# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import glob
import logging
from typing import Any
from typing import Literal

import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np
from anemoi.inference.processor import Processor
from scipy.stats import linregress

from ._operate_on_fields import apply_function_to_fields
from .modify_value import METHOD_FUNCTIONS

LOG = logging.getLogger(__name__)

VALID_METHODS = Literal["add", "subtract", "multiply", "divide", "replace"]

SEASON_MONTHS = {
    "DJF": ["12", "01", "02"],
    "JAS": ["07", "08", "09"],
}


def _haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two points (decimal degrees)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367.0 * 2 * np.arcsin(np.sqrt(a))


def _gaspari_cohn(lat_vec, lon_vec, bplat, bplon, locrad):
    """Compute Gaspari-Cohn localization function on the grid.

    Returns a 1-D array (same length as lat_vec) with values in [0, 1].
    """
    site_lat = lat_vec[bplat]
    site_lon = lon_vec[bplon]
    dists = np.asarray(_haversine(site_lon, site_lat, lon_vec, lat_vec), dtype=np.float64)

    hlr = 0.5 * locrad
    r = np.asarray(dists / hlr, dtype=np.float64)

    ind_inner = np.where(dists <= hlr)
    ind_outer = np.where(dists > hlr)
    ind_out = np.where(dists > 2.0 * hlr)

    cov = np.ones(lat_vec.shape[0], dtype=np.float64)

    cov[ind_inner] = (
        ((-0.25 * r[ind_inner] + 0.5) * r[ind_inner] + 0.625) * r[ind_inner] - (5.0 / 3.0)
    ) * (r[ind_inner] ** 2) + 1.0

    cov[ind_outer] = (
        (((r[ind_outer] / 12.0 - 0.5) * r[ind_outer] + 0.625) * r[ind_outer] + 5.0 / 3.0) * r[ind_outer] - 5.0
    ) * r[ind_outer] + 4.0 - 2.0 / (3.0 * r[ind_outer])

    cov[ind_out] = 0.0
    cov[cov < 0.0] = 0.0
    return cov


class RegressionPerturbationPlugin(Processor):
    """Compute regression-based initial-condition perturbations on-the-fly.

    Implements the Hakim & Masanam (2023) climatological regression method:
    data samples are loaded, a localized linear regression is computed against
    a reference variable at a chosen point, and the resulting perturbation
    field is applied to the model initial conditions.

    The perturbation is computed on the data source grid (``data_grid``) and
    regridded to the model target grid (``grid`` or checkpoint grid) via
    ``earthkit.regrid``.

    Example
    --------
    ```yaml
    pre_processors:
        - regression_perturbation:
            season: "JAS"
            data_path: "/path/to/data/grib/"
            data_grid: "N320"
            ylat: 15.0
            xlon: 320.0
            xlev: 5
            locrad: 1000.0
            amp: -1.0
            param_pl: ["z", "q", "t", "u", "v", "w"]
            level_pl: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            param_sfc: ["msl", "10u", "10v", "2t"]
            method: "add"
            file_glob_suffix: "_n320.grib"
    ```
    """

    def __init__(
        self,
        context,
        season: str,
        data_path: str,
        data_grid: str | list[float],
        ylat: float,
        xlon: float,
        xlev: int,
        locrad: float,
        amp: float,
        param_pl: list[str],
        level_pl: list[int],
        param_sfc: list[str],
        method: VALID_METHODS = "add",
        file_glob_suffix: str = ".grib",
        max_samples: int | None = None,
        rescale: float = 1.0,
    ):
        super().__init__(context)
        if season not in SEASON_MONTHS:
            raise ValueError(f"Invalid season: {season}. Must be one of {list(SEASON_MONTHS)}")
        if method not in METHOD_FUNCTIONS:
            raise ValueError(f"Invalid method: {method}. Valid methods: {', '.join(METHOD_FUNCTIONS)}")

        self._season = season
        self._data_path = data_path
        self._data_grid = data_grid
        self._ylat = ylat
        self._xlon = xlon
        self._xlev = xlev
        self._locrad = locrad
        self._amp = amp
        self._rescale = rescale
        self._param_pl = param_pl
        self._level_pl = level_pl
        self._param_sfc = param_sfc
        self._method = method
        self._file_glob_suffix = file_glob_suffix
        self._max_samples = max_samples

        # Build filter criteria for apply_function_to_fields
        self._fields: list[dict[str, Any]] = [
            {"shortName": p, "level": lev} for p in param_pl for lev in level_pl
        ] + [{"shortName": p} for p in param_sfc]

    @property
    def _target_grid(self):
        return self._data_grid if self._data_grid is not None else self.checkpoint.grid

    def _glob_data_files(self) -> np.ndarray:
        """Global GRIB files for the configured season."""
        months = SEASON_MONTHS[self._season]
        all_files: list[str] = []
        for month in months:
            pattern = f"{self._data_path}*{month}??_*{self._file_glob_suffix}"
            all_files.extend(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(
                f"No data files found in {self._data_path} for season {self._season} "
                f"with suffix {self._file_glob_suffix}"
            )
        return np.sort(all_files)

    def _compute_regression(self) -> dict[tuple[str, int | None], np.ndarray]:
        """Run the full climatological regression and return perturbation arrays.

        Returns a dict mapping (shortName, level) -> 1-D perturbation array
        on the data grid. Level is None for surface variables.
        """
        dates = self._glob_data_files()

        nvars_pl = len(self._param_pl)
        nlevs = len(self._level_pl)
        nvars_sfc = len(self._param_sfc)

        # --- coordinates from first PL file ---
        first_pl = [f for f in dates if "_pl" in f]
        if not first_pl:
            raise FileNotFoundError("No pressure-level files found (filename must contain '_pl')")
        latlon = ekd.from_source("file", first_pl[0]).sel(levtype="pl").to_latlon()
        lat, lon = latlon["lat"], latlon["lon"]
        n_gridpoints = lat.shape[0]

        # --- base point and localization ---
        bplat = int(np.argmin(np.abs(lat - self._ylat)))
        bplon = int(np.argmin(np.abs(lon - self._xlon)))
        LOG.info("Regression base point: lat=%.2f, lon=%.2f", lat[bplat], lon[bplon])

        locfunc = _gaspari_cohn(lat, lon, bplat, bplon, self._locrad)
        nonzeros = np.argwhere(locfunc > 0.0)[:, 0]
        n_nz = len(nonzeros)
        LOG.info("Localization: %d non-zero points out of %d", n_nz, n_gridpoints)

        # --- count files ---
        n_pl = sum(1 for f in dates if "_pl" in f)
        n_sfc = sum(1 for f in dates if "_sfc" in f)
        if self._max_samples is not None:
            n_pl = min(n_pl, self._max_samples)
            n_sfc = min(n_sfc, self._max_samples)
        nsamp = n_pl + n_sfc
        LOG.info("Using %d PL files and %d SFC files", n_pl, n_sfc)

        # --- load pressure-level data at non-zero indices ---
        regdat_pl = np.zeros((nvars_pl, n_pl, nlevs, n_nz), dtype=np.float32)
        k_pl = 0
        for infile in dates[: nsamp * 2]:
            if "_pl" not in infile or k_pl >= n_pl:
                continue
            LOG.debug("Reading PL file %d: %s", k_pl, infile)
            fields_pl = ekd.from_source("file", infile).sel(levtype="pl")
            fields_pl = fields_pl.sel(param=self._param_pl, level=self._level_pl)
            fields_pl = fields_pl.order_by(param=self._param_pl, level=self._level_pl)
            arr = fields_pl.to_numpy(dtype=np.float32).reshape((nvars_pl, nlevs, -1))
            for v in range(nvars_pl):
                regdat_pl[v, k_pl, :] = arr[v][:, nonzeros]
            k_pl += 1

        # --- load surface data at non-zero indices ---
        regdat_sfc = np.zeros((nvars_sfc, n_sfc, n_nz), dtype=np.float32)
        k_sfc = 0
        for infile in dates[: nsamp * 2]:
            if "_sfc" not in infile or k_sfc >= n_sfc:
                continue
            LOG.debug("Reading SFC file %d: %s", k_sfc, infile)
            fields_sfc = ekd.from_source("file", infile).sel(levtype="sfc")
            fields_sfc = fields_sfc.sel(param=self._param_sfc)
            fields_sfc = fields_sfc.order_by(param=self._param_sfc)
            arr = fields_sfc.to_numpy(dtype=np.float32)
            for v in range(nvars_sfc):
                regdat_sfc[v, k_sfc, :] = arr[v, nonzeros]
            k_sfc += 1

        # --- center ---
        regdat_pl -= np.mean(regdat_pl, axis=1, keepdims=True)
        regdat_sfc -= np.mean(regdat_sfc, axis=1, keepdims=True)

        # --- independent variable ---
        midlatlon = int(np.argmin(np.abs(nonzeros - np.mean(nonzeros))))
        if self._season == "DJF":
            xvar = regdat_pl[0, :, self._xlev, midlatlon]
        else:
            xvar = regdat_sfc[0, :, midlatlon]
        xvar = xvar / np.std(xvar)

        # --- regression: pressure levels ---
        regf_pl = np.zeros((nvars_pl, nlevs, n_nz), dtype=np.float64)
        for v in range(nvars_pl):
            for k in range(nlevs):
                for j in range(n_nz):
                    slope, intercept, *_ = linregress(xvar, regdat_pl[v, :, k, j])
                    regf_pl[v, k, j] = slope * self._amp + intercept
                regf_pl[v, k, :] *= locfunc[nonzeros]

        # --- regression: surface ---
        regf_sfc = np.zeros((nvars_sfc, n_nz), dtype=np.float64)
        for v in range(nvars_sfc):
            for j in range(n_nz):
                slope, intercept, *_ = linregress(xvar, regdat_sfc[v, :, j])
                regf_sfc[v, j] = slope * self._amp + intercept
            regf_sfc[v, :] *= locfunc[nonzeros]

        # --- expand to full data grid ---
        perturbation_map: dict[tuple[str, int | None], np.ndarray] = {}

        for vi, param in enumerate(self._param_pl):
            for ki, lev in enumerate(self._level_pl):
                full = np.zeros(n_gridpoints, dtype=np.float64)
                full[nonzeros] = regf_pl[vi, ki, :]
                perturbation_map[(param, lev)] = full

        for vi, param in enumerate(self._param_sfc):
            full = np.zeros(n_gridpoints, dtype=np.float64)
            full[nonzeros] = regf_sfc[vi, :]
            perturbation_map[(param, None)] = full

        return perturbation_map

    @functools.cached_property
    def _perturbation_map(self) -> dict[tuple[str, int | None], np.ndarray]:
        """Lazily compute the regression and regrid to the target grid."""
        raw_map = self._compute_regression()

        target_grid = self._target_grid
        need_regrid = self._data_grid != target_grid
        if need_regrid:
            LOG.info("Regridding perturbations from %s to %s", self._data_grid, target_grid)

        result: dict[tuple[str, int | None], np.ndarray] = {}
        for key, arr in raw_map.items():
            if need_regrid:
                arr = ekr.interpolate(arr, {"grid": self._data_grid}, {"grid": target_grid})
            result[key] = arr
        return result

    def _apply_perturbation(self, field: ekd.Field) -> ekd.Field:
        """Apply the regression perturbation to a single field."""
        metadata = dict(field.metadata())
        short_name = metadata.get("shortName")
        level = metadata.get("level")

        key = (short_name, level)
        if key not in self._perturbation_map:
            # Try surface key (level=None)
            key = (short_name, None)

        if key not in self._perturbation_map:
            LOG.warning("No perturbation found for %s (level=%s), skipping.", short_name, level)
            return field

        data = field.to_numpy()
        perturbation = self._perturbation_map[key] * self._rescale
        data = METHOD_FUNCTIONS[self._method](data, perturbation)
        return ekd.ArrayField(data, field.metadata())  # type: ignore

    def process(self, state: dict) -> dict:
        """Process the state and apply regression perturbations."""
        state["fields"] = apply_function_to_fields(
            self._apply_perturbation,
            state["fields"],
            filter=self._fields,
        )
        return state

    def __repr__(self):
        return (
            f"RegressionPerturbationPlugin(season='{self._season}', "
            f"ylat={self._ylat}, xlon={self._xlon}, locrad={self._locrad}, "
            f"amp={self._amp}, rescale={self._rescale}, data_grid='{self._data_grid}', "
            f"grid='{self._target_grid}', method='{self._method}')"
        )
