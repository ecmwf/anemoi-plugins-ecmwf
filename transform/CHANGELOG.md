# Changelog

## [0.3.0](https://github.com/ecmwf/anemoi-plugins-ecmwf/compare/transform-0.2.0...transform-0.3.0) (2026-11-02)


### ⚠ BREAKING CHANGES

* **transform/regrid:** build the MIR target grid with mir.Grid so latitude/longitude grids are regridded, MIRRegrid no longer takes a method
* **transform/regrid:** move the named grids to transform, resolved by MIRRegrid

### Features

* **inference/mir_templates:** make templates for unstructured and stretched-grid (cutout) output ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **transform/regrid:** move the named grids to transform, resolved by MIRRegrid ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))


### Bug Fixes

* **inference/fdb_plus:** drop the method argument removed from MIRRegrid ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **inference/mir_templates:** choose the template grid with mir.Grid and check it against the output points ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **inference/regrid:** take flat state coordinates from the regridded grid ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **transform/regrid:** build the MIR target grid with mir.Grid so latitude/longitude grids are regridded, MIRRegrid no longer takes a method ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **transform/regrid:** enable eckit-geo in eccodes and its grid cache ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))
* **transform/regrid:** stamp the grid uid on unstructured MIR output ([1d57367](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1d57367bddb371f0b6499946607d68d2c7988b2c))

## [0.2.0](https://github.com/ecmwf/anemoi-plugins-ecmwf/compare/transform-0.1.0...transform-0.2.0) (2026-07-24)


### Features

* FDB Plus input ([#102](https://github.com/ecmwf/anemoi-plugins-ecmwf/issues/102)) ([1ef32fe](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/1ef32fe4d8b48d3f51619665dffce901336c549a))
* Spectral Transforms ([#100](https://github.com/ecmwf/anemoi-plugins-ecmwf/issues/100)) ([19fd371](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/19fd371dbb759754a82bcabbb3da548074288578))

## 0.1.0 (2026-07-13)


### Features

* Virtual temperature filter ([#99](https://github.com/ecmwf/anemoi-plugins-ecmwf/issues/99)) ([eaf8ae7](https://github.com/ecmwf/anemoi-plugins-ecmwf/commit/eaf8ae7a1d86405c4a285931a1ad52a96d3175e8))
