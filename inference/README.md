# anemoi inference plugins by ECMWF

This package includes all plugins developed for anemoi inference developed by ECMWF

## Plugins

| Plugin    | Description                        |
|-----------|------------------------------------|
| `multio`  | Write out to a multio plan         |
| `polytope`| Use polytope as an input source    |
| `opendata`| Use ecmwf-opendata as an input source    |
| `mir`     | Use mir to create grib templates   |
| `regrid`  | Use `earthkit-regrid` to preprocess|
| `dynamics`| Use `earthkit-regrid` to preprocess|

See the underlying code for a more detailed `README`.

## Installation

All plugins are listed as optional dependencies

```bash
pip install anemoi-plugins-ecmwf-inference[PLUGIN]
```
