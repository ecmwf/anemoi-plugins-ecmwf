# anemoi inference plugins by ECMWF

This package includes all plugins developed for anemoi inference developed by ECMWF

## Plugins

| Extra      | Entry point (group)                                                                              | Description                                                   |
|------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `multio`   | `multio`, `multio.fdb`, `multio.grib`, `multio.plan` (`outputs`)                                 | Write out with multio, to a plan, FDB or GRIB files           |
| `polytope` | `polytope` (`inputs`)                                                                            | Use polytope as an input source                               |
| `opendata` | `opendata` (`inputs`)                                                                            | Use ecmwf-opendata as an input source                         |
| `fdbplus`  | `fdbplus` (`inputs`)                                                                             | FDB input with spectral to grid-point transform and regrid    |
| `thredds`  | `thredds_profile` (`inputs`)                                                                     | Use a Thredds profile as an input source                      |
| `mir`      | `mir` (`grib.templates`)                                                                         | Use mir to create grib templates                              |
| `regrid`   | `regrid` (`pre_processors`)                                                                      | Use `mir` to regrid the input fields                          |
| `dynamics` | `array_overlay`, `modify_value`, `regression-perturbation` (`pre_processors`)                    | Modify initial conditions to explore model dynamics           |
|            | `subtract_tendency` (`mid_processors`), `steady-state` (`runners`)                               |                                                               |

The entry point groups are prefixed with `anemoi.inference.`, e.g. `anemoi.inference.inputs`.

See the underlying code for a more detailed `README`.

## Installation

All plugins are listed as optional dependencies. The extras are
`all`, `dynamics`, `fdbplus`, `mir`, `multio`, `opendata`, `polytope`, `regrid` and `thredds`.
`all` installs `mir`, `multio`, `opendata` and `polytope`.

```bash
pip install anemoi-plugins-ecmwf-inference[PLUGIN]
```
