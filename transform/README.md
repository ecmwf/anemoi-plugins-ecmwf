# anemoi transform plugins by ECMWF

This package includes all plugins developed for anemoi transform developed by ECMWF

## Plugins

| Plugin    | Description                        |
|-----------|------------------------------------|
| `virtual_temperature` | Compute virtual temperature from q and t |
| `mir_regrid` | Regrid fields to a target grid and optional area using MIR |
| `sh_to_gg` | Transform spectral scalar fields (spherical harmonics) to grid-point |
| `vordiv_to_uv` | Transform spectral vorticity and divergence to u and v |
| `uv_to_vordiv` | Transform u and v to vorticity and divergence (reverse of `vordiv_to_uv`) |

All are registered in the `anemoi.transform.filters.fields` entry point group.

See the underlying code for a more detailed `README`.

## Installation

The only optional dependency extra is `mir` (also installed by `all`),
which is required by `mir_regrid`.

```bash
pip install anemoi-plugins-ecmwf-transform[mir]
```
