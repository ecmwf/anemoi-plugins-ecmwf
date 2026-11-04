# FDB Plus Input

A subclass of the `fdb` input to enable efficient retrieval from high resolution spectral encoded data, and ensure it's compatability with model initial conditions.

It uses `vordiv_to_uv`, `sh_to_gg` and `mir_regrid` to truncate, transform and regrid the raw fdb fields: vorticity and divergence are converted to u and v, other spectral fields are transformed to grid point space, and the remaining grid point fields are regridded to the target grid.

Install with `pip install anemoi-plugins-ecmwf-inference[fdbplus]`.

```yaml
input:
  fdbplus:
    class: od
    stream: oper
```

## Options

In addition to the options of the `fdb` input (`fdb_config`, `fdb_userconfig`, `variables`, and any other keys as MARS request keywords):

- `grid`: target grid. Defaults to the grid of the checkpoint; an error is raised if neither is available.
- `spectral_backend`: backend used for the spectral transforms. Defaults to the first available backend.
