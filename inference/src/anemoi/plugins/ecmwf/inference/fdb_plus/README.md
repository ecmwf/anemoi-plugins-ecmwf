# FDB Plus Input

A subclass of the `fdb` input to enable efficient retrieval from high resolution spectral encoded data, and ensure it's compatability with model initial conditions.

It uses `vordiv_to_uv` and `mir_regrid` to truncate, transform and regrid the raw fdb fields.

```yaml
input:
  fdbplus:
    class: od
    stream: oper
```
