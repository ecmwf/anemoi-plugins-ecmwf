# Virtual Temperature

A post processor that take `q` and `t` to compute `vtmp` for nudging workflows.

```math
vtmp=(1+0.61*q)*T
```

It is registered as the `virtual_temperature` filter in the `anemoi.transform.filters.fields` entry point group.
The input and output parameter names can be changed with the following options (defaults shown),
and all input fields are returned alongside the computed field.

```yaml
- virtual_temperature:
    virtual_temperature: vtmp
    specific_humidity: q
    temperature: t
```
