# An anemoi-inference virtual temperature post-processor

Replace temperature `T` with virtual temperature `Tv` on model levels.

For every temperature field named `t_<level>` for which a matching specific
humidity field `q_<level>` exists, the temperature is replaced by

```text
Tv = T * (1 + 0.61 * q)
```

Temperature fields without a matching humidity field are left unchanged.

## Install

This post-processor has no extra dependencies beyond `anemoi-inference`, so it is
available with a plain install of the package:

```bash
pip install anemoi-plugins-ecmwf-inference
```

## Usage

This post-processor can be used like any other for anemoi inference. Below is an
example config.

```yaml
checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: mars

post_processors:
  - compute_tv
```

To run, just like any other

```bash
anemoi-inference run config.yaml
```
