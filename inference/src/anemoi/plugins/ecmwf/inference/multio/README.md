# An anemoi-inference multio output plugin

Write `anemoi-inference` with multio.

## Install

```python

pip install anemoi-plugins-ecmwf-inference[multio]

```

The recommended `pymultio` is `pymultio>=2.5.1.dev20250627`

## Usage

This output can be used like any other for anemoi inference, below is shown an example config.

> [!TIP]
> The following example requires, `anemoi-models==0.3.1` and `flash_attn` installed.

```yaml

checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: mars

output:
  multio:
    plan: # The plan can also be written to a file and referenced here
        plans:
        - name: output-to-file
          actions:
          - type: encode-mtg2
          - type: sink
            sinks:
            - type: file
              append: true
              per-server: false
              path: 'output.grib'
      type: 'fc'
      class: 'ai'
      expver: '0001'
      model: 'aifs-single'
      stream: 'oper'
      # generatingProcessIdentifier: 4
      # number: 1
      # numberOfForecastsInEnsemble: 50
```

To run, just like any other

```bash
anemoi-inference run config.yaml
```

> [!NOTE]
> If you are using an ensemble, set `number` to the ensemble number and `numberOfForecastsInEnsemble` to the total number.

### Preset Plans

Additionally, two preset plans are provided

- `multio.grib` : Write to grib
- `multio.fdb`  : Write to FDB

```yaml
checkpoint:
  huggingface: ecmwf/aifs-single-1.0

date: 2020-01-01

input: mars

output:
  multio.grib:
      path: 'output.grib'
      type: 'fc'
      class: 'ai'
      expver: '0001'
      model: 'aifs-single'
      stream: 'oper'
      # generatingProcessIdentifier: 4
      # number: 1
      # numberOfForecastsInEnsemble: 50
```
