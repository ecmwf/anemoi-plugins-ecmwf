# An anemoi-inference multio output plugin

Write `anemoi-inference` with multio.

## Install

```bash

pip install anemoi-plugins-ecmwf-inference[multio]

```

The `multio` extra installs `pymultio[all]>=2.12.0.33`.

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

`type`, `class`, `expver` and `stream` are required, `model` is optional.
`expver` must be a string of digits or an integer (an integer is zero-padded to four digits, e.g. `1` becomes `0001`).
`stream` can be set to `ORIGIN` to take it from each variable's metadata in the checkpoint (falling back to `oper`).
Steps must be whole hours, sub-hourly steps are rejected.

The usual output options (`variables`, `post_processors`, `output_frequency`, `write_initial_state`) are also accepted.
`initial_state_diagnostics_grib` can point to a GRIB file whose fields are added to the step 0 output.
`archive_requests` (with `path`, and optionally `extra`, `patch` and `indent`) writes a JSON file with the MARS archive request for the fields written.

To run, just like any other

```bash
anemoi-inference run config.yaml
```

> [!NOTE]
> If you are using an ensemble, set `number` to the ensemble number and `numberOfForecastsInEnsemble` to the total number.

### Preset Plans

Additionally, two preset plans are provided

- `multio.grib` : Write to grib. Takes `path`, and optionally `append` and `per_server` (both default `false`).
- `multio.fdb`  : Write to FDB. Takes `fdb_config`, the path to the FDB configuration.

Both accept `debug: true` to add print actions before and after encoding.

`multio.plan` takes a `plan` (a dict or a path to a plan file) and an optional list of `sinks` to append to it.
If the plan already contains sinks, `sinks` cannot be used.
The plain `multio` output picks one of these from the arguments given, and exactly one of `plan`, `path` or `fdb_config` must be set.

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

### Hindcasts

To use the multio output for hindcasts, set `hindcast_reference_date` in the user provided metadata, to the reference date.
This will then be used as the date key, and `hdate` inserted into the multio write.

```yaml
output:
  multio.grib:
      path: 'output.grib'
      # ... other metadata as above ...
      hindcast_reference_date: 20260119
```

If creating a hindcast for a reference date of a leap day, (29th Feb), set the `hindcast_reference_date` to said leap day,
As the initial conditions for non leap years must fall on a non leap-day, maintain that for all years of the hindcast.
This will result in `date` being set correctly as the leap day, and hdate being the true initial condition day.

### FDB indexing with gribjump

Set `FDB_ENABLE_GRIBJUMP=1` to import `pygribjump` when the plugin is loaded, so that FDB writes are indexed for gribjump.
