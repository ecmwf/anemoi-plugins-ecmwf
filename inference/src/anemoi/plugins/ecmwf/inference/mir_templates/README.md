# An anemoi-inference mir grib templates plugin

Use `mir` to dynamically create grib templates of the correct geometry.
Install with `pip install anemoi-plugins-ecmwf-inference[mir]`, the template provider is then available as `mir`.

## Base templates

The grid is applied to a base field from `base_handles.yaml` (GRIB2, CCSDS packed, with no grid or area keys):
one for `levtype: pl` and one for everything else.
A different file, in the same templates yaml format, can be given with the `path` option of the provider.

## Grids

The grid and area come from the template lookup.
A bare number such as `0.5` (what anemoi-inference makes of `[0.5, 0.5]`) is
read as `0.5/0.5`.

For unstructured output grids (e.g. stretched grids built with a cutout), the provider builds the MIR
grid from the `latitudes` / `longitudes` of the output state. The grid is
chosen up front: the size of the lookup grid and area is computed with
`mir.Grid` and compared with the number of output points. If it differs (for a
cutout the lookup grid and area describe only one of the sources), the state
coordinates are used as an unstructured grid instead, dropping the area. If the
lookup has no grid the state coordinates are used directly. If the lookup grid is a
named grid (e.g. `meps`) or given as coordinates (a dict with `latitudes` and `longitudes`), the state
coordinates are used as the output points, and are not compared. A grid or area MIR
cannot build raises. The base field is then regridded once.

If there is no grid and the state has no latitudes/longitudes, the provider
returns `None` so the next template provider is tried. Errors from MIR on the
final regrid are raised, as is a `ValueError` if the template does not have
one point per output point.

### Lambert grids

A Lambert conformal conic grid is given as a dict of GRIB keys with `gridType: lambert`, as the lookup
`grid` of the provider and as the `grid` of `MIRRegrid`. A dict of `latitudes` and `longitudes` is a set of
coordinates; a dict with `gridType: lambert` is a Lambert grid. This is the MEPS grid (2.5 km, 849 x 969
points, a sphere of radius 6371000 m, first point the south-west corner):

```yaml
grid:
  gridType: lambert
  Nx: 849
  Ny: 969
  latitudeOfFirstGridPointInDegrees: 51.633816
  longitudeOfFirstGridPointInDegrees: 1.60491
  LaDInDegrees: 63.3
  LoVInDegrees: 15.0
  Latin1InDegrees: 63.3
  Latin2InDegrees: 63.3
  DxInMetres: 2500
  DyInMetres: 2500
  scanningMode: 64
  shapeOfTheEarth: 1
  scaleFactorOfRadiusOfSphericalEarth: 0
  scaledValueOfRadiusOfSphericalEarth: 6371000
  projectionCentreFlag: 0
  resolutionAndComponentFlags: 48
  latitudeOfSouthernPoleInDegrees: -90.0
  longitudeOfSouthernPoleInDegrees: 0.0
```

The size of the grid for the comparison with the output points is `Nx * Ny`, read from the keys (no
`mir.Grid` is built from it, and no MIR is needed for the size). The keys are validated when the
`MIRRegrid` or the lookup is made; a `ValueError` names what is wrong.

**Keys.**

- Required: `Nx`, `Ny`, `latitudeOfFirstGridPointInDegrees` and `longitudeOfFirstGridPointInDegrees` (the
  south-west corner), `LaDInDegrees`, `LoVInDegrees`, `DxInMetres`, `DyInMetres`.
- Optional, with their default: `Latin1InDegrees` and `Latin2InDegrees` (`LaDInDegrees`), `scanningMode` (64),
  `shapeOfTheEarth` (6), `projectionCentreFlag` (0), `resolutionAndComponentFlags` (48),
  `latitudeOfSouthernPoleInDegrees` (-90) and `longitudeOfSouthernPoleInDegrees` (0).
- `scaleFactorOfRadiusOfSphericalEarth` and `scaledValueOfRadiusOfSphericalEarth` go with `shapeOfTheEarth: 1`
  (radius = scaled value / 10^scale factor, in metres) and are required for it.
- `gridDefinitionTemplateNumber` (it must be 30) and `tablesVersion` of a GRIB message are accepted and
  ignored. Any other key raises a `ValueError`.

**Supported subset.** Spheres only: `shapeOfTheEarth` 0 (radius 6367470 m), 1 (given radius), 6 (6371229 m) and 8
(6371200 m); tangent grids only (`Latin1InDegrees` = `Latin2InDegrees` = `LaDInDegrees`); `scanningMode` 64
(first point south-west, rows running north); `projectionCentreFlag` 0 and the southern pole at (-90, 0), so no
rotation. Anything else raises a `ValueError`. Other scanning modes are refused because ecCodes
(`ECCODES_ECKIT_GEO=0`) does not place the points of a Lambert grid that scans from north to south as the
GRIB definition says.

**What MIR does, and how it is compensated.** MIR 1.30.2 builds a Lambert grid from a typed-grid string
whose first point is the north-west corner, and it puts the points on a fixed sphere of radius 6371229 m
whatever the earth of the input field or of the grid (the header only copies the earth of the input). The
plugin therefore asks MIR for the grid that has the points of yours on that sphere: the north-west corner
(computed from your south-west corner with spherical Lambert conformal conic maths at the radius of your
earth) and the steps `DxInMetres` and `DyInMetres` scaled by 6371229 / radius, in full precision. The input
message is given `shapeOfTheEarth=6` on a copy (the input fields are not modified). The rows of the output
are then reversed to scan from south to north and the header is written from your keys: the first point is
your south-west corner exactly as given, with `scanningMode` 64, the earth, `LaDInDegrees`, `LoVInDegrees`,
the steps, the flags and the other keys of the table above. ecCodes decodes the points of the exact
geometry of your grid: for MEPS they equal pyproj's `+proj=lcc +lat_0=63.3 +lat_1=63.3 +lat_2=63.3
+lon_0=15 +R=6371000` to better than 1e-10 degrees. The values are the interpolation of the input at
those points; the error depends on the input resolution (for a field whose value is the latitude or
longitude of its point: about 140 m from an O320 field, 40 m from an O640 field on MEPS).

**GRIB edition.** GRIB1 and GRIB2 input fields work, and the output is GRIB2. MIR only takes the
`writeLaDInDegrees` key for GRIB2 input (otherwise it fails or writes LaD as 0), so the plugin adds it for
GRIB2 input fields only. Missing values (a bitmap) are reversed with the values, and the packing type
is kept.

**No area.** The keys of a Lambert grid define its domain and MIR cannot crop a Lambert grid. `MIRRegrid` and
`mir_regrid` raise a `ValueError` for an `area` given with one (at construction for `MIRRegrid`), and the
regridded fields report no MARS area (`mars_area()` is `None`). In the template lookup the provider ignores any
`area` for a Lambert grid and logs it at debug level: when the regridded fields report no area,
`TemplateManager.load_template` falls back to the area of the checkpoint metadata, which would otherwise be
handed to MIR and fail (`Representation::croppedRepresentation() not implemented`).

### MIRRegrid as a pre-processor to a Lambert grid

The `MIRRegrid` filter of `anemoi-plugins-ecmwf-transform` can regrid the input of a
model to a Lambert grid, and this provider then builds the matching output template:

```yaml
input:
  grib:
    path: input.grib

pre_processors:
  - forward_transform_filter:
      filter:
        mir_regrid:
          grid:
            gridType: lambert
            Nx: 849
            Ny: 969
            latitudeOfFirstGridPointInDegrees: 51.633816
            longitudeOfFirstGridPointInDegrees: 1.60491
            LaDInDegrees: 63.3
            LoVInDegrees: 15.0
            DxInMetres: 2500
            DyInMetres: 2500
            shapeOfTheEarth: 1
            scaleFactorOfRadiusOfSphericalEarth: 0
            scaledValueOfRadiusOfSphericalEarth: 6371000

output:
  grib:
    path: data.grib
    templates:
      - 'mir'
```

The optional keys are left out here, so they have their default (tangent at `LaDInDegrees`, `scanningMode` 64,
`projectionCentreFlag` 0, `resolutionAndComponentFlags` 48, southern pole at (-90, 0)).

anemoi-inference takes the grid and area of the state (`state["_geography"]`)
from the `mars_grid()` and `mars_area()` of the regridded fields, and looks up
the template with them. The fields `MIRRegrid` returns for a Lambert grid therefore
report the Lambert grid as their MARS grid and `None` as their MARS area. The grid is the validated dict with every key
(the defaults filled in) in a fixed order and with the ignored keys left out, so equal grids
give equal results whatever the order of the keys: anemoi-inference compares the `str` of the
grid of each field and drops the grid if they differ. Everything else of the fields is
unchanged. anemoi-inference passes a dict grid on unchanged (`TemplateManager._grid` only
changes strings and two-element lists), so the template provider gets the Lambert dict as the `grid`
of the lookup and makes a Lambert template of the same grid: the same header and the same points
as the regridded fields.

**Set `ECCODES_ECKIT_GEO=0` in the environment** (before starting Python, ecCodes
reads it when it initialises) to use a Lambert grid. `anemoi.plugins.ecmwf.transform.regrid`
sets `ECCODES_ECKIT_GEO=1` unless it is already set (needed for unstructured grids
such as `meps`), and with it ecCodes cannot build the geography of a Lambert
field (`Factory(grid) has no builder for lambert`), so reading its grid points fails.
`MIRRegrid` logs a warning when it regrids to a Lambert grid with
`ECCODES_ECKIT_GEO=1`.

## Example

Use with the grib output

```yaml

output:
  grib:
    path: data.grib
    templates:
      - 'mir'

```
