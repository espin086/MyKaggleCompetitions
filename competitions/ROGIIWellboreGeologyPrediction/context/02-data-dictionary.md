# Data dictionary (from the Data tab, 2026-07-13)

The data is horizontal well trajectories + vertical reference logs (typewells). Organized as
`train/` and `test/`, each well identified by a unique 8-char hash (e.g. `015fe0d2`). ~2327
files total, ~1.33 GB. Not flat train/test CSVs. one set of files **per well**.

## `train/{WELLNAME}__horizontal_well.csv`. the lateral

Trajectory, geological surfaces, and log data along the horizontal well.

| Column | Meaning |
|---|---|
| `WELLNAME` | unique well identifier |
| `MD` | Measured Depth (ft). total length of wellbore from surface (the sequence axis) |
| `X` | Easting (ft). horizontal spatial coordinate |
| `Y` | Northing (ft). horizontal spatial coordinate |
| `Z` | True Vertical Depth (ft). vertical distance below sea level |
| `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` | Predicted depth of various geological formations. **Training only.** |
| `TVT` | True Vertical Thickness (ft). manually interpreted geologic position for each 1 ft of lateral. **The target. Training only.** |
| `GR` | Gamma Ray (API). log measuring natural radioactivity of rock (the signal to match) |
| `TVT_input` | Input Target (ft). a copy of `TVT` provided as a feature; **`NaN` in the evaluation zone.** This mask defines what you predict. |

## `train/{WELLNAME}__typewell.csv`. vertical reference log

The "template" column for geological correlation.

| Column | Meaning |
|---|---|
| `TVT` | Vertical Depth Index (ft). primary depth reference; corresponds to the geologic position (`TVT`) of the associated horizontal well |
| `GR` | Gamma Ray (API). the vertical GR signature used for correlation |
| `Geology` | Formation label (categorical, e.g. `EGFDL`, `BUDA`) |

## `train/{WELLNAME}.png`

Visualization of the well path and geological cross-section. Training only. Good for EDA and
for sanity-checking that a predicted `tvt` trace is geologically plausible.

## `test/` (~200 wells)

Two files per well: `__horizontal_well.csv` (with `TVT` hidden. `NaN`. in the eval zone)
and `__typewell.csv`. No formation-depth columns, no `TVT`, no `.png`.

> **Important:** the visible `test/` folder holds only a few instances copied from the
> training set as examples for authoring your notebook. On rerun against the hidden test set,
> these are replaced with the real ~200 test wells. So: do not overfit to the visible test
> wells, and make the notebook path-agnostic over whatever wells appear in `test/`.

## `sample_submission.csv`

Correct output format. `id = {WELLNAME}_{row_index}` (e.g. `015fe0d2_1654`), `tvt` = your
predicted True Vertical Thickness (ft).

## Key modeling implications

- **Eval-zone mask** = `horizontal_well.TVT_input.isna()`. Those rows (in test) are what you
  score on. In train, `TVT` is known everywhere so you can construct your own eval zones.
- **`TVT_input` is a strong feature** outside the eval zone. the known geologic position on
  the approach into the zone anchors the prediction.
- **Typewell is the label space:** the typewell's `TVT` axis + `Geology` labels define the
  column you're locating the bit within. Aligning lateral `GR` to typewell `GR` is the crux.
- Formation-depth columns exist in train only. usable as auxiliary training targets /
  features, but you cannot rely on them at test time (derive equivalents from the typewell).
- See discussion thread "Formation Columns Are Derived from Typewell, Not Independent 3D
  Surfaces". the formation columns are not independent info, they come from the typewell.
