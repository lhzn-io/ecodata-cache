# Ecodata Cache Reference

## Mission

Python microservice providing real-time and historical coastal boundary conditions,
initial conditions, and structural nudging telemetry to the `coastal-sim` Julia
physics engine.

## Environment

- **Python 3.11+** via `uv`. Always use `uv run` — never `pip install`.
- **FastAPI** microservice in `service/ecodata_serve/`.
- Tests run with `uv run pytest`.

## Architecture

### Core Pipeline

```text
HTTP Request → service/ecodata_serve/main.py
             → src/ecodata_cache/dispatcher.py
             → fetchers/*.py
             → regridder.py
             → Zarr (~/.cache/ecodata-cache/)
```

- **Entry point**: `service/ecodata_serve/main.py` — FastAPI application.
- **Dispatcher**: `src/ecodata_cache/dispatcher.py` — interprets
  requests and routes to the appropriate fetcher via tiered fallback.
  Uses `_rank_ic_candidates(bbox)` to select the best IC source by
  resolution and domain overlap.
- **Fetchers**: `src/ecodata_cache/fetchers/` — isolated modules per data source.
- **Regridder**: `src/ecodata_cache/regridder.py` — spatial
  interpolation to CF-compliant xarray Zarr stores.

## Fetcher Tier (IC/OBC Priority Order)

| Fetcher | Resolution | Domain | Notes |
|---|---|---|---|
| `nyofs.py` | ~100 m | NY/NJ Harbor | POM curvilinear grid, C-grid stagger |
| `necofs.py` | ~200 m | New England | FVCOM unstructured |
| `tides_tmd.py` | ~0.1° | Global | GOT4.10c/EOT20 pyTMD harmonics; raw amplitudes and phases |
| `hycom.py` | ~9 km | Global | Final fallback (Operational + Historical) |

*Note: Because the architecture is modular, a future roadmap item includes the integration of TPXO (via `tpxo.py`) given usage rights.*

Atmospheric forcing: HRRR (18h) → ERA5T (3mo) → ERA5 (Historical).
Volumetric atmospheric forcing (3D) is provided by ERA5 Pressure Levels.

## Testing

```bash
# Unit tests (no network I/O, fast)
uv run pytest tests/unit/ -v

# Single test file
uv run pytest tests/unit/test_nyofs.py -v

# Integration tests (live APIs, slower)
uv run pytest tests/integration/ -v

# Pre-commit checks (ruff + mypy) on modified files
uv run pre-commit run --files <file1> <file2> ...
```

## Commands

```bash
# Sync dependencies
uv sync

# Start data service
uv run python service/run_server.py

# Run one-off fetch
uv run python -c "from ecodata_cache.fetchers import nyofs; print(nyofs.get_metadata())"
```

## Gotchas

- **Telemetry performance**: Use `.csvp` endpoints for ERDDAP
  tabular time-series to avoid heavy NetCDF binary parsing overhead.
- **OPeNDAP engine**: Always pass `engine="pydap"` to `xr.open_dataset` for THREDDS
  endpoints. The default netcdf4 engine fails on remote DAP URLs.
- **NYOFS FMRC vs NCEI tiering**: FMRC aggregation covers rolling 7-day window
  (< 31 days). Older data requires per-hour NCEI file enumeration with naming
  conventions that changed on 2024-09-09.
- **C-grid interpolation**: NYOFS u/v are on Arakawa C-grid face
  points. After `.where(mask, drop=True)`, the subset is already
  zero-indexed — use the `_c_grid_to_rho()` helper that averages
  adjacent cells on the subset array, not the old absolute-index
  approach.
- **Mocking strategy**: Dispatcher unit tests check tuple/dict return
  boundaries carefully. When modifying mocked fetchers, track
  keyword-argument vs positional argument boundaries.
- **GOT4.10c/EOT20 model files must be downloaded separately** — fetcher raises `RuntimeError` with instructions if absent; model directory read from `PYTMD_DATA_DIR` env var or `~/.pytmd/`
- **Grid normalization**: `coastal-sim` expects elevations positive
  up (LMSL/NAVD88). Normalize any inverted datasets in the fetcher
  tier before the dispatcher sees them.
- **Cache keys**: Cache Zarr keys are deterministic hashes of bbox
  - time window. Changing fetcher output schema (variable names, dims)
  will miss existing cache entries — purge
  `~/.cache/ecodata-cache/` when making breaking schema changes.
