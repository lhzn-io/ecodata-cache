import os
import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pyTMD
    import pyTMD.io
    import pyTMD.predict
    import pyTMD.astro

    _PYTMD_AVAILABLE = True
except ImportError:
    _PYTMD_AVAILABLE = False

logger = logging.getLogger(__name__)

# pyTMD database key for "GOT4.10c" is "GOT4.10"; support both spellings.
_MODEL_ALIASES: dict[str, str] = {
    "GOT4.10c": "GOT4.10",
    "GOT4.10": "GOT4.10",
    "EOT20": "EOT20",
}

DEFAULT_CONSTITUENTS: list[str] = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]

# Days from MJD epoch (1858-11-17) to tide epoch (1992-01-01).
# pyTMD.predict.time_series expects t in days since 1992-01-01T00:00:00.
_TIDE_EPOCH = pd.Timestamp("1992-01-01T00:00:00")

_DEFAULT_MODEL_DIR = os.environ.get(
    "PYTMD_MODEL_DIR",
    os.path.expanduser("~/.local/share/tide_models"),
)

_DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get(
        "ECODATA_CACHE_CACHE_DIR",
        os.path.expanduser("~/.cache/ecodata-cache"),
    ),
    "tidal_harmonics",
)


def get_metadata(model: str = "GOT4.10c") -> dict:
    """Return metadata descriptor for the requested tidal model."""
    db_key = _MODEL_ALIASES.get(model, model)
    model_metadata: dict[str, dict] = {
        "GOT4.10": {
            "id": "got4.10c",
            "name": "GOT4.10c (NASA GSFC)",
            "resolution_approx_m": 55000.0,
            "type_desc": "Global tidal harmonic model (elevation only)",
            "domain_bbox": [-180.0, -90.0, 180.0, 90.0],
            "reference": "https://ntrs.nasa.gov/citations/19990089548",
            "constituents": DEFAULT_CONSTITUENTS,
        },
        "EOT20": {
            "id": "eot20",
            "name": "EOT20 (DGFI-TUM)",
            "resolution_approx_m": 28000.0,
            "type_desc": "Global tidal harmonic model (elevation only)",
            "domain_bbox": [-180.0, -90.0, 180.0, 90.0],
            "reference": "https://doi.org/10.17882/79489",
            "constituents": DEFAULT_CONSTITUENTS,
        },
    }
    return model_metadata.get(db_key, model_metadata["GOT4.10"])


def supports_bbox(bbox: list[float]) -> bool:
    """GOT4.10c and EOT20 have global coverage."""
    return True


def fetch_tidal_boundary_conditions(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    model: str = "GOT4.10c",
    constituents: list[str] | None = None,
    model_dir: str = _DEFAULT_MODEL_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    cache_bust: bool = False,
    dt_minutes: int = 15,
) -> Optional[xr.Dataset]:
    """
    Predict tidal boundary conditions using GOT4.10c or EOT20 via pyTMD.

    Generates tidal elevation (zeta_tidal) and placeholder barotropic
    currents (u_tidal, v_tidal = 0) over a regular lat/lon grid spanning
    the bounding box. GOT4.10c and EOT20 publish elevation constituents only;
    current constituents from an OTIS-family model would be needed for
    non-zero u/v tidal predictions.

    Parameters
    ----------
    start_date : str
        ISO8601 start date/time (e.g. "2024-06-01" or "2024-06-01T00:00:00").
    duration_hours : int
        Length of the prediction window in hours.
    bbox : list[float]
        Domain bounding box [min_lon, min_lat, max_lon, max_lat].
    model : str
        Tidal model key.  "GOT4.10c" (default) or "EOT20".
    constituents : list[str] or None
        Tidal constituents to include.  Defaults to DEFAULT_CONSTITUENTS.
    model_dir : str
        Path to local directory containing downloaded pyTMD model files.
    cache_dir : str
        Directory used for caching tidal prediction results.
    cache_bust : bool
        If True, ignore cached results and re-predict.
    dt_minutes : int
        Time resolution in minutes for the prediction grid.

    Returns
    -------
    xr.Dataset or None
        Dataset with variables ``zeta_tidal``, ``u_tidal``, ``v_tidal``
        on dimensions ``(time, lat, lon)``.

    Raises
    ------
    RuntimeError
        If the required model files are not found in *model_dir*.
    """
    if constituents is None:
        constituents = DEFAULT_CONSTITUENTS

    db_key = _MODEL_ALIASES.get(model, model)
    if db_key not in ("GOT4.10", "EOT20"):
        raise ValueError(
            f"Unsupported tidal model '{model}'. "
            "Supported values: 'GOT4.10c', 'GOT4.10', 'EOT20'."
        )

    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Cache key
    # -------------------------------------------------------------------------
    c_str = ",".join(sorted(constituents))
    cache_key_str = (
        f"{db_key}_{start_date}_{duration_hours}_{bbox[0]}_{bbox[1]}"
        f"_{bbox[2]}_{bbox[3]}_{c_str}_{dt_minutes}"
    )
    cache_hash = hashlib.sha256(cache_key_str.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tidal_{cache_hash}.zarr")

    if not cache_bust and os.path.exists(cache_path):
        logger.info(f"Tidal harmonics cache hit: {cache_path}")
        try:
            return xr.open_zarr(cache_path).load()
        except Exception as e:
            logger.warning(
                f"Failed to load tidal cache ({cache_path}): {e}. Re-predicting."
            )

    logger.info(
        f"Predicting tidal boundary conditions with {db_key} "
        f"({start_date}, {duration_hours}h, bbox={bbox})"
    )

    # -------------------------------------------------------------------------
    # Build time axis (15-min resolution by default)
    # -------------------------------------------------------------------------
    start_dt = pd.to_datetime(start_date).tz_localize(None)
    end_dt = start_dt + pd.Timedelta(hours=duration_hours)
    time_index = pd.date_range(start_dt, end_dt, freq=f"{dt_minutes}min")
    np_times = np.array(time_index, dtype="datetime64[us]")

    # -------------------------------------------------------------------------
    # Build a regular spatial grid over the bbox
    # -------------------------------------------------------------------------
    min_lon, min_lat, max_lon, max_lat = bbox
    grid_res = 0.1  # degrees
    lons = np.arange(min_lon, max_lon + grid_res * 0.5, grid_res)
    lats = np.arange(min_lat, max_lat + grid_res * 0.5, grid_res)
    lon2d, lat2d = np.meshgrid(lons, lats)
    lon_flat = lon2d.ravel()
    lat_flat = lat2d.ravel()
    n_pts = len(lon_flat)

    logger.info(
        f"Tidal grid: {len(lons)}×{len(lats)} = {n_pts} points, "
        f"{len(time_index)} time steps"
    )

    # -------------------------------------------------------------------------
    # Load pyTMD model
    # -------------------------------------------------------------------------
    if not _PYTMD_AVAILABLE:
        raise ImportError(
            "pyTMD is required for tidal predictions. "
            "Install it with: pip install pytmd>=3.0.6"
        )

    model_obj = pyTMD.io.model(model_dir)
    try:
        model_obj = model_obj.from_database(db_key)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load tidal model '{db_key}' from pyTMD database: {err}. "
            "Ensure pyTMD is installed and the database is available."
        ) from err

    # -------------------------------------------------------------------------
    # Open harmonic constants and interpolate to grid points
    # -------------------------------------------------------------------------
    try:
        ds_harmonics = model_obj.open_dataset(
            group="z",
            constituents=constituents,
            use_default_units=True,
        )
    except (FileNotFoundError, OSError) as err:
        _model_subdir = "GOT4.10c" if db_key == "GOT4.10" else db_key
        raise RuntimeError(
            f"Tidal model files for '{db_key}' not found in '{model_dir}'. "
            f"Download instructions:\n"
            f"  1. Register at https://cddis.nasa.gov/ (GOT4.10c) or "
            f"     https://www.dgfi.tum.de/en/datei/ (EOT20)\n"
            f"  2. Place unpacked model files under: {model_dir}/{_model_subdir}/\n"
            f"  3. Re-run this request.\n"
            f"Original error: {err}"
        ) from err
    except Exception as err:
        raise RuntimeError(
            f"Unexpected error opening tidal model '{db_key}': {err}"
        ) from err

    logger.info(
        f"Loaded {db_key} harmonic constants for constituents: "
        f"{ds_harmonics.tmd.constituents}"
    )

    # Interpolate harmonic constants to boundary grid points
    try:
        ds_interp = ds_harmonics.tmd.interp(lon_flat, lat_flat, extrapolate=True)
    except Exception as err:
        raise RuntimeError(
            f"Failed to interpolate {db_key} harmonics to boundary grid: {err}"
        ) from err

    # -------------------------------------------------------------------------
    # Predict tidal elevation at each grid point
    # -------------------------------------------------------------------------
    ts = pyTMD.astro.timescale.from_datetime(np_times)
    t_tide = ts.tide  # days relative to 1992-01-01

    logger.info(f"Running tidal prediction for {len(t_tide)} time steps…")

    try:
        tide_da = pyTMD.predict.time_series(
            t_tide,
            ds_interp,
            corrections=model_obj.format,
        )
    except Exception as err:
        raise RuntimeError(
            f"pyTMD tidal prediction failed for '{db_key}': {err}"
        ) from err

    # tide_da shape: (time, points); reshape to (time, lat, lon)
    zeta_vals = np.array(tide_da, dtype=np.float32)
    if zeta_vals.ndim == 1:
        # Scalar point - unlikely but guard anyway
        zeta_vals = zeta_vals[:, np.newaxis]
    nt, n_pts_pred = zeta_vals.shape
    assert n_pts_pred == n_pts, (
        f"Prediction point count mismatch: {n_pts_pred} != {n_pts}"
    )
    zeta_grid = zeta_vals.reshape(nt, len(lats), len(lons))

    # Tidal currents: GOT/EOT models only publish elevation constituents.
    # Return zero arrays so the blending step is a valid no-op for u/v.
    zeros_grid = np.zeros_like(zeta_grid)

    # -------------------------------------------------------------------------
    # Assemble output Dataset
    # -------------------------------------------------------------------------
    ds_out = xr.Dataset(
        {
            "zeta_tidal": (
                ("time", "lat", "lon"),
                zeta_grid,
                {"units": "m", "long_name": "tidal sea-surface elevation"},
            ),
            "u_tidal": (
                ("time", "lat", "lon"),
                zeros_grid,
                {
                    "units": "m s-1",
                    "long_name": "tidal barotropic eastward current (zero - elevation-only model)",
                },
            ),
            "v_tidal": (
                ("time", "lat", "lon"),
                zeros_grid,
                {
                    "units": "m s-1",
                    "long_name": "tidal barotropic northward current (zero - elevation-only model)",
                },
            ),
        },
        coords={
            "time": ("time", time_index.values.astype("datetime64[us]")),
            "lat": ("lat", lats.astype(np.float32)),
            "lon": ("lon", lons.astype(np.float32)),
        },
        attrs={
            "source": db_key,
            "tidal_model": model,
            "constituents": ",".join(ds_harmonics.tmd.constituents),
            "start_date": start_date,
            "duration_hours": duration_hours,
            "type": "tidal_boundary_conditions",
        },
    )

    # -------------------------------------------------------------------------
    # Cache to Zarr
    # -------------------------------------------------------------------------
    logger.info(f"Caching tidal prediction to {cache_path}")
    try:
        import shutil

        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        ds_out.to_zarr(cache_path, mode="w", zarr_format=2)
    except Exception as e:
        logger.warning(f"Failed to cache tidal prediction: {e}")

    return ds_out
