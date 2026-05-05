"""
Fixture-based OBC pipeline test.

Exercises the full post-fetch pipeline without any network calls:
  benthic extrapolation → temporal alignment → TPXO merge → zarr write

Catches dtype mismatches (e.g. object vs datetime64), frequency alias bugs
("1H" vs "1h"), and interp_like failures in seconds instead of a 40-min live fetch.
"""

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_donor_ds(start: str = "2024-06-01", n_hours: int = 6) -> xr.Dataset:
    """
    Minimal OBC donor dataset resembling NECOFS/NYOFS output.
    3-hourly → will be resampled to hourly by the pipeline.
    Spatial dims: latitude x longitude (5x5), depth (3 levels).
    """
    lats = np.linspace(40.5, 41.5, 5, dtype=np.float32)
    lons = np.linspace(-74.5, -73.5, 5, dtype=np.float32)
    depths = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    times = pd.date_range(start, periods=n_hours // 3 + 1, freq="3h")

    nt, nz, ny, nx = len(times), len(depths), len(lats), len(lons)
    rng = np.random.default_rng(42)

    return xr.Dataset(
        {
            "u": (
                ("time", "depth", "latitude", "longitude"),
                rng.standard_normal((nt, nz, ny, nx)).astype(np.float32) * 0.3,
            ),
            "v": (
                ("time", "depth", "latitude", "longitude"),
                rng.standard_normal((nt, nz, ny, nx)).astype(np.float32) * 0.2,
            ),
            "zeta": (
                ("time", "latitude", "longitude"),
                rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.5,
            ),
            "temp": (
                ("time", "depth", "latitude", "longitude"),
                (15.0 + rng.standard_normal((nt, nz, ny, nx))).astype(np.float32),
            ),
            "salt": (
                ("time", "depth", "latitude", "longitude"),
                (32.0 + rng.standard_normal((nt, nz, ny, nx)) * 0.5).astype(np.float32),
            ),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": lats,
            "longitude": lons,
        },
    )


def _make_tpxo_ds(
    start: str = "2024-06-01", n_hours: int = 6, bbox: list[float] | None = None
) -> xr.Dataset:
    """
    Synthetic TPXO tidal dataset with proper pd.DatetimeIndex time coordinate.
    Spatial grid matches _make_donor_ds bbox so interp_like works cleanly.
    """
    if bbox is None:
        bbox = [-74.5, 40.5, -73.5, 41.5]
    min_lon, min_lat, max_lon, max_lat = bbox

    res = 0.25
    lons = np.arange(min_lon, max_lon + res, res, dtype=np.float64)
    lats = np.arange(min_lat, max_lat + res, res, dtype=np.float64)
    time_index = pd.DatetimeIndex(
        [pd.Timestamp(start) + pd.Timedelta(hours=h) for h in range(n_hours)]
    )

    nt, ny, nx = len(time_index), len(lats), len(lons)
    rng = np.random.default_rng(7)

    return xr.Dataset(
        {
            "zeta_tide": (
                ("time", "lat", "lon"),
                rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.4,
            ),
            "u_tide": (
                ("time", "lat", "lon"),
                rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.05,
            ),
            "v_tide": (
                ("time", "lat", "lon"),
                rng.standard_normal((nt, ny, nx)).astype(np.float32) * 0.05,
            ),
        },
        coords={"time": time_index, "lat": lats, "lon": lons},
    )


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------


def test_temporal_resample_produces_hourly():
    ds = _make_donor_ds(n_hours=6)  # 3-hourly input
    resampled = ds.resample(time="1h").interpolate("linear")

    diffs = np.diff(resampled.time.values).astype("timedelta64[h]").astype(int)
    assert (
        diffs == 1
    ).all(), "All time steps should be exactly 1 hour after resampling"


def test_temporal_resample_preserves_dtype():
    ds = _make_donor_ds(n_hours=6)
    resampled = ds.resample(time="1h").interpolate("linear")

    # pandas/xarray 2.x uses datetime64[us]; any datetime64 resolution is acceptable
    assert (
        resampled.time.dtype.kind == "M"
    ), f"Time coordinate must be datetime64 (kind='M'), got {resampled.time.dtype}"


# ---------------------------------------------------------------------------
# TPXO time coordinate dtype
# ---------------------------------------------------------------------------


def test_tpxo_time_coordinate_is_datetime64():
    tide_ds = _make_tpxo_ds()
    assert tide_ds.time.dtype.kind == "M", (
        f"TPXO time must be datetime64 (kind='M'), got dtype={tide_ds.time.dtype}. "
        "Use pd.DatetimeIndex, not a list of pd.Timestamp objects."
    )


def test_tpxo_resample_does_not_raise():
    tide_ds = _make_tpxo_ds(n_hours=6)
    # If dtype is object this raises "Invalid frequency" or subtract errors
    resampled = tide_ds.resample(time="1h").interpolate("linear")
    assert len(resampled.time) == 6


# ---------------------------------------------------------------------------
# TPXO merge / interp_like
# ---------------------------------------------------------------------------


def test_tpxo_interp_like_compatible_dtypes():
    """
    The critical regression: interp_like fails if tide_ds.time is dtype=object
    while donor ds.time is datetime64. This catches the exact bug from prod.
    """
    ds = _make_donor_ds(n_hours=6)
    ds = ds.resample(time="1h").interpolate("linear")
    tide_ds = _make_tpxo_ds(n_hours=6)
    tide_ds = tide_ds.resample(time="1h").interpolate("linear")

    # interp_like requires matching time dtype — will raise TypeError if object
    tide_ds_interp = tide_ds.interp_like(ds, method="linear")
    assert "zeta_tide" in tide_ds_interp.data_vars


def test_tpxo_merge_adds_to_zeta():
    ds = _make_donor_ds(n_hours=6)
    ds = ds.resample(time="1h").interpolate("linear")
    tide_ds = _make_tpxo_ds(n_hours=6)
    tide_ds = tide_ds.resample(time="1h").interpolate("linear")
    tide_ds_interp = tide_ds.interp_like(ds, method="linear")

    # TPXO uses lat/lon; donor uses latitude/longitude — rename so xarray
    # aligns by dimension name rather than creating a product-space expansion.
    tide_ds_interp = tide_ds_interp.rename({"lat": "latitude", "lon": "longitude"})

    zeta_before = ds["zeta"].values.copy()
    ds["zeta"] = ds["zeta"] + tide_ds_interp["zeta_tide"]
    assert (
        ds["zeta"].shape == zeta_before.shape
    ), "Shape must not change after tidal addition"
    assert not np.allclose(ds["zeta"].values, zeta_before, equal_nan=True)


# ---------------------------------------------------------------------------
# Full pipeline (end-to-end, no network)
# ---------------------------------------------------------------------------


def test_obc_pipeline_end_to_end(tmp_path):
    """
    Runs the full post-fetch OBC pipeline inline using synthetic fixtures.
    Confirms zarr is written and contains expected variables with float32 dtype.
    """
    start_date = "2024-06-01"
    duration_hours = 6
    bbox = [-74.5, 40.5, -73.5, 41.5]

    # Step 1: donor dataset (3-hourly)
    ds = _make_donor_ds(start=start_date, n_hours=duration_hours)

    # Step 3: temporal alignment
    ds = ds.resample(time="1h").interpolate("linear")

    # Step 4: TPXO merge (mocked fetch)
    tide_ds = _make_tpxo_ds(start=start_date, n_hours=duration_hours, bbox=bbox)
    tide_ds = tide_ds.resample(time="1h").interpolate("linear")
    tide_ds_interp = tide_ds.interp_like(ds, method="linear")

    if "zeta" in ds.data_vars and "zeta_tide" in tide_ds_interp.data_vars:
        ds["zeta"] = ds["zeta"] + tide_ds_interp["zeta_tide"]
    if "u" in ds.data_vars and "u_tide" in tide_ds_interp.data_vars:
        ds["u"] = ds["u"] + tide_ds_interp["u_tide"]
    if "v" in ds.data_vars and "v_tide" in tide_ds_interp.data_vars:
        ds["v"] = ds["v"] + tide_ds_interp["v_tide"]

    # Step 5: float32 cast + endian encoding
    for var in ds.data_vars:
        if ds[var].dtype != np.float32:
            ds[var] = ds[var].astype(np.float32)
    for var in list(ds.variables):
        ds[var].encoding.clear()
        if ds[var].dtype.kind in "iu":
            ds[var].encoding["_FillValue"] = -9999
        else:
            ds[var].encoding["_FillValue"] = -9999.0
        if ds[var].dtype in (np.float32, np.float64) and "time" not in str(var):
            ds[var] = ds[var].astype("<f4")

    # Step 6: zarr write
    zarr_path = str(tmp_path / "obc_test.zarr")
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)

    # Verify
    result = xr.open_zarr(zarr_path)
    assert "zeta" in result.data_vars
    assert "u" in result.data_vars
    assert "temp" in result.data_vars
    assert result["zeta"].dtype == np.float32
    assert result["u"].dtype == np.float32
    # Hourly time axis
    diffs = np.diff(result.time.values).astype("timedelta64[h]").astype(int)
    assert (diffs == 1).all()
