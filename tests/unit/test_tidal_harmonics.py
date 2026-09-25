"""
Unit tests for ecodata_cache.fetchers.tidal_harmonics.

All tests use mocked pyTMD to avoid requiring local model files.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BBOX = [-74.5, 40.5, -73.5, 41.5]
START_DATE = "2024-06-01"
DURATION_HOURS = 6


def _make_mock_tide_da(n_time: int, n_pts: int) -> xr.DataArray:
    """Synthetic tidal prediction DataArray: shape (time, n_pts)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_time, n_pts)).astype(np.float32) * 0.3
    return xr.DataArray(data, dims=["time", "points"])


def _make_mock_harmonics_ds(constituents: list[str]) -> MagicMock:
    """A pyTMD-style Dataset mock with a .tmd accessor."""
    ds = MagicMock(spec=xr.Dataset)
    ds.tmd = MagicMock()
    ds.tmd.constituents = constituents
    ds.tmd.interp.return_value = ds  # interp returns same mock
    return ds


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


def test_get_metadata_got4_returns_correct_schema():
    from ecodata_cache.fetchers.tidal_harmonics import get_metadata

    meta = get_metadata("GOT4.10c")
    assert meta["id"] == "got4.10c"
    assert meta["name"] == "GOT4.10c (NASA GSFC)"
    assert meta["domain_bbox"] == [-180.0, -90.0, 180.0, 90.0]
    assert "M2" in meta["constituents"]
    assert isinstance(meta["resolution_approx_m"], float)


def test_get_metadata_eot20_returns_correct_schema():
    from ecodata_cache.fetchers.tidal_harmonics import get_metadata

    meta = get_metadata("EOT20")
    assert meta["id"] == "eot20"
    assert meta["name"] == "EOT20 (DGFI-TUM)"
    assert meta["domain_bbox"] == [-180.0, -90.0, 180.0, 90.0]
    assert "K1" in meta["constituents"]


def test_get_metadata_alias_got4_10c_equals_got4_10():
    from ecodata_cache.fetchers.tidal_harmonics import get_metadata

    meta_c = get_metadata("GOT4.10c")
    meta = get_metadata("GOT4.10")
    assert meta_c["id"] == meta["id"]


# ---------------------------------------------------------------------------
# supports_bbox
# ---------------------------------------------------------------------------


def test_supports_bbox_always_true():
    from ecodata_cache.fetchers.tidal_harmonics import supports_bbox

    assert supports_bbox([-180.0, -90.0, 180.0, 90.0]) is True
    assert supports_bbox([-74.5, 40.5, -73.5, 41.5]) is True
    assert supports_bbox([100.0, -5.0, 120.0, 5.0]) is True


# ---------------------------------------------------------------------------
# fetch_tidal_boundary_conditions — mocked pyTMD
# ---------------------------------------------------------------------------


def _patch_pytmd(n_time: int, n_pts: int, constituents: list[str]):
    """
    Returns a context manager stack that patches pyTMD model loading and
    prediction so no real model files are needed.

    Patches the actual pyTMD modules directly so they work even if the
    fetcher uses module-level imports.
    """
    mock_model = MagicMock()
    mock_model.format = "GOT-ascii"
    mock_model.open_dataset.return_value = _make_mock_harmonics_ds(constituents)

    mock_model_cls = MagicMock()
    mock_model_cls.return_value.from_database.return_value = mock_model

    mock_tide_da = _make_mock_tide_da(n_time, n_pts)

    mock_ts = MagicMock()
    mock_ts.tide = np.linspace(0, 1, n_time)

    return mock_model_cls, mock_tide_da, mock_ts


# Patch targets — pyTMD modules directly (work with both lazy and eager imports)
_PATCH_IO_MODEL = "pyTMD.io.model"
_PATCH_PREDICT_TS = "pyTMD.predict.time_series"
_PATCH_ASTRO_TS = "pyTMD.astro.timescale.from_datetime"


def test_fetch_returns_dataset_with_expected_variables(tmp_path):
    """fetch_tidal_boundary_conditions returns a Dataset with zeta_tidal, u_tidal, v_tidal."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2", "S2", "K1", "O1"]

    # Build expected grid size
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    # Time: 15-min steps from 2024-06-01 to 2024-06-01 + 6h
    time_idx = pd.date_range(START_DATE, periods=25, freq="15min")
    n_time = len(time_idx)

    mock_model_cls, mock_tide_da, mock_ts = _patch_pytmd(n_time, n_pts, constituents)

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        ds = fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="GOT4.10c",
            constituents=constituents,
            cache_dir=str(tmp_path),
            cache_bust=True,
        )

    assert ds is not None
    assert "zeta_tidal" in ds.data_vars
    assert "u_tidal" in ds.data_vars
    assert "v_tidal" in ds.data_vars


def test_fetch_dataset_dimensions(tmp_path):
    """Output dataset must have dims (time, lat, lon)."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2", "S2"]
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    n_time = 25  # 6h at 15min

    mock_model_cls, mock_tide_da, mock_ts = _patch_pytmd(n_time, n_pts, constituents)

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        ds = fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="GOT4.10c",
            constituents=constituents,
            cache_dir=str(tmp_path),
            cache_bust=True,
        )

    assert set(ds["zeta_tidal"].dims) == {"time", "lat", "lon"}
    assert set(ds["u_tidal"].dims) == {"time", "lat", "lon"}
    assert ds["zeta_tidal"].shape == (n_time, len(lats), len(lons))


def test_fetch_zeta_tidal_is_nonzero(tmp_path):
    """zeta_tidal must contain non-zero values (from synthetic harmonic prediction)."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2", "S2"]
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    n_time = 25

    mock_model_cls, mock_tide_da, mock_ts = _patch_pytmd(n_time, n_pts, constituents)

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        ds = fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="GOT4.10c",
            constituents=constituents,
            cache_dir=str(tmp_path),
            cache_bust=True,
        )

    assert not np.allclose(ds["zeta_tidal"].values, 0.0), (
        "zeta_tidal should be non-zero for non-trivial harmonic predictions"
    )
    assert np.allclose(ds["u_tidal"].values, 0.0), (
        "u_tidal must be zero: GOT/EOT models publish elevation only"
    )


def test_fetch_cache_hit(tmp_path):
    """Second call with same params should load from cache without calling pyTMD."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2"]
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    n_time = 25

    mock_model_cls, mock_tide_da, mock_ts = _patch_pytmd(n_time, n_pts, constituents)

    common_kwargs = dict(
        start_date=START_DATE,
        duration_hours=DURATION_HOURS,
        bbox=BBOX,
        model="GOT4.10c",
        constituents=constituents,
        cache_dir=str(tmp_path),
    )

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        # First call — populates cache
        fetch_tidal_boundary_conditions(**common_kwargs, cache_bust=True)

    # Second call — should hit cache; pyTMD.predict must NOT be invoked
    with patch(_PATCH_PREDICT_TS) as mock_predict_second:
        ds2 = fetch_tidal_boundary_conditions(**common_kwargs, cache_bust=False)

    mock_predict_second.assert_not_called()
    assert ds2 is not None
    assert "zeta_tidal" in ds2.data_vars


def test_fetch_missing_model_files_raises_runtime_error(tmp_path):
    """If model files are not found, a RuntimeError with download instructions is raised."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.format = "GOT-ascii"
    mock_model_instance.open_dataset.side_effect = FileNotFoundError(
        "No such file or directory: GOT4.10c/grids_oceantide/m2.d"
    )
    mock_model_cls.return_value.from_database.return_value = mock_model_instance

    with patch(
        _PATCH_IO_MODEL,
        mock_model_cls,
    ):
        with pytest.raises(RuntimeError, match="not found in"):
            fetch_tidal_boundary_conditions(
                start_date=START_DATE,
                duration_hours=DURATION_HOURS,
                bbox=BBOX,
                model="GOT4.10c",
                cache_dir=str(tmp_path),
                cache_bust=True,
            )


def test_fetch_invalid_model_raises_value_error(tmp_path):
    """Requesting an unsupported model name raises ValueError."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    with pytest.raises(ValueError, match="Unsupported tidal model"):
        fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="TPXO10",
            cache_dir=str(tmp_path),
            cache_bust=True,
        )


def test_fetch_eot20_model_uses_fes_format(tmp_path):
    """EOT20 requests should use model key 'EOT20' and return valid Dataset."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2", "K1"]
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    n_time = 25

    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.format = "FES-netcdf"
    mock_model_instance.open_dataset.return_value = _make_mock_harmonics_ds(
        constituents
    )
    mock_model_cls.return_value.from_database.return_value = mock_model_instance

    mock_tide_da = _make_mock_tide_da(n_time, n_pts)
    mock_ts = MagicMock()
    mock_ts.tide = np.linspace(0, 1, n_time)

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        ds = fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="EOT20",
            constituents=constituents,
            cache_dir=str(tmp_path),
            cache_bust=True,
        )

    # from_database must be called with "EOT20"
    mock_model_cls.return_value.from_database.assert_called_once_with("EOT20")
    assert ds is not None
    assert ds.attrs.get("tidal_model") == "EOT20"
    assert "zeta_tidal" in ds.data_vars


def test_fetch_time_coordinate_is_datetime64(tmp_path):
    """Output time coordinate must be datetime64 (compatible with xarray resample)."""
    from ecodata_cache.fetchers.tidal_harmonics import fetch_tidal_boundary_conditions

    constituents = ["M2"]
    bbox = BBOX
    lons = np.arange(bbox[0], bbox[2] + 0.05, 0.1)
    lats = np.arange(bbox[1], bbox[3] + 0.05, 0.1)
    n_pts = len(lons) * len(lats)
    n_time = 25

    mock_model_cls, mock_tide_da, mock_ts = _patch_pytmd(n_time, n_pts, constituents)

    with (
        patch(
            _PATCH_IO_MODEL,
            mock_model_cls,
        ),
        patch(
            _PATCH_PREDICT_TS,
            return_value=mock_tide_da,
        ),
        patch(
            _PATCH_ASTRO_TS,
            return_value=mock_ts,
        ),
    ):
        ds = fetch_tidal_boundary_conditions(
            start_date=START_DATE,
            duration_hours=DURATION_HOURS,
            bbox=BBOX,
            model="GOT4.10c",
            constituents=constituents,
            cache_dir=str(tmp_path),
            cache_bust=True,
        )

    assert ds["time"].dtype.kind == "M", (
        f"Time coordinate must be datetime64 (kind='M'), got {ds['time'].dtype}"
    )
    # Must be resampleable without error
    resampled = ds.resample(time="1h").interpolate("linear")
    assert len(resampled.time) == DURATION_HOURS + 1
