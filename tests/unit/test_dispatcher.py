import pytest
from datetime import datetime
from unittest.mock import patch
from ecodata_cache.dispatcher import dispatch_forcing_request


# We will mock the actual fetchers so we don't hit Copernicus or S3
@pytest.fixture
def mock_fetchers():
    with (
        patch("ecodata_cache.dispatcher.fetch_hrrr_surface_forcing") as mock_hrrr,
        patch("ecodata_cache.dispatcher.fetch_era5_surface_forcing") as mock_era5,
    ):
        mock_hrrr.return_value = "/tmp/mocked_hrrr.grib2"
        mock_era5.return_value = "/tmp/mocked_era5.grib"

        yield mock_hrrr, mock_era5


@patch("ecodata_cache.dispatcher.datetime")
def test_dispatch_tier_iv_hrrr(mock_datetime, mock_fetchers):
    mock_hrrr, mock_era5 = mock_fetchers

    # Mock 'now' to be today, and target to be 10 hours ago
    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    # 10 hours ago falls cleanly in Tier IV (HRRR)
    target_date = "2026-03-03"

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert mock_hrrr.called
    assert not mock_era5.called


@patch("ecodata_cache.dispatcher.datetime")
def test_dispatch_tier_i_era5_final(mock_datetime, mock_fetchers):
    mock_hrrr, mock_era5 = mock_fetchers

    # Mock 'now' to be today; target is pre-2014 (before HRRR archive start).
    # ERA5 Final is the only available source for pre-archive dates.
    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    target_date = "2013-06-01"  # pre-2014, outside HRRR archive

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert not mock_hrrr.called
    assert mock_era5.called

    import os

    mock_era5.assert_called_with(
        target_date,
        [-74.0, 40.0, -73.0, 41.0],
        os.path.join(
            os.environ.get(
                "ECODATA_CACHE_CACHE_DIR",
                os.path.expanduser("~/.cache/ecodata-cache"),
            ),
            "era5",
            "era5_sfc_2013-06-01.grib",
        ),
        preliminary=False,
        cache_bust=False,
    )


@patch("ecodata_cache.dispatcher.datetime")
def test_dispatch_tier_iv_hrrr_archive_replaces_era5t(mock_datetime, mock_fetchers):
    """Dates 5-90 days old within the HRRR archive now dispatch to HRRR, not ERA5T."""
    mock_hrrr, mock_era5 = mock_fetchers

    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    target_date = "2026-02-15"  # ~16 days ago, formerly dispatched to ERA5T

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert mock_hrrr.called
    assert not mock_era5.called


@patch("ecodata_cache.dispatcher.datetime")
def test_dispatch_tier_iv_hrrr_archive_56_day_hindcast(mock_datetime, mock_fetchers):
    """A 56-day hindcast dispatches to HRRR archive at 3 km, not ERA5T at 31 km."""
    mock_hrrr, mock_era5 = mock_fetchers

    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    target_date = "2026-01-06"  # 56 days before mock 'now'

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert mock_hrrr.called
    assert not mock_era5.called


@pytest.fixture
def mock_ic_fetchers():
    with (
        patch(
            "ecodata_cache.fetchers.dbofs.fetch_dbofs_initial_conditions"
        ) as mock_dbofs,
        patch(
            "ecodata_cache.fetchers.necofs.fetch_necofs_initial_conditions"
        ) as mock_necofs,
        patch(
            "ecodata_cache.fetchers.neracoos.fetch_neracoos_initial_conditions"
        ) as mock_neracoos,
        patch(
            "ecodata_cache.fetchers.maracoos.fetch_maracoos_initial_conditions"
        ) as mock_maracoos,
        patch(
            "ecodata_cache.fetchers.hycom.fetch_hycom_initial_conditions"
        ) as mock_hycom,
    ):
        yield mock_dbofs, mock_necofs, mock_neracoos, mock_maracoos, mock_hycom


@patch("ecodata_cache.dispatcher.os.path.exists")
@patch("shutil.rmtree")
def test_dispatch_ic_neracoos(mock_rmtree, mock_exists, mock_ic_fetchers, mocker):
    mock_dbofs, mock_necofs, mock_neracoos, mock_maracoos, mock_hycom = mock_ic_fetchers

    mock_exists.return_value = False

    mock_ds = mocker.MagicMock()
    mock_var = mocker.MagicMock()
    mock_var.dtype.kind = "f"
    mock_ds.variables = {"u": mock_var}
    mock_ds.__getitem__.return_value = mock_var
    mock_neracoos.return_value = mock_ds

    from ecodata_cache.dispatcher import dispatch_ic_request

    # Use bbox within NERACOOS domain [-71.5, 42.0, -69.0, 44.5]
    zarr_path = dispatch_ic_request("2026-03-03", [-71.0, 43.0, -70.0, 44.0])

    assert mock_neracoos.called
    assert not mock_maracoos.called
    assert not mock_hycom.called
    assert mock_ds.to_zarr.called
    assert zarr_path.endswith(".zarr")


@patch("ecodata_cache.dispatcher.os.path.exists")
@patch("shutil.rmtree")
def test_dispatch_ic_fallback_to_hycom(
    mock_rmtree, mock_exists, mock_ic_fetchers, mocker
):
    mock_dbofs, mock_necofs, mock_neracoos, mock_maracoos, mock_hycom = mock_ic_fetchers

    mock_exists.return_value = False

    # Force regional failures
    mock_dbofs.side_effect = Exception("DBOFS out of bounds")
    mock_necofs.side_effect = Exception("NECOFS out of bounds")
    mock_neracoos.side_effect = Exception("NERACOOS out of bounds")
    mock_maracoos.side_effect = Exception("MARACOOS out of bounds")

    mock_ds = mocker.MagicMock()
    mock_var = mocker.MagicMock()
    mock_var.dtype.kind = "f"
    mock_ds.variables = {"water_u": mock_var}
    mock_ds.__getitem__.return_value = mock_var
    mock_hycom.return_value = mock_ds

    from ecodata_cache.dispatcher import dispatch_ic_request

    # Use bbox supported by NERACOOS, NECOFS, MARACOOS, and HYCOM
    _ = dispatch_ic_request("2026-03-03", [-70.8, 42.9, -70.5, 43.1])

    assert mock_neracoos.called
    assert mock_maracoos.called
    assert mock_hycom.called
    assert mock_ds.to_zarr.called


def test_nyofs_expanded_bbox_covers_offshore_nj():
    """NYOFS domain_bbox min_lat should be 40.2 to accurately restrict its active domain."""
    from ecodata_cache.fetchers import nyofs

    meta = nyofs.get_metadata()
    assert (
        meta["domain_bbox"][1] >= 40.0
    ), "NYOFS domain_bbox min_lat should be ~40.2 for realistic coverage"


def test_nyofs_rejects_offshore_nj_bbox():
    """NYOFS supports_bbox must REJECT a bbox in the expanded NY Bight region."""
    from ecodata_cache.fetchers import nyofs

    # A bbox fully within [-74.3, 39.5, -73.3, 41.1]
    bbox = [-74.1, 39.6, -73.5, 39.9]
    assert not nyofs.supports_bbox(bbox)


def test_dbofs_rejects_mid_atlantic_bight_ic_due_to_hydro_mask():
    """DBOFS must reject the mid_atlantic_bight_100m bbox because it falls outside the active hydro mask."""
    from ecodata_cache.dispatcher import _rank_ic_candidates

    bbox = [-73.961826826, 39.724415971, -73.78395905, 39.877567191]
    ranked = _rank_ic_candidates(bbox)

    assert len(ranked) > 0
    assert (
        ranked[0][2]["id"] != "dbofs"
    ), "Expected DBOFS to be rejected for this IC donor due to offshore masking"


def test_dispatch_station_profiles_request_wlis(mocker):
    import os
    from ecodata_cache.dispatcher import dispatch_station_profiles_request

    mock_fetcher = mocker.patch(
        "ecodata_cache.fetchers.erddap.fetch_erddap_station_profiles"
    )
    mock_fetcher.return_value = {"surface": {"2024-05-01T00:00:00": 10.0}}

    result = dispatch_station_profiles_request(
        "WLIS", "2024-05-01T00:00:00Z", "2024-05-01T06:00:00Z"
    )
    assert "surface" in result
    mock_fetcher.assert_called_once_with(
        station_id="WLIS",
        start_time="2024-05-01T00:00:00Z",
        end_time="2024-05-01T06:00:00Z",
        cache_dir=os.path.join(
            os.environ.get(
                "ECODATA_CACHE_CACHE_DIR",
                os.path.expanduser("~/.cache/ecodata-cache"),
            ),
            "erddap",
        ),
        cache_bust=False,
    )
