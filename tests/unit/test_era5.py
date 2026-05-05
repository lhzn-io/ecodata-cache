from unittest.mock import patch
from ecodata_cache.fetchers.era5 import fetch_era5_pressure_levels


@patch("ecodata_cache.fetchers.era5.cdsapi.Client")
def test_era5_pressure_levels(mock_cds):
    """Test ERA5 pressure level fetcher call."""
    bbox = [41.0, -74.0, 40.0, -73.0]
    output = "test_era5_vol.grib"

    with patch("os.path.exists", return_value=False):
        fetch_era5_pressure_levels("2021-08-01", bbox, output)

        mock_cds.return_value.retrieve.assert_called_once()
        args = mock_cds.return_value.retrieve.call_args[0]
        assert args[0] == "reanalysis-era5-pressure-levels"
        assert "pressure_level" in args[1]
        assert len(args[1]["pressure_level"]) > 10
