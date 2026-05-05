import os
import cdsapi
import logging
from datetime import datetime

from .validation import verify_sources

logger = logging.getLogger(__name__)


def fetch_era5_surface_forcing(
    target_date: str,
    bbox: list[float],
    output_path: str,
    preliminary: bool = False,
    cache_bust: bool = False,
) -> str:
    """
    Fetches ERA5 or ERA5T (Preliminary) hourly single levels for wind and pressure forcing.
    """
    verify_sources(
        source_name="ERA5/HRES" if not preliminary else "ERA5T",
        url="https://cds.climate.copernicus.eu/api",
        timestamp=datetime.utcnow().isoformat(),
        spatial_res="30km",
        requires_auth=True,
    )

    if not cache_bust and os.path.exists(output_path):
        logger.info(
            f"Cache hit: {output_path} already exists. Skipping CDS API request."
        )
    else:
        client = cdsapi.Client()

        logger.info(
            f"Fetching {'ERA5T' if preliminary else 'ERA5'} surface forcing for {target_date} over bbox {bbox}..."
        )

        dt = datetime.strptime(target_date, "%Y-%m-%d")
        year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

        dataset = "reanalysis-era5-single-levels"

        pad = 0.5
        padded_bbox = [
            bbox[0] + pad,  # North
            bbox[1] - pad,  # West
            bbox[2] - pad,  # South
            bbox[3] + pad,  # East
        ]

        request_params = {
            "product_type": "reanalysis",
            "data_format": "grib",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": [f"{str(hour).zfill(2)}:00" for hour in range(24)],
            "area": padded_bbox,
        }

        try:
            client.retrieve(dataset, request_params, output_path)
            logger.info(
                f"Successfully downloaded Surface GRIB forcing to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to fetch ERA5 surface data: {e}")
            raise e

    return output_path


def fetch_era5_pressure_levels(
    target_date: str,
    bbox: list[float],
    output_path: str,
    cache_bust: bool = False,
) -> str:
    """
    Fetches ERA5 hourly pressure level data for volumetric atmospheric forcing.
    Required variables: u, v, w, T, geopotential (Z), and specific humidity (q).
    """
    if not cache_bust and os.path.exists(output_path):
        logger.info(
            f"Cache hit: {output_path} already exists. Skipping CDS API request."
        )
        return output_path

    client = cdsapi.Client()
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    dataset = "reanalysis-era5-pressure-levels"

    pad = 0.5
    padded_bbox = [bbox[0] + pad, bbox[1] - pad, bbox[2] - pad, bbox[3] + pad]

    # Pressure levels from 1000 hPa up to 1 hPa (tropopause and beyond)
    levels = [
        "1",
        "2",
        "3",
        "5",
        "7",
        "10",
        "20",
        "30",
        "50",
        "70",
        "100",
        "125",
        "150",
        "175",
        "200",
        "225",
        "250",
        "300",
        "350",
        "400",
        "450",
        "500",
        "550",
        "600",
        "650",
        "700",
        "750",
        "775",
        "800",
        "825",
        "850",
        "875",
        "900",
        "925",
        "950",
        "975",
        "1000",
    ]

    request_params = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "temperature",
            "geopotential",
            "specific_humidity",
        ],
        "pressure_level": levels,
        "year": year,
        "month": month,
        "day": day,
        "time": [f"{str(hour).zfill(2)}:00" for hour in range(24)],
        "area": padded_bbox,
    }

    try:
        logger.info(f"Fetching ERA5 pressure-level data for {target_date}...")
        client.retrieve(dataset, request_params, output_path)
        logger.info(
            f"Successfully downloaded Pressure Level GRIB forcing to {output_path}"
        )
    except Exception as e:
        logger.error(f"Failed to fetch ERA5 pressure level data: {e}")
        raise e

    return output_path
