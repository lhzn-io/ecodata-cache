import sys
import logging
from ecodata_cache.fetchers.era5 import fetch_era5_surface_forcing

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Test bounding box (Throgs Neck)
    bbox = [40.85, -73.85, 40.75, -73.75]  # N, W, S, E
    target_date = "2024-03-01"  # Past date to guarantee ERA5 is available
    import os

    output_path = os.path.join(
        os.path.expanduser("~/.cache/ecodata-cache"), "test_era5_arco.zarr"
    )

    try:
        res = fetch_era5_surface_forcing(
            target_date, bbox, output_path, preliminary=False
        )
        print(f"SUCCESS! Downloaded to: {res}")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
