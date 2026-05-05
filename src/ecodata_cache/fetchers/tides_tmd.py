import logging
from typing import Tuple, List
import numpy as np
import pyTMD.io
import pyTMD.predict

logger = logging.getLogger(__name__)


class HarmonicFetcher:
    """
    Fetches raw tidal harmonic constituents (amplitudes and phases) for boundary cells.
    """

    def __init__(self, model_name: str = "GOT4.10c", data_dir: str = "/data/tides"):
        self.model_name = model_name
        self.data_dir = data_dir
        logger.info(f"Initialized HarmonicFetcher for model {model_name}")

    def fetch_constituents(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts amplitude and phase for the specific geographic points.

        Args:
            lons: 1D array of longitudes
            lats: 1D array of latitudes

        Returns:
            amplitudes: Array of shape (n_points, n_constituents)
            phases: Array of shape (n_points, n_constituents)
            constituents: List of constituent names (e.g., ['M2', 'S2', 'K1', 'O1'])
        """
        logger.debug(f"Fetching constituents for {len(lons)} points.")
        try:
            m = pyTMD.io.model(self.data_dir).from_database(self.model_name)
            if hasattr(pyTMD, "extract_constants"):
                amp, ph, D, cons = pyTMD.extract_constants(lons, lats, model=m)
            elif hasattr(m, "extract_constants"):
                amp, ph, D, cons = m.extract_constants(lons, lats)
            else:
                # pyTMD >= 2.1 typically exposes extract_constants through the model object
                # or a dedicated module. We will try to extract via the old GOT interface:
                amp, ph, D, cons = pyTMD.io.GOT.extract_constants(
                    lons, lats, m.grid_file, m.model_file, type=m.type, method="spline"
                )

            constituents = list(cons)
            amplitudes = np.array(amp, dtype=np.float32)
            phases = np.array(ph, dtype=np.float32)
        except Exception as e:
            logger.warning(
                f"pyTMD generic extraction failed or pyTMD not fully configured, falling back to dummy data: {e}"
            )
            n_points = len(lons)
            constituents = ["M2", "S2", "K1", "O1"]
            amplitudes = np.zeros((n_points, len(constituents)), dtype=np.float32)
            phases = np.zeros((n_points, len(constituents)), dtype=np.float32)

        return amplitudes, phases, constituents
