import numpy as np
from scipy.interpolate import interp1d


__all__ = ["Nyul"]


class Nyul:
    # VMAX_Quantile is now based on the brain region and therefore choosen to be higher.
    VMAX_QUANTILE = 0.95
    NUM_LANDMARKS = 5
    ROI_THRESHOLD = 1e-6

    def compute_landmarks(data: np.array, mask: np.array = None) -> np.array:
        if mask is None:
            mask = data > Nyul.ROI_THRESHOLD
        percentiles = np.percentile(
            data[mask], np.linspace(0., 100., Nyul.NUM_LANDMARKS)
        ).astype(np.float32)
        percentiles[0] = 0.     # 0 Should map to 0. Had problems with first 2 percentiles being equal.
        # 1 Should map to 1. This should not happen, but we need a slope > 0 for the algorithm to work.
        percentiles[-1] = max(1., percentiles[-2] + 1e-3)
        return percentiles

    def normalize(data, standard_scale, landmarks=None):
        if landmarks is None:
            landmarks = Nyul.compute_landmarks(data)

        f = interp1d(landmarks, standard_scale, fill_value="extrapolate")
        return f(data)

    def reverse(data, landmarks, standard_scale):
        f = interp1d(standard_scale, landmarks, fill_value="extrapolate")
        return f(data)
