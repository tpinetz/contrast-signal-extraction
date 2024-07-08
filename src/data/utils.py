import numpy as np
# import pyfftw
import SimpleITK as sitk

__all__ = ["scan_mask", "fft", "ifft", "estimate_local_mean_std"]

# define fft backend
fft_backend = np.fft
# fft_backend = pyfftw.interfaces.numpy_fft
# pyfftw.config.NUM_THREADS = 16


def scan_mask(mask: sitk.Image, radius: int = 3) -> sitk.Image:
    """ filter scan mask

    Args:
        mask: input mask
        radius: (3) radius of closing structure element (ball)

    Returns:
        sitk.Image: binary mask
    """
    rr = [radius] * mask.GetDimension()
    return sitk.BinaryMorphologicalClosing(mask, kernelRadius=rr, kernelType=sitk.sitkBall,
                                           safeBorder=True)


def fft(x: np.ndarray) -> np.ndarray:
    return fft_backend.fftshift(fft_backend.fftn(x, axes=(-3, -2, -1)))


def ifft(x: np.ndarray) -> np.ndarray:
    return fft_backend.ifftn(fft_backend.ifftshift(x), axes=(-3, -2, -1)).real


def estimate_local_mean_std(
        x: np.ndarray,
        sigma_est: float,
        guassian_std: float = 16
        ) -> np.ndarray:

    def filter(x):
        yy = sitk.DiscreteGaussian(
            sitk.GetImageFromArray(x),
            variance=guassian_std**2,
            maximumKernelWidth=guassian_std*4+1
        )
        return sitk.GetArrayFromImage(yy)

    mask = np.ones_like(x)
    mask[x >= +2*sigma_est] = 0
    mask[x <= -2*sigma_est] = 0
    M = filter(mask)

    mean = filter(mask * x) / (M + 1e-8)
    std = np.sqrt(filter(mask * x**2) / (M + 1e-8) - mean**2)

    return mean, std


def sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(under='ignore', over='ignore'):
        return 1/(1 + np.exp(-x))
