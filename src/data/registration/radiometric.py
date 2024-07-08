import logging
import numpy as np


def radiometric(reference: np.ndarray, image: np.ndarray,
                mask: np.ndarray = None,
                alpha: float = 0.05,
                diff_quantile: float = None,
                max_iter: int = 1000) -> dict:
    """ Performs affine registration of intensities.

        Soves:
            min_s 1/2 \langel \rho( M*(x - (s*y)), 1 \rangle,
        where `x` is the reference image, `y` the input image,
        and `M` is a binary mask obtained by computing a quantile
        of `y`. The penalty function `\rho` is defined as
           \rho(x) = \log(1+x^2/\alpha).

        Args:
            reference: reference image.
            image: to be transformed.
            mask: optional region of interest
            diff_quantile: optional quantile that determines the threshold
                for masking out all differences above
            alpha: turning point of robust statisitcs method
            max_iter: maximal number of iterations
        Returns:
            dictionary with the entries:
                - out: radiometrically registered image
                - error: energy
                - params: resulting scale s
    """

    x = image
    y = reference

    # select only reference pixels with "large" intensities
    if mask is None:
        mask = (y > 0.01) & (x > 0.01)

    if diff_quantile:
        diff = x - y
        th = np.quantile(diff[mask], diff_quantile)
        mask = mask & (diff < th)

    if np.sum(mask) < 1000:
        raise RuntimeError("Insufficient number of pixels to register")
    x = x[mask].ravel()[::8]  # subsample for speed
    y = y[mask].ravel()[::8]  # subsample for speed
    logging.info(f"Radiometric registration: using {x.size} samples and {alpha=}")

    # normalize the inputs (=> Lipschitz constant = 1)
    x_norm2 = np.sum(x**2)

    def energy_grad(s, x, y, alpha):
        alpha_ = 1/alpha**2
        diff = s*x - y
        # compute rho
        denom = 1 + alpha_*diff**2
        energy = np.sum(np.log(denom)/2/alpha_)/x_norm2
        grad = np.sum(x*diff/denom)/x_norm2
        return energy, grad

    # overrelaxation parameter
    beta = 0.7
    # initial scale estimate
    s = np.array(1.0)

    s_old = s.copy()
    E = np.zeros((max_iter,), dtype=np.float32)
    for i in range(max_iter):
        # overrelax
        s_hat = s + beta*(s-s_old)
        s_old = s.copy()
        # gradient step
        energy, grad = energy_grad(s_hat, x, y, alpha=alpha)
        s = s_hat - grad
        E[i] = energy
        logging.debug(f"Radiometric registration: {i:2d}: {energy=:.3e} {grad=:.3e} {s=:.5f}")
        # stopping criterion
        if np.abs(s - s_old) < 1e-5:
            E = E[:i+1]
            break

    logging.info(f"Radiometric registration: scale {s:.5f}; final energy {E[-1]:.2e} after {i} steps")

    if i > 0 and E[0] < E[-1]:
        raise RuntimeError("Radiometric registration did not converge!")

    # compute the output
    out = image*s
    return {
        "out": out,
        "energy": E,
        "params": s,
    }
