import numpy as np
from sklearn.mixture import _gaussian_mixture as _gmm


__all__ = ["CGaussianMixture", "fit_cgmm"]


def _estimate_cgaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.
    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.
    means : array-like of shape (n_components, n_features)
        The centers of the current components.
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # enforce that the mean of the first component is zero
    means[0, 0] = 0
    # and the remaining means are above the estimated standard deviation of the noise
    # sigma_hat = np.sqrt(np.mean(X[X<=0]**2))
    # means[1:, 0] = np.clip(means[1:, 0], a_min=3*sigma_hat, a_max=10)
    covariances = {
        "full": _gmm._estimate_gaussian_covariances_full,
        "tied": _gmm._estimate_gaussian_covariances_tied,
        "diag": _gmm._estimate_gaussian_covariances_diag,
        "spherical": _gmm._estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)

    return nk, means, covariances


def _estimate_gaussian_prob(X, means, precisions_chol):
    """Estimate the Gaussian probability.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    Returns
    -------
    prob : array, shape (n_samples, n_components)
    """
    precisions = precisions_chol**2
    log_prob = (X - means.T)**2 * precisions.T
    return np.sqrt(1/2/np.pi*precisions.T) * np.exp(-log_prob/2)


class CGaussianMixture(_gmm.GaussianMixture):
    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        random_state=None,
        verbose=0,
        verbose_interval=10,
    ):

        means_init = np.concatenate([[0,], np.linspace(.5, 2, n_components-1)])
        weights_init = np.ones_like(means_init)
        weights_init[0] = .8
        weights_init[1:] = (1 - weights_init[0])/(n_components-1)
        sigmas_init = np.ones_like(means_init) * .25
        sigmas_init[0] = .1

        super().__init__(
            n_components=n_components,
            covariance_type="diag",
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params="kmeans",
            weights_init=weights_init,
            means_init=means_init[:, None],
            precisions_init=1/sigmas_init[:, None]**2,
            random_state=random_state,
            warm_start=False,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_cgaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _gmm._compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_weighted_prob(self, X):
        """Estimate the weighted probabilities, log P(X | Z) * weights.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return _estimate_gaussian_prob(X, self.means_, self.precisions_cholesky_) * self.weights_[None, :]

    def _estimate_prob(self, X):
        """Estimate probabilities for each sample.
        Compute the probabilities, weighted probabilities per
        component for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
             responsibilities
        """
        weighted_prob = self._estimate_weighted_prob(X)
        return weighted_prob / np.sum(weighted_prob, 1, keepdims=True)

    def predict_proba_signal(self, X):
        """Evaluate the components' density for each sample.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        resp : array, shape (n_samples, )
            Probability for each sample in X not originating from noise.
        """
        X = self._validate_data(X, reset=False)
        proba_noise = self._estimate_prob(X)[:, 0]
        proba_noise[X[:, 0] <= 0] = 1.
        return 1 - proba_noise


def fit_cgmm(data, n_components=5, N=100_000):
    X = np.random.choice(data, size=N, replace=False)
    return CGaussianMixture(n_components=n_components, tol=1e-4).fit(X[:, None])
