import numpy as np
from scipy.special import erf
from typing import Callable

from multimodal_emg.models.basic_terms import gauss_term, asymm_term


def gaussian_envelope_model(
        alpha: float = None,
        mu: float = 0,
        sigma: float = 1,
        eta: float = 0,
        f_c: int = None,
        phi: int = None,
        exp_fun: Callable = np.exp,
        erf_fun: Callable = erf,
        cos_fun: Callable = np.cos,
        x: np.ndarray = np.linspace(-1e3, 1e3, int(2e3)),
        ):
    """
    Synthesis of univariate Gaussian model with the classical parameterization.

    :param alpha: normalization factor (amplitude)
    :param mu: mean (position)
    :param sigma: spread (width)
    :param eta: skew parameter (placeholder only)
    :param f_c: central frequency (placeholder only)
    :param phi: phase shift (placeholder only)
    :param x: sample positions
    :return: synthesized univariate gaussian
    """

    alpha = 1 if alpha is None else alpha
    gauss = gauss_term(x, mu, sigma, exp_fun)

    return alpha * gauss


def emg_envelope_model(
        alpha: float = None,
        mu: int = 0,
        sigma: float = 1.0,
        eta: float = 0,
        f_c: int = None,
        phi: int = None,
        exp_fun: Callable = np.exp,
        erf_fun: Callable = erf,
        cos_fun: Callable = np.cos,
        x: np.ndarray = np.linspace(-1e3, 1e3, int(2e3)),
        ):
    """
    Synthesis of univariate Exponentially-Modified-Gaussian (EMG) with the classical parameterization.

    :param alpha: normalization factor (amplitude)
    :param mu: mean (position)
    :param sigma: spread (width)
    :param eta: skew parameter
    :param f_c: central frequency (placeholder only)
    :param phi: phase shift (placeholder only)
    :param exp_fun: exponential function considering data type
    :param erf_fun: error function considering data type
    :param x: sample positions
    :return: synthesized univariate exponentially modified gaussian
    """

    alpha = 1 if alpha is None else alpha
    gauss = gauss_term(x, mu, sigma, exp_fun)
    asymm = asymm_term(x, mu, sigma, eta, erf_fun)

    return alpha * gauss * asymm
