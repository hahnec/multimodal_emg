import numpy as np
from scipy.special import erf
from typing import Callable

from multimodal_emg.models.basic_terms import gauss_term, asymm_term, oscil_term


def gaussian_wave_model(
        alpha: float = 1,
        mu: float = 0,
        sigma: float = 1e-5,
        eta: float = 0,      
        fkhz: int = 1e-1,
        phi: int = 0,
        exp_fun: Callable = np.exp,
        erf_fun: Callable = erf,
        cos_fun: Callable = np.cos,
        x: np.ndarray = np.linspace(-1e3, 1e3, int(2e3)),
            ):
    """
    Synthesis of univariate Gaussian model with the oscillation term.

    :param alpha: normalization factor (amplitude)
    :param mu: mean value (position)
    :param sigma: spread (width)
    :param eta: skew parameter (placeholder only)
    :param f_c: central frequency
    :param phi: phase shift
    :param t: sample positions
    :return: synthesized univariate oscillating Gaussian
    """

    alpha = 1 if alpha is None else alpha
    gauss = gauss_term(x, mu, sigma, exp_fun)
    oscil = oscil_term(x, mu, fkhz*1e3, phi, cos_fun)

    return alpha * gauss * oscil


def emg_wave_model(
        alpha: float = None,
        mu: int = 0,
        sigma: float = 1.0,
        eta: float = 0,
        fkhz: int = 1e-1,
        phi: int = 0,
        exp_fun: Callable = np.exp,
        erf_fun: Callable = erf,
        cos_fun: Callable = np.cos,
        x=np.linspace(-1e3, 1e3, int(2e3)),
            ):
    """
    Synthesis of univariate Exponentially-Modified-Gaussian (EMG) model with oscillation term.

    :param alpha: normalization factor (amplitude)
    :param mu: mean (position)
    :param sigma: spread (width)
    :param eta: skew parameter
    :param f_c: central frequency
    :param phi: phase shift
    :param t: sample positions
    :return: synthesized univariate oscillating exponentially modified gaussian
    """

    alpha = 1 if alpha is None else alpha
    gauss = gauss_term(x, mu, sigma, exp_fun)
    oscil = oscil_term(x, mu, fkhz*1e3, phi, cos_fun)
    asymm = asymm_term(x, mu, sigma, eta, erf_fun)

    return alpha * gauss * oscil * asymm
