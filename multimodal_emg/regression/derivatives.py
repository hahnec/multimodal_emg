from typing import Callable, Union
import numpy as np
import torch
from scipy.special import erf

from multimodal_emg.regression.lib_handler import get_lib
from multimodal_emg.models.basic_terms import gauss_term, asymm_term, oscil_term, PI

# partial differential equations of gaussian
pd_gauss_wrt_mu = lambda x, mu, sigma, exp_fun=np.exp: gauss_term(x, mu, sigma, exp_fun) * (x-mu)/sigma**2
pd_gauss_wrt_sigma = lambda x, mu, sigma, exp_fun=np.exp: gauss_term(x, mu, sigma, exp_fun) * (x-mu)**2/sigma**3

# partial differential equations of exponentially modified term
erf_term = lambda x, mu, sigma, eta, exp_fun=np.exp: exp_fun(-.5*eta**2*(x-mu)**2/sigma**2)
pd_asymm_wrt_mu = lambda x, mu, sigma, eta, exp_fun=np.exp: erf_term(x, mu, sigma, eta, exp_fun) * -1 * (2/PI)**.5 * eta/sigma
pd_asymm_wrt_sigma = lambda x, mu, sigma, eta, exp_fun=np.exp: erf_term(x, mu, sigma, eta, exp_fun) * -1 * (2/PI)**.5 * eta*(x-mu)/sigma**2
pd_asymm_wrt_eta = lambda x, mu, sigma, eta, exp_fun=np.exp: erf_term(x, mu, sigma, eta, exp_fun) * (2/PI)**.5 * (x-mu)/sigma

# partial differential equation of oscillation term
pd_oscil_wrt_mu = lambda x, mu, f_c, phi, sin_fun=np.sin: 2 * PI * f_c * sin_fun(2 * PI * f_c * (x - mu) + phi)

def pd_emg_wrt_alpha(x, mu, sigma, eta, exp_fun=np.exp, erf_fun=erf):

    return gauss_term(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun)

def pd_emg_wrt_mu(x, mu, sigma, eta, exp_fun=np.exp, erf_fun=erf):

    product_rule_term_a = gauss_term(x, mu, sigma, exp_fun) * pd_asymm_wrt_mu(x, mu, sigma, eta, exp_fun)
    product_rule_term_b = pd_gauss_wrt_mu(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun)

    return product_rule_term_a + product_rule_term_b

def pd_emg_wrt_sigma(x, mu, sigma, eta, exp_fun=np.exp, erf_fun=erf):

    product_rule_term_a = gauss_term(x, mu, sigma, exp_fun) * pd_asymm_wrt_sigma(x, mu, sigma, eta, exp_fun)
    product_rule_term_b = pd_gauss_wrt_sigma(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun)

    return product_rule_term_a + product_rule_term_b

def pd_emg_wrt_eta(x, mu, sigma, eta, exp_fun=np.exp):

    return gauss_term(x, mu, sigma, exp_fun) * pd_asymm_wrt_eta(x, mu, sigma, eta, exp_fun)

def pd_emg_wrt_mu_scratch(x, mu, sigma, eta, exp_fun=np.exp, erf_fun=erf):

    term_a = (exp_fun(-(x-mu)**2/(2*sigma**2)*(1+eta**2))*eta*(2/PI)**.5/sigma * asymm_term) / sigma**3
    term_b = ((x-mu)/sigma**2) * asymm_term(x, mu, sigma, eta, erf_fun) * gauss_term(x, mu, sigma, exp_fun)

    return term_a - term_b

def pd_emg_wrt_sigma_scratch(x, mu, sigma, eta, exp_fun=np.exp):
    
    term_a = ((2/PI)**.5 * eta * (x-mu) * exp_fun(-1*eta**2*(x-mu)**2/(2*sigma**2)-(x-mu)**2/(2*sigma**2))) / sigma**2
    term_b = (x-mu)**2*exp_fun(-(x-mu)**2/(2*sigma**2)/(2*sigma**2))

    return term_a - term_b

def pd_oemg_wrt_mu(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, erf_fun=erf, cos_fun=np.cos, sin_fun=np.sin):

    product_rule_term_a = pd_asymm_wrt_mu(x, mu, sigma, eta, exp_fun) * gauss_term(x, mu, sigma, exp_fun) * oscil_term(x, mu, f_c, phi, cos_fun)
    product_rule_term_b = pd_gauss_wrt_mu(x, mu, sigma, exp_fun) * oscil_term(x, mu, f_c, phi, cos_fun) * asymm_term(x, mu, sigma, eta, erf_fun)
    product_rule_term_c = pd_oscil_wrt_mu(x, mu, f_c, phi, sin_fun) * gauss_term(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun)

    return product_rule_term_a + product_rule_term_b + product_rule_term_c

def pd_oemg_wrt_sigma(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, erf_fun=erf, cos_fun=np.cos):

    product_rule_term_a = pd_asymm_wrt_sigma(x, mu, sigma, eta, exp_fun) * gauss_term(x, mu, sigma, exp_fun)
    product_rule_term_b = pd_gauss_wrt_sigma(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun)

    return (product_rule_term_a + product_rule_term_b) * oscil_term(x, mu, f_c, phi, cos_fun)

def pd_oemg_wrt_eta(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, cos_fun=np.cos):

    return gauss_term(x, mu, sigma, exp_fun) * oscil_term(x, mu, f_c, phi, cos_fun) * pd_asymm_wrt_eta(x, mu, sigma, eta, exp_fun)

def pd_oemg_wrt_f_c(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, erf_fun=erf, sin_fun=np.sin):

    return gauss_term(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun) * -2 * PI * (x-mu) * sin_fun(2*PI*f_c*(x-mu)+phi)

def pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, erf_fun=erf, sin_fun=np.sin):

    return gauss_term(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun) * -1 * sin_fun(2*PI*f_c*(x-mu)+phi)

def pd_oemg_wrt_alpha(x, mu, sigma, eta, f_c, phi, exp_fun=np.exp, erf_fun=erf, cos_fun=np.cos):

    return gauss_term(x, mu, sigma, exp_fun) * asymm_term(x, mu, sigma, eta, erf_fun) * oscil_term(x, mu, f_c, phi, cos_fun)

def gaussian_jac(alpha=None, mu=None, sigma=None, x=None, exp_fun=np.exp, erf_fun=erf):

    lib = get_lib(x)

    jacobian = -1 * lib.vstack([
        gauss_term(x, mu, sigma),
        alpha*pd_gauss_wrt_mu(x, mu, sigma),
        alpha*pd_gauss_wrt_sigma(x, mu, sigma),
    ])

    return jacobian.T

def emg_jac(alpha=None, mu=None, sigma=None, eta=None, x=None, exp_fun=np.exp, erf_fun=erf):

    lib = get_lib(x)

    jacobian = -1 * lib.vstack([
        pd_emg_wrt_alpha(x, mu, sigma, eta, exp_fun, erf_fun),
        alpha*pd_emg_wrt_mu(x, mu, sigma, eta, exp_fun, erf_fun),
        alpha*pd_emg_wrt_sigma(x, mu, sigma, eta, exp_fun, erf_fun),
        alpha*pd_emg_wrt_eta(x, mu, sigma, eta, exp_fun),
    ])

    return jacobian.T

def oemg_jac(alpha=None, mu=None, sigma=None, eta=None, fkhz=None, phi=None, x=None, exp_fun=np.exp, erf_fun=erf):

    f_c = fkhz * 1e3

    lib = get_lib(x)

    jacobian = -1 * lib.vstack([
        pd_oemg_wrt_alpha(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_mu(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, cos_fun=lib.cos, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_sigma(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_eta(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_f_c(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, sin_fun=lib.sin),
    ])

    return jacobian.T

def wav_jac(alpha=None, mu=None, sigma=None, eta=None, fkhz=None, phi=None, x=None, exp_fun=np.exp, erf_fun=erf):

    f_c = fkhz * 1e3

    lib = get_lib(x)

    jacobian = -1 * lib.vstack([
        lib.zeros(len(x)),
        lib.zeros(len(x)),
        lib.zeros(len(x)),
        lib.zeros(len(x)),
        alpha*pd_oemg_wrt_f_c(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=exp_fun, erf_fun=erf_fun, sin_fun=lib.sin),
    ])

    return jacobian.T

def components_jac(
        p, 
        model: Callable, 
        data: Union[np.ndarray, torch.Tensor], 
        components: int,
        jac_component_with_args: Callable,
    ):

    lib = get_lib(data)

    n = len(p) // components
    c = 2*(data-model(p))[..., None]   # chain rule term

    d = []
    for i in range(components):
        jac = jac_component_with_args(p[i*n:(i+1)*n])
        d.append(c*jac)

    return lib.hstack(d)
