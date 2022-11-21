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

def gaussian_jac(alpha=None, mu=None, sigma=None, x=None):

    lib = get_lib(x)

    jacobian = -1 * lib.stack([
        gauss_term(x, mu, sigma, lib.exp),
        alpha*pd_gauss_wrt_mu(x, mu, sigma, lib.exp),
        alpha*pd_gauss_wrt_sigma(x, mu, sigma, lib.exp),
    ])

    return jacobian

def emg_jac(alpha=None, mu=None, sigma=None, eta=None, x=None):

    lib = get_lib(x)

    erf_fun = torch.erf if lib.__name__ == 'torch' else erf

    jacobian = -1 * lib.stack([
        pd_emg_wrt_alpha(x, mu, sigma, eta, lib.exp, erf_fun),
        alpha*pd_emg_wrt_mu(x, mu, sigma, eta, lib.exp, erf_fun),
        alpha*pd_emg_wrt_sigma(x, mu, sigma, eta, lib.exp, erf_fun),
        alpha*pd_emg_wrt_eta(x, mu, sigma, eta, lib.exp),
    ])

    return jacobian

def oemg_jac(alpha=None, mu=None, sigma=None, eta=None, fkhz=None, phi=None, x=None):

    f_c = fkhz * 1e3

    lib = get_lib(x)

    erf_fun = torch.erf if lib.__name__ == 'torch' else erf

    jacobian = -1 * lib.stack([
        pd_oemg_wrt_alpha(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_mu(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, cos_fun=lib.cos, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_sigma(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_eta(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, cos_fun=lib.cos),
        alpha*pd_oemg_wrt_f_c(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, sin_fun=lib.sin),
    ])

    return jacobian

def wav_jac(alpha=None, mu=None, sigma=None, eta=None, fkhz=None, phi=None, x=None):

    f_c = fkhz * 1e3

    lib = get_lib(x)

    erf_fun = torch.erf if lib.__name__ == 'torch' else erf

    jacobian = -1 * lib.stack([
        lib.zeros_like(x),
        lib.zeros_like(x),
        lib.zeros_like(x),
        lib.zeros_like(x),
        alpha*pd_oemg_wrt_f_c(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, sin_fun=lib.sin),
        alpha*pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, sin_fun=lib.sin),
    ])

    return jacobian

def phi_jac(alpha=None, mu=None, sigma=None, eta=None, fkhz=None, phi=None, x=None):

    f_c = fkhz * 1e3

    lib = get_lib(x)

    erf_fun = torch.erf if lib.__name__ == 'torch' else erf

    jacobian = -1 * lib.stack([
        lib.zeros_like(x),
        lib.zeros_like(x),
        lib.zeros_like(x),
        lib.zeros_like(x),
        lib.zeros_like(x),
        alpha*pd_oemg_wrt_phi(x, mu, sigma, eta, f_c, phi, exp_fun=lib.exp, erf_fun=erf_fun, sin_fun=lib.sin),
    ])

    return jacobian

def components_jac(
        p, 
        model: Callable, 
        data: Union[np.ndarray, torch.Tensor], 
        components: int,
        jac_component_with_args: Callable,
    ):

    lib = get_lib(data)

    feats_num = len(p) // components
    c = 2*(data-model(p))[..., None]   # chain rule term

    d = []
    for i in range(components):
        jac = jac_component_with_args(p[i*feats_num:(i+1)*feats_num])
        d.append(c*jac.T)

    return lib.hstack(d)

def batch_components_jac(
        p, 
        model: Callable, 
        data: torch.Tensor, 
        components: int,
        jac_component_with_args: Callable,
    ):

    c = 2*(data.unsqueeze(1)-model(p))[..., None]   # chain rule term

    feats_num = p.shape[-1] // components
    feats = p.view(-1, components, feats_num)    # reshape to batch x components x features 
    if feats_num > 4:
        # phase in (-pi, pi] constraint by wrapping values into co-domain
        feats[..., 5][feats[..., 5] < -PI] += 2*PI
        feats[..., 5][feats[..., 5] > +PI] -= 2*PI
    feats = feats.view(-1, feats_num).T.unsqueeze(-1)    # features x batch*components

    jac = jac_component_with_args(feats)
    #jac = torch.swapaxes(jac.view(data.shape[-1], -1, components, feats_num), 0, 1).flatten(start_dim=2)   # with transpose
    #jac = torch.swapaxes(jac.reshape(data.shape[-1], -1, components*feats_num), 0, 1)  # with transpose
    #jac = torch.swapaxes(torch.swapaxes(jac.reshape(components*feats_num, -1, data.shape[-1]), 0, 1), 1, 2)  # without transpose
    jac = jac.swapaxes(0, 2).reshape(data.shape[-1], -1, components*feats_num).swapaxes(0, 1)
    jac = torch.nan_to_num(jac, nan=0)
    d = c * jac.unsqueeze(1)

    return d
