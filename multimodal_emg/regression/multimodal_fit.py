import numpy
import torch
from typing import Callable, Union
from scipy.special import erf
from scipy.optimize import least_squares

from multimodal_emg.regression.lib_handler import get_lib
from multimodal_emg import gaussian_envelope_model, emg_envelope_model, gaussian_wave_model, emg_wave_model
from multimodal_emg.regression.derivatives import components_jac, emg_jac
from multimodal_emg.regression.losses import l2_norm
from multimodal_emg.models.basic_terms import PI

def multimodal_fit(
        data: Union[numpy.ndarray, torch.Tensor],
        features: Union[list, tuple, numpy.ndarray, torch.Tensor],
        components: int = None,
        max_iter: int = None,
        x: Union[numpy.ndarray, torch.Tensor] = None,
        fun: Union[Callable, str] = None,
        jac_fun: Union[Callable, str] = None,
        loss_fun: Callable = None,
    ):

    lib = get_lib(data)

    # prepare variables
    if isinstance(features, (list, tuple)): features = torch.Tensor(features) if lib.__str__().__contains__('torch') else numpy.array(features)
    p = lib.reshape(features, (-1,))
    x = lib.arange(len(data)) if x is None else x
    x = x.cpu() if isinstance(x, torch.Tensor) else x
    data = data.cpu() if isinstance(data, torch.Tensor) else data
    p = p.cpu() if isinstance(p, torch.Tensor) else p
    components = features.shape[0] if components is None and len(features.shape) == 2 else components
    fun = emg_envelope_model if fun is None else fun
    jac_fun = emg_jac if jac_fun is None else jac_fun
    loss_fun = l2_norm if loss_fun is None else loss_fun

    # component number assertion
    comp_num = len(p) / components
    if fun == gaussian_envelope_model: assert comp_num == 3, 'Gaussian regression requires 3 parameters per component'
    if fun == emg_envelope_model: assert comp_num == 4, 'EMG regression requires 4 parameters per component'
    if fun == gaussian_wave_model: assert comp_num == 5, 'Gaussian wave regression requires 5 parameters per component'
    if fun == emg_wave_model: assert comp_num in (2, 6), 'Wave-EMG regression requires 2 or 6 parameters per component'

    # pass args to functions
    model = lambda alpha, mu, sigma, eta=None, f_c=None, phi=None: fun(alpha, mu, sigma, eta, f_c, phi, x=x)
    components_model_with_args = lambda p: multimodal_model(p, model, components)
    cost_fun = lambda p: loss_fun(data, components_model_with_args(p))

    # pass args to jacobian function
    if isinstance(jac_fun, Callable):
        emg_jac_with_args = lambda p: jac_fun(*p, x=x)
        jac_fun_with_args = lambda p: components_jac(p, components_model_with_args, data, components, emg_jac_with_args)
    else:
        jac_fun_with_args = '2-point'

    # optimization
    p_star = least_squares(cost_fun, p, jac=jac_fun_with_args, max_nfev=max_iter).x

    # infer result
    result = components_model_with_args(p_star)

    return p_star, result

def multimodal_model(
        p: Union[list, tuple, numpy.ndarray, torch.Tensor],
        model: Callable,
        components: int = 1,
            ):

    n = len(p) // components
    d = model(*p[:n])

    for i in range(1, components):

        # phase in (-pi, pi] constraint by wrapping values into co-domain
        if n > 4 and not(-PI < p[i*n-1] < PI): 
            p[i*n-1] += 2*PI if p[i*n-1] < 0 else -2*PI

        d += model(*p[i*n:(i+1)*n])

    return d
