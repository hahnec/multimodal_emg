import numpy
import torch
from typing import Callable, Union
from torchimize.functions import lsq_lma_parallel

from multimodal_emg import gaussian_envelope_model, emg_envelope_model, gaussian_wave_model, emg_wave_model
from multimodal_emg.regression.derivatives import batch_components_jac, emg_jac
from multimodal_emg.regression.losses import l2_norm
from multimodal_emg.models.basic_terms import PI

from multimodal_emg.util.nested_list_levels import level_enumerate

def batch_multimodal_fit(
        data: Union[numpy.ndarray, torch.Tensor],
        features: Union[list, tuple, numpy.ndarray, torch.Tensor],
        components: int = None,
        max_iter: int = None,
        x: Union[numpy.ndarray, torch.Tensor] = None,
        fun: Union[Callable, str] = None,
        jac_fun: Union[Callable, str] = None,
        loss_fun: Callable = None,
    ):

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # consider inconsistent number of components across batches
    if isinstance(features, (list, tuple)):

        dims = max(level_enumerate(features))
        assert dims > 1, 'features must have at least 2 dimensions'
        
        if dims == 2:
            batch_feats_num = [len(f) for f in features]
            p = torch.zeros((len(features), max(batch_feats_num)), dtype=torch.float64, device=device)
        elif dims == 3:
            batch_comps_num = [len(f) for f in features]        # first dimension is components
            batch_feats_num = [len(f[0]) for f in features]     # second dimension is features
            p = torch.zeros((len(features), max(batch_comps_num), max(batch_feats_num)), dtype=torch.float64, device=device)

        feat_mask = torch.ones(p.shape, dtype=bool, device=device)
        for i in range(len(features)):
            p[i, :len(features[i])] = torch.tensor(features[i], dtype=torch.float64)
            m = batch_feats_num[i] - p.shape[1]
            if m < 0:
                feat_mask[i, m:] = 0

    elif isinstance(features, (numpy.ndarray, torch.Tensor)):

        assert len(features.shape) > 1, 'features must have at least 2 dimensions'
        p = features if isinstance(features, torch.Tensor) else torch.tensor(features)

        feat_mask = torch.ones(p, dtype=bool, device=device)
    
    else:
        raise Exception('Feature dimensionality not recognized')

    # prepare variables
    p = p.view(p.shape[0], -1)
    p = p.to(device) if isinstance(p, torch.Tensor) else p
    data = data.to(device, dtype=torch.float64) if isinstance(data, torch.Tensor) else torch.tensor(data, device=device, dtype=torch.float64)
    batch_size = p.shape[0]
    components = features.shape[1] if components is None and len(features.shape) == 3 else components
    x = torch.arange(len(data)) if x is None else x
    x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
    x = x.repeat(batch_size*components, 1) if len(x.shape) == 1 else x
    x = x.to(device) if isinstance(x, torch.Tensor) else x
    fun = emg_envelope_model if fun is None else fun
    jac_fun = emg_jac if jac_fun is None else jac_fun
    loss_fun = l2_norm if loss_fun is None else loss_fun
    wvec = torch.ones(1, dtype=p.dtype).to(device)

    # component number assertion
    params_num = p.shape[-1] / components
    if fun == gaussian_envelope_model: assert params_num == 3, 'Gaussian regression requires 3 parameters per component'
    if fun == emg_envelope_model: assert params_num == 4, 'EMG regression requires 4 parameters per component'
    if fun == gaussian_wave_model: assert params_num == 5, 'Gaussian wave regression requires 5 parameters per component'
    if fun == emg_wave_model: assert params_num in (2, 6), 'Wave-EMG regression requires 2 or 6 parameters per component'

    # pass args to functions
    model = lambda alpha, mu, sigma, eta=None, f_c=None, phi=None: fun(alpha, mu, sigma, eta, f_c, phi, exp_fun=torch.exp, erf_fun=torch.erf, cos_fun=torch.cos, x=x)
    components_model_with_args = lambda p: batch_multimodal_model(p, model, components, batch_size)
    cost_fun = lambda p: loss_fun(data.unsqueeze(1), components_model_with_args(p))

    # pass args to jacobian function
    if isinstance(jac_fun, Callable):
        emg_jac_with_args = lambda p: jac_fun(*p, x=x)
        jac_fun_with_args = lambda p: batch_components_jac(p, components_model_with_args, data, components, emg_jac_with_args)
    else:
        jac_fun_with_args = '2-point'

    # optimization
    p_list = lsq_lma_parallel(p, cost_fun, jac_function=jac_fun_with_args, wvec=wvec, max_iter=max_iter, ftol=0)

    # infer result
    result = components_model_with_args(p_list[-1])

    return p_list[-1], result

def batch_multimodal_model(
        p: torch.Tensor,
        model: Callable,
        components: torch.Tensor = 1,
        batch_size: int = 1,
            ):

    feats_num = p.shape[-1] // components
    feats = p.view(-1, components, feats_num)    # view to batch x components x features 
    if feats_num > 4:
        # phase in (-pi, pi] constraint by wrapping values into co-domain
        feats[..., 5][feats[..., 5] < -PI] += 2*PI
        feats[..., 5][feats[..., 5] > +PI] -= 2*PI

    # alpha positive constraint
    feats[..., 0][feats[..., 0] < 0] = abs(feats[..., 0][feats[..., 0] < 0])

    # sigma positive constraint
    feats[..., 2][feats[..., 2] <= 0] = 1e-2
    
    feats = feats.view(-1, feats_num).T.unsqueeze(-1)    # features x batch*components

    d = model(*feats)
    
    # split into batch and components while accumulating all components per batch
    d = torch.nansum(d.view(batch_size, components, -1), -2)    # nansum to exclude masked components 

    return d.unsqueeze(1)
