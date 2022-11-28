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
        device: Union[str, torch.device] = None,
    ):

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")if device is None else device

    # consider inconsistent number of components across batches
    if isinstance(features, (list, tuple)):

        dims = max(level_enumerate(features))
        assert dims > 1, 'features must have at least 2 dimensions'
        
        if dims == 2:
            batch_feats_num = [len(f) for f in features]
            p_init = torch.zeros((len(features), max(batch_feats_num)), dtype=torch.float64, device=device)
        elif dims == 3:
            batch_comps_num = [len(f) for f in features]        # first dimension is components
            batch_feats_num = [len(f[0]) for f in features]     # second dimension is features
            p_init = torch.zeros((len(features), max(batch_comps_num), max(batch_feats_num)), dtype=torch.float64, device=device)

        feat_mask = torch.ones(p_init.shape, dtype=bool, device=device)
        for i in range(len(features)):
            p_init[i, :len(features[i])] = torch.tensor(features[i], dtype=torch.float64)
            m = batch_feats_num[i] - p_init.shape[1]
            if m < 0:
                feat_mask[i, m:] = 0

    elif isinstance(features, (numpy.ndarray, torch.Tensor)):

        assert len(features.shape) > 1, 'features must have at least 2 dimensions'
        p_init = features if isinstance(features, torch.Tensor) else torch.tensor(features)

        feat_mask = torch.ones(p_init.shape, dtype=bool, device=device)
    
    else:
        raise Exception('Feature dimensionality not recognized')

    # prepare variables
    p_init = p_init.view(p_init.shape[0], -1)
    p_init = p_init.to(device) if isinstance(p_init, torch.Tensor) else p_init
    data = data.to(device, dtype=torch.float64) if isinstance(data, torch.Tensor) else torch.tensor(data, device=device, dtype=torch.float64)
    batch_size = p_init.shape[0]
    components = features.shape[1] if components is None and len(features.shape) == 3 else components
    x = torch.arange(len(data)) if x is None else x
    x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
    x = x.repeat(batch_size*components, 1) if len(x.shape) == 1 else x
    x = x.to(device) if isinstance(x, torch.Tensor) else x
    fun = emg_envelope_model if fun is None else fun
    jac_fun = emg_jac if jac_fun is None else jac_fun
    loss_fun = l2_norm if loss_fun is None else loss_fun
    wvec = torch.ones(1, dtype=p_init.dtype).to(device)

    # component number assertion
    params_num = p_init.shape[-1] / components
    if fun == gaussian_envelope_model: assert params_num == 3, 'Gaussian regression requires 3 parameters per component'
    if fun == emg_envelope_model: assert params_num == 4, 'EMG regression requires 4 parameters per component'
    if fun == gaussian_wave_model: assert params_num == 5, 'Gaussian wave regression requires 5 parameters per component'
    if fun == emg_wave_model: assert params_num in (2, 6), 'Wave-EMG regression requires 2 or 6 parameters per component'

    # constraint parameters
    sigma_threshold = p_init[:, 2::int(params_num)].nanmean() * 10
    mu_references = p_init[:, 1::int(params_num)]

    # pass args to functions
    model = lambda alpha, mu, sigma, eta=None, f_c=None, phi=None: fun(alpha, mu, sigma, eta, f_c, phi, exp_fun=torch.exp, erf_fun=torch.erf, cos_fun=torch.cos, x=x)
    components_model_with_args = lambda p: batch_multimodal_model(p, model, components, batch_size, sigma_threshold=sigma_threshold, mu_references=mu_references)
    cost_fun = lambda p: loss_fun(data.unsqueeze(1), components_model_with_args(p))

    # pass args to jacobian function
    if isinstance(jac_fun, Callable):
        emg_jac_with_args = lambda p: jac_fun(*p, x=x)
        jac_fun_with_args = lambda p: batch_components_jac(p, components_model_with_args, data, components, emg_jac_with_args)
    else:
        jac_fun_with_args = '2-point'

    # optimization
    p_list = lsq_lma_parallel(p_init, cost_fun, jac_function=jac_fun_with_args, wvec=wvec, max_iter=max_iter, ftol=0)

    # infer result
    result = components_model_with_args(p_list[-1])

    # infer components
    data = model(*p_list[-1].view(-1, p_list[-1].shape[-1] // components).T.unsqueeze(-1)).view(result.size(0), -1, result.size(-1))

    # infer confidences for each component
    confidences = 1 / torch.abs(data - result).sum(-1).nan_to_num(nan=torch.inf)

    return p_list[-1], result, confidences

def batch_multimodal_model(
        p: torch.Tensor,
        model: Callable,
        components: torch.Tensor = 1,
        batch_size: int = 1,
        sigma_threshold: float = 5,
        mu_references: torch.Tensor = None,
            ):

    feats_num = p.shape[-1] // components
    feats = p.view(-1, components, feats_num)    # view to batch x components x features 

    # positivity constraints for alpha, mu and sigma 
    feats[..., :3] = abs(feats[..., :3])

    # mu constraint
    if mu_references is not None:
        feats[..., 1][feats[..., 1] < mu_references-sigma_threshold/2] = mu_references[feats[..., 1] < mu_references-sigma_threshold/2] - sigma_threshold/2
        feats[..., 1][feats[..., 1] > mu_references+sigma_threshold/2] = mu_references[feats[..., 1] > mu_references+sigma_threshold/2] + sigma_threshold/2

    # phase in (-pi, pi] constraint by wrapping values into co-domain
    if feats_num > 4:
        feats[..., 5][feats[..., 5] < -PI] += 2*PI
        feats[..., 5][feats[..., 5] > +PI] -= 2*PI
    
    feats = feats.view(-1, feats_num).T.unsqueeze(-1)    # features x batch*components

    data = model(*feats)
    
    # split into batch and components while accumulating all components per batch
    data = torch.nansum(data.view(batch_size, components, -1), -2)    # nansum to exclude masked components 

    return data.unsqueeze(1)
