import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict
import time

from multimodal_emg import batch_multimodal_fit
from multimodal_emg import emg_envelope_model, emg_wave_model, emg_jac, wav_jac, phi_jac, oemg_jac
from multimodal_emg.util.torch_hilbert import hilbert_transform
from multimodal_emg.util.peak_detect import grad_peak_detect


def batch_staged_memgo(
        data_arr: torch.Tensor,
        x: torch.Tensor,
        cfg: Dict = {},
        max_iter_per_stage: int = None,
        plot_opt: bool = False,
        print_opt: bool = False,
        ):

    start = time.time()

    max_iter_per_stage = 8 if max_iter_per_stage is None else max_iter_per_stage
    batch_size = data_arr.shape[-1]
    batch_data_arr = data_arr.clone().T

    # echo detection, tbd: test adapative threshold: remove threshold argument
    batch_hilbert_data = abs(hilbert_transform(batch_data_arr))
    batch_echoes = grad_peak_detect(batch_hilbert_data, grad_step=cfg.enlarge_factor/6*5, ival_smin=0, ival_smax=500*cfg.enlarge_factor)
    echo_num = batch_echoes.shape[1]
    
    # add amplitude and width approximations
    amplitudes = batch_echoes[..., -1]
    batch_echo_feats = torch.stack([amplitudes, batch_echoes[..., 1], (batch_echoes[..., 1]-batch_echoes[..., 0])/2.5, torch.zeros(amplitudes.shape, device=data_arr.device)]).swapaxes(0, -1).swapaxes(0, 1)

    if print_opt: print('MEMGO preparation: %s' % str(round(time.time()-start, 4)))

    # multimodal optimization
    p_star, result, _ = batch_multimodal_fit(
        batch_hilbert_data,
        features=batch_echo_feats,
        components=echo_num,
        x=torch.arange(0, len(x)),
        max_iter=max_iter_per_stage,
        fun=emg_envelope_model,
        jac_fun=emg_jac,
        device=data_arr.device,
    )

    if print_opt: print('MEMGO step 1: %s' % str(round(time.time()-start, 4)))

    # add oscillating estimates frequency and phase for optimization
    batch_echo_feats = torch.dstack([batch_echo_feats, torch.zeros(batch_echo_feats.shape[:2], device=x.device), torch.zeros(batch_echo_feats.shape[:2], device=x.device)])
    fkHz = cfg.fs/1e3
    for i in range(len(batch_echo_feats)):
        feats = [p_star[i][j*4:(j+1)*4].cpu().numpy().tolist() + [fkHz, 0] for j in range(echo_num)]
        #batch_echo_feats[i] = [[feat[0], feat[1]/(cfg.fs*cfg.enlarge_factor), feat[2]/(cfg.fs*cfg.enlarge_factor), feat[3], feat[4], feat[5]] for feat in feats]
        batch_echo_feats[i] = torch.tensor([[feat[0], feat[1]/(cfg.fs*cfg.enlarge_factor), feat[2]/(cfg.fs*cfg.enlarge_factor), feat[3], feat[4], feat[5]] for feat in feats])

    p = init_phase(batch_echo_feats, batch_data_arr[:, ::1], x[::1])

    if print_opt: print('MEMGO phase approx.: %s' % str(round(time.time()-start, 4)))

    # multimodal optimization
    p_star, result, _ = batch_multimodal_fit(
        batch_data_arr,
        features=p,
        components=echo_num,
        x=x,
        max_iter=max_iter_per_stage,
        fun=emg_wave_model,
        jac_fun=wav_jac,
        device=data_arr.device,
    )

    if print_opt: print('MEMGO step 2: %s' % str(round(time.time()-start, 4)))

    # add oscillating estimates frequency and phase for optimization
    for i in range(len(batch_echo_feats)):
        feats = [p_star[i][j*6:(j+1)*6].cpu().numpy().tolist() for j in range(echo_num)]
        batch_echo_feats[i] = torch.tensor([[feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]] for feat in feats])

    if print_opt: print('MEMGO add oscil params: %s' % str(round(time.time()-start, 4)))

    # multimodal optimization
    p_star, result, conf = batch_multimodal_fit(
        batch_data_arr,
        features=batch_echo_feats,
        components=echo_num,
        x=x,
        max_iter=max_iter_per_stage,
        fun=emg_wave_model,
        jac_fun=oemg_jac,
        device=data_arr.device,
    )

    if print_opt: print('MEMGO step 3: %s' % str(round(time.time()-start, 4)))

    # clear storage
    del batch_data_arr, batch_echo_feats
    torch.cuda.empty_cache()

    # align dimensions
    result = result.squeeze(1)
    p_star = p_star.view(p_star.shape[0], p_star.shape[1]//6, 6)

    # set amplitude, mean and confidence of outlying components to zero
    p_star[..., 0][(p_star[..., 1] < x[0]) & (x[-1] < p_star[..., 1])] = 0
    conf[(p_star[..., 1] < x[0]) & (x[-1] < p_star[..., 1])] = 0

    if print_opt: print('MEMGO completion: %s' % str(round(time.time()-start, 4)))

    return p_star, result, conf, batch_echoes


def init_phase(batch_echo_feats, data, x, steps=8*4):

    p = batch_echo_feats.to(data.device)
    phi_candidates = torch.linspace(-torch.pi, torch.pi, steps, device=data.device)
    
    stack = []
    for val in phi_candidates:
        p[..., 5] = val
        fit = emg_wave_model(*p.view(-1, 6).T.unsqueeze(-1), exp_fun=torch.exp, erf_fun=torch.erf, cos_fun=torch.cos, x=x)
        losses = torch.sum(abs(data.unsqueeze(1) - fit.view(*p.shape[:2], fit.shape[-1])), axis=-1)
        stack.append(losses)

    idcs = torch.argmin(torch.stack(stack), 0)
    p[..., 5] = phi_candidates[idcs]

    return p
