import numpy
import torch
from typing import Union
from scipy.signal import hilbert, butter, sosfiltfilt
import warnings

from multimodal_emg import multimodal_fit
from multimodal_emg import emg_envelope_model, emg_wave_model
from multimodal_emg.regression.derivatives import emg_jac, wav_jac, oemg_jac
from multimodal_emg.regression.lib_handler import get_lib


def three_stage_fit(        
        data: Union[numpy.ndarray, torch.Tensor],
        features: Union[numpy.ndarray, torch.Tensor],
        max_iter: int = None,
        x: Union[numpy.ndarray, torch.Tensor] = None,
    ):

    lib = get_lib(data)

    assert isinstance(features, (numpy.ndarray, torch.tensor)), 'features argument needs to be of type numpy.ndarray or torch.tensor'
    features = features.reshape(-1, 6) if features.shape[-1] != 6 else features

    # 1st stage: EMG model optimization
    p_emg, _ = multimodal_fit(
        lib.abs(hilbert(bandpass_filter(data))),
        features=features[:, :4],
        components=features.shape[0],
        x=x,
        max_iter=max_iter,
        fun=emg_envelope_model,
        jac_fun=emg_jac,
    )

    # add intermediate results to features
    features[:, :4] = p_emg.reshape(-1, 4)

    # 2nd stage: wave model optimization
    p_wav, _ = multimodal_fit(
        data,
        features=features,
        components=features.shape[0],
        x=x,
        max_iter=max_iter,
        fun=emg_wave_model,
        jac_fun=wav_jac,
    )

    # add intermediate results to features
    features[:, 4:] = p_wav.reshape(-1, 6)[:, 4:]

    # 3rd stage: oscillating EMG
    p_star, wav_result = multimodal_fit(
        data,
        features=features,
        components=features.shape[0],
        x=x,
        max_iter=max_iter,
        fun=emg_wave_model,
        jac_fun=oemg_jac,
    )

    return p_star, wav_result


def bandpass_filter(channel_data):
    """
    automatic bandpass-filtering around prevalent frequency component
    """

    # detect relative frequency
    main_freq = detect_freq(channel_data)

    if main_freq <= 0:
        warnings.warn('Skip bandpass filter due to invalid frequency')
        return channel_data

    # set cut-off frequencies (where amplitude drops by 3dB)
    sw = 0.5
    lo, hi = numpy.array([sw, (2-sw)]) * main_freq
    lo, hi = max(0, lo), min(1-numpy.spacing(1), hi)

    sos = butter(5, [lo, hi], btype='band', output='sos')
    y = sosfiltfilt(sos, channel_data)

    return y

def detect_freq(channel_data):

    w = numpy.fft.fft(channel_data)
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = abs(freqs[idx]) * 2

    return freq
