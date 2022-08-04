import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import warnings


pow_law_fun = lambda x, a=140.1771, b=1.1578: a*x**-b
sample2dist = lambda x, c=343, fkHz=175, sample_rate=1: c/2 * x / sample_rate / fkHz
dist2sample = lambda d, c=343, fkHz=175, sample_rate=1: 2/c * d * fkHz * sample_rate


class ChannelData(object):

    def __init__(self, *args, **kwargs):

        self._hilbert_data = kwargs['hilbert_data'] if 'hilbert_data' in kwargs else np.zeros(1024)
        self._channel_data = kwargs['channel_data'] if 'channel_data' in kwargs else np.zeros(len(self.hilbert_data))
        self._log_amp_data = kwargs['log_amp_data'] if 'log_amp_data' in kwargs else np.zeros(1024)

        # channel frequency rates as scalars (default values from a k-Wave simulation)
        self._tsample_rate = kwargs['tsample_rate'] if 'tsample_rate' in kwargs else 265830000
        self._sigfreq_rate = kwargs['sigfreq_rate'] if 'sigfreq_rate' in kwargs else 1e6
        self._nbursts_rate = kwargs['nbursts_rate'] if 'nbursts_rate' in kwargs else 7

        self._signal_width = self._nbursts_rate * self._tsample_rate/self._sigfreq_rate

    @property
    def channel_data(self) -> np.ndarray:
        return self._channel_data

    @property
    def hilbert_data(self) -> np.ndarray:
        return self._hilbert_data

    def logarithm(self):

        self._hilbert_data = np.log(self._hilbert_data)
        self._hilbert_data = (self._hilbert_data-min(self._hilbert_data))/(max(self._hilbert_data)-min(self._hilbert_data))

    def hilbert_transform(self, gauss_opt=False) -> np.ndarray:

        bp_filters = self.bandpass_filter() if self.oscillates else self._channel_data

        self._hilbert_data = np.abs(signal.hilbert(bp_filters))

        if gauss_opt: self._hilbert_data = gaussian_filter(self._hilbert_data, sigma=(13-1)/6)

        return self._hilbert_data
    
    @property
    def oscillates(self) -> bool:

        # analyze frequency
        _, _, z = signal.stft(self._channel_data, fs=1.0, window='hann', nperseg=len(self._channel_data)//4)

        # see if maximum of the signal's fft-magnitude is above certain threshold
        ret_val = abs(z).max() > 1e-6

        return ret_val

    def bandpass_filter(self):
        """
        automatic bandpass-filtering around prevalent frequency component
        """

        # detect relative frequency
        main_freq = self.detect_freq()

        if main_freq <= 0:
            warnings.warn('Skip bandpass filter due to invalid frequency')
            return self._channel_data

        # set cut-off frequencies (where amplitude drops by 3dB)
        sw = 0.5
        lo, hi = np.array([sw, (2-sw)]) * main_freq
        lo, hi = max(0, lo), min(1-np.spacing(1), hi)

        sos = signal.butter(5, [lo, hi], btype='band', output='sos') # , sr=self._sigfreq_rate)
        y = signal.sosfiltfilt(sos, self._channel_data)

        return y

    def detect_freq(self):

        w = np.fft.fft(self._channel_data)
        freqs = np.fft.fftfreq(len(w))
        idx = np.argmax(np.abs(w))
        freq = abs(freqs[idx]) * 2

        return freq
    
    def log_amplify(self) -> np.ndarray:
        
        self._log_amp_data = np.log(self._hilbert_data)

        return self._log_amp_data

    def path_conv(self, tsample_pos, velocity=346.130):

        return velocity * np.array(tsample_pos) / self._tsample_rate

    def compensate_pow_law(self, x=None, a=140.1771, b=1.1578, c=343, fkHz=175, sample_rate=1.):

        # compute sample positions in millimeter distances
        if x is None:
            x = sample2dist(np.arange(len(self.hilbert_data)) + np.spacing(1), c=c, fkHz=fkHz, sample_rate=sample_rate)

        self._channel_data /= pow_law_fun(x, a=a, b=b)
        self._hilbert_data /= pow_law_fun(x, a=a, b=b)
