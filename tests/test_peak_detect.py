import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

from multimodal_emg import multimodal_fit
from multimodal_emg import emg_envelope_model, emg_wave_model, emg_jac, wav_jac
from multimodal_emg.util.torch_hilbert import hilbert_transform
from multimodal_emg.util.peak_detect import grad_peak_detect, gaussian_filter_1d


class PeakDetectTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PeakDetectTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)
        self.plot_opt = False

        # synthesize echoes
        self.fs = 3e5
        self.time_interval = 5e-3
        self.t = np.arange(0, self.time_interval, 1/self.fs)
        self.gt_data = np.zeros(len(self.t))
        self.gt_params = []
        self.echo_num = 4
        self.channel_num = 8
        
        alphas = np.random.standard_normal(size=self.echo_num) + 5.5
        mus = np.random.uniform(low=.001, high=self.time_interval*.9, size=self.echo_num)
        sigmas = np.random.uniform(low=5e-5, high=1e-4, size=self.echo_num)
        etas = np.random.uniform(low=2, high=6, size=self.echo_num)
        fkhzs = 4e1 + (np.random.uniform(size=self.echo_num)-.5)*1e-2
        phis = np.random.standard_normal(size=self.echo_num) * .1

        # generate echoes
        for i in range(self.echo_num):
            self.gt_data += emg_wave_model(alpha=alphas[i], mu=mus[i], sigma=sigmas[i], eta=etas[i], fkhz=fkhzs[i], phi=phis[i], x=self.t)
            self.gt_params.append([alphas[i], mus[i], sigmas[i], etas[i], fkhzs[i], phis[i]])

        # duplicate data and add noise to disturb echo detection
        self.data_arr = []
        for i in range(self.channel_num):
            data_channel = self.add_noise(self.gt_data, 2e-1)
            self.data_arr.append(data_channel)
        self.data_arr = np.stack(self.data_arr)

        self.gt_echo_list = [
            [317, 354, 0.10693694309391295, 11.532769257925278, 0.2653333333333333, 74], 
            [432, 467, 0.08253485310187075, 9.501325744020871, 0.2653333333333333, 70], 
            [724, 763, 0.09610781524018126, 6.833534653582119, 0.2653333333333333, 78], 
            [1173, 1205, 0.07668806616800042, 10.132450921878117, 0.2653333333333333, 64]
        ] 

        # add estimates (alpha, mu, sigma, skew) for mixture model optimization
        self.echo_feats = list(chain(*[[self.gt_echo_list[i][3], self.gt_echo_list[i][1]/self.fs, (self.t[self.gt_echo_list[i][1]]-self.t[self.gt_echo_list[i][0]])/3, 3] for i in range(len(self.gt_echo_list))]))

    @staticmethod
    def add_noise(data: np.ndarray, scale: float) -> np.ndarray:
        return data.copy() + scale * np.random.standard_normal(len(data))
 
    def test_grad_peak_detect(self):

        # prepare variables
        self.data_arr = torch.tensor(self.data_arr)
        grad_step = 50
        threshold = 0.00025
        batch_hilbert_data = abs(hilbert_transform(self.data_arr))

        # find peaks
        batch_echoes = grad_peak_detect(batch_hilbert_data, grad_step=grad_step, threshold=threshold)

        if self.plot_opt:
            batch_grad_data = torch.gradient(batch_hilbert_data, spacing=grad_step, dim=-1)[0]
            batch_grad_data = gaussian_filter_1d(batch_grad_data, sigma=(grad_step*2-1)/6)
            self.plot_grad(batch_grad_data[0, ...], thres_pos=threshold, thres_neg=-threshold/4, show_opt=True)

        assert len(batch_echoes) == len(self.data_arr), 'Batch dimension mismatch'

        for echo_list in batch_echoes:
            np.testing.assert_array_almost_equal(echo_list[:, 0].numpy(), np.array(self.gt_echo_list)[:, 0], decimal=-2)
            np.testing.assert_array_almost_equal(echo_list[:, 1].numpy(), np.array(self.gt_echo_list)[:, 1], decimal=-2)
    
    def plot_grad(self, hilbert_data, thres_pos, thres_neg, show_opt=True):

        grad_plus = hilbert_data > thres_pos
        grad_minu = hilbert_data < thres_neg

        t = np.arange(len(hilbert_data))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hilbert_data, label='gradient')
        ax.plot(thres_pos*np.ones(len(hilbert_data)), label='pos. threshold')
        ax.plot(hilbert_data.max()*grad_plus, label='positive breakthroughs')
        ax.plot(hilbert_data.max()*grad_minu, label='negative breakthroughs')
        ax.plot(thres_neg*np.ones(len(hilbert_data)), label='neg. threshold')
        ax.legend()
        plt.show() if show_opt else None

    def main(self):

        self.test_synth_multimodal_model()

if __name__ == '__main__':
    unittest.main()
