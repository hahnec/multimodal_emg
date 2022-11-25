import unittest
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from itertools import chain
import torch

from multimodal_emg import batch_multimodal_fit
from multimodal_emg import emg_envelope_model, emg_wave_model, emg_jac, wav_jac


class BatchMultiModalPlot:

    def batch_multimodal_plot(self, fitted_data=None, title=None):

        fig, axs = plt.subplots(nrows=self.batch_size, ncols=1)
        axs[0].set_title(title)
        for i in range(self.batch_size):
            axs[i].plot(self.t, self.batch_data_arr[i, ...], label='raw data')
            axs[i].plot(self.t, self.batch_hilbert_data[i, ...], label='envelope')
            axs[i].plot(self.t, fitted_data[i, ...], linestyle=':', label='fitted')
        plt.legend()
        plt.tight_layout()
        plt.show()


class BatchEchoMultiModalTester(unittest.TestCase, BatchMultiModalPlot):

    def __init__(self, *args, **kwargs):
        super(BatchEchoMultiModalTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)
        self.plot_opt = True

        # synthesize echoes
        self.fs = 3e5
        self.time_interval = 5e-3
        self.t = np.arange(0, self.time_interval, 1/self.fs)
        self.gt_data = np.zeros(len(self.t))
        self.gt_data_alt = np.zeros(len(self.t))
        self.gt_params = []
        self.gt_params_alt = []
        self.echo_num = 4
        
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

        # add noise to disturb echo detection
        self.data_arr = self.add_noise(self.gt_data, 2e-1)

        self.hilbert_data = self.hilbert_transform(self.data_arr)

        echo_list = [
            [317, 354, 0.10693694309391295, 11.532769257925278, 0.2653333333333333, 74], 
            [432, 467, 0.08253485310187075, 9.501325744020871, 0.2653333333333333, 70], 
            [724, 763, 0.09610781524018126, 6.833534653582119, 0.2653333333333333, 78], 
            [1173, 1205, 0.07668806616800042, 10.132450921878117, 0.2653333333333333, 64]
        ]

        # add estimates (alpha, mu, sigma, skew) for mixture model optimization
        self.echo_feats = list(chain(*[[echo_list[i][3], echo_list[i][1]/self.fs, (self.t[echo_list[i][1]]-self.t[echo_list[i][0]])/3, 3] for i in range(len(echo_list))]))

        for i in range(self.echo_num-1):
            self.gt_data_alt += emg_wave_model(alpha=alphas[i], mu=mus[i], sigma=sigmas[i], eta=etas[i], fkhz=fkhzs[i], phi=phis[i], x=self.t)
            self.gt_params_alt.append([alphas[i], mus[i], sigmas[i], etas[i], fkhzs[i], phis[i]])
        self.data_arr_alt = self.add_noise(self.gt_data_alt, 2e-1)
        self.hilbert_data_alt = self.hilbert_transform(self.data_arr_alt)

        self.batch_size = 3

        self.batch_data_arr = torch.tensor(np.array([self.data_arr.copy() for _ in range(self.batch_size-1)] + [self.data_arr_alt]))
        self.batch_hilbert_data = torch.tensor(np.array([self.hilbert_data.copy() for _ in range(self.batch_size-1)] + [self.hilbert_data_alt]))
        self.batch_echo_feats = [self.echo_feats.copy() for _ in range(self.batch_size-1)] + [self.echo_feats[:-4]]

    @staticmethod
    def hilbert_transform(data) -> np.ndarray:

        hilbert_data = np.abs(signal.hilbert(data))

        return hilbert_data

    @staticmethod
    def add_noise(data: np.ndarray, scale: float) -> np.ndarray:
        return data.copy() + scale * np.random.standard_normal(len(data))
 
    def test_synth_batch_multimodal_model(self):

        # multimodal optimization
        p_star, result = batch_multimodal_fit(
            self.batch_hilbert_data,
            features=self.batch_echo_feats,
            components=self.echo_num,
            x=self.t,
            max_iter=30,
            fun=emg_envelope_model,
            jac_fun=emg_jac,
        )[:2]

        result = result.squeeze(1).cpu().numpy()

        # results
        self.batch_multimodal_plot(fitted_data=result, title='Multi-Modal EMG plot') if self.plot_opt else None
        squared_error = np.sum((self.batch_hilbert_data.cpu().numpy() - result)**2) / result.size
        print('Multi-Modal EMG fit squared error amounts to  %s' % round(squared_error, 4))

        # error assertion
        self.assertTrue(p_star.shape[0] == len(self.batch_echo_feats), msg='Number of input and output parameters varies')
        self.assertTrue(squared_error < 1, msg='Channel data and fit curve deviate')

        # add oscillating estimates frequency and phase for optimization
        for i in range(len(self.batch_echo_feats)):
            self.batch_echo_feats[i] = [p_star[i][j*4:(j+1)*4].cpu().numpy().tolist() + [4e1, 0] for j in range(4)]

        # multimodal optimization
        p_star, result = batch_multimodal_fit(
            self.batch_data_arr,
            features=self.batch_echo_feats,
            components=self.echo_num,
            x=self.t,
            max_iter=30,
            fun=emg_wave_model,
            jac_fun=wav_jac,
        )[:2]

        result = result.squeeze(1).cpu().numpy()

        # results
        self.batch_multimodal_plot(fitted_data=result, title='Multi-Modal EMG with oscillation plot') if self.plot_opt else None
        squared_error = np.sum((self.batch_data_arr.cpu().numpy() - result)**2) / result.size
        print('Multi-Modal EMG oscillation fit squared error amounts to  %s' % round(squared_error, 4))

        # error assertion
        self.assertTrue(p_star.shape[0] == len(self.batch_echo_feats), msg='Number of input and output parameters varies')
        self.assertTrue(squared_error < 1, msg='Channel data and fit curve deviate')

    def main(self):

        self.test_synth_batch_multimodal_model()


if __name__ == '__main__':
    unittest.main()
