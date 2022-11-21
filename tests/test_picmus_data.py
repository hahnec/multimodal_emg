import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import h5py

from multimodal_emg import batch_multimodal_fit
from multimodal_emg import emg_envelope_model, emg_wave_model, emg_jac, phi_jac, oemg_jac
from peak_detector import GradPeakDetector
from peak_detector.plot_hyst import plot_hyst
from peak_detector.plot_grad import plot_grad
from tests.test_batch_multimodal_emg import BatchMultiModalPlot


rmse = lambda reference, result: (((reference/reference.max() - result/reference.max())**2).sum() / result.size)**.5


class TestPicmusData(unittest.TestCase, BatchMultiModalPlot):

    def __init__(self, *args, **kwargs):
        super(TestPicmusData, self).__init__(*args, **kwargs)

    def setUp(self):
        
        self.plot_opt = False

        self.fname = './data/in_vivo/carotid_cross/carotid_cross_expe_dataset_rf.hdf5'
        f = h5py.File(self.fname, 'r')
        print(list(f.keys()))
        data_re = np.array(f['US']['US_DATASET0000']['data']['real'])
        data_im = np.array(f['US']['US_DATASET0000']['data']['imag'])

        self.fs = float(np.array(f['US']['US_DATASET0000']['sampling_frequency']))
        self.mod_freq = self.fs/5

        self.upsample_factor = 8 # over-sampling helps with frequency and phase estimates
        self.batch_size = 4
        self.max_iter = 50

        self.t = np.arange(0, len(data_re[0, 0, ...])/self.fs, 1/self.fs/self.upsample_factor)

        self.echo_list = []
        hilbert_list = []
        self.data_arr = np.zeros((data_re[0, ...].shape[0], len(data_re[0, 0, ...])*self.upsample_factor), dtype=data_re.dtype)
        for i in range(self.batch_size):
            
            self.data_arr[i, ...] = np.interp(x=self.t, xp=np.arange(0, len(data_re[0, 0, ...])/self.fs, 1/self.fs), fp=data_re[0, i, ...]) if self.upsample_factor > 1 else data_re[0, i, ...]
            det = GradPeakDetector(channel_data=self.data_arr[i, ...], grad_step=2*self.upsample_factor)
            det._grad_step = 2 if self.upsample_factor == 1 else 1.5 * self.upsample_factor
            det._thres_pos = 1*2e-3 if self.upsample_factor == 1 else 1*2e-3 / (self.upsample_factor*4)
            det._ival_list = [0, 500*self.upsample_factor]

            candidates = det.gradient_hysteresis()

            if self.plot_opt:
                plot_grad(det)
                plot_hyst(det)
                plt.show()
        
            self.echo_list.append(candidates)
            hilbert_list.append(det.hilbert_data)

        self.batch_hilbert_data = torch.tensor(np.array(hilbert_list))
        self.batch_data_arr = torch.tensor(self.data_arr[:self.batch_size, ...])
        self.echo_num = max([len(candidates) for candidates in self.echo_list])

    def test_picmus_data(self):

        self.batch_echo_feats = []
        for echoes in self.echo_list:
            feats = [[echo[3], echo[1], (echo[1]-echo[0])/3, 0] for echo in echoes]
            #feats = [[echo[3], echo[1]/fs, (echo[1]-echo[0])/(3*fs), 0] for echo in echoes]
            self.batch_echo_feats.append(feats)

        # multimodal optimization
        p_star, result = batch_multimodal_fit(
            self.batch_hilbert_data,
            features=self.batch_echo_feats,
            components=self.echo_num,
            x=np.arange(0, len(self.data_arr[0, ...])),#self.t,
            max_iter=self.max_iter,
            fun=emg_envelope_model,
            jac_fun=emg_jac,
        )

        result = result.squeeze(1).cpu().numpy()

        self.batch_multimodal_plot(fitted_data=result, title=Path(self.fname).name) if self.plot_opt else None
        squared_error = rmse(self.batch_hilbert_data.cpu().numpy(), result)
        print('Multi-Modal EMG fit squared error amounts to  %s' % round(squared_error, 4))

        # error assertion
        self.assertTrue(p_star.shape[0] == len(self.batch_echo_feats), msg='Number of input and output parameters varies')
        self.assertTrue(squared_error < 1, msg='Channel data and fit curve deviate')

        # add oscillating estimates frequency and phase for optimization
        fkHz = self.mod_freq/1e3 if False else 4642.9   # the latter is obtained by measurement
        for i in range(len(self.batch_echo_feats)):
            feats = [p_star[i][j*4:(j+1)*4].cpu().numpy().tolist() + [fkHz, 0] for j in range(self.echo_num)]
            self.batch_echo_feats[i] = [[feat[0], feat[1]/(self.fs*self.upsample_factor), feat[2]/(self.fs*self.upsample_factor), feat[3], feat[4], feat[5]] for feat in feats]
            #self.batch_echo_feats[i] = [[feat[0], feat[1], feat[2], feat[3], (1/5)/1000, feat[5]] for feat in feats]    #

        # multimodal optimization
        p_star, result = batch_multimodal_fit(
            self.batch_data_arr,
            features=self.batch_echo_feats,
            components=self.echo_num,
            x=self.t,
            max_iter=self.max_iter,
            fun=emg_wave_model,
            jac_fun=phi_jac,
        )

        result = result.squeeze(1).cpu().numpy()

        # results
        self.batch_multimodal_plot(fitted_data=result, title='Multi-Modal EMG with oscillation plot') if self.plot_opt else None
        squared_error = rmse(self.batch_data_arr.cpu().numpy(), result)
        print('Multi-Modal EMG oscillation fit squared error amounts to  %s' % round(squared_error, 4))

        # error assertion
        self.assertTrue(p_star.shape[0] == len(self.batch_echo_feats), msg='Number of input and output parameters varies')
        self.assertTrue(squared_error < 1, msg='Channel data and fit curve deviate')

        # add oscillating estimates frequency and phase for optimization
        for i in range(len(self.batch_echo_feats)):
            feats = [p_star[i][j*6:(j+1)*6].cpu().numpy().tolist() for j in range(self.echo_num)]
            self.batch_echo_feats[i] = [[feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]] for feat in feats]    #

        # multimodal optimization
        p_star, result = batch_multimodal_fit(
            self.batch_data_arr,
            features=self.batch_echo_feats,
            components=self.echo_num,
            x=self.t,
            max_iter=self.max_iter,
            fun=emg_wave_model,
            jac_fun=oemg_jac,
        )

        result = result.squeeze(1).cpu().numpy()

        # results
        self.batch_multimodal_plot(fitted_data=result, title='Multi-Modal EMG with oscillation plot') if self.plot_opt else None
        squared_error = rmse(self.batch_data_arr.cpu().numpy(), result)
        print('Multi-Modal EMG oscillation fit squared error amounts to  %s' % round(squared_error, 4))

        # error assertion
        self.assertTrue(p_star.shape[0] == len(self.batch_echo_feats), msg='Number of input and output parameters varies')
        self.assertTrue(squared_error < 1, msg='Channel data and fit curve deviate')        

    def main(self):

        self.test_picmus_data()

if __name__ == '__main__':
    unittest.main()
