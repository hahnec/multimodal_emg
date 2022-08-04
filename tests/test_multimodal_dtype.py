import unittest
import numpy
import torch
import matplotlib.pyplot as plt

from multimodal_emg.regression.derivatives import gauss_term, asymm_term
from multimodal_emg import multimodal_fit
from multimodal_emg.regression.losses import huber_loss


class MultimodalTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MultimodalTester, self).__init__(*args, **kwargs)

    def setUp(self):

        self.plot_opt = False

        numpy.random.seed(3008)

        # synthesize echoes
        self.step = .3e-5
        self.interval = 5e-3
        self.t = numpy.arange(0, self.interval, self.step)
        self.gt_data = numpy.zeros(len(self.t))
        self.gt_params = []

        self.components = 4
        alphas = numpy.random.standard_normal(size=self.components) + 5.5
        mus = numpy.random.uniform(low=.001, high=self.interval*.9, size=self.components)
        sigmas = numpy.random.uniform(low=5e-5, high=1e-4, size=self.components)
        etas = numpy.random.uniform(low=2, high=6, size=self.components)

        # generate echoes
        for i in range(self.components):
            emg = alphas[i] * gauss_term(self.t, mus[i], sigmas[i]) * asymm_term(self.t, mus[i], sigmas[i], etas[i])
            self.gt_data += emg
            self.gt_params.append([alphas[i], mus[i], sigmas[i], etas[i]])

        # add noise to disturb echo detection
        self.data_arr = self.add_noise(self.gt_data, 2e-1)

        # add estimates (alpha, mu, sigma, skew) for mixture model optimization
        self.features = (numpy.asarray(self.gt_params) + 1e-5*numpy.random.randn(self.components, 4)).flatten()

    @staticmethod
    def add_noise(data: numpy.ndarray, scale: float) -> numpy.ndarray:
        return data.copy() + scale * numpy.random.standard_normal(len(data))
 
    def test_synth_multimodal_numpy(self):

        for jac_fun in [None, '2-point']:
            # multimodal optimization
            p_star, result = multimodal_fit(
                self.data_arr,
                features=self.features,
                components=self.components,
                x=self.t,
                max_iter=180,
                jac_fun=jac_fun,
                loss_fun=huber_loss,
            )

            # results
            self.multimodal_plot(fitted_data=result, title='MM-EMG plot') if self.plot_opt else None
            squared_error = numpy.sum((self.data_arr - result)**2) / len(result)
            print('Multi-modal EMG squared error amounts to  %s' % round(squared_error, 4))

            # error assertion of mixture model
            self.assertTrue(len(p_star) == len(self.features), msg='Number of input and output parameters varies')
            self.assertTrue(squared_error < 0.039, msg='Channel data and MM-EMG curve deviate')

    def test_synth_multimodal_torch(self):

        for device in ['cpu', 'cuda']:

            data_arr = torch.Tensor(self.data_arr).to(device)
            features = torch.Tensor(self.features).to(device)
            t = torch.Tensor(self.t).to(device)

            # multimodal optimization
            p_star, result = multimodal_fit(
                data_arr,
                features=features,
                components=self.components,
                x=t,
                max_iter=180,
                loss_fun=huber_loss,
            )

            result = result.cpu().numpy()

            # results
            self.multimodal_plot(fitted_data=result, title='MM-EMG plot') if self.plot_opt else None
            squared_error = numpy.sum((self.data_arr - result)**2) / len(result)
            print('Multi-modal EMG squared error amounts to  %s' % round(squared_error, 4))

            # error assertion of mixture model
            self.assertTrue(len(p_star) == len(features), msg='Number of input and output parameters varies')
            self.assertTrue(squared_error < 0.039, msg='Channel data and MM-EMG curve deviate')

    def test_synth_multimodal_feature_types(self):
        
        l = list(self.features.copy())
        t = tuple(self.features.copy())
        n = numpy.array(self.features.copy().reshape(self.components, 4))
        t = torch.Tensor(self.features.copy().reshape(self.components, 4))

        for feats, m in [(l, self.components), (t, self.components), (n, self.components), (t, self.components), (n, None), (t, None), ]:

            # multimodal optimization
            p_star, result = multimodal_fit(
                self.data_arr,
                features=feats,
                components=m,
                x=self.t,
            )

            # results
            self.multimodal_plot(fitted_data=result, title='MM-EMG plot') if self.plot_opt else None
            squared_error = numpy.sum((self.data_arr - result)**2) / len(result)
            print('Multi-modal EMG squared error amounts to  %s' % round(squared_error, 4))

            # error assertion of mixture model
            self.assertTrue(len(p_star) == len(self.features), msg='Number of input and output parameters varies')
            self.assertTrue(squared_error < 0.039, msg='Channel data and MM-EMG curve deviate')


    def multimodal_plot(self, fitted_data=None, title=None):

        plt.figure()
        plt.plot(self.data_arr, label='raw data')
        plt.plot(self.gt_data, label='ground-truth data')
        plt.plot(fitted_data, linestyle=':', label='fitted')
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def main(self):

        self.test_synth_multimodal_numpy()
        self.test_synth_multimodal_torch()
        self.test_synth_multimodal_feature_types()

if __name__ == '__main__':
    unittest.main()
