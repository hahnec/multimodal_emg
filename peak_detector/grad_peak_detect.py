__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2021 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy import ndimage

from peak_detector.channel_data import ChannelData

class GradPeakDetector(ChannelData):

    def __init__(self, *args, **kwargs):
        super(GradPeakDetector, self).__init__(*args, **kwargs)

        # perform Hilbert transform for oscillating input
        self.hilbert_transform() if self.oscillates and self.hilbert_data.sum() == 0 else None

        # default parameters
        self._grad_step = 50
        self._thres_pos = (self.hilbert_grad.max()+self.hilbert_grad.mean())/10
        self._thres_neg = -self._thres_pos/4

        # parameters from input arguments (defaults are heuristics)
        self._echo_list = kwargs['echo_list'] if 'echo_list' in kwargs else []
        self._grad_step = kwargs['grad_step'] if 'grad_step' in kwargs else self._grad_step
        self._thres_pos = kwargs['threshold'] if 'threshold' in kwargs else self._thres_pos
        self._thres_neg = -self._thres_pos/4
        # interval list defines the minimum and maximum width between start and peak of echo
        self._ival_list = [self._grad_step//2, self._grad_step*3]
        self._ival_list[0] = kwargs['ival_smin'] if 'ival_smin' in kwargs else self._ival_list[0]
        self._ival_list[1] = kwargs['ival_smax'] if 'ival_smax' in kwargs else self._ival_list[1]

    @property
    def echo_list(self):
        """
        echo list with length representing number of echos and element attributes as follows:
            ['PosStart', 'PosPeak', 'AmpStart', 'AmpPeak', 'FreqDetect' 'EchoWidth']
        """

        f = self.detect_freq()

        # per echo feature containing: start sample position, peak sample position, max-peak, frequency, sample width
        echo_list = [[e[0], e[1], self.hilbert_data[e[0]], self.hilbert_data[e[1]], f, 2*(e[1]-e[0])] for e in self._echo_list]

        return echo_list

    @property
    def hilbert_grad(self):

        grad = np.gradient(self.hilbert_data, self._grad_step)
        gfil = ndimage.gaussian_filter(grad, sigma=(self._grad_step*2-1)/6)

        return gfil

    def gradient_hysteresis(self, grad_step: int=None, threshold: float=None, ival_smin: int=None, ival_smax: int=None):

        # parameter init (defaults are heuristics)
        self._grad_step = grad_step if grad_step is not None else self._grad_step
        self._thres_pos = threshold if threshold is not None else self._thres_pos
        self._ival_list = [ival_smin, ival_smax] if ival_smin is not None and ival_smax is not None else self._ival_list

        # gradient analysis
        grad_curv = self.hilbert_grad
        grad_plus = grad_curv > self._thres_pos
        grad_minu = grad_curv < self._thres_neg

        # remove isolated threshold breakthroughs
        filt_plus = grad_plus#ndimage.binary_opening(grad_plus, iterations=4)
        filt_minu = grad_minu#ndimage.binary_opening(grad_minu, iterations=4)

        # get echo positions
        peak_plus = np.diff((filt_plus==1).astype(int), axis=0)
        peak_minu = np.diff((filt_minu==1).astype(int), axis=0)
        args_plus = np.argwhere(peak_plus==1).flatten()
        args_minu = np.argwhere(peak_minu==1).flatten()

        # hysteresis via interval analysis from differences between start and stop indices
        self.echo_hysteresis(args_plus, args_minu, grad_curv)

        return self.echo_list

    def echo_hysteresis(self, args_plus, args_minu, grad_curv, win_method=None):

        win_method = "min_dist" if win_method is None else win_method

        # reset detected echoes
        self._echo_list = []
        
        # iterate through echo peaks (neg. grad) and look for echo start (pos. grad)
        # intuition: hysteresis relies on history and only looks into past
        for c_m in args_minu:
            # compute distance between echo peak and echo start
            dist = c_m - args_plus
            # validate that echo width is within pre-defined range (ival_list)
            mask = (dist > self._ival_list[0]) & (dist < self._ival_list[1])
            if sum(mask) > 0:
                idcs = args_plus[mask]
                if win_method == "min_dist":
                    # select closest positive gradient
                    winner = idcs[np.argmin(c_m-idcs)]
                elif win_method == "max_grad":
                    # select strongest positive gradient
                    winner = idcs[np.argmax(grad_curv[idcs])]
                # save pair of negative and positive gradients
                self._echo_list.append([winner, c_m])
                # remove positive gradient candidate (exclude it from remainding search)
                args_plus = np.delete(args_plus, np.argmax(grad_curv[idcs]))

        return self.echo_list
    
    def replace_echo_start(self):
        
        #np.array(self._echo_list)[:, 1] - self._tsample_rate*self._nbursts_rate/2
        self._echo_list = [[echo[-1] - int(self._signal_width/2), echo[-1]] for echo in self._echo_list]

        return self.echo_list
    
    def get_distances(self):
        # convert sample position to path length
        return [self.path_conv(echo) for echo in self._echo_list]
