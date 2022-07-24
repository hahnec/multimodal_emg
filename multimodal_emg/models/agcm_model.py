import numpy as np


def gaussian_enve_alt(t, alpha, r, tau, beta=1):

        k = 1
        return beta*np.exp(-alpha*(1-r*np.tanh(k*(t-tau)))*(t-tau)**2)


def gaussian_wave_alt(t, alpha, r, tau, fc, Phi, phi, beta=1):

        res = gaussian_enve_alt(t, alpha, r, tau, beta) * np.cos(2*np.pi*fc*(t-tau)+Phi*(t-tau)**2+phi)
        return res