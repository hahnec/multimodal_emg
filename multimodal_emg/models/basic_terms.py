import numpy as np
from scipy.special import erf

PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062

alpha_term = lambda sigma: 1 / (sigma * 2**.5 * PI)
gauss_term = lambda x, mu, sigma, exp_fun=np.exp: exp_fun(-.5 * (x - mu) ** 2 / sigma ** 2)
asymm_term = lambda x, mu, sigma, eta, erf_fun=erf: 1 + erf_fun(eta * (x - mu) / (sigma * 2**.5))
oscil_term = lambda x, mu, f_c, phi, cos_fun=np.cos: cos_fun(2 * PI * f_c * (x - mu) + phi)
