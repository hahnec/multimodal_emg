import numpy as np
from scipy import ndimage

def plot_grad(detector, show_opt=True, t=None):

    grad_curv = detector.hilbert_grad
    grad_plus = grad_curv > detector._thres_pos
    grad_minu = grad_curv < detector._thres_neg

    # remove isolated threshold breakthroughs
    filt_plus = ndimage.binary_opening(grad_plus, iterations=4)
    filt_minu = ndimage.binary_opening(grad_minu, iterations=4)

    t = np.arange(len(grad_curv)) if t is None else t

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(grad_curv, label='gradient')
    ax.plot(detector._thres_pos*np.ones(len(grad_curv)), label='pos. threshold')
    ax.plot(grad_curv.max()*filt_plus, label='positive breakthroughs')
    ax.plot(grad_curv.max()*filt_minu, label='negative breakthroughs')
    ax.plot(detector._thres_neg*np.ones(len(grad_curv)), label='neg. threshold')
    ax.legend()
    plt.show() if show_opt else None
