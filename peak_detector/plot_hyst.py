import numpy as np


def plot_hyst(detector, feat_list=None, grad_opt=False, circ_opt=False, show_opt=False, gray_opt=False, title='', ax=None, t=None):

    feat_list = detector.echo_list if feat_list is None else feat_list

    [c1, c2, c3, c4, c5] = [None]*5 if gray_opt else ['orange', 'blue', 'purple', 'green', 'red']

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    #plt.rc('text', usetex=True)
    #plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True

    max_val = max(detector.hilbert_data)
    t = np.arange(len(detector.channel_data)) if t is None else t

    _, ax = plt.subplots(figsize=(15, 5)) if ax is None else (None, ax)
    l1, = ax.plot(t, detector.channel_data, color=c1, linestyle='-', label=r'Wave signal $A(t)$')#/max(detector.channel_data)
    l2, = ax.plot(t, detector.hilbert_data, color=c2, linestyle='-.', label=r'Hilbert norm $f(t)$')#/max(detector.hilbert_data)
    if grad_opt:
        grad_curv = detector.hilbert_grad
        g1 = ax.plot(t, grad_curv/max(grad_curv), color=c3, label=r'$\nabla f(t, \delta t)$ Gradient')

    fsize = 18
    ax.set_xlabel(r'Time $t$ [samples]', fontsize=fsize)
    ax.set_ylabel(r'Amplitudes [a.u.]', fontsize=fsize)
    ax.set_title(title.rstrip('\r\n'), fontsize=fsize, y=1, loc='left') if isinstance(title, str) else None
    if np.array(feat_list).size != 0:
        p1 = [ax.plot([t[feat_arg[0]], t[feat_arg[0]]], [.5*max_val, 0], color=c4, linestyle=':', label='Echo Start') for feat_arg in feat_list]
        p2 = [ax.plot([t[feat_arg[1]], t[feat_arg[1]]], [1*max_val, 0], color=c5, linestyle='--', label='Echo Peak') for feat_arg in feat_list]
        if circ_opt:
            idx = -1
            radius = feat_list[idx][1] - feat_list[idx][0]
            e1 = Ellipse((feat_list[idx][1], 0), width=radius*2, height=2*max_val, fill=False, color=None, linestyle=(0, (3, 5, 1, 5, 1, 5)), label='Corresponding triplet')
            ax.add_artist(e1)
            ax.legend(handles=[l1, l2, p1[0][0], p2[0][0], e1], loc='lower right', fontsize=fsize, title_fontsize=fsize)
        else:
            ax.legend(handles=[l1, l2, p1[0][0], p2[0][0]], loc='lower right', fontsize=fsize, title_fontsize=fsize)
    else:
        ax.legend(handles=[l1, l2], loc='lower right', fontsize=fsize-2, title_fontsize=fsize)
    
    plt.tight_layout()
    plt.show() if show_opt else None

    return ax