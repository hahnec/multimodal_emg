import torch
from torch.nn.functional import conv1d
from torch.distributions import Normal


def grad_peak_detect(data, grad_step: int=None, threshold: float=None, ival_smin: int=None, ival_smax: int=None):

    batch_size = data.shape[0]

    # hilbert data preparation
    grad_step = grad_step if grad_step is not None else 2    
    grad_data = torch.gradient(data, spacing=grad_step, dim=-1)[0]
    grad_data = gaussian_filter_1d(grad_data, sigma=(grad_step*2-1)/6)

    # parameter init (defaults are heuristics)
    thres_pos = threshold if threshold is not None else (grad_data.max()+grad_data.mean())/15
    thres_neg = -thres_pos/4
    ival_list = [ival_smin, ival_smax] if ival_smin is not None and ival_smax is not None else [grad_step//2, grad_step*3]

    # gradient analysis
    grad_plus = grad_data > thres_pos
    grad_minu = grad_data < thres_neg

    # get potential echo positions
    peak_plus = torch.diff((grad_plus==1).int(), axis=-1)
    peak_minu = torch.diff((grad_minu==1).int(), axis=-1)
    args_plus = torch.argwhere(peak_plus==1)#.flatten()
    args_minu = torch.argwhere(peak_minu==1)#.flatten()

    # hysteresis via interval analysis from differences between start and stop indices
    peak_list = []
    max_len = 0
    for i in range(batch_size):
        ap = args_plus[args_plus[:, 0]==i, 1].unsqueeze(1)#.float()
        am = args_minu[args_minu[:, 0]==i, 1].unsqueeze(0)#.float()
        if ap.numel() == 0 or am.numel() == 0:
            peak_list.append(torch.tensor([], device=data.device))
            continue

        dmat = am - ap.repeat((1, am.shape[1]))
        dmat[dmat<0] = 2**32 # constraint that only differences for ap occuring before am are valid
        echo_peak_idcs = torch.argmin(abs(dmat), dim=0)
        candidates = torch.hstack([ap[echo_peak_idcs], am.T])

        # constraint that only differences for ap occuring before am are valid
        gaps = candidates.diff(1).squeeze()
        candidates = candidates[(gaps>ival_list[0]) & (gaps<ival_list[1]), :]

        if candidates.numel() == 0:
            peak_list.append(torch.tensor([], device=data.device))
            continue

        # gradient peak uniqueness constraint
        apu, uniq_idcs = torch.unique(candidates[:, 0], return_inverse=True)
        amu = candidates[torch.diff(uniq_idcs.flatten(), prepend=torch.tensor([-1], device=apu.device))>0, 1]
        peaks = torch.stack([apu.flatten(), amu.flatten()]).T

        peak_list.append(peaks)
        if len(peaks) > max_len: max_len = len(peaks)

    # convert list to tensor: batch_size x echo_num x (xy)-coordinates
    batch_peaks = torch.tensor([torch.hstack([echoes, data[i, echoes[:, 1][:, None]]]).tolist()+[[0,0,0],]*(max_len-len(echoes)) if len(echoes) > 0 else [[0,0,0],]*max_len for i, echoes in enumerate(peak_list)], dtype=data.dtype, device=data.device)

    return batch_peaks


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    
    radius = int(num_sigmas * sigma)+1 # ceil
    support = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    w /= w.sum()#max()#
    #norm_term = (1/(2*torch.pi)**.5*std)#/w.sum()
    return w#/norm_term


def gaussian_filter_1d(data: torch.Tensor, sigma: float) -> torch.Tensor:
    
    kernel_1d = gaussian_kernel_1d(sigma).to(data.device)  # Create 1D Gaussian kernel
    #kernel_1d = gaussian_fn(int(3*sigma)+1, sigma).double().to(data.device)  # Create 1D Gaussian kernel
    
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    data = data.unsqueeze(1)#.unsqueeze_(0)  # Need 4D data for ``conv2d()``
    # Convolve along columns and rows
    #data = conv1d(data, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    #data = conv2d(data, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    data = conv1d(data, weight=kernel_1d.view(1, 1, -1), padding=padding)
    return data.squeeze(1)#.squeeze_(0)  # Make 2D again