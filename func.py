import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

import Data


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def spatial_laplacian(y, x, temp_idx=2):
    """
    get divergence of y with respect to x excpet for 'temp_idx'
    :param y: tensor, element to diffrentiate
    :param x: tensor, element to derive according to
    :param temp_idx: index of variable to omit from divergence
    :return: spatial divergence, dy/dt
    """
    grad = gradient(y, x)
    return divergence(grad, x, temp_idx), grad[:, temp_idx]


def divergence(y, x, temp_idx=-1):
    div = 0.
    for i in range(y.shape[-1]):
        if i == temp_idx:
            continue
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def gaussian_conv(t, f0, turncate=4):
    """
    implementation of the gaussian convolution with respect to the defined grid.
    :param N: int,
    :param t: float, sqrt(4t) is the standrad deviation of the gaussian.
    :param f0: 2d array, input signal to convolve with gaussian
    :param turncate: int, number of std to turncate kernel after.
    """
    # define the kernel domain
    sigma = np.sqrt(2 * t)
    N = f0.shape[0]
    domain = Data.temporal_slice(t=0, N=N)[..., :-1]

    x_kernel_bound = np.logical_and(domain[..., 0] < turncate * sigma,
                                    domain[..., 0] > -turncate * sigma)
    L_x = x_kernel_bound.sum() // N  # size in x dim

    y_kernel_bound = np.logical_and(domain[..., 1] < turncate * sigma,
                                    domain[..., 1] > -turncate * sigma)
    L_y = y_kernel_bound.sum() // N  # size in y dim

    kernel_bounds = np.logical_and(x_kernel_bound, y_kernel_bound)
    kernel_domain = domain[kernel_bounds].reshape(L_x, L_y, 2)

    # define kernel
    x = kernel_domain[..., 0] ** 2 + kernel_domain[..., 1] ** 2
    x = -x / (2 * (sigma ** 2))
    kernel = np.exp(x) / (4 * np.pi * t)

    # convolve
    delta_x = domain[0, 1:, 0, 0] - domain[0, :-1, 0, 0]
    delta_x = delta_x.mean()
    delta_y = domain[1:, 0, 0, 1] - domain[:-1, 0, 0, 1]
    delta_y = delta_y.mean()
    kernel = kernel * delta_x * delta_y

    res = convolve2d(f0, kernel, mode='same', boundary='symm')
    return res


def eval_res(model, t_max=1, x_min=-1, x_max=1, y_min=-1, y_max=1, N=200, sampler='uniform', mode='eval', save=False,
             camman=True, path='.'):
    """

    :param model:
    :param t_max:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param N:
    :param sampler:
    :param mode:
    :param save:
    :param camman:
    :param path:
    :return:
    """
    num_rows = 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols)
    fig.set_size_inches([25, 15])
    fig.tight_layout(pad=3.2)

    print('generating spatial grid with bounds: [{},{}]x[{},{}]'.format(x_min,
                                                                        x_max,
                                                                        y_min,
                                                                        y_max))

    T = Data.sampler(start=0, stop=t_max, num_samples=int(num_rows * num_cols), method=sampler)
    coord = Data.temporal_slice(t=0,
                                N=N,
                                with_boundary=True,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max)

    f0 = Data.init_f0(cameraman_im=camman)
    f0 = Data.img_sampler(f0,
                          coord,
                          x_min=x_min,
                          x_max=x_max,
                          y_min=y_min,
                          y_max=y_max)

    counter = 0
    for i in range(num_rows):
        row = axes[i]
        for ax in row:
            coord[..., -1] = T[counter]

            if mode == 'eval':
                out, _ = model(torch.from_numpy(coord))
                out = out.view(N, N).detach().numpy()
                title = f't={T[counter]:.5f}'

            if mode == 'gt':
                if T[counter] <= 0:
                    out = f0
                if T[counter] > 0:
                    out = gaussian_conv(T[counter], f0)
                title = f't={T[counter]:.5f}'

            if mode == 'diff':
                # Ground Truth
                if T[counter] <= 0:
                    gt = f0
                if T[counter] > 0:
                    gt = gaussian_conv(T[counter], f0)

                out, _ = model(torch.from_numpy(coord).unsqueeze(dim=1))

                out = out.view(N, N).detach().numpy()
                out = np.abs(out - gt)
                title = 't={:.3e};'.format(T[counter] * t_max)
                title += "\n image range [{:.2f},{:.2f}]".format(gt.min(), gt.max())
                title += '\n mean error ={:.3f}+-{:.3f}'.format(out.mean(), out.std())
                title += '\n max error ={:.3f}'.format(out.max())

            ax.imshow(out, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
            counter += 1

    plt.tight_layout()
    if save:
        plt.savefig(path + '/' + mode)

    plt.draw()


def eval_metric(model, t_max: float, metric, x_min=-1, x_max=1, y_min=-1, y_max=1, N: int = 200, num_points: int = 10,
              sampler: str = 'uniform',
              path: str = '.', camman: bool = True, save: bool = False):
    """
    Create a plot of SSIM values as a function of t from 0 to t_max
    :param model: model generating images to evaluate
    :param t_max: max value of time
    :param N: image size (square images). (optional), default 200.
    :param num_points: number of points in time to evaluate. (optional), default 10.
    :param sampler: sampling method of time points. (optional), default 'uniform'.
    :param path: path to directory to save results if save if True. (optional), default current dir.
    :param camman: if to use cameraman image. (optional), default True.
    :param save: if to save plot. (optional), default False.
    """

    print('generating spatial grid with bounds: [{},{}]x[{},{}]'.format(x_min,
                                                                        x_max,
                                                                        y_min,
                                                                        y_max))
    if metric == 'ssim':
        name = '/ssim_plot'
        T, err = ssim_by_time(N, model, t_max, num_points, camman, sampler, x_max, x_min, y_max, y_min)
    if metric == 'psnr':
        name = '/psnr_plot'
        T, err = psnr_by_time(N, model, t_max, num_points, camman, sampler, x_max, x_min, y_max, y_min)

    plt.figure(figsize=[16, 8])
    x_ticks = [t for t in range(len(T))]
    plt.plot(x_ticks, err, c='tab:blue')
    plt.scatter(x_ticks, err, c='tab:blue')

    x_labels = ['{:.1e}'.format(t * t_max) for t in T]
    plt.xticks(ticks=x_ticks, labels=x_labels)

    x_title = 'Time - exp spacing' if sampler == 'exp' else 'Time'
    plt.xlabel(x_title)
    plt.ylabel(metric)
    plt.suptitle('{} vs. t'.format(metric), y=1.05)

    plt.tight_layout()
    plt.savefig(path + name) if save else None
    plt.draw()


def ssim_by_time(N, model, t_max, num_points=10, camman=True, sampler='uniform', x_max=1, x_min=-1, y_max=1, y_min=-1):
    T = Data.sampler(start=0, stop=t_max, num_samples=num_points, method=sampler)
    coord = Data.temporal_slice(t=0,
                                N=N,
                                with_boundary=True,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max)
    f0 = Data.init_f0(cameraman_im=camman)
    f0 = Data.img_sampler(f0,
                          coord,
                          x_min=x_min,
                          x_max=x_max,
                          y_min=y_min,
                          y_max=y_max)
    err = np.zeros_like(T)
    for counter in range(len(T)):
        # Ground Truth
        if T[counter] <= 0:
            gt = f0
        if T[counter] > 0:
            gt = gaussian_conv(T[counter], f0)

        # estim
        coord[..., -1] = T[counter]

        out, _ = model(torch.from_numpy(coord))

        out = out.view(N, N).detach().numpy()
        err[counter] = ssim(gt, out)
    return T, err


def psnr_by_time(N, model, t_max, num_points=10, camman=True, sampler='uniform', x_max=1, x_min=-1, y_max=1, y_min=-1):
    T = Data.sampler(start=0, stop=t_max, num_samples=num_points, method=sampler)
    coord = Data.temporal_slice(t=0,
                                N=N,
                                with_boundary=True,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max)
    f0 = Data.init_f0(cameraman_im=camman)
    f0 = Data.img_sampler(f0,
                          coord,
                          x_min=x_min,
                          x_max=x_max,
                          y_min=y_min,
                          y_max=y_max)
    err = np.zeros_like(T)
    for counter in range(len(T)):
        # Ground Truth
        if T[counter] <= 0:
            gt = f0
        if T[counter] > 0:
            gt = gaussian_conv(T[counter], f0)

        # estim
        coord[..., -1] = T[counter]

        out, _ = model(torch.from_numpy(coord))

        out = out.view(N, N).detach().numpy()

        # calc psnr
        max_I = gt.max()
        mse =  np.square(gt-out).mean()
        psnr = 20*np.log10(max_I) - 10*np.log10(mse)
        err[counter] = psnr
    return T, err
