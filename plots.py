import numpy as np
import torch
from matplotlib import pyplot as plt
import Data
import func


def show_to_user(config_dict, epoch, model, device=torch.device('cpu'), hist_dict=None, show=True):
    if hist_dict is not None:
        print('\nEpoch {} loss: {:.5f}\tfidelity: {:.5f}\tic: {:.5f}\tbc: {:.5f}\n'
              .format(epoch, hist_dict['train_loss'][-1], hist_dict['mean_fid'][-1],
                      hist_dict['mean_ic'][-1], hist_dict['mean_bc'][-1]))

    if show:
        with torch.no_grad():
            omega = Data.temporal_slice(x_min=config_dict['x_min'], x_max=config_dict['x_max'],
                                        y_min=config_dict['y_min'], y_max=config_dict['y_max'],
                                        t=config_dict['y_max'], N=config_dict['N_ic'],
                                        with_boundary=True)
            omega = omega.reshape(-1, 3)
            omega = torch.from_numpy(omega)
            omega = omega.to(device)
            T = np.array([0])

            title = 'image evaluated at time t=' + str(T)
            T = torch.from_numpy(T).to(device)
            T = torch.ones_like(omega[..., -1]) * T
            omega[..., -1] = T
            out, _ = model(omega)
            plt.figure()
            plt.imshow(out.cpu().numpy().reshape(config_dict['N_ic'], config_dict['N_ic']), cmap='gray')
            plt.suptitle(title, fontsize=12)
            plt.axis('off')

            plt.show()


def history_plot(hist, start=20, end=None, max=False, save=False, path='.', figname='history', show: bool = False):
    if max:
        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches([24, 14])
        axes[0].plot(hist['max_fid'][start:])
        axes[0].set_title('max fidelity')
        axes[0].grid('on')

        axes[1].plot(hist['max_ic'][start:])
        axes[1].set_title('max initial condition')
        axes[1].grid('on')

        axes[2].plot(hist['max_bc'][start:])
        axes[2].set_title('max boundary condition')
        axes[2].grid('on')

    else:
        fig, axes = plt.subplots(4, 1)
        fig.set_size_inches([24, 14])
        axes[0].plot(hist['train_loss'][start:end])
        axes[0].set_title('train loss')
        axes[0].grid('on')

        axes[1].plot(hist['mean_fid'][start:end])
        axes[1].set_title('mean fidelity')
        axes[1].grid('on')

        axes[2].plot(hist['mean_ic'][start:end])
        axes[2].set_title('mean initial condition')
        axes[2].grid('on')

        axes[3].plot(hist['mean_bc'][start:end])
        axes[3].set_title('mean boundary condition')
        axes[3].grid('on')

    fig.tight_layout(h_pad=1)
    plt.subplots_adjust()
    if save:
        fig.savefig(path + '/' + figname)
    if show:
        plt.draw()
    plt.close('all')


def grad_dist_plot(model, epoch, save=False, path='', show=False):
    num_layers = 0
    for name, param in model.named_parameters():
        if param.dim() > 1 and param.requires_grad:  # only for none bias weights with gradients
            num_layers += 1

    fig, axes = plt.subplots(num_layers)
    fig.set_size_inches([24, 9])

    fig_num = 0
    for name, param in model.named_parameters():
        if param.dim() > 1 and param.requires_grad: #only for none bias weights with gradients
            grad = param.grad.flatten()
            grad = grad.cpu().detach().numpy()

            num_bins = np.min(np.array([len(grad), 10]))
            hist, edges = np.histogram(grad, bins=num_bins)
            hist = hist / hist.sum()
            w = edges[1] - edges[0]
            axes[fig_num].bar(edges[1:], hist, width=w)

            ticks = np.linspace(edges.min(), edges.max(), num=20)
            axes[fig_num].set_xticks(ticks)
            axes[fig_num].set_title('{} gradient distribution'.format(name.split('.')[:-1]))
            fig_num += 1

    fig.suptitle('gradient distribution at epoch {}'.format(epoch), y=1.05)
    fig.tight_layout(h_pad=1)
    fig.savefig(fname=path + '/grad dist') if save else None
    plt.show() if show else plt.close(fig)


def display_hyper_parameters(config_dict):
    print('\nGrid specs:(x,y,t)= [{},{}]x[{},{}]x[0,{}]'.format(config_dict['x_min'],
                                                                config_dict['y_max'],
                                                                config_dict['y_min'],
                                                                config_dict['y_max'],
                                                                config_dict['t_max']))
    print('\nSample specs: N_spat={}, N_bound={}, N_temp={}, N_ic={}'.format(config_dict['N_spat'],
                                                                             config_dict['N_bound'],
                                                                             config_dict['N_temp'],
                                                                             config_dict['N_ic']))
    print('\nTraining hyper parameters: lr={}, num epochs={}, schedule params={}, lambdas={}'.format(config_dict['optimizer_config']['lr'],
                                                                                                     config_dict[
                                                                                                         'MAX_EPOCH'],
                                                                                                     config_dict[
                                                                                                         'sched_dict'],
                                                                                                     config_dict['C']))
    print('\nModel hyper parameters: num layers={}, num hidden units={}'.format(config_dict['num_layers'],
                                                                                config_dict['hidden_features']))


def evaluate(config_dict, exp_dir, model, save, show):
    func.eval_res(model=model,
                  t_max=config_dict['t_max'],
                  x_min=config_dict['x_min'],
                  x_max=config_dict['x_max'],
                  y_min=config_dict['y_min'],
                  y_max=config_dict['y_max'],
                  camman=config_dict['cam_man'],
                  N=config_dict['N_ic'],
                  mode='eval',
                  save=save,
                  path=exp_dir)
    func.eval_res(model=model,
                  t_max=config_dict['t_max'],
                  x_min=config_dict['x_min'],
                  x_max=config_dict['x_max'],
                  y_min=config_dict['y_min'],
                  y_max=config_dict['y_max'],
                  camman=config_dict['cam_man'],
                  N=config_dict['N_ic'],
                  mode='gt',
                  save=save,
                  path=exp_dir)
    func.eval_res(model=model,
                  t_max=config_dict['t_max'],
                  x_min=config_dict['x_min'],
                  x_max=config_dict['x_max'],
                  y_min=config_dict['y_min'],
                  y_max=config_dict['y_max'],
                  camman=config_dict['cam_man'],
                  N=config_dict['N_ic'],
                  mode='diff',
                  save=save,
                  path=exp_dir)
    func.eval_metric(model=model, t_max=config_dict['t_max'], metric='ssim', x_min=config_dict['x_min'],
                     x_max=config_dict['x_max'], y_min=config_dict['y_min'], y_max=config_dict['y_max'],
                     N=config_dict['N_ic'], path=exp_dir, save=save, show=show)
    func.eval_metric(model=model, t_max=config_dict['t_max'], metric='psnr', x_min=config_dict['x_min'],
                     x_max=config_dict['x_max'], y_min=config_dict['y_min'], y_max=config_dict['y_max'],
                     N=config_dict['N_ic'], path=exp_dir, save=save, show=show)


def history_plots(config_dict, exp_dir, hist, save, show):
    history_plot(hist, start=20, save=save, path=exp_dir, figname='mean_value_hist', show=show)
    history_plot(hist, start=config_dict['MAX_EPOCH'] // 2, save=save, path=exp_dir,
                       figname='mean_value_hist_midway', show=show)
    history_plot(hist, max=True, save=save, path=exp_dir, figname='max_value_hist', show=show)
    history_plot(hist, start=config_dict['MAX_EPOCH'] // 2, max=True, save=save, path=exp_dir,
                       figname='max_value_hist_midway', show=show)