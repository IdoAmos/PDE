import pickle

import matplotlib.pyplot as plt
import torch
import torchinfo

import Data
import func
import plots
import train
import utils
from plots import display_hyper_parameters


def main(save=True, load=False, dest='.', source='', config_dict=None):
    def_params = utils.default_parameter_generator()
    if config_dict is not None:
        for key in def_params.keys():
            if key not in config_dict:
                config_dict[key] = def_params[key]

    exp_dir = '' if not save else utils.make_exp_dir(path=dest, exp_params=config_dict)
    print('Experiment directory:' + exp_dir)

    print('\n\nBegin sequence with passed hyper parameters:')
    display_hyper_parameters(config_dict)

    print('\nGenerating dataset, dataloaders...')
    dloader_int, dloader_bc, dloader_ic, f0 = Data.generate_train_data(config_dict)

    # show the initial condition image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(f0, cmap='gray')
    ax.set_title('Anchor Image')
    plt.axis('off')

    print('\nInitializing a {} model'.format(config_dict['model_name']))
    model = utils.assign_model(config_dict)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    torchinfo.summary(model, (1, 3), verbose=1)

    print('\nBegin training...')
    if load:
        f = open(source + '/hist', 'rb')
        hist = pickle.load(f)
        f.close()
    else:
        hist = utils.init_hist_dict()

    train.call_method(model=model, int_loader=dloader_int, bc_loader=dloader_bc, ic_loader=dloader_ic, hist_dict=hist,
                      config_dict=config_dict, grad_dist=True, save=save, path=exp_dir, load=load)

    history_plots(config_dict, exp_dir, hist, save)
    if save:
        utils.save_exp(exp_dir, hist, model)
        plots.grad_dist_plot(model, epoch=config_dict['MAX_EPOCH'], save=True, path=exp_dir)

    model.to('cpu')
    evaluate(config_dict, exp_dir, model, save)
    plt.show()
    print('Finished!')

    return model, hist


def evaluate(config_dict, exp_dir, model, save):
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
    func.eval_metric(model=model,
                     metric='ssim',
                     t_max=config_dict['t_max'],
                     x_min=config_dict['x_min'],
                     x_max=config_dict['x_max'],
                     y_min=config_dict['y_min'],
                     y_max=config_dict['y_max'],
                     N=config_dict['N_ic'],
                     save=save,
                     path=exp_dir)
    func.eval_metric(model=model,
                     metric='psnr',
                     t_max=config_dict['t_max'],
                     x_min=config_dict['x_min'],
                     x_max=config_dict['x_max'],
                     y_min=config_dict['y_min'],
                     y_max=config_dict['y_max'],
                     N=config_dict['N_ic'],
                     save=save,
                     path=exp_dir)


def history_plots(config_dict, exp_dir, hist, save):
    plots.history_plot(hist, start=20, save=save, path=exp_dir, figname='mean_value_hist')
    plots.history_plot(hist, start=config_dict['MAX_EPOCH'] // 2, save=save, path=exp_dir,
                       figname='mean_value_hist_midway')
    plots.history_plot(hist, max=True, save=save, path=exp_dir, figname='max_value_hist')
    plots.history_plot(hist, start=config_dict['MAX_EPOCH'] // 2, max=True, save=save, path=exp_dir,
                       figname='max_value_hist_midway')
