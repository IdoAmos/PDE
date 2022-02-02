import matplotlib.pyplot as plt
import torch
import torchinfo

import Data
import plots
import train
import utils
from argparse import ArgumentParser
import yaml


def main(config, save=False, dest='.', exp_name='test', load=False, source='', show=True):

    with open(config, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    def_params = utils.default_parameter_generator()
    if config_dict is not None:
        for key in def_params.keys():
            if key not in config_dict:
                config_dict[key] = def_params[key]

    exp_dir = '' if not save else utils.make_exp_dir(path=dest + '/' + exp_name, exp_params=config_dict)
    print('Experiment directory:' + exp_dir)
    print('Experiment Type:', config_dict['train_scheme'])

    print('\n\nBegin sequence with passed hyper parameters:')
    plots.display_hyper_parameters(config_dict)

    print('\nGenerating dataset, dataloaders...')
    dloader_int, dloader_bc, dloader_ic, f0 = Data.generate_train_data(config_dict)

    # show the initial condition image
    print('\nInitializing a {} model'.format(config_dict['model_name']))
    model = utils.assign_model(config_dict)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    torchinfo.summary(model, (1, 3), verbose=1)

    print('\nBegin training...')
    # if args.load:
    #     f = open(args.source + '/hist', 'rb')
    #     hist = pickle.load(f)
    #     f.close()
    # else:
    #     hist = utils.init_hist_dict()
    if load:
        config_dict['checkpoint_path'] = source
    hist = utils.init_hist_dict()

    hist = train.call_method(model=model, int_loader=dloader_int, bc_loader=dloader_bc, ic_loader=dloader_ic,
                             hist_dict=hist,
                             config_dict=config_dict, grad_dist=True, save=save, path=exp_dir, load=load, show=True)

    plots.history_plots(config_dict, exp_dir, hist, save, show)

    if save:
        utils.save_exp(exp_dir, hist, model)

    model.to('cpu')
    plots.evaluate(config_dict, exp_dir, model, save, show)
    if show:
        plt.show()
    else:
        plt.close('all')
    print('Finished!')
