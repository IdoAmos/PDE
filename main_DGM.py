import pickle

import matplotlib.pyplot as plt
import torch
import torchinfo

import Data
import plots
import train
import utils
from argparse import ArgumentParser
import yaml

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--config", dest='config', required=True, help='path to config_BASELINE.yml file')
    parser.add_argument("--exp_name", dest='exp_name', required=False, default='experiment', type=str)
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--dest", dest='dest', default='', required=False, type=str, help='path to checkpoint.pt')
    parser.add_argument("--source", dest='source', default='', required=False, type=str)
    parser.add_argument("--show", action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    def_params = utils.default_parameter_generator()
    if config_dict is not None:
        for key in def_params.keys():
            if key not in config_dict:
                config_dict[key] = def_params[key]

    exp_dir = '' if not args.save else utils.make_exp_dir(path=args.dest+'/'+args.exp_name, exp_params=config_dict)
    print('Experiment directory:' + exp_dir)

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
    if args.load:
        config_dict['checkpoint_path'] = args.source
    hist = utils.init_hist_dict()

    hist = train.call_method(model=model, int_loader=dloader_int, bc_loader=dloader_bc, ic_loader=dloader_ic, hist_dict=hist,
                      config_dict=config_dict, grad_dist=True, save=args.save, path=exp_dir, load=args.load)

    plots.history_plots(config_dict, exp_dir, hist, args.save, args.show)

    if args.save:
        utils.save_exp(exp_dir, hist, model)

    model.to('cpu')
    plots.evaluate(config_dict, exp_dir, model, args.save, args.show)
    if args.show:
        plt.show()
    else:
        plt.close('all')
    print('Finished!')

