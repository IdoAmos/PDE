import os
import pickle
import torch
import torchinfo
import Models

def init_hist_dict():
    d = {'train_loss': [],
         'mean_ic': [],
         'mean_bc': [],
         'mean_fid': [],
         'max_ic': [],
         'max_bc': [],
         'max_fid': []}
    return d


def save_checkpoint(net, path, epoch=None, optimizer=None, loss=None):
    torch.save({
        'epoch': epoch if epoch is not None else None,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss if loss is not None else None,
    }, path)


def load_checkpoint(net, path, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if optimizer is not None else ''
    if checkpoint['epoch'] is not None:
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    if checkpoint['loss'] is not None:
        loss = checkpoint['loss']
    else:
        loss = -1
    return epoch, loss


def default_parameter_generator():
    C = dict(fidelity=1, ic=1, bc=1)

    P = dict(fidelity=2, ic=1, bc=1)

    sched_dict = dict(type='step', step=500, gamma=0.8)
    optimizer_config = dict(type='Adam', lr=1e-6, weight_decay=0)

    def_params = dict(N_spat=128, N_temp=8, N_bound=128, N_ic=128, cam_man=True,
                      MAX_EPOCH=6000, batch_size=2048, sched_dict=sched_dict, C=C, P=P,
                      checkpoint=True, checkpoint_path='',
                      model_name='Siren', num_layers=1, hidden_features=256, optimizer_config=optimizer_config,
                      t_max=0.001, x_max=1, x_min=-1, y_max=1, y_min=-1, sample_method='uniform')
    return def_params


def make_exp_dir(path, exp_params):
    exp_num = 0
    if not os.path.exists(path):
        os.mkdir(path)
    exp_dir = path + '/config0' + str(exp_num)
    while os.path.exists(exp_dir):
        exp_num += 1
        exp_dir = path + '/config{}'.format(exp_num) + str(exp_num)
    os.mkdir(exp_dir)
    os.mkdir(exp_dir + '/checkpoints') if exp_params['checkpoint'] else ''

    f = open(exp_dir + '/exp_config.txt', 'a')
    for key, item in exp_params.items():
        f.write(key + ': {}\n'.format(item))
    f.close()

    f = open(exp_dir + '/config_dict', 'wb')
    pickle.dump(exp_params, f)
    f.close()
    return exp_dir


def save_exp(exp_dir, hist, model):
    print('saving experiment to:\n' + exp_dir)
    f = open(exp_dir + '/hist', 'wb')
    pickle.dump(hist, f)
    f.close()

    torch.save(model.state_dict(), exp_dir + '/weights')
    sum_str = torchinfo.summary(model, (1, 3), verbose=0)
    f = open(exp_dir + '/model_config.txt', 'a')
    f.write(str(sum_str))
    f.close()


def assign_model(config_dict) -> torch.nn.Module:
    name = config_dict['model_name']

    if name == 'Siren':
        model = Models.Siren(in_features=3,
                             out_features=1,
                             hidden_features=config_dict['hidden_features'],
                             hidden_layers=config_dict['num_layers'],
                             outermost_linear=True)
    if name == 'ResOpHidden':
        # init model
        model = Models.ResOpHidden(in_features=3,
                                   out_features=1,
                                   hidden_features=config_dict['hidden_features'],
                                   ic_layers=config_dict['ic_layers'],
                                   op_layers=config_dict['op_layers'])
    return model
