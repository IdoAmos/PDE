import torch
from torch import optim as optim
from tqdm.auto import trange
import os

import Data
import plots
import utils
import Models
from func import gradient, spatial_laplacian
from plots import show_to_user, grad_dist_plot
from utils import save_checkpoint
from fast_tensor_data_loader import FastTensorDataLoader as fLoader


def call_method(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict,
                show=True, grad_dist=False, save=True, path='', load=False):
    if config_dict['model_name'] == 'Siren':
        hist_dict = train(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict,
                          show=show, grad_dist=grad_dist, save=save, path=path, load=load)
    if config_dict['model_name'] == 'ResOpHidden':
        train_res_op(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict,
                     show=show, grad_dist=grad_dist, save=save, path=path, load=load)
    return hist_dict


def train(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict
          , show=True, grad_dist=False, save=True, path='', load=False):
    """

    :param model:
    :param int_loader:
    :param bc_loader:
    :param ic_loader:
    :param hist_dict:
    :param config_dict:
    :param show:
    :param grad_dist:
    :param save:
    :param path:
    :param load:
    :return:
    """
    device, hist_dict, optimizer, scheduler, start_epoch = pre_training_setup(config_dict=config_dict,
                                                                              hist_dict=hist_dict,
                                                                              load=load,
                                                                              model=model,
                                                                              path=path,
                                                                              save=save)
    if config_dict['train_scheme'] == 'ic_grad':
        neural_img = Models.Siren(in_features=2, hidden_features=512, hidden_layers=2,
                                  out_features=1, outermost_linear=True)
        neural_img.to(device)
        state_dict_path = os.getcwd() + '/neural_camman'  # TODO: CHANGE TO SOMETHING MORE GENERAL
        neural_img.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        neural_img = None

    model.train()
    for epoch in trange(start_epoch, config_dict['MAX_EPOCH']):
        for key, item in hist_dict.items():
            item.append(0)
        if config_dict['train_scheme'] == 'DGM':
            int_loader, bc_loader, ic_loader = Data.DGM_sample(batch_size=config_dict['batch_size'],
                                                               x_min=config_dict['x_min'],
                                                               x_max=config_dict['x_max'],
                                                               y_min=config_dict['y_min'],
                                                               y_max=config_dict['y_max'],
                                                               t_max=config_dict['t_max'])

        train_epoch(bc_loader=bc_loader, config_dict=config_dict, device=device, hist_dict=hist_dict,
                    ic_loader=ic_loader, int_loader=int_loader, model=model, optimizer=optimizer, neural_img=neural_img)

        post_epoch_ops(config_dict=config_dict, device=device, epoch=epoch, grad_dist=grad_dist,
                       hist_dict=hist_dict, int_loader=int_loader, model=model, optimizer=optimizer,
                       path=path, save=save, show=show, scheduler=scheduler)

    model.to("cpu")
    model.eval()
    return hist_dict


def pre_training_setup(config_dict, hist_dict, load, model, path, save):
    # move model and data to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # move to device
    model.to(device)
    if save:
        os.mkdir(path + '/gradient dist', )
    # init optimizer scheduler
    optimizer = init_optimizer(config_dict, model)
    scheduler = scheduler_wrapper(config_dict, optimizer)

    if load:
        start_epoch, loss = utils.load_checkpoint(model, config_dict['checkpoint_path'], device, optimizer)
        optimizer.zero_grad()
        if loss != -1:
            hist_dict = loss
    else:
        start_epoch = 0
    return device, hist_dict, optimizer, scheduler, start_epoch


def post_epoch_ops(config_dict, device, epoch, grad_dist, hist_dict, int_loader, model, optimizer, path, save,
                   scheduler, show=False):
    with torch.no_grad():
        hist_dict['train_loss'][-1] /= len(int_loader)
        hist_dict['mean_fid'][-1] /= len(int_loader)
        hist_dict['mean_ic'][-1] /= len(int_loader)
        hist_dict['mean_bc'][-1] /= len(int_loader)

        # print some stuff to the screen
        if epoch % 5 == 0:
            if epoch % 500 == 0:
                show_to_user(config_dict=config_dict, epoch=epoch, model=model, device=device, hist_dict=hist_dict,
                             show=show)
            else:
                show_to_user(config_dict=config_dict, epoch=epoch, model=model, device=device, hist_dict=hist_dict,
                             show=False)

    scheduler.step(hist_dict['train_loss'][-1])

    if save:
        if epoch % 500 == 0:
            save_checkpoint(model, path + '/checkpoints/model.pt', epoch, optimizer, hist_dict)
        if epoch == 5:
            save_checkpoint(model, path + '/checkpoints/initial_model.pt', epoch, optimizer, hist_dict)
            grad_dist_plot(model, epoch=epoch, save=save, path=path + '/gradient dist', show=False,
                           img_name='grad_dist_init') if grad_dist else None
        if epoch == config_dict['MAX_EPOCH'] // 2:
            save_checkpoint(model, path + '/checkpoints/mid_model.pt', epoch, optimizer, hist_dict)
            grad_dist_plot(model, epoch=epoch, save=save, path=path + '/gradient dist', show=False,
                           img_name='grad_dist_mid') if grad_dist else None
        if epoch == config_dict['MAX_EPOCH'] - 1:
            save_checkpoint(model, path + '/checkpoints/trained_model.pt', config_dict['MAX_EPOCH'], optimizer,
                            hist_dict)
            grad_dist_plot(model, epoch=epoch, save=save, path=path + '/gradient dist',
                           show=False, img_name='grad_dist_final') if grad_dist else None


def train_epoch(bc_loader, config_dict, device, hist_dict, ic_loader, int_loader, model, optimizer, neural_img=None):
    for batch in zip(int_loader, ic_loader, bc_loader):
        # extract data from batches, move to device
        loss = eval_loss(batch, config_dict, device, hist_dict, model, neural_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record values to history
        with torch.no_grad():
            hist_dict['train_loss'][-1] += loss.item()


def eval_loss(batch, config_dict, device, hist_dict, model, neural_img=None):
    with torch.no_grad():
        int_batch, ic_batch, bc_batch = batch

        X_int = int_batch[0].to(device) if config_dict['C']['fidelity'] != 0 else None
        X_ic = ic_batch[0].to(device) if config_dict['C']['ic'] != 0 else None
        y_ic = ic_batch[1].to(device) if config_dict['C']['ic'] != 0 else None
        X_top = bc_batch[0].to(device) if config_dict['C']['bc'] != 0 else None
        X_bottom = bc_batch[1].to(device) if config_dict['C']['bc'] != 0 else None
        X_left = bc_batch[2].to(device) if config_dict['C']['bc'] != 0 else None
        X_right = bc_batch[3].to(device) if config_dict['C']['bc'] != 0 else None
    # fidelity term
    if config_dict['C']['fidelity'] != 0:
        fidelity = fidelity_term(X_int, hist_dict, model, config_dict['P'], config_dict['C'])
    else:
        fidelity = torch.Tensor([0]).to(device)[0]
    # initial condition term
    if config_dict['C']['ic'] != 0:
        ic = initial_cond(X_ic, hist_dict, model, config_dict['P'], y_ic, config_dict['C'], neural_img)
    else:
        ic = torch.Tensor([0]).to(device)[0]
    if config_dict['optimizer_config']['type'] == 'lbfgs':  # pass l2 penalty explicitly
        weight_decay = 0
        for w in model.parameters():
            weight_decay += w.norm() ** 2
    else:
        weight_decay = torch.Tensor([0]).to(device)[0]
    # initial condition term
    if config_dict['C']['bc'] != 0:
        bc = boundary_cond(X_bottom, X_left, X_right, X_top, hist_dict, model, config_dict['P'],
                           config_dict['C'])
    else:
        bc = torch.Tensor([0]).to(device)[0]
    # calc loss and backward
    loss = fidelity + ic + bc + weight_decay
    return loss


def boundary_cond(X_bottom, X_left, X_right, X_top, hist_dict, model, P, C):
    # eval on boundaries
    out_t, in_t = model(X_top)
    out_b, in_b = model(X_bottom)
    out_l, in_l = model(X_left)
    out_r, in_r = model(X_right)

    grad_t = gradient(out_t, in_t)[..., 1]
    grad_b = gradient(out_b, in_b)[..., 1]
    grad_l = gradient(out_l, in_l)[..., 0]
    grad_r = gradient(out_r, in_r)[..., 0]

    grad_on_bound = torch.cat([grad_t, grad_b, grad_l, grad_r], dim=0)
    bc = torch.abs(grad_on_bound) ** P['bc']

    # store max values
    if bc.max() > hist_dict['max_bc'][-1]:
        hist_dict['max_bc'][-1] = bc.max().detach().cpu()

    # compute mean for loss
    bc = torch.mean(bc)
    hist_dict['mean_bc'][-1] += bc.detach().to("cpu")
    bc = C['bc'] * torch.mean(bc)
    return bc


def initial_cond(X, hist_dict, model, P, y, C, neural_img=None):
    """
    Get initial condition term for model and input
    Args:
        X_ic: input
        hist_dict: history dictionary to store values
        model: predictor
        P: powers for initial condition terms
        y_ic: label or label generating function
        C: parameters for initial condition
        IC_GRAD: if to include intial condition gradient in loss, if True 'y_ic' must be a torch.nn.Module

    Returns:
        torch scalar tensor
        :param y:
        :param neural_img:
    """
    if neural_img is not None:
        y, neural_coords = neural_img(X[..., :-1])
        gt_grad = gradient(y=y, x=neural_coords)

    out, coords = model(X)
    ic = torch.abs(out - y) ** P['ic']

    # store max values
    if ic.max() > hist_dict['max_ic'][-1]:
        hist_dict['max_ic'][-1] = ic.max().detach().cpu()

    # get the top k values if using inf norm
    if 'ic_inf' in C.keys():
        ic_inf = C['ic_inf'] * torch.topk(ic, k=P['ic_inf'], dim=0).values
    else:
        ic_inf = torch.Tensor([0]).to(ic.device)[0]

    if not neural_img:
        ic_grad = torch.Tensor([0]).to(ic.device)[0]
    # get loss on gradient if using gradient of IC as well
    else:
        predicted_grad = gradient(y=out, x=coords)[..., :-1]  # we want the spatial gradient of model to match GT
        ic_grad = C['ic_grad'] * torch.norm(gt_grad - predicted_grad, p=P['ic_grad'],
                                            dim=1)  # TODO: IS DIM 1 GOOD HERE?

    # compute mean for loss
    ic = torch.mean(ic)
    hist_dict['mean_ic'][-1] += ic.detach().to("cpu")
    ic = C['ic'] * ic.mean() + ic_inf.mean() + ic_grad.mean()
    return ic


def fidelity_term(X_int, hist_dict, model, P, C):
    # fidelity term
    # track gradients for model input
    out, coords = model(X_int)
    laplacian, dt = spatial_laplacian(out, coords)
    fidelity = torch.abs(laplacian.flatten() - dt) ** P['fidelity']

    # store the max value
    if fidelity.max() > hist_dict['max_fid'][-1]:
        hist_dict['max_fid'][-1] = fidelity.max().detach().cpu()

    # get the top k values if using inf norm
    if 'fid_inf' in P.keys():
        fidelity_inf = C['fid_inf'] * torch.topk(fidelity, k=P['fid_inf'], dim=0).values
    else:
        fidelity_inf = torch.Tensor([0]).to(fidelity.device)[0]

    # compute mean for loss, update history and return
    fidelity = torch.mean(fidelity)
    hist_dict['mean_fid'][-1] += fidelity.detach().to("cpu")
    fidelity = C['fidelity'] * fidelity + torch.mean(fidelity_inf)
    return fidelity


def train_res_op(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict
                 , show=True, grad_dist=False, save=True, path='', load=False):
    print('\nTraining initial condition solver...')
    train_init_cond(model, 5e-5, 1000, config_dict)
    model.full_train_mode()
    print('\nTraining fidelity solver...')
    train(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict
          , show=show, grad_dist=grad_dist, save=save, path=path, load=load)


def train_init_cond(model, learning_rate, max_epoch, config_dict):
    # generate high-res init cond
    f0 = Data.init_f0(cameraman_im=config_dict['cam_man'])
    data, labels = Data.initial_cond_dataset(N_ic=f0.shape[0],
                                             f0=f0,
                                             x_min=config_dict['x_min'],
                                             x_max=config_dict['x_max'],
                                             y_min=config_dict['y_min'],
                                             y_max=config_dict['y_max'])
    data = data.view(-1, 3)
    labels = labels.view(-1, 1)

    # generate a dataloader
    ic_loader = fLoader(data,
                        labels,
                        batch_size=10000,
                        shuffle=True)

    # move model and data to GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    # move to device
    model.to(device)

    omega = Data.temporal_slice(x_min=config_dict['x_min'],
                                x_max=config_dict['x_max'],
                                y_min=config_dict['y_min'],
                                y_max=config_dict['y_max'],
                                t=0,
                                N=f0.shape[0],
                                with_boundary=True)
    omega = omega.reshape(-1, 3)
    omega = torch.from_numpy(omega)
    omega = omega.to(device)

    # init optimizer scheduler
    loss_fn = torch.nn.L1Loss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(max_epoch):
        for batch in ic_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            res, _ = model(x)
            loss = loss_fn(res, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if epoch % 200 == 0:
                print('epoch: {}\tloss: {}'.format(epoch, loss.item()))
                plots.show_to_user(device, epoch, model, omega)


def get_params_to_update(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


def train_with_ic_grad(model, int_loader, bc_loader, ic_loader, hist_dict, config_dict
                       , show=True, grad_dist=False, save=True, path='', load=False):
    """

    :param model:
    :param int_loader:
    :param bc_loader:
    :param ic_loader:
    :param hist_dict:
    :param config_dict:
    :param show:
    :param grad_dist:
    :param save:
    :param path:
    :param load:
    :return:
    """
    device, hist_dict, optimizer, schedule, scheduler, start_epoch = pre_training_setup(config_dict, hist_dict, load,
                                                                                        model, path, save)

    model.train()
    for epoch in trange(start_epoch, config_dict['MAX_EPOCH']):
        for key, item in hist_dict.items():
            item.append(0)

        train_epoch(bc_loader, config_dict, device, hist_dict, ic_loader, int_loader, model, optimizer, neural_img)

        post_epoch_ops(config_dict, device, epoch, grad_dist, hist_dict, int_loader, model, optimizer, path, save,
                       scheduler, show)

    model.to("cpu")
    model.eval()


class scheduler_wrapper:
    def __init__(self, config, optimizer):
        self.type = ''
        if config['sched_dict'] is not None:
            schedule_dict = config['sched_dict']
            self.type = schedule_dict['type']
            if self.type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                           step_size=schedule_dict['step'],
                                                           gamma=schedule_dict['gamma'])
            if self.type == 'reduce_on_plat':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      factor=schedule_dict['gamma'],
                                                                      patience=schedule_dict['patience'],
                                                                      threshold=schedule_dict['threshold'],
                                                                      verbose=True)

    def step(self, metric):
        if self.type == 'step':
            self.scheduler.step()
        if self.type == 'reduce_on_plat':
            self.scheduler.step(metric)


def init_optimizer(config, model):
    optim_config = config['optimizer_config']
    type = optim_config['type']
    lr = optim_config['lr']
    l2_pen = optim_config['weight_decay']
    if type == 'Adam':
        if 'beta1' in optim_config.keys():
            beta1 = optim_config['beta1']
            beta2 = optim_config['beta2']
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=l2_pen)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_pen)

    if type == 'lbfgs':
        optimizer = optim.LBFGS(params=model.parameters(), lr=lr)

    return optimizer
