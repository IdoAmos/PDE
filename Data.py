import torch
import numpy as np
from torchvision.transforms import ToTensor
import pickle
from PIL import Image
import os

I0_SIZE = 512  # size of base cameraman image


def get_img_tensor(path='camerman.jpeg'):
    '''
    Returns a numpy array representing the cameraman (square) image,
    with dimensions (N, N)

    Args:
        path:
    '''
    image_path = os.getcwd() + '/' + path
    img = Image.open(image_path)
    transform = ToTensor()
    img = transform(img)
    img = torch.permute(img, (1, 2, 0))
    return img.numpy()


def img_sampler(f0, coords, x_min=-1, x_max=1, y_min=-1, y_max=1):
    """
    Sample image f0 according to general coordinates in R2.
    Sampling is done by transforming coords domain to image range [0,num_rows)x[0,num_cols) and NN sampling
    :param f0: image to sample from
    :param data: coordinates of sample
    :return: resampled version of f0
    """
    # coords should be 2D
    data = coords.copy()
    if data.shape[-1] > 2:
        data = data[..., :2]
    # ---------------------------------------------------

    # scale coords to the dimension of f0
    num_cols = f0.shape[1] - 1
    num_rows = f0.shape[0] - 1

    # scale x, y
    data[..., 0] = data[..., 0] - x_min
    data[..., 0] = data[..., 0] / (x_max - x_min)
    data[..., 0] = data[..., 0] * num_cols
    data[..., 1] = data[..., 1] - y_min
    data[..., 1] = data[..., 1] / (y_max - y_min)
    data[..., 1] = data[..., 1] * num_rows

    data = np.rint(data).astype(np.int)

    data = data[..., [1, 0]]  # change to image orientation (y,x)
    sample_points = data.reshape(-1, 2)
    if len(coords.shape) == 2:
        f0_sampled = f0[sample_points[:, 0], sample_points[:, 1]].reshape(len(coords), -1)  # sample f0 in coordinates
    else:
        f0_sampled = f0[sample_points[:, 0], sample_points[:, 1]].reshape(data.shape[:2])  # sample f0 in coordinates

    return f0_sampled


def init_f0(cameraman_im=True) -> np.ndarray:
    if cameraman_im:
        f0 = get_img_tensor('camerman.jpeg')[..., 0]

    # normalize image to [-1,1]
    f0 = (f0 - f0.min()) / f0.max()  # to [0,1]
    f0 = 2 * f0 - 1

    return f0.astype(np.float32)


def tile_time(omega, t):
    """
    Increase dimension of last axis in input array by 1 and fill with values of t for each
    entry in omega
    """
    # repeat each point in omega len(t) times
    omega = np.tile(omega, len(t)).reshape(omega.shape[0],
                                           omega.shape[1],
                                           len(t),
                                           omega.shape[-1])

    # repeat each point in t to concantenate to the omega values
    T = np.tile(t, omega.shape[0] * omega.shape[1]).reshape(omega.shape[0], omega.shape[1], -1)

    T = np.expand_dims(T, axis=len(T.shape))

    # add t values to each point on the grid
    omega = np.concatenate([omega, T], axis=len(omega.shape) - 1)

    return omega


def domain_interior(t_max: float, N_spat: int, N_temp: int,
                    x_min: float = -1, x_max: float = 1, y_min: float = -1, y_max: float = 1,
                    sample_method='uniform'):
    """
    creates the 3D grid defined by the 1D domains: [x_min,x_max]x[y_min,y_max]x[0,t_max].
    Number of points in the gird is N_spat x N_spat x N_temp.
    :param t_max: grid boundary
    :param N_spat: int, square root of number of samples in spatial dimnesion
    :param N_temp: int, number of samples in temporal dimension.
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return: ndarray with shape N_spat x N_spat x N_temp x 3 with values (x,y,t).
    """
    # add points so we can erase the boudnary
    N_spat += 2
    N_temp += 1
    x = sampler(x_min, x_max, N_spat).astype(np.float32)
    y = sampler(y_min, y_max, N_spat).astype(np.float32)
    t = sampler(0, t_max, N_temp, method=sample_method).astype(np.float32)

    # erase boundary
    x = x[1:-1]
    y = y[1:-1]
    t = t[1:]

    # create spatial grid
    omega = np.stack(np.meshgrid(x, y), axis=2)

    # add time
    omega = tile_time(omega=omega, t=t)

    return omega


def interior_dataset(t_max, N_spat, N_temp, x_min=-1, x_max=1, y_min=-1, y_max=1, t_sample_method='uniform'):
    """
    Generate the interior dataset by domain interior as torch tensor,
    """
    omega = domain_interior(t_max=t_max,
                            N_spat=N_spat,
                            N_temp=N_temp,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            sample_method=t_sample_method)
    return torch.from_numpy(omega)


def spatial_boundary(t_max, N_bound, N_temp, x_min=-1, x_max=1, y_min=-1, y_max=1, t_sample_method='uniform'):
    # generate boundary meta-grid (the grid points)
    x = sampler(x_min, x_max, N_bound).astype(np.float32)
    y = sampler(y_min, y_max, N_bound).astype(np.float32)
    t = sampler(0, t_max, N_temp, method=t_sample_method).astype(np.float32)

    # stack as a 3D boundary
    top = np.expand_dims(a=np.stack([x, np.ones_like(x) * y[0]], axis=1),
                         axis=1)
    bottom = np.expand_dims(a=np.stack([x, np.ones_like(x) * y[-1]], axis=1),
                            axis=1)
    left = np.expand_dims(np.stack([np.ones_like(y) * x[0], y], axis=1),
                          axis=1)
    right = np.expand_dims(np.stack([np.ones_like(y) * x[-1], y], axis=1),
                           axis=1)

    # add time
    top = tile_time(top, t)
    bottom = tile_time(bottom, t)
    left = tile_time(left, t)
    right = tile_time(right, t)

    return top, bottom, left, right


def boundary_dataset(t_max, N_bound, N_temp, x_min=-1, x_max=1, y_min=-1, y_max=1, t_sample_method='uniform'):
    """
    Creates a dataset of spatial boundary points of the form:
    data= { (x_i,y_i,t_i) } where x_i,y_i are on the spatial boundary
    The function samples uniformly on each grid, the samples are then shuffled
    and joined into a dataset.
    :param t_max:
    :param N_bound:
    :param N_temp:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return:
    """
    top, bottom, left, right = spatial_boundary(t_max=t_max,
                                                N_bound=N_bound,
                                                N_temp=N_temp,
                                                x_min=x_min,
                                                x_max=x_max,
                                                y_min=y_min,
                                                y_max=y_max,
                                                t_sample_method=t_sample_method)
    ds_bc = torch.from_numpy(np.stack([top, bottom, left, right], axis=0))

    return ds_bc


def temporal_slice(t, N, with_boundary=True, x_min=-1, x_max=1, y_min=-1, y_max=1):
    if with_boundary:
        x = sampler(x_min, x_max, N).astype(np.float32)
        y = sampler(y_min, y_max, N).astype(np.float32)
    else:
        N += 2
        x = sampler(x_min, x_max, N).astype(np.float32)
        y = sampler(y_min, y_max, N).astype(np.float32)
        x = x[1:-1]
        y = y[1:-1]

    # create the spatial grid, add degenerate temporal axis
    omega = np.expand_dims(np.stack(np.meshgrid(x, y), axis=2), axis=2)

    # add a temporal slice
    t = np.ones(shape=(omega.shape[0], omega.shape[1], omega.shape[2], 1), \
                dtype=np.float32) * t

    temp_slice = np.concatenate([omega, t], axis=len(omega.shape) - 1)

    return temp_slice


def x_slice(x, t_max, N_spat, N_temp, with_boundary=True, y_min=-1, y_max=1):
    if with_boundary:
        t = sampler(0, t_max, N_temp).astype(np.float32)
        y = sampler(y_min, y_max, N_spat).astype(np.float32)
    else:
        N_spat += 2
        N_temp += 1
        t = sampler(0, t_max, N_temp).astype(np.float32)
        y = sampler(y_min, y_max, N_spat).astype(np.float32)
        t = t[1:]
        y = y[1:-1]

    x = np.ones_like(y) * x
    # create the spatial grid, add degenerate temporal axis
    omega = np.expand_dims(np.stack([x, y], axis=1), axis=1)
    print(omega.shape)

    # add a temporal axis
    spat_slice = tile_time(omega, t)

    return spat_slice


def y_slice(y, t_max, N_spat, N_temp, with_boundary=True, x_min=-1, x_max=1):
    if with_boundary:
        t = sampler(0, t_max, N_temp).astype(np.float32)
        x = sampler(x_min, x_max, N_spat).astype(np.float32)
    else:
        N_spat += 2
        N_temp += 1
        t = sampler(0, t_max, N_temp).astype(np.float32)
        x = sampler(x_min, x_max, N_spat).astype(np.float32)
        t = t[1:]
        x = x[1:-1]

    y = np.ones_like(x) * y
    # create the spatial grid, add degenerate temporal axis
    omega = np.expand_dims(np.stack(np.meshgrid(x, y), axis=2), axis=2)

    # add a temporal axis
    spat_slice = tile_time(omega, t)

    return spat_slice


def initial_cond_dataset(N_ic, f0, x_min=-1, x_max=1, y_min=-1, y_max=1):
    data = temporal_slice(t=0,
                          N=N_ic,
                          x_min=x_min,
                          x_max=x_max,
                          y_min=y_min,
                          y_max=y_max)
    labels = img_sampler(f0, data, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    data = data.astype(np.float32)
    return torch.from_numpy(data), torch.from_numpy(labels)


def make_datasets(t_max, N_spat, N_temp, N_ic, N_bound, f0,
                  x_min=-1, x_max=1, y_min=-1, y_max=1, t_sample_method='uniform', save=False, path=''):
    '''
    generates the 3 datasets for the interior points.
    Returns:
    interior, boundary, initial condition datasets
    '''
    interior = interior_dataset(t_max=t_max,
                                N_spat=N_spat,
                                N_temp=N_temp,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max,
                                t_sample_method=t_sample_method)
    interior = interior.reshape(-1, 3)

    boundary = boundary_dataset(t_max=t_max,
                                N_bound=N_bound,
                                N_temp=N_temp,
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max,
                                t_sample_method=t_sample_method)
    top = boundary[0].reshape(-1, 3)
    bottom = boundary[1].reshape(-1, 3)
    left = boundary[2].reshape(-1, 3)
    right = boundary[3].reshape(-1, 3)
    boundary = torch.stack([top, bottom, left, right], dim=0)

    init_cond, labels = initial_cond_dataset(N_ic=N_ic,
                                             f0=f0,
                                             x_min=x_min,
                                             x_max=x_max,
                                             y_min=y_min,
                                             y_max=y_max)
    init_cond = init_cond.reshape(-1, 3)
    labels = labels.reshape(-1, 1)

    if save:
        f = open(path + '/data', 'wb')
        pickle.dump([interior, boundary, init_cond, labels], f)
        f.close()
        f = open(path + '/README.txt', 'w')
        f.write("A list holding the tensor datasets and the f0 image ordered as:\n")
        f.write("[interior, boundary, init_cond, f0]")
        f.close()

    return interior, boundary, init_cond, labels


def dataloaders(interior_data, boundary_data, init_cond_data, init_cond_labels,
                int_batch_size, shuff=True, show=True):
    '''
    generates the 3 dataloaders for the interior points. we need to match
    the number of batches for each dataset.
    Returns:
    int_dloader,bc_dloader ,ic_dloader
    :param show:
    '''
    int_dloader = FastTensorDataLoader(interior_data, \
                                       batch_size=int_batch_size, shuffle=shuff)

    # match batch size
    num_batches = len(int_dloader)
    bc_batch_size = boundary_data.shape[1] // num_batches
    ic_batch_size = init_cond_data.shape[0] // num_batches

    bc_dloader = FastTensorDataLoader(boundary_data[0],
                                      boundary_data[1],
                                      boundary_data[2],
                                      boundary_data[3],
                                      batch_size=bc_batch_size,
                                      shuffle=shuff)

    ic_dloader = FastTensorDataLoader(init_cond_data,
                                      init_cond_labels,
                                      batch_size=ic_batch_size,
                                      shuffle=shuff)

    if len(int_dloader) != len(bc_dloader) or len(int_dloader) != len(ic_dloader):
        raise ValueError('Size mismatch in number of batches!')

    if show:
        print('batch sizes: (interior, boundary, initial condition):')
        print(int_batch_size, bc_batch_size, ic_batch_size)
        print('number of batchs:', len(int_dloader))

    return int_dloader, bc_dloader, ic_dloader


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def sampler(start: float, stop: float, num_samples: int = 100, method: str = 'uniform', beta=-5):
    """
    Wrapper for 1D sampling methods
    :param start: startpoint of sampling
    :param stop: endpoint of sampling
    :param method: sampling method. 'uniform', 'exp'. 'exp' samples exponenets in [0,1] and then
    transforms to the desired range [start,stop]
    :param beta: decay parameter for exponential sampling
    :return: 1D array
    """

    if method == 'exp':
        samples = np.linspace(0, 1, num=num_samples)
        samples = np.exp(beta * samples)
        samples = np.flip((samples - samples[-1]) * stop / (1 - samples[-1]) + start)

    if method == 'uniform':
        samples = np.linspace(start=start, stop=stop, num=num_samples)

    return samples


def generate_train_data(config_dict):
    print('\nGenerating dataset, dataloaders...')
    f0 = init_f0(config_dict['cam_man'])

    interior, boundary, init_cond, labels = make_datasets(t_max=config_dict['t_max'],
                                                          x_min=config_dict['x_min'],
                                                          y_min=config_dict['y_min'],
                                                          x_max=config_dict['x_max'],
                                                          y_max=config_dict['y_max'],
                                                          N_spat=config_dict['N_spat'],
                                                          N_temp=config_dict['N_temp'],
                                                          N_ic=config_dict['N_ic'],
                                                          N_bound=config_dict['N_bound'],
                                                          f0=f0,
                                                          t_sample_method=config_dict['sample_method'])
    dloader_int, dloader_bc, dloader_ic = dataloaders(interior, boundary, init_cond, labels, config_dict["batch_size"])

    return dloader_int, dloader_bc, dloader_ic, f0


def DGM_sample(batch_size, x_min, x_max, y_min, y_max, t_max, img_name='cam_man'):
    """
    Sample the interior and boundary
    :param config_dict: confguration of expriments
    :return:
    """
    # interior sample
    X_int = torch.from_numpy(range_sampler(x_min, x_max, y_min, y_max, t_max, size=batch_size, eps=1e-6))

    # intial condition sampler
    X_ic = range_sampler(x_min, x_max, y_min, y_max, t_max=0, size=batch_size, eps=1e-6)
    if img_name == 'cam_man':
        f0 = init_f0()
        y_ic = img_sampler(f0=f0, coords=X_ic, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    X_ic = torch.from_numpy(X_ic)
    y_ic = torch.from_numpy(y_ic)

    # boundary condition sampler
    top = torch.from_numpy(range_sampler(x_min, x_max, y_max, y_max, t_max=0, size=batch_size//4, eps=0))
    bottom = torch.from_numpy(range_sampler(x_min, x_max, y_min, y_min, t_max=0, size=batch_size//4, eps=0))
    left = torch.from_numpy(range_sampler(x_min, x_min, y_min, y_max, t_max=0, size=batch_size//4, eps=0))
    right = torch.from_numpy(range_sampler(x_max, x_max, y_min, y_max, t_max=0, size=batch_size // 4, eps=0))
    X_bound = torch.stack([top, bottom, left, right], dim=0)

    int_dloader, bc_dloader, ic_dloader = dataloaders(X_int, X_bound, X_ic, y_ic, batch_size, show=False)
    return int_dloader, bc_dloader, ic_dloader


def range_sampler(x_min, x_max, y_min, y_max, t_max, size, t_min=0, eps=1e-6):
    sample = -1 * (np.random.rand(size, 3) - 1)  # random sample from (0, 1]
    if x_min == x_max:
        sample[..., 0] = x_max
    if y_min == y_max:
        sample[..., 1] = y_max
    if t_min == t_max:
        sample[..., 2] = t_max

    # transform sample to open domain
    sample[..., 0] = (x_max - x_min - eps) * sample[..., 0] + x_min
    sample[..., 1] = (y_max - y_min - eps) * sample[..., 1] + y_min
    sample[..., 2] = (t_max - t_min - eps) * sample[..., 2] + t_min

    return sample.astype(np.float32)
