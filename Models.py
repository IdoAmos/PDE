import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.num_layers = hidden_layers + 2  # 2 for first and last
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ResOp(nn.Module):
    def __init__(self, in_features, hidden_features, ic_layers, op_layers, out_features,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.train_init = True
        self.ic_head = []  # this head aims to solve only the initial condition
        self.op_head = []  # this head aims to solve the residual of the i.c from the current state

        # ic head does not expect time axis
        self.ic_head.append(SineLayer(in_features - 1, hidden_features,
                                      is_first=True, omega_0=first_omega_0))
        for i in range(ic_layers - 1):
            self.ic_head.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))

        self.op_head.append(SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0))

        for i in range(op_layers - 1):
            self.op_head.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))

        ic_final_layer = nn.Linear(hidden_features, out_features)
        op_final_layer = nn.Linear(hidden_features, out_features)

        with torch.no_grad():
            ic_final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                           np.sqrt(6 / hidden_features) / hidden_omega_0)
            op_final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / (1.5 * hidden_omega_0),
                                           np.sqrt(6 / hidden_features) / (1.5 * hidden_omega_0))

        self.ic_last = ic_final_layer
        self.op_last = op_final_layer
        self.ic_head = nn.Sequential(*self.ic_head)
        self.op_head = nn.Sequential(*self.op_head)

    def full_train_mode(self):
        self.train_init = False

        for w in self.ic_head.parameters():
            w.requires_grad = False
        for w in self.ic_last.parameters():
            w.requires_grad = False

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        intermediate = self.ic_head(coords[..., :-1])
        output = self.ic_last(intermediate)

        if not self.train_init:
            output = output + self.op_last(self.op_head(coords))

        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class ResOpHidden(nn.Module):
    def __init__(self, in_features, hidden_features, ic_layers, op_layers, out_features,
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.train_init = True
        self.ic_head = []  # this head aims to solve only the initial condition
        self.op_head = []  # this head aims to solve the residual of the i.c from the current state

        # ic head does not expect time axis
        self.ic_head.append(SineLayer(in_features - 1, hidden_features,
                                      is_first=True, omega_0=first_omega_0))
        for i in range(ic_layers - 1):
            self.ic_head.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))

        self.spatial_head = SineLayer(in_features - 1, hidden_features,
                                      is_first=True, omega_0=first_omega_0)
        self.temporal_head = SineLayer(in_features=1, out_features=hidden_features,
                                       is_first=True, omega_0=first_omega_0, bias=False)

        for i in range(op_layers - 1):
            self.op_head.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0, bias=False))

        ic_final_layer = nn.Linear(hidden_features, out_features)

        with torch.no_grad():
            ic_final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                           np.sqrt(6 / hidden_features) / hidden_omega_0)

        self.ic_last = ic_final_layer
        self.ic_head = nn.Sequential(*self.ic_head)
        self.op_head = nn.Sequential(*self.op_head)

    def full_train_mode(self):
        self.train_init = False

        for w in self.ic_head.parameters():
            w.requires_grad_(False)
        for w in self.ic_last.parameters():
            w.requires_grad_(False)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        intermediate = self.ic_head(coords[..., :-1])

        if not self.train_init:
            h_spatial = self.spatial_head(coords[..., :-1])
            h_temporal = self.temporal_head(coords[..., -1].unsqueeze(dim=-1))
            h = torch.mul(h_spatial, h_temporal)
            intermediate = intermediate + self.op_head(h)

        output = self.ic_last(intermediate)

        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
