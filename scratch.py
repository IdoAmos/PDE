from Models import *
import Data

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import numpy.fft as fft
import scipy.stats as stats


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def format_x_ticks(x, pos):
    """Format odd tick positions
    """
    return eformat(x, 0, 1)


def format_y_ticks(x, pos):
    """Format odd tick positions
    """
    return eformat(x, 0, 1)


def get_spectrum(activations):
    n = activations.shape[0]

    spectrum = fft.fft(activations.numpy().astype(np.double).sum(axis=-1), axis=0)[:n // 2]
    spectrum = np.abs(spectrum)

    max_freq = 100
    freq = fft.fftfreq(n, 2. / n)[:n // 2]
    return freq[:max_freq], spectrum[:max_freq]


def plot_all_activations_and_grads(activations):
    num_cols = 4
    num_rows = len(activations)

    fig_width = 5.5
    fig_height = num_rows / num_cols * fig_width
    fig_height = 9

    fontsize = 5

    fig, axs = plt.subplots(num_rows, num_cols, gridspec_kw={'hspace': 0.3, 'wspace': 0.2},
                            figsize=(fig_width, fig_height), dpi=300)

    axs[0][0].set_title("Activation Distribution", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][1].set_title("Activation Spectrum", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][2].set_title("Gradient Distribution", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][3].set_title("Gradient Spectrum", fontsize=7, fontfamily='serif', pad=5.)

    x_formatter = matplotlib.ticker.FuncFormatter(format_x_ticks)
    y_formatter = matplotlib.ticker.FuncFormatter(format_y_ticks)

    spec_rows = []
    for idx, (key, value) in enumerate(activations.items()):
        grad_value = value.grad.cpu().detach().squeeze(0)
        flat_grad = grad_value.view(-1)
        axs[idx][2].hist(flat_grad, bins=256, density=True)

        value = value.cpu().detach().squeeze(0)  # (1, num_points, 256)
        n = value.shape[0]
        flat_value = value.view(-1)

        axs[idx][0].hist(flat_value, bins=256, density=True)

        if idx > 1:
            if not (idx) % 2:
                x = np.linspace(-1, 1., 500)
                axs[idx][0].plot(x, stats.arcsine.pdf(x, -1, 2),
                                 linestyle=':', markersize=0.4, zorder=2)
            else:
                mu = 0
                variance = 1
                sigma = np.sqrt(variance)
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 500)
                axs[idx][0].plot(x, stats.norm.pdf(x, mu, sigma),
                                 linestyle=':', markersize=0.4, zorder=2)

        activ_freq, activ_spec = get_spectrum(value)
        axs[idx][1].plot(activ_freq, activ_spec)

        grad_freq, grad_spec = get_spectrum(grad_value)
        axs[idx][-1].plot(grad_freq, grad_spec)

        for ax in axs[idx]:
            ax.tick_params(axis='both', which='major', direction='in',
                           labelsize=fontsize, pad=1., zorder=10)
            ax.tick_params(axis='x', labelrotation=0, pad=1.5, zorder=10)

            ax.xaxis.set_major_formatter(x_formatter)
            ax.yaxis.set_major_formatter(y_formatter)



num_ch = [64, 32, 16]
hidden_layers = len(num_ch) - 1

model = ConvSiren(in_features=2, num_ch=num_ch, hidden_layers=hidden_layers,
                  out_features=1, nhood=3)

input_signal = torch.from_numpy(Data.range_sampler(-1, 1, -1, 1, 0.001, size=100))[..., :-1]
activations = model.forward_with_activations(input_signal, retain_grad=True)
output = activations[next(reversed(activations))]

# Compute gradients. Because we have retain_grad=True on
# activations, each activation stores its own gradient!
output.mean().backward()

plot_all_activations_and_grads(activations)