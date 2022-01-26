import glob

import numpy as np

import func
import pickle
import utils
import matplotlib.pyplot as plt
import torch

path = '/Users/idoamos/Desktop/Projects/Nir - Project/exp_eff'

ssim_plots = []
psnr_plots = []
titles = []
for counter, subdir in enumerate(glob.glob(path+'/*')):

    print(subdir)
    configs = pickle.load(open(subdir+'/config_dict', 'rb'))
    titles.append('depth/width: {}/{}'.format(configs['num_layers'], configs['hidden_features']))
    model = utils.assign_model(configs)
    model.load_state_dict(torch.load(subdir+'/weights', map_location=torch.device('cpu')))

    ssim_plot = func.ssim_by_time(N=configs['N_ic'], model=model, t_max=configs['t_max'], num_points=15)
    ssim_plots.append(ssim_plot)
    psnr_plot = func.psnr_by_time(N=configs['N_ic'], model=model, t_max=configs['t_max'], num_points=15)
    psnr_plots.append(psnr_plot)



color = plt.cm.rainbow(np.linspace(0, 1, len(ssim_plots)))
print(color)
plt.figure()
i = 0
for plot, label in zip(ssim_plots, titles):
    plt.plot(plot[0], plot[1], label=label, c=color[i])
    plt.scatter(plot[0], plot[1], c=color[i])
    i += 1
plt.grid('on')
plt.legend()
plt.suptitle('SSIM vs. T for Varying Model Capacity')
plt.draw()

color = plt.cm.rainbow(np.linspace(0, 1, len(ssim_plots)))
plt.figure()
i = 0
for plot, label in zip(psnr_plots, titles):
    plt.plot(plot[0], plot[1], label=label, c=color[i])
    plt.scatter(plot[0], plot[1], c=color[i])
    i += 1

plt.grid('on')
plt.legend()
plt.draw()
plt.suptitle('PSNR vs. T for Varying Model Capacity')

plt.show()