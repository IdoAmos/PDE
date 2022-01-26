import utils
import main

config = utils.default_parameter_generator()

config['N_spat'] = 8
config['N_temp'] = 8
config['N_bound'] = 8
config['N_ic'] = 8
config['MAX_EPOCH'] = 5
config['batch_size'] = 1000
config['lr'] = 5e-5

config['model_name'] = 'Siren'
config['num_layers'] = 2
config['hidden_features'] = 256

# config['C'].pop('fid_inf')
# config['C'].pop('ic_inf')
# config['P'].pop('fid_inf')
# config['P'].pop('ic_inf')
config['weight_decay'] = 1e-6
config['sched_dict'] = dict(type='reduce_on_plat', patience=20, gamma=0.8, threshold=0.01)

config['checkpoint'] = True
config['checkpoint_path'] = 'model.pt'

for key, item in config.items():
    print(key, item)

exp_name = 'test'
model, hist = main.main(save=True, dest=exp_name, config_dict=config, load=False, show=True)
