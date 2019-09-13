import tensorflow as tf
import yaml
import sys

with open('../hyperparams.yaml', 'r') as f:
    configs=yaml.load(f)

model_dir=configs['model_dir']
shared_scripts_dir=configs['shared_scripts_dir']
sys.path.append(shared_scripts_dir)
import model as model

learning_rate=configs['train']['learning_rate']

timesteps=configs['model']['timesteps']
encoding_vec=configs['model']['encoding_vec']
num_labels=configs['model']['num_labels']

model_utils=model.ModelInitializer(timesteps, encoding_vec, num_labels, learning_rate)
model=model_utils.architecture()

model.load_weights(model_dir+model_utils.name+'model.02-0.03.hdf5')

model.save((model_dir+model_utils.name+'model.h5'))