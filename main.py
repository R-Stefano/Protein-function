import tensorflow as tf
import pickle
import numpy as np
import sys
from absl import app
from absl import flags
import yaml
'''
Let's suppose I have a dataset of 

X data (13802, 512, 25)
Y data (13802, 3)

How I have them ready to feed into my network?

.TFRecordDataset(filenames) if files in .tfrecord format
.TextLineDataset(filenames) if files in .txt format

filter out some data. could be useful to avoid to store the wholle matricies on disk
dataset = dataset.filter(filter_fn)
'''

with open("hyperparams.yaml", 'r') as stream:
    hyperparams = yaml.safe_load(stream)


FLAGS = flags.FLAGS

flags.DEFINE_integer('epoches', 2, 'Training epoches')
flags.DEFINE_integer('batch_size_train', 32, 'Size of batch size for training')
flags.DEFINE_integer('batch_size_test', 64, 'Size of batch size for testing')
flags.DEFINE_integer('shuffle_buffer_size', 1000, 'Number of examples to load before shuffling')
flags.DEFINE_boolean('reshuffle_iteration', False, 'Shuffle examples at each epoch')

flags.DEFINE_integer('num_layers', 1, 'Number of layers in the Transformer')
flags.DEFINE_integer('d_model', 512, 'Dimension of the vecs in Transfomer layer, must be d_model%num_heads=0')
flags.DEFINE_integer('num_heads', 8, 'Number of attention heads')
flags.DEFINE_integer('fc', 2048, 'Number neurons in the fully connected layer after attention')

flags.DEFINE_integer('unique_labels', hyperparams['unique_labels'], 'Length of the label hot vec')
flags.DEFINE_integer('unique_aminos', len(hyperparams['unique_aminos']), 'Length of the label hot vec')
flags.DEFINE_integer('max_length_aminos', hyperparams['max_length_aminos'], 'Length of the label hot vec')

flags.DEFINE_string('dataPath', 'prepare/data/', 'Path for the data')
flags.DEFINE_string('mode', None, '\ntrain: to start training the model')
flags.mark_flag_as_required('mode')

FLAGS(sys.argv)

import train.trainer as trainer
import prepare.createDataset as createDataset


def main(argvs):
    if (FLAGS.mode=='train'):
        trainer.train()
    elif (FLAGS.mode=='createdata'):
        createDataset.createDataset()
    else:
        print('No mode selected')

if __name__ == '__main__':
    app.run(main)


