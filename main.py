import tensorflow as tf
import pickle
import numpy as np
import prepare.tfapiConverter as tfconv
import os

from absl import app
from absl import flags
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


FLAGS = flags.FLAGS

flags.DEFINE_integer('epoches', 2, 'Training epoches')
flags.DEFINE_integer('num_examples', 6, 'Number of examples in the dataset')
flags.DEFINE_float('ratio_test_examples', 0.4, 'Ratio of examples for testing')
flags.DEFINE_integer('batch_size_train', 32, 'Size of batch size for training')
flags.DEFINE_integer('batch_size_test', 64, 'Size of batch size for testing')
flags.DEFINE_integer('shuffle_buffer_size', 50000, 'Number of examples to load before shuffling')
flags.DEFINE_boolean('reshuffle_iteration', False, 'Shuffle examples at each epoch')


def memoryDataset():
    print('Loading all dataset in memroy method...')
    with open("data/dataset_batch_0", "rb") as fp:
        fileData=pickle.load(fp)

    #file data is a dict X: [examples, aminos, hot] Y:[examples, hot]
    #.batch(N) allows, when dataset is iterated to retrieve N examples
    dataset = tf.data.Dataset.from_tensor_slices((fileData)).batch(32)#.shuffle(10000).batch(32)

    #transformations of dataset, check Dataset.map(), Dataset.flat_map(), and Dataset.filter()

    #the number of examples in batch has been defined earlier with .batch()
    for batch in dataset:
        print(batch['X'].shape)
        print(batch['Y'].shape)
        break

def shardsDataset(filenames):
    test_size=int(FLAGS.num_examples*FLAGS.ratio_test_examples)
    train_size=FLAGS.num_examples - test_size

    print('train size:',train_size)
    print('test size:',test_size)    

    raw_dataset=tf.data.TFRecordDataset(filenames) 

    #convert tfrecords to tensors
    dataset = raw_dataset.map(tfconv.decodeTFRecord)

    #Shuffle dataset once
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=FLAGS.reshuffle_iteration, seed=0)

    #Number of iterations (repeat the whole dataset)
    #dataset=dataset.repeat(epoches)

    #create train/test datasets and create batches
    test_set = dataset.take(test_size).batch(FLAGS.batch_size_test)
    train_set = dataset.skip(test_size).batch(FLAGS.batch_size_train)

    #Store in memory buffer_size examples to be ready to be fed to the GPU
    #dataset = dataset.prefetch(AUTOTUNE)

    for ep in range(FLAGS.epoches):
        print('\n\nEpoch', ep)
        print('Training..')
        for i, batch in enumerate(train_set):
            print('\nTrain batch:', i)       
            print('Input data:',batch['X'].shape)
            print('Label data:',batch['Y'].shape)

        print('Testing..')
        for i, batch in enumerate(test_set):
            print('\nTest batch:', i)       
            print('Input data:',batch['X'].shape)
            print('Label data:',batch['Y'].shape)


def main(argvs):
    '''
    for i,f in enumerate(['test1', 'test2', 'test3']):
        examples=np.asarray([[np.ones((3,5), dtype=np.uint8)*i, np.identity(5, dtype=np.uint8)[i]],
                            [np.ones((3,5), dtype=np.uint8)*i, np.identity(5, dtype=np.uint8)[i]]
                            ])

        examples_x=examples[:,0]
        examples_y=examples[:,1]
        tfconv.generateTFRecord(examples_x, examples_y, (f+'.tfrecords'))
    
    filenames=['test1.tfrecords', 'test2.tfrecords', 'test3.tfrecords']
    '''
    filenames=['prepare/data/'+fn for fn in os.listdir('prepare/data')]
    
    shardsDataset(filenames)

if __name__ == '__main__':
    app.run(main)


