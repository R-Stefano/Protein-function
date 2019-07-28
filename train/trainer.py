import tensorflow as tf
import os
import multiprocessing
from absl import flags
import time
import numpy as np
from tensorflow import keras

import prepare.tfapiConverter as tfconv
import evaluate.custom_metrics as custom

FLAGS = flags.FLAGS

dataPath=FLAGS.dataPath
ckptsPath='train/ckpt'
savedModelPath=FLAGS.savedModelPath
logsPath='train/logs'

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adadelta()

#Loss Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

#Metrics
thresholds=np.arange(start=0.1, stop=1.0, step=0.1)

train_precision = tf.keras.metrics.Precision(name='train_precision')
train_recall = tf.keras.metrics.Recall(name='train_recall')
train_f1_max =custom.F1MaxScore(thresholds, name="train_f1_max")

test_precision= tf.keras.metrics.Precision(name='test_precision')
test_recall= tf.keras.metrics.Recall(name='test_recall')
test_f1_max=custom.F1MaxScore(thresholds, name="test_f1_max")

def train(model):
    @tf.function
    def train_step(x_batch, y_batch):
        #GradientTape traces operations to compute gradients later
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_object(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_precision(y_batch, predictions)
        train_recall(y_batch, predictions)
        train_f1_max(y_batch, predictions)

    @tf.function
    def test_step(x_batch, y_batch):
        predictions = model(x_batch, training=False)
        t_loss = loss_object(y_batch, predictions)

        test_loss(t_loss)
        test_precision(y_batch, predictions)
        test_recall(y_batch, predictions)
        test_f1_max(y_batch, predictions)

    train_path=dataPath+'train/'
    test_path=dataPath+'test/'
    train_files=[train_path+fn for fn in os.listdir(train_path)]
    test_files=[test_path+fn for fn in os.listdir(test_path)]

    train_dataset=tf.data.TFRecordDataset(train_files)
    test_dataset=tf.data.TFRecordDataset(test_files)

    #convert batch data from tfrecord to tensors during iteration
    #num_parallel_calls: distribute the preprocessing (decoding, hot encoding) across cpus
    train_dataset = train_dataset.map(tfconv.decodeTFRecord, num_parallel_calls=multiprocessing.cpu_count())
    test_dataset = test_dataset.map(tfconv.decodeTFRecord, num_parallel_calls=multiprocessing.cpu_count())
    #or tf.data.experimental.AUTOTUNEif you prefer

    #Shuffle dataset once
    train_dataset = train_dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=FLAGS.reshuffle_iteration, seed=0)

    #Number of iterations (repeat the whole dataset)
    #dataset=dataset.repeat(epoches)

    #create create batches
    train_set = train_dataset.batch(FLAGS.batch_size_train)
    test_set = test_dataset.batch(FLAGS.batch_size_test)

    #Store in memory buffer_size examples to be ready to be fed to the GPU
    #load batch*2 examples for the train step (T+1) while GPU occupied with train step (T)
    train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckptsPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_summary_writer = tf.summary.create_file_writer(os.path.join(logsPath, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(logsPath, 'test'))

    for ep in range(1, FLAGS.epoches +1):
        print('\nEpoch ({}/{})'.format(ep, FLAGS.epoches))
        start=time.time()
        #training
        for i, batch in enumerate(train_set):
            train_step(batch['X'], batch['Y'])
            
            if (i%200==0):
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=int(ckpt.step))
                    tf.summary.scalar('precision', train_precision.result(), step=int(ckpt.step))
                    tf.summary.scalar('recall', train_recall.result(), step=int(ckpt.step))
                    tf.summary.scalar('f1_max', train_f1_max.result(), step=int(ckpt.step))

                    train_loss.reset_states()
                    train_precision.reset_states()
                    train_recall.reset_states()
                    train_f1_max.reset_states()
                    
            ckpt.step.assign_add(1)            

        #testing
        for i, batch in enumerate(test_set):
            test_step(batch['X'], batch['Y'])

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=int(ckpt.step))
            tf.summary.scalar('precision', test_precision.result(), step=int(ckpt.step))
            tf.summary.scalar('recall', test_recall.result(), step=int(ckpt.step))
            tf.summary.scalar('f1_max', test_f1_max.result(), step=int(ckpt.step))

            test_loss.reset_states()
            test_precision.reset_states()
            test_recall.reset_states()
            test_f1_max.reset_states()

        message='Epoch {} in {:.2f} secs'
        print(message.format(ep, (time.time()-start)))

        save_path = manager.save()
        print("Saved checkpoint | global step {}: {}".format(int(ckpt.step), save_path))

        if (ep%10==0):
            print('Saving weights')
            model.save_weights(savedModelPath, save_format='tf')

    print('Saving final model..')
    #How to save a model developed using Imperative style
    model.save_weights(savedModelPath, save_format='tf')
        
