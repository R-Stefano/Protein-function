import tensorflow as tf
import os
import multiprocessing
from absl import flags
import time
import numpy as np
from tensorflow import keras
import train.model as mod
import prepare.tfapiConverter as tfconv

FLAGS = flags.FLAGS

dataPath=FLAGS.dataPath
modelPath='./train/ckpt'
model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.unique_labels)

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(x_batch, y_batch):
    #GradientTape traces operations to compute gradients later
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        loss = loss_object(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_batch, predictions)

@tf.function
def test_step(x_batch, y_batch):
  predictions = model(x_batch)
  t_loss = loss_object(y_batch, predictions)

  test_loss(t_loss)
  test_accuracy(y_batch, predictions)

def train():
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
    manager = tf.train.CheckpointManager(ckpt, modelPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    #train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
    #with train_summary_writer.as_default():
        #tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)

    for ep in range(1, FLAGS.epoches +1):
        print('\nEpoch ({}/{})'.format(ep, FLAGS.epoches))
        start=time.time()
        #training
        for i, batch in enumerate(train_set):
            train_step(batch['X'], batch['Y'])
            ckpt.step.assign_add(1)

        #testing
        for i, batch in enumerate(test_set):
            test_step(batch['X'], batch['Y'])

        message='\nEpoch {} in {:.2f} secs | Loss: {:.2f} | Accuracy: {:.2f} | Test Loss: {:.2f} | Test Accuracy: {:.2f}'
        print(message.format(ep, (time.time()-start),train_loss.result(),train_accuracy.result()*100,
                                 test_loss.result(),test_accuracy.result()*100))

        save_path = manager.save()
        print("Saved checkpoint | global step {}: {}".format(int(ckpt.step), save_path))

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    print('Saving final model..')
    #How to save a model developed using Imperative style
    model.save_weights("train/saved_model/module_without_signature", save_format='tf')
        
