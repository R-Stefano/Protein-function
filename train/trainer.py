import tensorflow as tf
import prepare.tfapiConverter as tfconv

import os
import multiprocessing
from absl import app
from absl import flags
import train.model as mod

FLAGS = flags.FLAGS

dataPath=FLAGS.dataPath
modelPath='./train/ckpt'
model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.target_size)

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(x_batch, y_batch):
  with tf.GradientTape() as tape:
    predictions = model(x_batch, True, None)
    loss = loss_object(y_batch, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y_batch, predictions)

@tf.function
def test_step(x_batch, y_batch):
  predictions = model(x_batch, False, None)
  t_loss = loss_object(y_batch, predictions)

  test_loss(t_loss)
  test_accuracy(y_batch, predictions)

def train():
    filenames=[dataPath+fn for fn in os.listdir(dataPath)]
    test_size=int(FLAGS.num_examples*FLAGS.ratio_test_examples)
    train_size=FLAGS.num_examples - test_size

    raw_dataset=tf.data.TFRecordDataset(filenames)
    #raw_dataset = tf.data.Dataset.from_tensor_slices((fileData))

    print('train size:',train_size)
    print('test size:',test_size)    

    #convert batch data from tfrecord to tensors during iteration
    #num_parallel_calls: distribute the preprocessing (decoding, hot encoding) across cpus
    dataset = raw_dataset.map(tfconv.decodeTFRecord, num_parallel_calls=multiprocessing.cpu_count())
    #or tf.data.experimental.AUTOTUNEif you prefer

    #Shuffle dataset once
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=FLAGS.reshuffle_iteration, seed=0)

    #Number of iterations (repeat the whole dataset)
    #dataset=dataset.repeat(epoches)

    #create train/test datasets and create batches
    test_set = dataset.take(test_size).batch(FLAGS.batch_size_test)
    train_set = dataset.skip(test_size).batch(FLAGS.batch_size_train)

    #Store in memory buffer_size examples to be ready to be fed to the GPU
    #load batch*2 examples for the train step (T+1) while GPU occupied with train step (T)
    test_set = test_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, modelPath, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for ep in range(1, FLAGS.epoches +1):
        print('\nEpoch ({}/{})'.format(ep, FLAGS.epoches))
        print('Training..')
        for i, batch in enumerate(train_set):
            if i%500==0:
                print('Train batch ({}/{})'.format(i, train_size//FLAGS.batch_size_train))
            train_step(batch['X'], batch['Y'])

        print('Testing..')
        for i, batch in enumerate(test_set):
            if i%500==0:
                print('Test batch ({}/{})'.format(i, test_size//FLAGS.batch_size_test))
            test_step(batch['X'], batch['Y'])
        
        message='\nEpoch {} | Loss: {:.2f} | Accuracy: {:.2f} | Test Loss: {:.2f} | Test Accuracy: {:.2f} '
        print(message.format(ep, train_loss.result(),train_accuracy.result()*100,
                                 test_loss.result(),test_accuracy.result()*100))

        if int(ckpt.step)%10==0:
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))

        ckpt.step.assign_add(1)
        
