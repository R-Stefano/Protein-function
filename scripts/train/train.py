import tensorflow as tf
import os
import multiprocessing
import sys
import yaml

with open('../../hyperparams.yaml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']
train_data_dir=configs['train_data_dir']
test_data_dir=configs['test_data_dir']
model_dir=configs['model_dir']
seed=configs['seed']

epochs=configs['train']['epochs']
buffer_size=configs['train']['buffer_size']
batch_size=configs['train']['batch_size']
train_size=configs['train']['train_size']
test_size=configs['train']['test_size']
learning_rate=configs['train']['learning_rate']

timesteps=configs['model']['timesteps']
encoding_vec=configs['model']['encoding_vec']
num_labels=configs['model']['num_labels']

shared_scripts_dir=configs['shared_scripts_dir']
sys.path.append(shared_scripts_dir)
import tfapiConverter as tfconv
import model as model

model_utils=model.ModelInitializer(timesteps, encoding_vec, num_labels, learning_rate)
model=model_utils.architecture()

train_files=[train_data_dir+fn for fn in os.listdir(train_data_dir)]
test_files=[test_data_dir+fn for fn in os.listdir(test_data_dir)]

train_dataset=tf.data.TFRecordDataset(train_files)
test_dataset=tf.data.TFRecordDataset(test_files)

#convert batch data from tfrecord to tensors during iteration
#num_parallel_calls: distribute the preprocessing (decoding, hot encoding) across cpus
train_dataset = train_dataset.map(tfconv.decodeTFRecord, num_parallel_calls=multiprocessing.cpu_count())
test_dataset = test_dataset.map(tfconv.decodeTFRecord, num_parallel_calls=multiprocessing.cpu_count())

#Shuffle dataset once
train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True, seed=seed)

#Iterations, split in batches, set buffer size
train_dataset = train_dataset.repeat(epochs).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.repeat(epochs).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=model_dir+model_utils.name+'model.{epoch:02d}-{val_loss:.2f}.hdf5' ,
    monitor='val_loss', 
    load_weights_on_restart=True, 
    save_best_only=True)

model.fit(train_dataset,
    steps_per_epoch=(train_size//batch_size)-1,
    epochs=epochs,
    validation_data=test_dataset,
    validation_steps=(test_size//batch_size)-1,
    callbacks=[checkpoint_callback],
    verbose=2)

model.save(model_dir+model_utils.name+'model.h5')