import tensorflow as tf
import yaml
import sys
import numpy as np

with open('../../hyperparams.yaml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import CPC as Model
import CPC_tuned as Model_tuner

models_dir=hyperparams['models_dir']
data_dir=hyperparams['data_dir']

train_dataset_input=np.load(data_dir+'dataset/train/input_dataset_0.npy', allow_pickle=True)
train_dataset_label=np.load(data_dir+'dataset/train/label_dataset_0.npy', allow_pickle=True)

test_dataset_input=np.load(data_dir+'dataset/test/input_dataset_0.npy', allow_pickle=True)
test_dataset_label=np.load(data_dir+'dataset/test/label_dataset_0.npy', allow_pickle=True)

print('Train dataset:', train_dataset_input.shape)
print('Train dataset:', len(train_dataset_input[0]))
print('label dataset:', train_dataset_label.shape)
print('label dataset:', len(train_dataset_label[0]))

#Prepare CPC
model_utils=Model.Model()
custom_objects={
	'custom_xent':model_utils.custom_xent,
    'custom_accuracy': model_utils.custom_accuracy
}
print('>Loading model..')
model=tf.keras.models.load_model(models_dir+model_utils.name+'model.h5', custom_objects=custom_objects)
'''
model.summary()
for layer in model.layers:
    print(layer.name)
'''

learning_rate=0.00001
epochs=10
batch_size=128

#Instantiate fine_tuner
cpc_tuner_utils=Model_tuner.Model(learning_rate)
cpc_tuner=cpc_tuner_utils.architecture(model, 1918)

train_generator=cpc_tuner_utils.prepareBatch(train_dataset_input, train_dataset_label, batch_size)
test_generator=cpc_tuner_utils.prepareBatch(test_dataset_input, test_dataset_label, batch_size)

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=models_dir+cpc_tuner_utils.name+'logs/', histogram_freq=1, profile_batch = 3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=models_dir+cpc_tuner_utils.name+'model.{epoch:02d}-{val_loss:.2f}.hdf5' ,
        monitor='val_custom_accuracy', 
        load_weights_on_restart=True, 
        save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_custom_accuracy', patience=3)
]

cpc_tuner.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.save(models_dir+cpc_tuner_utils.name+'model.h5')