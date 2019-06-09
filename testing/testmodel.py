import numpy as np
import model as mod
import tensorflow as tf

inputData=np.ones((10,300,25), dtype=np.float32)
labelData=np.ones((10,3))

model=mod.Transformer(num_layers=1, d_model=512, num_heads=8, dff=2048, target_size=3)
#out=model(inputData, False, None)
#print('Output', out.shape)

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

@tf.function
def train_step(x_batch, y_batch):
  with tf.GradientTape() as tape:
    predictions = model(x_batch, False, None)
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

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")

for ep in range(1):
    train_step(inputData, labelData)
    ckpt.step.assign_add(1)
    print('Loss: {:.2f} | accuracy: {:.2f}'.format(train_loss.result(),train_accuracy.result()*100))
    if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))
