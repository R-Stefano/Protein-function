import yaml
import tensorflow as tf
import numpy as np
bce = tf.keras.losses.BinaryCrossentropy()

labels=[0,1]
preds=[0.2,0.8]

labels=tf.cast(labels, tf.float32)
preds=tf.cast(preds, tf.float32)



loss = bce(labels, preds)
print('Loss: ', loss.numpy())  # Loss: 12.007

m = tf.keras.metrics.Precision(thresholds=[0.1,0.9])
m.update_state(labels, preds)
print('Precision result: ', m.result().numpy())  # Final result: 0.66

m = tf.keras.metrics.Recall()
m.update_state(labels, preds)
print('Recall result: ', m.result().numpy())  # Final result: 0.66