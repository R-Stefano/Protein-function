import tensorflow as tf
import models.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    num_labels=1918

    #CNN
    layers=[
        {'n':2, 'filters':32, 'kernel':9, 'padding':'SAME', 'stride':2},
        {'n':3, 'filters':64, 'kernel':9, 'padding':'SAME', 'stride':2},
        {'n':3, 'filters':128, 'kernel':9, 'padding':'SAME', 'stride':2}
    ]
    self.conv_layers=[]
    for layer_def in layers:
        for _ in range(layer_def['n']):
            self.conv_layers.append(utils.Conv1DLayer(layer_def['filters'], layer_def['kernel'], layer_def['padding'], layer_def['stride']))

    #FLAT
    self.flat=tf.keras.layers.Flatten()

    #FC
    self.final_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid')

  def call(self, x):
    for layer in self.conv_layers:
        x=layer(x)

    x=self.flat(x)
    x=self.final_layer(x)
    return x