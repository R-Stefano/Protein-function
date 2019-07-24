import tensorflow as tf
import models.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    num_labels=1918

    #TRANSFORMER
    self.num_trans_layers=3
    num_heads=8
    head_vec=64
    d_model=num_heads*head_vec
    fc=2048

    self.trans_layers=[utils.TransformerLayer(d_model, num_heads, fc) for _ in range(self.num_trans_layers)]

    #RNN
    self.rnn=tf.keras.layers.LSTM(256)

    #FC
    self.final_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid')

  def call(self, x):
    for i in range(self.num_trans_layers):
      x = self.trans_layers[i](x)
    
    print('Shape after transformer:', x.shape)
    x=self.rnn(x)
    print('shape after RNN:', x.shape)


    x = self.final_layer(x)
    print('Output shape:', x.shape)
    
    return x