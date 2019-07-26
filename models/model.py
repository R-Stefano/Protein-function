import tensorflow as tf
import models.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    num_labels=1918
    conv_type='1d' #sequential data: [batch, seq, vec]

    #TRANSFORMER
    head_vecs=64
    num_heads=8
    d_model=num_heads*head_vecs # 512=8*64
    dff=2048
    self.self_attention_layers=[
        utils.TransformerLayer(d_model=d_model, num_heads=num_heads, dff=dff),
        utils.TransformerLayer(d_model=d_model, num_heads=num_heads, dff=dff)
    ]

    #RNN
    hidden_units=512
    self.rnn_layers=[
      tf.keras.layers.LSTM(hidden_units)
    ]

    #FLAT
    self.flat=tf.keras.layers.Flatten()

    #FC
    self.fc_layers=[
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(num_labels, activation='sigmoid')
    ]
    

  def call(self, x, training):
    print('Input shape:', x.shape)

    #Positional encoding
    x += utils.positionalEncoding(x.shape[1], x.shape[-1])

    for i, layer in enumerate(self.self_attention_layers):
        x=layer(x, training)
        print('trans_{}: {}'.format(i, x.shape))

    #How to handle transformer output:
    '''
    IDEAS:
    -average
    -rnn
    -cnn
    '''

    x=tf.math.reduce_mean(x, axis=1)
    print('projected transformer output:', x.shape)
    for i,layer in enumerate(self.fc_layers):
      x=layer(x)
      print('fc_{}: {}'.format(i, x.shape))
    return x