import tensorflow as tf
import model.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    num_labels=1918
    conv_type='1d' #sequential data: [batch, seq, vec]

    self.conv_layers=[
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=9),
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=9, stride=2),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=9),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=9, stride=2),
        utils.ConvLayer(conv_type=conv_type, num_filters=128 ,filter_size=9),
        utils.ConvLayer(conv_type=conv_type, num_filters=128 ,filter_size=9, stride=2),
        utils.ConvLayer(conv_type=conv_type, num_filters=256 ,filter_size=9),
        utils.ConvLayer(conv_type=conv_type, num_filters=512 ,filter_size=9)
    ]

    self.rnn_layers=[
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256), merge_mode='concat') #forw-back outputs are concat
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

    for i, layer in enumerate(self.conv_layers):
        x=layer(x)
        print('conv_{}: {}'.format(i, x.shape))

    for i, layer in enumerate(self.rnn_layers):
        x=layer(x)
        print('rnn_{}: {}'.format(i, x.shape))

    for i,layer in enumerate(self.fc_layers):
      x=layer(x)
      print('fc_{}: {}'.format(i, x.shape))
    return x