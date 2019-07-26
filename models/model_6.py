import tensorflow as tf
import models.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

    num_labels=1918
    conv_type='1d' #sequential data: [batch, seq, vec]

    #CNN
    self.conv_layers=[
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=9, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=9, dilatation=2),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=9, dilatation=4),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=9, dilatation=8),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=9, dilatation=16),
        utils.ConvLayer(conv_type=conv_type, num_filters=256  ,filter_size=9, dilatation=32)

    ]

    self.res_layers=[
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=32  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=64  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=256  ,filter_size=1, dilatation=1),
    ]

    self.project_layers=[
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1),
        utils.ConvLayer(conv_type=conv_type, num_filters=128  ,filter_size=1, dilatation=1)
    ]

    #TRANSFORMER
    head_vecs=64
    num_heads=8
    d_model=num_heads*head_vecs # 512=8*64
    dff=2048
    self.self_attention_layers=[
        utils.TransformerLayer(d_model=d_model, num_heads=num_heads, dff=dff)
    ]

    #RNN
    hidden_units=512
    self.rnn_layers=[
      tf.keras.layers.LSTM(hidden_units)
    ]

    #FLAT
    self.flat=tf.keras.layers.Flatten()

    #TESTING:
    self.feature_wise_max_pooling=tf.keras.layers.MaxPool1D(pool_size=d_model, data_format='channels_first')

    #FC
    self.fc_layers=[
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(num_labels, activation='sigmoid')
    ]
    

  def call(self, x, training):
    print('Input shape:', x.shape)

    layers_outs=[x]

    for i, layer in enumerate(self.conv_layers):
        #layer dilated conv
        x_out=layer(x)
        #layer residual transformation
        x_in_transform=self.res_layers[i](x)
        #layer output
        x=x_out + x_in_transform
        layers_outs.append(x)
        print('conv_{}: {}'.format(i, x.shape))

    output_projections=[] #layer, batch, seq, vec
    for i, layer in enumerate(self.project_layers):
        output_projections.append(layer(layers_outs[i]))

        print('conv_1x1_{}: {}'.format(i, output_projections[i].shape))
    x=tf.transpose(output_projections, perm=[1,2,3,0]) #batch, seq, vec, layer

    '''
    TODO:
    -average
    -concatenate
    '''
    x=tf.math.reduce_mean(x, axis=-1)
    print('Handle projected values:', x.shape)

    '''
    #Positional encoding
    x += utils.positionalEncoding(x.shape[1], x.shape[-1])

    for i, layer in enumerate(self.self_attention_layers):
        x=layer(x, training)
        print('trans_{}: {}'.format(i, x.shape))
    '''
    for i, layer in enumerate(self.rnn_layers):
      x=layer(x)
      print('rnn_{}: {}'.format(i, x.shape))

    for i,layer in enumerate(self.fc_layers):
      x=layer(x)
      print('fc_{}: {}'.format(i, x.shape))
    return x