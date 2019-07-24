import tensorflow as tf
import models.utils as utils

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    '''
    This class is used to mix togheter different architecture layers. 
    At the moment there is:
      -Transformer (calle encoded layer, will be called self_attention layer) todo: merge heads option (?)
      -convolution (conv + batch + relu): todo: making them optional, add residual
    
    When initialized, you have to define: 
      self.num_layers: Total number of transformer layers. todo: rename to att_layer
      self.num_conv_layers: Total number of convolution layers 
    '''
    num_labels=1918

    #TRANSFORMER
    self.num_trans_layers=1
    num_heads=8
    head_vec=64
    d_model=num_heads*head_vec
    fc=2048

    self.trans_layers=[utils.TransformerLayer(d_model, num_heads, fc) for _ in range(self.num_trans_layers)]

    #CNN
    layers=[
        {'n':2, 'filters':32, 'kernel':5, 'padding':'valid', 'stride':2},
        {'n':2, 'filters':64, 'kernel':5, 'padding':'valid', 'stride':2},
        {'n':2, 'filters':128, 'kernel':5, 'padding':'valid', 'stride':2}
    ]

    self.conv_layers=[]
    for layer_def in layers:
        for _ in range(layer_def['n']):
            self.conv_layers.append(utils.Conv2DLayer(layer_def['filters'], layer_def['kernel'], layer_def['padding'], layer_def['stride']))

    #FLAT
    self.flat=tf.keras.layers.Flatten()

    #FC
    self.final_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid')

  def call(self, x):
    for i in range(self.num_trans_layers):
      x = self.trans_layers[i](x)
    
    print('Shape after transformer:', x.shape)
    #add the 4 dimension
    x=tf.expand_dims(x, -1)

    for layer in self.conv_layers:
      x=layer(x)
      print('Shape after convolution:', x.shape)
    
    x=self.flat(x)
    print('Shape after flatten it:', x.shape)

    x = self.final_layer(x)
    print('Output shape:', x.shape)
    
    return x