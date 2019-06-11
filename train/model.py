import tensorflow as tf
'''
https://www.tensorflow.org/beta/tutorials/text/transformer#encoder_and_decoder
'''

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Inputs:
    q(tensor): query shape (batch_size, num_heads, seq_len_v, depth)
    k(tensor): key shape (batch_size, num_heads, seq_len_v, depth)
    v(tensor): value shape (batch_size, num_heads, seq_len_v, depth)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). 
          Defaults to None.
    
  Returns:
    output, attention_weights
  """
  #compute query-key relation
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  # (batch_size, num_heads, seq_len, seq_len)  
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, v)
  # (batch_size, num_heads, seq_len, depth)  

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    #Compute n heads (d_model) from the input
    q = self.wq(q)  
    k = self.wk(k)
    v = self.wv(v) 
    # (batch_size, seq_len, d_model)
    
    #d_model contains all the heads. So split it in num_heads, depth
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)
    # (batch_size, num_heads, seq_len_v, depth)
    
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    # scaled_attention: (batch_size, num_heads, seq_len, depth)
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)

    #before d_model split in num_heads, depth. Now, merge num, heads, depth into d_model
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
    #concat_attention: (batch_size, seq_len, d_model)

    output = self.dense(concat_attention)  
    #output: (batch_size, seq_len_v, d_model)
        
    return output, attention_weights
  

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    '''
    Called to initialize the encoder layer. The encoder layer consists of
    - MultiHead Attention mechanism
    - Fully connected layer

    Inputs:
      d_model (int): vector dimension, obtained by num_heads*head_depth. 
                      The vector is going to contain all the computed 
                      heads for a given input
      num_heads (int): number of heads in the attention mechanism
      dff (int): Number neurons in the fcl d_model -> fcl -> d_model
    '''
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.converter = tf.keras.layers.Dense(d_model)


    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    ''' 
    Called to process input. Apply multihead attention and fully connected layers

    Inputs:
      x (tensor): The input tensor of shape batch_size, seq_len, d_vec
      training (bool): To disable dropout during inference
      mask (tensor): If None, mask is not applied
    '''
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    #Add input to multihead attention output
    #input last dimension must be transformed into d_model length
    out1 = self.layernorm1(self.converter(x) + attn_output)  
    #(batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  
    
    # (batch_size, input_seq_len, d_model)
    return out2

class ConvLayer(tf.keras.layers.Layer):
  def __init__(self, num_filters, filter_size, padding, stride):
    super(ConvLayer, self).__init__()
    '''
    Called to inizialize a convolution layer. The convolution layer consists of:
    -convolution 
    -batch normalization
    -activation function
    '''

    self.conv=tf.keras.layers.Conv2D(num_filters, filter_size, padding=padding, strides=stride)
    self.norm=tf.keras.layers.BatchNormalization()
    self.activation=x =tf.keras.activations.relu
  
  def call(self, x):
    x=self.conv(x)
    x_norm=self.norm(x)
    output=self.activation(x_norm)
    return output

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_size, rate=0.1):
    super(Transformer, self).__init__()
    '''
    This class is used to mix togheter different architecture layers. 
    At the moment there is:
      -Transformer (calle encoded layer, will be called self_attention layer) todo: merge heads option (?)
      -convolution (conv + batch + relu): todo: making them optional
    
    When initialized, you have to define: 
      self.num_layers: num transformer layers. todo: rename to att_layer
      self.num_conv_layers: number of convolution layers 
    '''

    self.num_layers=num_layers
    self.num_conv_layers=6
    self.num_filters_layers=[32, 32, 64, 64, 128]
    #if num_conv_layers is higher than the filters defined, repeat the last filter dimension
    if (self.num_conv_layers!=len(self.num_filters_layers)):
      for _ in range(self.num_conv_layers-len(self.num_filters_layers)):
        self.num_filters_layers.append(self.num_filters_layers[-1])
    filter_size=5
    padding='valid'
    stride=2

    #Define transformers layers
    self.trans_layers=[EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
    #Define convolutional layers
    self.conv_layers=[]
    for i in range(self.num_conv_layers):
      self.conv_layers.append(ConvLayer(self.num_filters_layers[i], filter_size, padding, stride))

    #Define fully connected layer
    self.final_layer = tf.keras.layers.Dense(target_size)
    
  def call(self, x, training, mask):
    
    for i in range(self.num_layers):
      x = self.trans_layers[i](x, training, mask)
    
    print('Shape after transformer:', x.shape)
    #add the 4 dimension
    x=tf.expand_dims(x, -1)

    for i in range(self.num_conv_layers):
      x=self.conv_layers[i](x)
      print('Shape after convolution',i,':', x.shape)
    
    x=tf.reshape(x, (x.shape[0],-1))
    print('Shape after flatten it:', x.shape)
    print('\n\n')

    final_output = self.final_layer(x)
    
    return final_output