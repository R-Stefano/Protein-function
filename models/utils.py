import tensorflow as tf
import numpy as np
class ConvLayer(tf.keras.layers.Layer):
  def __init__(self, conv_type, num_filters, filter_size, padding='SAME', stride=1, activation=True, pooling=False, dilatation=1):
    super(ConvLayer, self).__init__()
    '''
    Called to inizialize a convolution layer. The convolution layer consists of:
    -convolution (1D|2D)
    -batch normalization
    -activation function (optional)
    -max pooling (1D|2D) (optional)
    '''
    self.applyActivation=activation
    self.applyPooling=pooling

    if (conv_type=='1d'):
      self.conv=tf.keras.layers.Conv1D(num_filters, filter_size, padding=padding, strides=stride, dilation_rate=dilatation)
    else:
      self.conv=tf.keras.layers.Conv2D(num_filters, filter_size, padding=padding, strides=stride, dilation_rate=dilatation)

    self.norm=tf.keras.layers.BatchNormalization()
    
    self.activation=tf.keras.activations.relu

    if (conv_type=='1d'):
      self.pooling=tf.keras.layers.MaxPool1D()
    else:
      self.pooling=tf.keras.layers.MaxPool2D()
  
  def call(self, x):
    x=self.conv(x)
    x=self.norm(x)

    if (self.applyActivation):
      x=self.activation(x)

    if (self.applyPooling):
      x=self.pooling(x)
    return x

class ResidualLayer(tf.keras.layers.Layer):
  def __init__(self, conv_type, num_filters, filter_size, pooling, dilatation=1):
    super(ResidualLayer, self).__init__()
    self.applyPooling=pooling

    self.conv_block_1=ConvLayer(conv_type, num_filters, filter_size, dilatation=dilatation)
    self.conv_block_2=ConvLayer(conv_type, num_filters, filter_size, activation=False, dilatation=dilatation)

    #Transform residual block input: (apply 1x1 convolution)
    #-[10, 20, 32] and [10, 20, 25] != channel
    if (conv_type=='1d'):
      self.transform=tf.keras.layers.Conv1D(num_filters, 1, padding='SAME', strides=1, dilation_rate=dilatation)
    else:
      self.transform=tf.keras.layers.Conv2D(num_filters, 1, padding='SAME', strides=1, dilation_rate=dilatation)

    self.residual_activation=tf.keras.activations.relu

    if (conv_type=='1d'):
      self.pooling=tf.keras.layers.MaxPool1D()
    else:
      self.pooling=tf.keras.layers.MaxPool2D()


  def call(self, x_in):
    x=self.conv_block_1(x_in)
    x=self.conv_block_2(x)

    x_in=self.transform(x_in)

    x=x + x_in

    x=self.residual_activation(x)

    if (self.applyPooling):
      x=self.pooling(x)
    return x



class TransformerLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
    '''
    Called to initialize the encoder layer. The encoder layer consists of
    - MultiHead Attention mechanism
    - Fully connected layer

    Inputs:
      d_model (int):  heads vector dimension size. [batch, seq, d_model]. 
                      The vector is going to contain all the computed 
                      heads for a given input
      num_heads (int): number of heads in the attention mechanism
      dff (int): Number neurons in the FC layer d_model -> fcl -> d_model
    '''
    super(TransformerLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = self.point_wise_feed_forward_network(d_model, dff)
    self.converter = tf.keras.layers.Dense(d_model)

    #Layer normalization is similar batch normalization, but the mean and variances are 
    #calculated along the last dimension
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

  def point_wise_feed_forward_network(self, d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

  def call(self, x, training):
    ''' 
    Inputs:
      x (tensor): The input tensor of shape [batch_size, seq_len, d_vec]
      training (boolean): If True, apply dropout
    '''

    #1. Multi Head attention:
    attn_output, _ = self.mha(x, x, x)  # (batch_size, seq_len, d_model)

    attn_output = self.dropout1(attn_output, training=training)

    #2. Add & Norm
    #input tensor last dimension must be transformed into d_model length
    out1 = self.layernorm1(self.converter(x) + attn_output)  #(batch_size, seq_len, d_model)
    
    #3. Point-wise FNN
    ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)

    ffn_output = self.dropout2(ffn_output, training=training)

    #4. Add & Norm
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
       
    return out2

class MultiHeadAttention(tf.keras.layers.Layer):
  '''
  MultiHeadAttention takes as input the sequence data: [batch, seq, vec].
  The input is projected into num_heads query, value, key vectors. Compute 
  attention for each head and then concatenate the results
  Args:
  -d_model (int): d_model= num_heads * q|v|k vector dimension
  -num_heads (int): num of parallel attention layers 
  '''
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    #Define projection [batch, seq, vec] -> [batch, seq, d_model]
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
  
  def scaled_dot_product_attention(self, q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    
    Inputs:
      q(tensor): query shape (batch_size, num_heads, seq_len_v, depth)
      k(tensor): key shape (batch_size, num_heads, seq_len_v, depth)
      v(tensor): value shape (batch_size, num_heads, seq_len_v, depth)
          
    Returns:
      output, attention_weights
    """
    #compute query-key relation
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # (batch_size, num_heads, seq_len, seq_len)  
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    # (batch_size, num_heads, seq_len, depth)  

    return output, attention_weights

  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    
    #v=k=q=x size: [batch, seq, input_size]
    q = self.wq(q)  
    k = self.wk(k)
    v = self.wv(v) 
    # (batch_size, seq_len, d_model)
    
    #d_model contains all the heads. So split it in num_heads, depth
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)
    # (batch_size, num_heads, seq_len_v, depth)
    
    scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
    # scaled_attention: (batch_size, num_heads, seq_len, depth)
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)

    #before d_model split in num_heads, depth. Now, merge num_heads, depth into d_model
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
    #concat_attention: (batch_size, seq_len, d_model)

    output = self.dense(concat_attention)  
    #output: (batch_size, seq_len, d_model)
        
    return output, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positionalEncoding(seq_length, d_model):
  '''
  '''
  pos=np.expand_dims(np.arange(seq_length), -1) #[seq_length, 1]
  i=np.expand_dims(np.arange(d_model), 0) #[1, d_model]
  angle_rads = get_angles(pos,i,d_model) #[seq_length, d_model]

  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)

  #add batch dimension
  pos_encoding=np.expand_dims(pos_encoding, 0)

  return tf.cast(pos_encoding, dtype=tf.float32)