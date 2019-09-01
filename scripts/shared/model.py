import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics 


class ModelInitializer():
  def __init__(self, timesteps, encoding_vec, num_labels, learning_rate):
    self.name='CNN/'
    self.kernels=[32,32,64,64,128,128,256,512]
    self.strides=[1,2,1,2,1,2,1,1]
    self.num_labels=num_labels
    self.timesteps=timesteps
    self.encoding_vec=encoding_vec
    self.learning_rate=learning_rate

  def Precision(self):
    return None
  
  def Recall(self):
    return None

  def F1Score(self):
    return None
  
  def architecture(self):
    x_input=layers.Input((self.timesteps, self.encoding_vec), name='input')

    x=x_input

    for k, s in zip(self.kernels, self.strides):
        x=layers.Conv1D(k, 9, strides=s, padding='SAME', activation='linear')(x)
        x=layers.BatchNormalization()(x)
        x=layers.LeakyReLU()(x)

    x=layers.Flatten()(x)

    for n_fc in [1024]:
      x=layers.Dense(n_fc, activation='relu')(x)
    
    out=layers.Dense(self.num_labels, activation='sigmoid')(x)

    model = models.Model(inputs=x_input, outputs=out, name='model')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['binary_accuracy']
    )
    model.summary()

    return model

  def prepareBatch(batch):
    return None