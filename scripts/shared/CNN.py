import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics 


class ModelInitializer():
  def __init__(self, timesteps, encoding_vec, num_labels, learning_rate):
    self.name='CNN/'
    self.kernels=[32,32,64,64,128,128,256,512]
    self.strides=[2,2,2,2,2,2,2,2]
    self.num_labels=num_labels
    self.timesteps=timesteps
    self.encoding_vec=encoding_vec
    self.learning_rate=learning_rate

  def precision(self, y_true, y_pred, threshold=0.5):
    y_pred=tf.cast(y_pred>=threshold, tf.float32)

    tp=tf.math.reduce_sum(y_true*y_pred)
    fp=tf.math.reduce_sum((1-y_true)*y_pred)

    return tp/(tp+fp+1e-9)
  
  def recall(self, y_true, y_pred, threshold=0.5):
    y_pred=tf.cast(y_pred>=threshold, tf.float32)

    tp=tf.math.reduce_sum(y_true*y_pred)
    fn=tf.math.reduce_sum((1-y_pred)*y_true)
    return tp/(tp+fn+1e-9)

  def f1score(self, y_true, y_pred):
    precision=self.precision(y_true, y_pred)
    recall=self.recall(y_true, y_pred)

    f1=2*precision*recall / (precision + recall)
    return f1
  
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
        metrics=['binary_accuracy', self.precision, self.recall, self.f1score]
    )
    model.summary()

    return model

  def prepareBatch(batch):
    return None