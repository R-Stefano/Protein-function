from absl import flags
import tensorflow as tf
from tensorflow import keras
import numpy as np
import train.model as mod
import evaluate.custom_metrics as custom

FLAGS = flags.FLAGS

'''
How to use a model developed using Imperative style for prediction.

symbolic or imperative?
https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021

how to save both models
https://www.tensorflow.org/beta/guide/keras/saving_and_serializing#saving_subclassed_models

how to make transformer in symbolic(functiona API)
https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2

how to make transformer in imperative
https://www.tensorflow.org/beta/tutorials/text/transformer
'''

thresholds=np.arange(start=0.1, stop=1.0, step=0.1)
myMetric=custom.F1MaxScore(thresholds, name="train_f1_max")
recall=tf.keras.metrics.Recall()
precision=tf.keras.metrics.Precision()

@tf.function
def updateMetrics(y_true, y_pred):
	myMetric(y_true, y_pred)
	recall(y_true, y_pred)
	precision(y_true, y_pred)



def evaluate():
	version="version_1"
	export_dir="evaluate/models/"+version+"/savedModel"
	model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.num_labels)
	model.load_weights(export_dir)

	inputData=np.ones((1,512,25))
	labelData=np.ones((1,3344), dtype=np.float32)
	out=model.predict(inputData)

	updateMetrics(labelData, out)
