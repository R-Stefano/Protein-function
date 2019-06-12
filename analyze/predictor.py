from absl import flags
import tensorflow as tf
from tensorflow import keras
import numpy as np
FLAGS = flags.FLAGS
import os
import train.model as mod


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
def predict():
	export_dir="train/saved_model/module_without_signature"

	model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.unique_labels)
	model.load_weights(export_dir)

	inputData=np.ones((1, 512, 25), dtype=np.float32)

	out=model.predict(inputData)
	print('output:', out.shape)
	print(out[0][:10])
