from absl import flags
import tensorflow as tf
from tensorflow import keras
import numpy as np
FLAGS = flags.FLAGS
import os
import train.model as mod

import analyze.custom_metrics as custom
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
prec=tf.keras.metrics.Precision()

@tf.function
def training(y_true, y_pred):
	myMetric(y_true, y_pred)

def predict():
	version="version_1"
	export_dir="analyze/models/"+version+"/module_without_signature"
	#model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.unique_labels)
	#model.load_weights(export_dir)
	#out=model.predict(inputData)

	batch_size=2
	labels=4

	np.random.seed(0)
	predictions=np.random.random(size=(batch_size,labels))
	labels=np.random.random(size=(batch_size,labels))
	labels[labels<0.5]=0
	labels[labels>=0.5]=1

	training(tf.Variable(labels), tf.Variable(predictions))
	training(tf.Variable(labels), tf.Variable(predictions))
	training(tf.Variable(labels), tf.Variable(predictions))

	print('Epoch finished, result', myMetric.result().numpy())
	print('Resetting state', myMetric.reset_state())
	print('Epoch finished, result', myMetric.result().numpy())

	training(tf.Variable(labels), tf.Variable(predictions))
	training(tf.Variable(labels), tf.Variable(predictions))
	training(tf.Variable(labels), tf.Variable(predictions))
	print('Epoch finished, result', myMetric.result().numpy())























def evaluation1(preds, labels, thres):
	'''
	This function computes the first evaluation metric. 
	The protein-centric evaluation.
	Compute f1-score for thresholds between 0.1 and 0.9 and use the 
	best f1-score as output.
	Protein-centric evaluation measures how accurately methods can assign functional terms to a protein.

	'''
	#generate thresholds
	#thresholds=np.arange(start=0.1, stop=1.0, step=0.1)
	#results=[]
	#for t in thresholds:
	#	results.append(f1_score(np.copy(preds), labels, thres))

	#print('Result evaluation 1: {:.2f}'.format(np.max(results)))

	return f1_score(np.copy(preds), labels, thres)

def f1_score(preds, labels, threshold):
	'''
	This function computes the F1-score for a given threshold.
	The precision is computed on the examples with at least 1 positive
	prediction. If the example has each element in the prediction vector below
	the threshold, it is discarded for the calculation of the precision.

	Args:
		preds (tensor): [batch_size, preds]. Preds have values between 0 and 1
		labels (tensor): [batch_size, labels]. Labels have values 0 or 1
		threshold (float): if pred>threshold, pred is considered as 1
	
	more info about the evalution
	https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6
	'''
	assert len(np.asarray(preds).shape)==2
	assert len(np.asarray(labels).shape)==2

	num_labels=np.asarray(preds).shape[-1]
	exclude=True

	#Apply threshold
	preds[preds>=threshold]=1
	preds[preds!=1]=0

	print('Preds', preds)
	print('Labels', labels)

	#EXCLUDE EXAMPLES WITH 0 POSITIVE PREDICTIONS FOR PRECISION ONLY
	if (exclude):
		#compute which column to keep
		mask=np.argwhere(np.sum(preds, axis=-1)>0)
		#apply the mask
		prec_preds=np.reshape(preds[mask], (-1, num_labels))
		prec_labels=np.reshape(labels[mask], (-1, num_labels))
	else:
		prec_preds=preds
		prec_labels=labels
	#Compute precision
	prec.update_state(prec_labels, prec_preds)
	avg_prec=prec.result().numpy()

	print('precision', avg_prec)

	#Compute recall


	recall.update_state(labels, preds)
	avg_rec=recall.result().numpy()

	print('recall', avg_rec)

	f1_score=(2*avg_prec * avg_rec)/(avg_prec+avg_rec +1e-9)
	return f1_score
