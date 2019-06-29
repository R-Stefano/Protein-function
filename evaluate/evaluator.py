from absl import flags
import tensorflow as tf
import numpy as np
import obonet
import pickle
import os
import yaml

import train.model as mod
import evaluate.custom_metrics as custom
import prepare.tfapiConverter as tfconv

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

dataPath=FLAGS.dataPath

thresholds=np.arange(start=0.1, stop=1.0, step=0.1)
#BLAST metrics
blast_myMetric=custom.F1MaxScore(thresholds, name="blast_f1_max")
blast_recall=tf.keras.metrics.Recall()
blast_precision=tf.keras.metrics.Precision()
#Model metrics
model_myMetric=custom.F1MaxScore(thresholds, name="model_f1_max")
model_recall=tf.keras.metrics.Recall()
model_precision=tf.keras.metrics.Precision()

with open("hyperparams.yaml", 'r') as f:
	hyperparams=yaml.safe_load(f)

graph = obonet.read_obo('extract/go.obo')

def analysizeResults():
	print('BLAST f1 score:  {:.2f}'.format(blast_myMetric.result().numpy()))
	print('BLAST recall:    {:.2f}'.format(blast_recall.result().numpy()))
	print('BLAST precision: {:.2f}'.format(blast_precision.result().numpy()))
	print('Model f1 score:  {:.2f}'.format(model_myMetric.result().numpy()))
	print('Model recall:    {:.2f}'.format(model_recall.result().numpy()))
	print('Model precision: {:.2f}'.format(model_precision.result().numpy()))

@tf.function
def updateMetrics(y_true, y_pred_blast, y_pred_model):
	blast_myMetric(y_true, y_pred_blast)
	blast_recall(y_true, y_pred_blast)
	blast_precision(y_true, y_pred_blast)

	model_myMetric(y_true, y_pred_model)
	model_recall(y_true, y_pred_model)
	model_precision(y_true, y_pred_model)

def preprocessBLASTpredictions():
	'''
	This function loads the BLAST predictions as GO notations and 
	create the encoded predictions. 
	For each example:
	from GO:003248, GO:456392 -> [0,0,1,0,0,1]

	Return:
		a tensor of shape [dataset_size, num_go_labels]
	'''
	folder_path='evaluate/BLAST_baseline/blast_predictions/'
	files_name=os.listdir(folder_path)

	BLAST_results=[]
	for f in files_name:
		data_batch=pickle.load(open(folder_path+f, 'rb'))
		for example in data_batch:
			goes_idxs=[]
			for go in example:
				try:
					goes_idxs.append(hyperparams['available_goes'].index(go))
				except:
					continue
			#generate example encoded predictions
			example_encoded=np.zeros((len(hyperparams['available_goes']))).astype(np.float32)
			#assign 1 to the predicted go notations 
			if len(goes_idxs)>0:
				example_encoded[np.asarray(goes_idxs)]=1

			BLAST_results.append(example_encoded)
	
	return np.asarray(BLAST_results)
	

def evaluate():
	#params
	batch_size=64

	#0. PREPARE BLAST PREDICTIONS
	blast_predictions=preprocessBLASTpredictions()

	#1. LOAD MODEL
	version="version_1"
	export_dir="evaluate/models/"+version+"/savedModel"
	model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.num_labels)
	model.load_weights(export_dir)

	#2. LOAD TEST EXAMPLES
	ex_folder=os.path.join(dataPath,'test/')
	test_files=[ex_folder+fn for fn in os.listdir(ex_folder)]
	dataset=tf.data.TFRecordDataset(test_files)

	dataset= dataset.map(tfconv.decodeTFRecord)
	dataset=dataset.batch(batch_size)

	for idx, batch in enumerate(dataset):
		start=idx*batch_size
		end=start+batch_size

		#BLAST prediction:
		blast_preds=blast_predictions[start:end]
		#DeepFunc model prediction:
		model_preds=model.predict(batch['X'])

		#keep tyrack of the results
		updateMetrics(batch['Y'], blast_preds, model_preds)

	analysizeResults()
