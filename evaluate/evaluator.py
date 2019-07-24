from absl import flags
import tensorflow as tf
import numpy as np
import obonet
import pickle
import os
import yaml

import models.model_1 as model
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

This script is used to assess the quality of the model's 
predictios using the metrics defined in the CAFA challenge.

TODO:
-Define and add term-centri metric
-Threshold for predictions to 1 or 0. pred_threshold. Which value should have? 
'''

dataPath=FLAGS.dataPath
pred_threshold=0.8
model_version="version_1"

with open("hyperparams.yaml", 'r') as f:
	hyperparams=yaml.safe_load(f)

graph = obonet.read_obo('extract/go-basic.obo')

thresholds=np.arange(start=0.1, stop=1.0, step=0.1)

#1. PROTEIN-CENTRIC METRICS
#BLAST metrics
#blast_myMetric=custom.F1MaxScore(thresholds, name="blast_f1_max")
#blast_recall=tf.keras.metrics.Recall()
#blast_precision=tf.keras.metrics.Precision()
#Model metrics
model_f1max=custom.F1MaxScore(thresholds, name="model_f1_max")
model_recall=tf.keras.metrics.Recall()
model_precision=tf.keras.metrics.Precision()

#2. TERM-CENTRIC METRICS

#3. GO CLASS-CENTRIC METRICS
#Map each GO term (label) to its GO Class.
#Each GO class contains the labels idxs of the GO term belonging to the class.
#Used to evaluate the predictions for each GO class
go_classes_idxs={
	'cellular_component':[],
	'biological_process':[],
	'molecular_function':[]
}

for idx, go_term in enumerate(hyperparams['available_gos']):
	go_classes_idxs[graph.node[go_term]['namespace']].append(idx)

#Stores the 3 metrics for each go class
go_classes_metrics={
	'cellular_component':{},
	'biological_process':{},
	'molecular_function':{}
}

#Create the 3 metrics for each go class
for go_class in go_classes_metrics:
	go_classes_metrics[go_class]['f1_max']=custom.F1MaxScore(thresholds, name=go_class+"_f1_max")
	go_classes_metrics[go_class]['recall']=tf.keras.metrics.Recall(name=go_class+"_recall")
	go_classes_metrics[go_class]['precision']=tf.keras.metrics.Precision(name=go_class+"_precision")

def protein_centric_metric(y_true, y_pred):
	model_f1max(y_true, y_pred)
	model_recall(y_true, y_pred)
	model_precision(y_true, y_pred)

def go_class_centric_metric(y_true, y_pred):
	'''
	This function computes the F1 max score, precision and recall for BP, CC and MF

	'''

	bp_idxs=np.asarray(go_classes_idxs['biological_process'])
	cc_idxs=np.asarray(go_classes_idxs['cellular_component'])
	mf_idxs=np.asarray(go_classes_idxs['molecular_function'])

	#get predictions and labels for each go class
	y_trues_bp=np.transpose(np.transpose(y_true)[bp_idxs])
	y_trues_cc=np.transpose(np.transpose(y_true)[cc_idxs])
	y_trues_mf=np.transpose(np.transpose(y_true)[mf_idxs])

	y_preds_bp=np.transpose(np.transpose(y_pred)[bp_idxs])
	y_preds_cc=np.transpose(np.transpose(y_pred)[cc_idxs])
	y_preds_mf=np.transpose(np.transpose(y_pred)[mf_idxs])

	#Update the metrics
	for metric_name in go_classes_metrics['cellular_component']:
		metric_obj=go_classes_metrics['cellular_component'][metric_name]
		metric_obj(y_trues_cc, y_preds_cc)

	for metric_name in go_classes_metrics['biological_process']:
		metric_obj=go_classes_metrics['biological_process'][metric_name]
		metric_obj(y_trues_bp, y_preds_bp)
	
	for metric_name in go_classes_metrics['molecular_function']:
		metric_obj=go_classes_metrics['molecular_function'][metric_name]
		metric_obj(y_trues_mf, y_preds_mf)

def displayResults():
	print('\n\nRESULTS:\n')
	print('Protein-centric results:')
	print('Model f1_max score:  {:.2f}'.format(model_f1max.result().numpy()))
	print('Model recall:    {:.2f}'.format(model_recall.result().numpy()))
	print('Model precision: {:.2f}'.format(model_precision.result().numpy()))

	print('\nGO class-centric results:')
	for go_class in go_classes_metrics:
		print(go_class)
		for metric_name in go_classes_metrics[go_class]:
			metric_obj=go_classes_metrics[go_class][metric_name]
			print('> '+metric_name, ' {:.2f}'.format(metric_obj.result().numpy()))
		print('\n')

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
	

def extractPredictedGOs(predictions):
	'''
	This function maps the predictions to the GO idxs.

	Args:
		predictions (array): an array of shape [batch_size, num_labels]
	
	Returns:
		An array of shape batch_size, go_terms_predicted with the GO id terms

	'''
	idxs=np.where(predictions>pred_threshold, 1, 0)

	go_terms_predictions=[]
	for pred_idxs in idxs:
		GO_terms_predicted= np.asarray(hyperparams['available_gos'])[np.reshape(np.argwhere(pred_idxs==1), -1)]
		go_terms_predictions.append(GO_terms_predicted)

	return go_terms_predictions



def evaluate():
	#params
	batch_size=64

	#1. LOAD MODEL
	#export_dir="evaluate/models/"+model_version+"/savedModel"
	#model=mod.Transformer(num_layers=FLAGS.num_layers, d_model=FLAGS.d_model, num_heads=FLAGS.num_heads, dff=FLAGS.fc, target_size=FLAGS.num_labels)
	#model.load_weights(export_dir)

	#2. LOAD TEST EXAMPLES
	ex_folder=os.path.join(dataPath,'test/')
	test_files=[ex_folder+fn for fn in os.listdir(ex_folder)]
	dataset=tf.data.TFRecordDataset(test_files)

	dataset= dataset.map(tfconv.decodeTFRecord)
	dataset=dataset.batch(batch_size)

	for idx, batch in enumerate(dataset):
		#DeepFunc model prediction:
		#model_preds=model.predict(batch['X'])

		model_preds=np.random.random(size=(64, 1918)).astype(np.float32)

		extractPredictedGOs(model_preds)

		#NOT NEEDED. Metrics already squash RIGHT(?) squash predictions to 0 or 1 based on threshold
		#model_preds=np.where(model_preds > pred_threshold, 1, 0)
		protein_centric_metric(batch['Y'], model_preds)
		#2. term_centric_metric(batch['Y'], model_preds)
		go_class_centric_metric(batch['Y'], model_preds)
		break

	displayResults()
