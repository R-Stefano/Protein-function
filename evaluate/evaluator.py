from absl import flags
import tensorflow as tf
import numpy as np
import obonet
import pickle
import os
import yaml
import importlib.util

import evaluate.custom_metrics as custom
import prepare.tfapiConverter as tfconv

FLAGS = flags.FLAGS

'''
This script is used to assess the quality of the model's 
predictios using the metrics defined in the CAFA challenge.

TODO:
-Define and add term-centri metric
-Threshold for predictions to 1 or 0. pred_threshold. Which value should have? 
'''

dataPath=FLAGS.dataPath
pred_threshold=0.8
model_version="version_2"
modelPath='evaluate/models/'+model_version

with open("hyperparams.yaml", 'r') as f:
	hyperparams=yaml.safe_load(f)

graph = obonet.read_obo('extract/go-basic.obo')

thresholds=np.arange(start=0.1, stop=1.0, step=0.1)

#1. PROTEIN-CENTRIC METRICS
model_f1max=custom.F1MaxScore(thresholds, name="model_f1_max")
model_recall=tf.keras.metrics.Recall()
model_precision=tf.keras.metrics.Precision()

#2. TERM-CENTRIC METRICS
go_terms_f1max=[]
go_terms_recall=[]
go_terms_precision=[]
#create the metrics for each GO term
for i in range(len(hyperparams['available_gos'])):
	go_terms_f1max.append(custom.F1MaxScore(thresholds, name="model_f1_max"))
	go_terms_recall.append(tf.keras.metrics.Recall())
	go_terms_precision.append(tf.keras.metrics.Precision())


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

def term_centric_metric(y_true, y_pred):
	'''
	This function computes the F1 score, precision and recall for each go term
	'''
	print('GO terms:', y_true.shape[1])
	for go_term_idx in range(y_true.shape[1]):
		go_term_y_true=np.reshape(y_true[:, go_term_idx], (-1, 1))
		go_term_y_pred=np.reshape(y_pred[:, go_term_idx], (-1, 1))

		print('number of examples per go term (should not change', go_term_y_true.shape)

		go_terms_f1max[go_term_idx](go_term_y_true, go_term_y_pred)
		go_terms_recall[go_term_idx](go_term_y_true, go_term_y_pred)
		go_terms_precision[go_term_idx](go_term_y_true, go_term_y_pred)



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
	print('1. Protein-centric results:')
	print('Model f1_max score:  {:.2f}'.format(model_f1max.result().numpy()))
	print('Model recall:    {:.2f}'.format(model_recall.result().numpy()))
	print('Model precision: {:.2f}'.format(model_precision.result().numpy()))

	print('\n3. GO class-centric results:')
	for go_class in go_classes_metrics:
		print(go_class)
		for metric_name in go_classes_metrics[go_class]:
			metric_obj=go_classes_metrics[go_class][metric_name]
			print('> '+metric_name, ' {:.2f}'.format(metric_obj.result().numpy()))
		print('\n')

	results={
		'model':model_version,
		'protein_centric': {
			'f1':model_f1max.result().numpy(),
			'recall':model_recall.result().numpy(),
			'precision': model_precision.result().numpy()},
		'GO_class_centric': {}
	}

	for go_class in go_classes_metrics:
		results['GO_class_centric'][go_class]={}
		for metric_name in go_classes_metrics[go_class]:
			metric_obj=go_classes_metrics[go_class][metric_name]
			results['GO_class_centric'][go_class][metric_name]=metric_obj.result().numpy()

	with open(modelPath+"/results", 'wb') as f:
		pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
	
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
	#spec = importlib.util.spec_from_file_location("module.name", modelPath+"/model.py")
	#netModule = importlib.util.module_from_spec(spec)
	#spec.loader.exec_module(netModule)
	#model=netModule.Model()
	#model.load_weights(modelPath+"/savedModel")

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

		#3 METRICS:
		protein_centric_metric(batch['Y'], model_preds)
		term_centric_metric(batch['Y'], model_preds)
		go_class_centric_metric(batch['Y'], model_preds)
		break

	displayResults()
