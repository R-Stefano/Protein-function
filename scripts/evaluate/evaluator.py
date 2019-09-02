'''
This script is used to assess the quality of the model's 
predictios using the metrics defined in the CAFA challenge.
'''

import tensorflow as tf
import yaml
import sys

import custom_metrics as custom

with open("../../hyperparams.yaml", 'r') as f:
	configs=yaml.safe_load(f)

test_data_dir=configs['test_data_dir']
model_dir=configs['model_dir']+'CNN/'

shared_scripts_dir=configs['shared_scripts_dir']
sys.path.append(shared_scripts_dir)
import tfapiConverter as tfconv
import model as model

learning_rate=configs['train']['learning_rate']
batch_size=configs['train']['batch_size']

timesteps=configs['model']['timesteps']
encoding_vec=configs['model']['encoding_vec']
num_labels=configs['model']['num_labels']

model_utils=model.ModelInitializer(timesteps, encoding_vec, num_labels, learning_rate)

custom_objects={
	'precision':model_utils.precision,
	'recall':model_utils.recall,
	'f1score':model_utils.f1score,
}

print('>Loading model..')
model=tf.keras.models.load_model(model_dir+'model.h5', custom_objects=custom_objects)

print('>Loading metrics..')
evaluator=custom.Evaluator()

print('>Loading data..')
test_files=[test_data_dir+fn for fn in os.listdir(test_data_dir)]
dataset=tf.data.TFRecordDataset(test_files)

dataset= dataset.map(tfconv.decodeTFRecord)
dataset=dataset.batch(batch_size)

def displayResults():
	print('\n\nRESULTS:\n')
	protein_centric_data=evaluator.resultsProteinCentricMetric()
	go_term_centric_data=evaluator.resultsGOTermCentricMetric()
	go_class_centric_data=evaluator.resultsGOClassCentricMetric()

	results={
		'model':model_version,
		'protein_centric': protein_centric_data,
		'go_term_centric': go_term_centric_data,
		'go_class_centric': go_class_centric_data
	}

	with open(modelPath+"/results", 'wb') as f:
		pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print('>Processing data..')
for idx, batch in enumerate(dataset):
	print('>Batch', idx+1)
	x, y=batch
	#model prediction:
	model_preds,_=model.predict(x)

	#3 METRICS:
	evaluator.updateProteinCentricMetric(y, model_preds)
	evaluator.updateGOTermCentricMetric(y, model_preds)
	evaluator.updateGOClassCentricMetric(y, model_preds)

	break


displayResults()











