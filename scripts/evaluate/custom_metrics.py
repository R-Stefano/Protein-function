import tensorflow as tf
import numpy as np
import obonet
import yaml

class F1MaxScore():
	'''
	This function computes f1-score for thresholds between 0.1 and 0.9 and use the 
	best f1-score as output.
	Protein-centric evaluation measures how accurately methods can assign functional terms to a protein.
	'''
	def __init__(self, thres, name='f1_score', **kwargs):
		#stores the thresholds: 0.1, 0.2 .. 0.9
		self.thresholds=thres

		self.exclude_empty_prec=True
		
		#stores the f1 scores sum for each threshold.
		self.f1_thresholds=np.zeros((self.thresholds.shape[0]))
		#stores the number of times the thresholds have been updated (used to compute the average)
		self.counter=0

	def _f1_score(self, labels, preds, threshold):
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
		assert len(preds.shape)==2
		assert len(labels.shape)==2

		#cast everything to float32
		threshold=float(threshold)
		labels=labels.astype(np.float32)
		preds=preds.astype(np.float32)
		

		#Apply threshold
		casted_preds=(preds>threshold).astype(np.float32)

		FN = np.count_nonzero((casted_preds - 1) * labels, axis=-1).astype(np.float32)
		TP = np.count_nonzero(casted_preds * labels, axis=-1).astype(np.float32)
		FP = np.count_nonzero(casted_preds * (labels - 1), axis=-1).astype(np.float32)

		#Compute recall
		recall=np.mean(TP/(TP+FN+1e-9))

		#EXCLUDE EXAMPLES WITH 0 POSITIVE PREDICTIONS FOR PRECISION ONLY
		if (self.exclude_empty_prec):
			#compute which column to keep
			mask=np.sum(casted_preds, axis=-1)>0
			#apply mask
			TP=TP[mask]
			FP=FP[mask]

		#compute precision
		single_prec=TP/(TP+FP+1e-9)

		if len(single_prec)==0:
			single_prec=[0]


		precision=np.mean(single_prec)

		f1_score=(2*precision * recall)/(precision+recall +1e-9)

		return f1_score

	def __call__(self, y_true, y_pred):
		for idx, thre in enumerate(self.thresholds):
			f1_score=self._f1_score(y_true, y_pred, thre)

			self.f1_thresholds[idx] += f1_score

		#update counter
		self.counter += 1


	def result(self):
		return np.max(self.f1_thresholds)/(self.counter + 1e-9)
	
	def reset_states(self):
		self.counter=0
		self.f1_thresholds*=0

class Evaluator():
	def __init__(self, data_dir):
		#thresholds for F1score
		self.thresholds=np.arange(start=0.1, stop=1.0, step=0.1)

		#Used to retrieve the go class of each go term
		with open(data_dir+"dataset_config.yaml", 'r') as f:
			self.dataset_config=yaml.safe_load(f)

		self.graph = obonet.read_obo(data_dir+'go-basic.obo')

		self.defineProteinCentricMetric()
		self.defineGOTermCentricMetric()
		self.defineGOClassCentricMetric()

	def defineProteinCentricMetric(self):
		self.model_f1max=F1MaxScore(self.thresholds, name="model_f1_max")
		self.model_recall=tf.keras.metrics.Recall()
		self.model_precision=tf.keras.metrics.Precision()

	def defineGOTermCentricMetric(self):
		self.go_terms_f1max_results=[[] for i in self.dataset_config['available_gos']]
		self.go_terms_recall_results=[[] for i in self.dataset_config['available_gos']]
		self.go_terms_precision_results=[[] for i in self.dataset_config['available_gos']]

		self.go_terms_f1max_results_times=[0 for i in self.dataset_config['available_gos']]
		self.go_terms_recall_results_times=[0 for i in self.dataset_config['available_gos']]
		self.go_terms_precision_results_times=[0 for i in self.dataset_config['available_gos']]

		self.go_terms_f1max=F1MaxScore(self.thresholds, name="GO_term_f1_max")
		self.go_terms_recall=tf.keras.metrics.Recall()
		self.go_terms_precision=tf.keras.metrics.Precision()

	def defineGOClassCentricMetric(self):
		#Stores the 3 metrics for each go class
		self.metrics_go_classes={
			'cellular_component':{},
			'biological_process':{},
			'molecular_function':{}
		}

		#Create the 3 metrics for each go class
		for go_class in self.metrics_go_classes:
			self.metrics_go_classes[go_class]['f1']=F1MaxScore(self.thresholds, name=go_class+"_f1_max")
			self.metrics_go_classes[go_class]['recall']=tf.keras.metrics.Recall(name=go_class+"_recall")
			self.metrics_go_classes[go_class]['precision']=tf.keras.metrics.Precision(name=go_class+"_precision")

		#Stores the GO term indexs in order to split the predictions in CC, BP and MF
		self.go_classes_idxs={
			'cellular_component':[],
			'biological_process':[],
			'molecular_function':[]
		}

		for idx, go_term in enumerate(self.dataset_config['available_gos']):
			self.go_classes_idxs[self.graph.node[go_term]['namespace']].append(idx)

	def updateProteinCentricMetric(self,y_true, y_pred):
		self.model_f1max(y_true.numpy(), y_pred)
		self.model_recall(y_true, y_pred)
		self.model_precision(y_true, y_pred)

	def updateGOTermCentricMetric(self,y_true, y_pred):
		'''
		This function computes the F1 score, precision and recall for each go term
		'''

		for go_term_idx in range(y_true.shape[1]):
			#Extract go term labels and predictions from the batch
			go_term_y_true=np.reshape(y_true[:, go_term_idx], (-1, 1))
			go_term_y_pred=np.reshape(y_pred[:, go_term_idx], (-1, 1))

			mask=go_term_y_true!=0
			go_term_y_true=np.reshape(go_term_y_true[mask], (-1, 1))
			go_term_y_pred=np.reshape(go_term_y_pred[mask], (-1, 1))

			if (len(go_term_y_true) !=0):
				#Compute the metrics for the go term
				self.go_terms_f1max(go_term_y_true, go_term_y_pred)
				self.go_terms_recall(go_term_y_true, go_term_y_pred)
				self.go_terms_precision(go_term_y_true, go_term_y_pred)

				#Save the results
				self.go_terms_f1max_results[go_term_idx].append(self.go_terms_f1max.result())
				self.go_terms_recall_results[go_term_idx].append(self.go_terms_recall.result().numpy())
				self.go_terms_precision_results[go_term_idx].append(self.go_terms_precision.result().numpy())
				
				self.go_terms_f1max_results_times[go_term_idx] += 1
				self.go_terms_recall_results_times[go_term_idx] += 1
				self.go_terms_precision_results_times[go_term_idx] += 1

				#Reset the metrics to not affect the next go term
				self.go_terms_f1max.reset_states()
				self.go_terms_recall.reset_states()
				self.go_terms_precision.reset_states()

	def updateGOClassCentricMetric(self,y_true, y_pred):
		'''
		This function computes the F1 score, precision and recall for BP, CC and MF
		'''

		bp_idxs=np.asarray(self.go_classes_idxs['biological_process'])
		cc_idxs=np.asarray(self.go_classes_idxs['cellular_component'])
		mf_idxs=np.asarray(self.go_classes_idxs['molecular_function'])

		#get predictions and labels for each go class
		y_trues_bp=np.transpose(np.transpose(y_true)[bp_idxs])
		y_trues_cc=np.transpose(np.transpose(y_true)[cc_idxs])
		y_trues_mf=np.transpose(np.transpose(y_true)[mf_idxs])

		y_preds_bp=np.transpose(np.transpose(y_pred)[bp_idxs])
		y_preds_cc=np.transpose(np.transpose(y_pred)[cc_idxs])
		y_preds_mf=np.transpose(np.transpose(y_pred)[mf_idxs])

		#Update the metrics
		for metric_name in self.metrics_go_classes['cellular_component']:
			metric_obj=self.metrics_go_classes['cellular_component'][metric_name]
			metric_obj(y_trues_cc, y_preds_cc)

		for metric_name in self.metrics_go_classes['biological_process']:
			metric_obj=self.metrics_go_classes['biological_process'][metric_name]
			metric_obj(y_trues_bp, y_preds_bp)
		
		for metric_name in self.metrics_go_classes['molecular_function']:
			metric_obj=self.metrics_go_classes['molecular_function'][metric_name]
			metric_obj(y_trues_mf, y_preds_mf)

	def resultsProteinCentricMetric(self):
		print('Protein-centric results:')
		print('>>Model f1_max score:  {:.2f}'.format(self.model_f1max.result()))
		print('>>Model recall:    {:.2f}'.format(self.model_recall.result().numpy()))
		print('>>Model precision: {:.2f}'.format(self.model_precision.result().numpy()))


		return {
			'f1':        self.model_f1max.result(),
			'recall':    self.model_recall.result().numpy(),
			'precision': self.model_precision.result().numpy(),
			'f1_list':self.model_f1max.f1_thresholds/self.model_f1max.counter
			}

	def resultsGOTermCentricMetric(self):
		print('\nGO term-centric results:')
		f1_scores=[]
		for go_term_values in self.go_terms_f1max_results:
			if len(go_term_values)!=0:
				f1_scores.append(np.mean(go_term_values))
			else:
				f1_scores.append(0)

		recall_scores=[]
		for go_term_values in self.go_terms_recall_results:
			if len(go_term_values)!=0:
				recall_scores.append(np.mean(go_term_values))
			else:
				recall_scores.append(0)

		precision_scores=[]
		for go_term_values in self.go_terms_precision_results:
			if len(go_term_values)!=0:
				precision_scores.append(np.mean(go_term_values))
			else:
				precision_scores.append(0)

		print('>>GO Terms average f1_max score:  {:.2f}'.format(np.mean(f1_scores)))
		print('>>GO Terms average recall:    {:.2f}'.format(np.mean(recall_scores)))
		print('>>GO Terms average precision: {:.2f}'.format(np.mean(precision_scores)))

		return {
			'f1':f1_scores,
			'f1_counter':self.go_terms_f1max_results_times,
			'recall':recall_scores,
			'recall_counter':self.go_terms_recall_results_times,
			'precision':precision_scores,
			'precision_counter':self.go_terms_precision_results_times,

		}

	def resultsGOClassCentricMetric(self):
		print('\nGO class-centric results:')
		metrics=[key for key in self.metrics_go_classes['cellular_component']]
		for metric in metrics:
			for go_class in self.metrics_go_classes:
				metric_obj=self.metrics_go_classes[go_class][metric]
				if (metric=='f1'):
					print('>>{} {}: {:.2f}'.format(go_class, metric, metric_obj.result()))
				else:
					print('>>{} {}: {:.2f}'.format(go_class, metric, metric_obj.result().numpy()))
			print('\n')
		'''
		results: {
			'cellular component': {
				'f1':score
				'recall':score
			},
			'biological process': {
				'f1': ...
			}
		}
		'''
		results={}
		for go_class in self.metrics_go_classes:
			results[go_class]={}
			for metric_name in self.metrics_go_classes[go_class]:
				metric_obj=self.metrics_go_classes[go_class][metric_name]
				if (metric_name=='f1'):
					results[go_class][metric_name]=metric_obj.result()
				else:
					results[go_class][metric_name]=metric_obj.result().numpy()


		return results



	