import tensorflow as tf

class F1MaxScore(tf.keras.metrics.Metric):
	'''
	This function computes the first evaluation metric. 
	The protein-centric evaluation.
	Compute f1-score for thresholds between 0.1 and 0.9 and use the 
	best f1-score as output.
	Protein-centric evaluation measures how accurately methods can assign functional terms to a protein.

	'''
	def __init__(self, thres, name='f1_score', **kwargs):
		super(F1MaxScore, self).__init__(name=name, **kwargs)
		self.f1_max_buffer=tf.Variable(0.)
		self.f1_max_sum=tf.Variable(0.)
		self.counter=tf.Variable(0.)

		self.thresholds=tf.Variable(thres, name='thresholds')
		self.exclude_empty_prec=True

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

		#Apply threshold
		casted_preds=tf.cast(preds>tf.cast(threshold, tf.float32), tf.float32)

		FN = tf.math.count_nonzero((casted_preds - 1) * labels, dtype=tf.float32, axis=-1)
		TP = tf.math.count_nonzero(casted_preds * labels, dtype=tf.float32, axis=-1)
		FP = tf.math.count_nonzero(casted_preds * (labels - 1), dtype=tf.float32, axis=-1)

		#Compute recall
		recall=tf.reduce_mean(tf.math.divide(TP,(tf.math.add(TP,FN)+1e-6)))

		#EXCLUDE EXAMPLES WITH 0 POSITIVE PREDICTIONS FOR PRECISION ONLY
		if (self.exclude_empty_prec):
			#compute which column to keep
			mask=tf.reduce_sum(casted_preds, axis=-1)>0
			#apply mask
			TP=tf.boolean_mask(TP,mask)
			FP=tf.boolean_mask(FP,mask)

		#compute precision
		single_prec=tf.math.divide(TP,(tf.math.add(TP,FP)+1e-6))

		size=tf.size(single_prec)
		#handle when there are no example with positive predictions
		def f1(): return tf.zeros([1], dtype=tf.float32)
		def f2(): return single_prec
		single_prec=tf.cond(tf.math.equal(tf.size(single_prec),0), f1, f2)

		precision=tf.reduce_mean(single_prec)

		f1_score=(2*precision * recall)/(precision+recall +1e-6)

		return f1_score


	@tf.function
	def update_state(self, y_true, y_pred, sample_weight=None):
		self.f1_max_buffer.assign(0)

		for thre in self.thresholds:
			f1_score=self._f1_score(y_true, y_pred, thre)
			#tf.print('Threshold:', thre, '| Score:', f1_score)
			tf.cond(
				tf.math.greater(f1_score, self.f1_max_buffer), 
				lambda: self.f1_max_buffer.assign(f1_score), 
				lambda: self.f1_max_buffer)
		
		self.f1_max_sum.assign_add(self.f1_max_buffer)
		self.counter.assign_add(1)

		return self.f1_max_buffer

	def result(self):
		return self.f1_max_sum/tf.math.add(self.counter, 1e-6)
	
	def reset_states(self):
		self.f1_max_sum.assign(0)
		self.counter.assign(0)
