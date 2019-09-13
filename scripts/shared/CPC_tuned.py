import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import Sequence 
import numpy as np

class Model():
    def __init__(self, learning_rate):
        self.name='CPC_tuner/'
        self.learning_rate=learning_rate
    
    def prepareBatch(self, x_set, y_set, batch_size):
        return BatchGenerator(x_set, y_set,  batch_size)

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

    def architecture(self, cpc, num_labels):
        cpc_model=tf.keras.Model(
            inputs=cpc.get_layer('encoder_input').input,
            outputs=cpc.get_layer('rnn').output,
            name='CPC'
        )

        cpc_model.trainable=False

        x_input=cpc_model.get_layer('encoder_input').input

        cpc_output=cpc_model(x_input)[:, -1] #get last output from [batch, timesteps, rnn_units]

        tuner_output=layers.Dense(units=num_labels, activation='sigmoid', name='tuner_output')(cpc_output)

        fine_tuner=tf.keras.Model(
            inputs=x_input,
            outputs=tuner_output,
            name='CPC_tuner'
        )


        # Compile model
        fine_tuner.compile(
            optimizer=tf.keras.optimizers.Adadelta(lr=self.learning_rate),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=['binary_accuracy', self.precision, self.recall, self.f1score]
        )

        fine_tuner.summary()

        return fine_tuner


class BatchGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        '''
        Used by fit or fit_generator to set the number of steps per epoch
        '''
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        '''
        Batch comes in shape [batch_size, sequence_aa, 1].
        This function returns 2 tensors:
        -InputData: [batch_size, sequence_length, window_size, encoding_length]
        -labels: [batch_size, num_functions] #multi label encoding
        '''
        b_start=idx * self.batch_size
        b_end=(idx + 1) * self.batch_size
        inds = self.indices[b_start:b_end]
        batch_input = self.x[inds]
        batch_label = self.y[inds]

        batch_size=self.batch_size
        sequence_aa=1078
        padding=10
        sequence_length=32
        num_predic_terms=4
        num_samples=4
        window_size=34
        encoding_length=1
        stride=34

        num_functions=1918

        #1. batch encoding
        #batch_encoded=np.zeros((batch_size, sequence_aa, encoding_length))
        #batch_encoded[:, :, batch[:,:]]=1
        batch_encoded=np.reshape(batch_input, (batch_size, sequence_aa, 1))

        #2. generate input data
        #padding
        if padding>0:
            pads=np.zeros((batch_size, padding, 1), dtype=np.int8)-1
            batch_encoded=np.concatenate((batch_encoded, pads), axis=1)
            
        inputData=np.zeros((batch_size, sequence_length, window_size, encoding_length), dtype=np.int8)-1
        for i in range(sequence_length):
            patch_start=i*stride
            patch=batch_encoded[:, patch_start: patch_start+window_size]
            inputData[:, i]=patch 

        #3. Generate labels
        labels=np.zeros((batch_size, num_functions))
        for idx, labels_idxs in enumerate(batch_label):
            labels[idx][labels_idxs]=1

        return inputData, labels

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)