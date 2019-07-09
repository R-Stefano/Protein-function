import pickle
import tensorflow as tf
import os
import numpy as np
import yaml

with open("../../hyperparams.yaml", 'r') as f:
	hyperparams=yaml.safe_load(f)

goes_notations=np.asarray(hyperparams['available_goes'])
#BLAST PREDICTION
data=pickle.load(open('blast_predictions/blast_goes_predicted_last', 'rb'))

print('BLAST prediction:', data[0])

#GROUND-TRUTH PREDICTION
def decodeTFRecord(example_proto):
    # Create a description of the features.
    feature_description = {
        'inputData': tf.io.FixedLenFeature([], tf.string),
        'labelData': tf.io.FixedLenFeature([], tf.string),
        'hot_aminos': tf.io.FixedLenFeature([], tf.int64),
        'hot_classes': tf.io.FixedLenFeature([], tf.int64),
        'pad_aminos': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the input tf.Example proto using the dictionary above.
    example=tf.io.parse_single_example(example_proto, feature_description)
    
    #one-hot encode amino acids
    inputData=tf.io.decode_raw(example['inputData'], tf.uint8)

    decodeLabel=tf.io.decode_raw(example['labelData'], tf.uint8)

    batch={
        'X':inputData,
        'Y':decodeLabel
    }

    return batch

test_folder_path='../../prepare/data/test/'
test_files=[test_folder_path+fn for fn in os.listdir(test_folder_path)]
dataset=tf.data.TFRecordDataset(test_files)

dataset=dataset.map(decodeTFRecord)
dataset=dataset.batch(1)

for idx, example in enumerate(dataset):
    example_goes=goes_notations[example['Y'].numpy()]
    print(example_goes)
    break