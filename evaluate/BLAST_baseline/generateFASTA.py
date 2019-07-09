import tensorflow as tf
import os 
import yaml
import numpy as np

with open("../../hyperparams.yaml", 'r') as f:
	hyperparams=yaml.safe_load(f)

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

    batch={
        'X':inputData
    }

    return batch

def BLASTPrediction(encoded_seqs, seq_idx):
	'''
	This function takes a sequence of indexed aminos and use the list of aminos to 
	convert the sequence into a string sequence. 
	The string sequences are saved in FASTA format to use query BLAST on Uniprot database.
	'''
	queries_file= open("queries.fasta","a+")
	aminos_letters=np.asarray(hyperparams['unique_aminos'])

	print('Single sequence shape', encoded_seqs.shape)
	aminos_list=np.reshape(aminos_letters[encoded_seqs[0]], -1).tolist()
	aminos=''.join(aminos_list)

	#save sequence as fasta format 
	queries_file.write(">seq{}\n{}\n".format(seq_idx, aminos))

	queries_file.close()

#1. LOAD TEST EXAMPLES
test_folder_path='../../prepare/data/test/'
test_files=[test_folder_path+fn for fn in os.listdir(test_folder_path)]
dataset=tf.data.TFRecordDataset(test_files)

dataset=dataset.map(decodeTFRecord)
dataset=dataset.batch(1)

count=0
for idx, batch in enumerate(dataset):
    if idx==500:
        break

    BLASTPrediction(batch['X'].numpy(), idx)
    count=idx
print('Number of test sequences (starting from 0):', count)