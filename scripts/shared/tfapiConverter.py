'''
generateTFRecord() takes as input a bach of examples (x,y) 
and output a single tfrecord. N:(x,y) -> N:(tf.examples) -> 1:tfrecord

decodeTFRecord() takes as input a tfrecord and output a banch of examples(x,y)
'''
import tensorflow as tf
import numpy as np
import yaml

with open('../../hyperparams.yaml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']

with open(data_dir+'dataset_config.yaml', 'r') as f:
    dataset_configs=yaml.load(f)

unique_labels=len(dataset_configs['available_gos'])
#get max length to use
max_length_amino=configs['max_length_aminos']

def generateTFRecord(examples_x, examples_y, filename):
    '''
    In a tf.train.Example, features can be only of 3 types:
        int64: int or bools
        bytes: use this for text and TENSORS using numpy array-string converter
        float
    '''
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def serialize_example(input_example, label):
        """
        Creates a tf.Example

        Args:
            input (array): np.array
            label (array): np.array

        Return:
            serialized proto example
        """
        if (type(input_example)!=type(np.asarray([]))):
            input_example=np.asarray(input_example, dtype=np.uint8)

        if (type(label)!=type(np.asarray([]))):
            label=np.asarray(label, dtype=np.uint8)

        shapeInput=input_example.shape
        shapeLabel=label.shape
        features = {
            'inputData': _bytes_feature(input_example.tostring()),
            'labelData':  _bytes_feature(label.tostring()),
            'hot_classes': _int64_feature(unique_labels),
        }

        # Create tf.train.Features obj asssigning encoded data
        tf_features=tf.train.Features(feature=features)

        # Create tf.train.Example assigning tf.train.Features obj
        example_proto = tf.train.Example(features=tf_features)

        return example_proto.SerializeToString()

    # Write the `tf.Example` observations to the file.
    with tf.io.TFRecordWriter(filename) as writer:
        for x,y in zip(examples_x, examples_y):
            example = serialize_example(x,y)

            writer.write(example)

def decodeTFRecord(example_proto):
    # Create a description of the features.
    feature_description = {
        'inputData': tf.io.FixedLenFeature([], tf.string),
        'labelData': tf.io.FixedLenFeature([], tf.string),
        'hot_classes': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the input tf.Example proto using the dictionary above.
    example=tf.io.parse_single_example(example_proto, feature_description)
    
    #one-hot encode amino acids
    inputData=tf.cast(tf.io.decode_raw(example['inputData'], tf.uint8), tf.float32)/25.
    inputData=tf.reshape(inputData, (max_length_amino,1))
    
    #one-hot encode gos notations
    decodeLabel=tf.io.decode_raw(example['labelData'], tf.uint8)
    hot_label=tf.one_hot(decodeLabel, tf.cast(example['hot_classes'], tf.int32), dtype=tf.float32)
    labelData=tf.math.reduce_max(hot_label, axis=0)

    return inputData, labelData