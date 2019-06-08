'''
generateTFRecord() takes as input a bach of examples (x,y) 
and output a single tfrecord. N:(x,y) -> N:(tf.examples) -> 1:tfrecord

decodeTFRecord() takes as input a tfrecord and output a banch of examples(x,y)
'''
import tensorflow as tf
import numpy as np
def generateTFRecord(examples_x, examples_y, filename):
    '''
    In a tf.train.Example, features can be only of 3 types:
        int64: int or bools
        bytes: use this for text and matricies using tf.serialize_tensor
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
        input (array): np.array
        label (array): np.array
        """
        shapeInput=input_example.shape
        shapeLabel=label.shape
        features = {
            'inputData': _bytes_feature(input_example.tostring()),
            'labelData':  _bytes_feature(label.tostring()),
            'input_aminos': _int64_feature(shapeInput[0]),
            'amino_hot_length': _int64_feature(shapeInput[1]),
            'label_hot_length': _int64_feature(shapeLabel[-1])
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
        'input_aminos': tf.io.FixedLenFeature([], tf.int64),
        'amino_hot_length': tf.io.FixedLenFeature([], tf.int64),
        'label_hot_length': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the input tf.Example proto using the dictionary above.
    example=tf.io.parse_single_example(example_proto, feature_description)
    
    decodeInput=tf.io.decode_raw(example['inputData'], tf.uint8)
    inputData=tf.reshape(decodeInput, (example['input_aminos'],example['amino_hot_length']))

    decodeLabel=tf.io.decode_raw(example['labelData'], tf.uint8)
    labelData=tf.reshape(decodeLabel, (example['label_hot_length'],))

    batch={
        'X':inputData,
        'Y':labelData
    }

    return batch