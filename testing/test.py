import tensorflow as tf
import numpy as np

data=np.arange(0,10)
dataset=tf.data.Dataset.from_tensor_slices(data)

print('dataset:', data)
dataset=dataset.shuffle(len(data),reshuffle_each_iteration=False, seed=0)
print('\nshuffled:')
print([i.numpy() for i in dataset])

dataset=dataset.repeat(2)
print('\nRepeat:')
print([i.numpy() for i in dataset])

test=dataset.take(int(data.shape[0]*0.3))
print('\nTest data:')
print([i.numpy() for i in test])

train=dataset.skip(int(data.shape[0]*0.3))
print('\nTrain data:')
print([i.numpy() for i in train])


