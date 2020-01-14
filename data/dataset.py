'''
1. generate a dataset

    tf.data.Dataset.from_tensor_slices
    tf.data.TextLineDataset()
    tf.data.FixedLengthRecordDataset()
    tf.data.TFRecordDataset()

    tf.contrib.data.make_csv_dataset

2. modify the dataset
    map
    batch
    shuffle
    repeat

3. use the dataset (iterate the dataset)
    none eager:
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()

    eager enabled by default in tf2.0:
        # iterate the dataset
        for x in dataset:
            print(x)
'''

import tensorflow as tf
import tempfile

# create dataset from tensor and file
ds_tensors = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
# iterate the dataset
for x in ds_tensors:
    print(x)

batched_dataset = ds_tensors.batch(2)
# iterate the dataset
for x in batched_dataset:
    print(x)

batched_dataset2 = batched_dataset.batch(2)
# iterate the dataset
for x in batched_dataset2:
    print(x)

shuffled_dataset = batched_dataset.shuffle(2)
# iterate the dataset
for x in shuffled_dataset:
    print(x)


# apply transformations to the dataset
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

# iterate the dataset
for x in ds_tensors:
    print(x)

# TextLineDataset example:
_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)
ds_file = ds_file.batch(2)

for x in ds_file:
    print(x)
    print(x.numpy())