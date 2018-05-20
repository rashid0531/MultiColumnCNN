import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import numpy as np
import MultiColumnCNN.MultiColumnCNN.Tensorflow.prepare as prepare

from matplotlib import pyplot as plt

############################### Parameters #############################################
batch_size = 1



############################### Input Paths #############################################

train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/temp/train"

train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/temp/train_density_maps"

# train_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"
#
# train_gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"


train_imgset, gt_train  = prepare.get_trainDataSet(train_path, train_gt_path)

print(gt_train[0])

# A vector of filenames for trainset.
images_input_train = tf.constant(train_imgset)
images_gt_train = gt_train

dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))

# At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
# So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

Batched_dataset_train = dataset_train.map(
        lambda img,gt: tf.py_func(prepare.read_npy_file, [img,gt], [img.dtype,tf.float32]))

Batched_dataset_train = Batched_dataset_train.map(prepare._parse_function)\
                        .batch(batch_size=batch_size)\
                        .repeat()

# # Iterator for train dataset.
iterator_train = Batched_dataset_train.make_one_shot_iterator()
#
train_images,train_gt = iterator_train.get_next()
#
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(train_images)

    print(output.shape)
