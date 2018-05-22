import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import numpy as np
import MultiColumnCNN.MultiColumnCNN.Tensorflow.prepare as prepare
import MultiColumnCNN.MultiColumnCNN.Tensorflow.MCNN as models
from matplotlib import pyplot as plt
import MultiColumnCNN.MultiColumnCNN.Tensorflow.config as config

############################### Parameters #############################################
batch_size = 4
learning_rate = 0.001

############################### Input Paths ############################################

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

train_images,train_gt = iterator_train.get_next()

############################### Graph Initialization ###################################


width = config.input_image_width
height = config.input_image_height
channels = config.input_image_channels

# Placeholder for input image.
X = tf.placeholder(tf.float32, [batch_size, width, height, channels])

# Place Holder for ground truth.
# !!!!!!! Need to change the shape for multichannel images. !!!!!!!!!!!!!!!!!!!!!!!!
Y = tf.placeholder(tf.float32, shape= [batch_size,config.ground_truth_width,config.ground_truth_height,channels],name = "ground_truth")

mcnn_model = models.MCNN(X)
predicted_density_map = mcnn_model.final_layer_output

############################### loss function ###########################################

with tf.name_scope("loss"):

    # Designing simple cost function.

    cost = tf.sqrt(tf.reduce_mean(tf.square(Y - predicted_density_map)))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cost)



############################### Graph execution ##########################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    images,gt = sess.run([train_images,train_gt])
    print(images.shape,gt.shape)

    output = sess.run(cost,feed_dict= {X: images,Y:gt})
    output = np.array(output)
    # print(output.shape)
    print(output)