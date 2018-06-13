import tensorflow as tf
import os
from PIL import Image, ImageFile
import numpy as np
import MultiColumnCNN.MultiColumnCNN.Tensorflow.prepare as prepare
import MultiColumnCNN.MultiColumnCNN.Tensorflow.MCNN as models
from matplotlib import pyplot as plt
import MultiColumnCNN.MultiColumnCNN.Tensorflow.config as config
from datetime import datetime
from tensorflow.python.client import timeline

############################### Parameters #############################################
batch_size = 3
learning_rate = 0.000001


############################### Input Paths ############################################

train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"

test_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

test_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"

log_path = "/home/mohammed/tf_logs"

if not os.path.exists(log_path):
    os.makedirs(log_path)

# train_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"
#
# train_gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"

train_imgset, gt_train  = prepare.get_trainDataSet(train_path, train_gt_path)
print(train_imgset[100],gt_train[100])

# A vector of filenames for trainset.
images_input_train = tf.constant(train_imgset)
images_gt_train = gt_train

dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))

# At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
# So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

Batched_dataset_train = dataset_train.map(
        lambda img,gt: tf.py_func(prepare.read_npy_file, [img,gt], [img.dtype,tf.float32]))

Batched_dataset_train = Batched_dataset_train\
                        .shuffle(buffer_size=4000)\
                        .map(prepare._parse_function)\
                        .batch(batch_size=batch_size)\
                        .repeat()

# Batched_dataset_train = Batched_dataset_train\
#                         .map(prepare._parse_function)\
#                         .batch(batch_size=batch_size)\
#                         .repeat()

# # Iterator for train dataset.
iterator_train = Batched_dataset_train.make_one_shot_iterator()

train_images,train_gt = iterator_train.get_next()


############################### Graph Initialization ###################################

width = config.input_image_width
height = config.input_image_height
channels = config.input_image_channels

# Placeholder for input image.
X = tf.placeholder(tf.float32, [None, width, height, channels])

# Place Holder for ground truth.
# !!!!!!! Need to change the shape for multichannel images. !!!!!!!!!!!!!!!!!!!!!!!!
Y = tf.placeholder(tf.float32, shape= [None,config.ground_truth_width,config.ground_truth_height,channels],name = "ground_truth")

mcnn_model = models.MCNN(X)
predicted_density_map = mcnn_model.final_layer_output

############################### loss function ###########################################

with tf.name_scope("loss"):

    # Designing mse as cost function.

    # cost = tf.losses.mean_squared_error(Y, predicted_density_map)

    cost = tf.reduce_mean((tf.reduce_sum(tf.square(tf.subtract(Y, predicted_density_map)),axis=[1,2,3],keepdims=True)))
    # cost = tf.reduce_mean(cost1)


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)

    # For evaluating the model, The cost funtion needs to be computed based on the the difference between sum of predicted density map and sum of ground truth density map.
    # It doesn't compute pixel wise differences.

    # Evaluate model

    sum_of_Y = tf.reduce_sum(Y,axis=[1,2,3],keepdims=True)

    sum_of_predicted_density_map = tf.reduce_sum(predicted_density_map,axis=[1,2,3],keepdims=True)

    mse  = tf.sqrt(tf.reduce_mean(tf.square(sum_of_Y - sum_of_predicted_density_map)))

############################### Log and Summary ###########################################

root_log_dir_for_tflog = log_path
num_steps = len(train_imgset)//batch_size
display_step = 10

# tf log initialization.
currenttime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/run-{}/".format(root_log_dir_for_tflog,currenttime)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())

# summary writter - mse.
cost_summary_train = tf.summary.scalar("Training loss", mse)
cost_summary_test = tf.summary.scalar("Testing loss", cost)

############################### Graph execution ##########################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epoch = 0

    # images, gt = sess.run([train_images, train_gt])
    # output = sess.run([sum_of_Y,sum_of_predicted_density_map], feed_dict={X: images, Y: gt})
    # print(output[0].shape,output[1].shape)

    while (epoch < 5):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        for step in range(1, num_steps + 1):

            images,gt = sess.run([train_images,train_gt])
            output = sess.run(train_op,feed_dict= {X: images,Y:gt})

            if (step % display_step == 0):

                # Collecting Trainset MSE loss.
                loss, loss_summary_train = sess.run([mse, cost_summary_train],
                                                  feed_dict={X: images,Y:gt})

                print("training loss(MSE) over batch: {}".format(loss))

                # print(np.array(loss))

                file_writer.add_summary(loss_summary_train, epoch*num_steps + step)

        epoch += 1



file_writer.close()