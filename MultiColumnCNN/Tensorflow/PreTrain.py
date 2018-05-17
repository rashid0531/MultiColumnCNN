import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import numpy as np
import MultiColumnCNN.Tensorflow.prepare as prepare

from matplotlib import pyplot as plt

train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

train_imgset, gt_train  = prepare.get_trainDataSet(train_path, train_gt_path)

# A vector of filenames for trainset.
images_input_train = tf.constant(train_imgset)
images_gt_train = tf.constant(gt_train)

dataset_train = tf.data.Dataset.from_tensor_slices(images_input_train, images_gt_train)