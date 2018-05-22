import glob
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import io
import MultiColumnCNN.MultiColumnCNN.Tensorflow.config as config
import cv2

def get_trainDataSet(image_path,gt_path):

    image_data_train = []
    groundTruth_data_train = []

    for filename in glob.glob(image_path+"/*"):

        image_data_train.append(filename)

    # The sorted() function doesn't work with list that contains string values.
    image_data_train= sorted(image_data_train)

    for filename in glob.glob(gt_path + "/*"):
        groundTruth_data_train.append(filename)

    groundTruth_data_train = sorted(groundTruth_data_train)

    # This count is to check if the image name and ground truth csv files are in same index order.
    count = 0
    for i in range(0,len(image_data_train)):
        im = (image_data_train[0].split("/")[-1]).split(".")[0]
        gt = groundTruth_data_train[0].split("/")[-1].split(".")[0]

        if im != gt:
            count += 1

    if (count!=0):
        print("ground truth and images are not in same index order")

    return image_data_train,groundTruth_data_train


def readcsv(input_csv):

    decoded_csv = pd.read_csv(input_csv).values
    decoded_csv = np.array(decoded_csv)
    return decoded_csv


def read_npy_file(image_name,item):


    # The ground truth density map needs to be downsampled because after beign processed through the MAX-POOL layers the input is downsized in half for each MAX-POOL layer.
    data = np.load(item.decode())
    width =  int(config.input_image_width/4)
    height = int(config.input_image_height/4)
    data = cv2.resize(data, (width, height))
    data = data * ((width * height) / (width * height))

    # !!!!!!!!!!!!!!!! This reshaping doesn't need to be done if the density map is multichanneled. !!!!!!!!!!!!!!!!!!!!!!
    data = np.reshape(data, [data.shape[1], data.shape[0], 1])
    return image_name,data.astype(np.float32)


def _parse_function(image_path,groundTruth_path):

    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=config.input_image_channels)
    # Due to the variable size of input images, resizing was done to scale all images into a fix size.
    image_resized = tf.image.resize_images(image_decoded, [config.input_image_width, config.input_image_height])
    image = tf.cast(image_resized, tf.float32)

    return image,groundTruth_path


if __name__ == "__main__":

    # train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"
    #
    # train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    train_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

    train_gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    np_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/temp/train_density_maps/100_1.npy"

    read_npy_file(train_gt_path,np_path)