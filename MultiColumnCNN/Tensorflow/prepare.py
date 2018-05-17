import glob
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import io

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


def _parse_function(image_path,groundTruth_csvPath):

    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)

    # image_decoded = tf.image.decode_jpeg(image_string,channels=3)

    # !!!!!!!!!!! Resizing maynot be necessary !!!!!!!!!!!
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image = tf.cast(image_resized, tf.float32)

    gt_string = tf.read_file(groundTruth_csvPath)
    gt_image_decoded = tf.image.decode_jpeg(gt_string, channels=1)

    # !!!!!!!!!!! Resizing maynot be necessary !!!!!!!!!!!
    gt_image_resized = tf.image.resize_images(gt_image_decoded, [224, 224])
    gt = tf.cast(gt_image_resized, tf.float32)


    return image,gt


if __name__ == "__main__":

    # train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"
    #
    # train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    train_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

    train_gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    get_trainDataSet(train_path,train_gt_path)