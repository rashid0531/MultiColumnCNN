import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image, ImageFile
import numpy as np
# import cv2
import pandas as pd

from matplotlib import pyplot as plt

class MCNN():

    def __init__(self,input_image):

        # column1_design : a tuple that contains the parameters for each layer of CNN.

        self.column1_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [16,9,2],
            'maxPool1':[2,1],
            'conv2': [32,7,2],
            'maxPool2': [2,1],
            'conv3': [16,7,2],
            'conv4': [8,7,2],

        }

        self.column1_output = self.Shallow(input_image,self.column1_design)
        # self.column2_output
        # self.column3_output


    def Shallow(self,input_image,properties):

        # First convolutional layer
        conv1 = tf.layers.conv2d(input_image, filters=properties['conv1'][0], kernel_size=properties['conv1'][1],
                                 strides=[properties['conv1'][2], properties['conv1'][2]], padding="SAME", activation=tf.nn.relu)

        # Max pool layer - 1st
        max_pool1 = tf.nn.max_pool(conv1, ksize=[1, properties['maxPool1'][0], properties['maxPool1'][0], 1],
                                   strides=[1, properties['maxPool1'][1],properties['maxPool1'][1], 1], padding="VALID")

        # 2nd convolutional layer
        conv2 = tf.layers.conv2d(max_pool1, filters=properties['conv2'][0], kernel_size=properties['conv2'][1],
                                 strides=[properties['conv2'][2], properties['conv2'][2]], padding="SAME",
                                 activation=tf.nn.relu)

        # Max pool layer - 2nd
        max_pool2 = tf.nn.max_pool(conv2, ksize=[1, properties['maxPool2'][0], properties['maxPool2'][0], 1],
                                   strides=[1, properties['maxPool2'][1], properties['maxPool2'][1], 1],
                                   padding="VALID")

        # 3rd convolutional layer
        conv3 = tf.layers.conv2d(max_pool2, filters=properties['conv3'][0], kernel_size=properties['conv3'][1],
                                 strides=[properties['conv3'][2], properties['conv3'][2]], padding="SAME",
                                 activation=tf.nn.relu)

        # 3rd convolutional layer
        conv4 = tf.layers.conv2d(conv3, filters=properties['conv4'][0], kernel_size=properties['conv4'][1],
                                 strides=[properties['conv4'][2], properties['conv4'][2]], padding="SAME",
                                 activation=tf.nn.relu)

        return conv4


def read_image_using_PIL(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(image)
    # image = image.resize((224,224))
    image = np.asarray(image, np.uint8)
    print(np.sum(image))
    # img = Image.fromarray(image)
    # img.show()
    return image


if __name__ == "__main__":

    X = tf.placeholder(tf.float32, [1, 224, 224, 3])
    ob1 = MCNN(X)


    input_path= "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train/1_9.jpg"
    input_path_3channels = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/original/shanghaitech/part_A_final/train_data/images/IMG_1.jpg"
    gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den/100_1.csv"

    # input_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train/1_9.jpg"
    # input_path_3channels = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/original/shanghaitech/part_A_final/train_data/images/IMG_1.jpg"
    # gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den/52_8.csv"



    den = pd.read_csv(gt_path).values
    den = np.array(den)
    print(np.sum(den))
    print(den)

    # plt.imshow(den)
    # plt.show()
    #
    image = read_image_using_PIL("/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps/100_1.png")

    print(den.shape)

    plt.imshow(den)
    plt.show()
    #
    # image = read_image_using_PIL(input_path_3channels)
    # print(np.shape(image))


    print(image)


    # gt_string = tf.read_file(groundTruth_csvPath)
    # gt_image_decoded = tf.image.decode_jpeg(gt_string, channels=1)
    #
    # # !!!!!!!!!!! Resizing maynot be necessary !!!!!!!!!!!
    # # gt_image_resized = tf.image.resize_images(gt_image_decoded, [160, 256])
    # gt = tf.cast(gt_image_decoded, tf.float32)
    # # print(np.shape(image))
    #
    # # config = tf.ConfigProto()
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #
    #
    # # with tf.Session() as sess:
    # #     # Initialize all variables
    # #     sess.run(tf.global_variables_initializer())
    # #     output = sess.run(ob1.column1_output, feed_dict={X: [image]})
    # #
    # # print(np.shape(output))
    # # output_1 = output[0,:,:,4]
    # #
    # # # print(type(output))
    # # img = np.array(output_1,dtype = np.uint8)
    # #
    # # plt.imshow(img, interpolation='nearest')
    # # plt.show()
    # # # print(img.shape)
    # # # # print(img)
    # # # img = Image.fromarray(np.uint8(img), mode="RGB")
    # # # img.show()