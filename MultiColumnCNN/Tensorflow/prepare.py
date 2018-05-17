import glob
import os

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


if __name__ == "__main__":

    train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

    train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    get_trainDataSet(train_path,train_gt_path)