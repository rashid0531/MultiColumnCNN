import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

def readcsv(input_csv,output_path):

    input_prefix = ((input_csv.split("/")[-1]).split(".")[0])

    decoded_csv = pd.read_csv(input_csv).values

    output_path = output_path+"/"+input_prefix
    decoded_csv = np.array(decoded_csv)
    plt.imsave(output_path,decoded_csv)


if __name__ == "__main__":

    train_gt_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_den"

    output_path = "/u1/rashid/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in glob.glob(train_gt_path + "/*"):
        readcsv(filename,output_path)
