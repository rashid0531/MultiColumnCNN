import tensorflow as tf

class MCNN():

    def __init__(self,imput_image,keep_prob):

        # column1_design : a tuple that contains the parameters for each layer of CNN.

        column1_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [16,9,2],
            'maxPool1':[2,1],
            'conv2': [32,7,2],
            'maxPool2': [2,1],
            'conv3': [16,7,2],
            'conv4': [8,7,2],

        }

        self.column1_output = self.Shallow(self.column1_design)
        self.column2_output;
        self.column3_output



    def Shallow(self,properties):






if __name__ == "__main__":

    ob1 = MCNN()
