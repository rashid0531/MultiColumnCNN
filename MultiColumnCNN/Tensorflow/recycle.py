# def read_npy_file(image_name,item):
#
#
#     # The ground truth density map needs to be downsampled because after beign processed through the MAX-POOL layers the input is downsized in half for each MAX-POOL layer.
#     data = np.load(item)
#     width =  int(config.input_image_width/4)
#     height = int(config.input_image_height/4)
#     data = cv2.resize(data, (width, height))
#     data = data * ((width * height) / (width * height))
#
#     temp = np.zeros((data.shape[1],data.shape[0],1))
#     temp = np.reshape(data,[data.shape[1],data.shape[0],1])
#
#     np.savetxt("too.csv", temp[:,:,0], delimiter=",")
#     np.savetxt("foo.csv", data, delimiter=",")
#
#     return image_name,data.astype(np.float32)

import tensorflow as tf

x = tf.constant([[1, 1, 1], [1, 1, 1]])
p = tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

with tf.Session() as sess:
    out = sess.run(p)
    print(tf.rank(out))