import tensorflow as tf
import MultiColumnCNN.MultiColumnCNN.Tensorflow.prepare as prepare
import MultiColumnCNN.MultiColumnCNN.Tensorflow.MCNN as model

############################### Parameters #############################################
batch_size = 30
learning_rate = 0.000001
number_of_epoch = 50

############################### Input Paths ############################################

train_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train"

train_gt_path = "/home/mohammed/Projects/CrowdCount/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/train_density_maps"

def training_dataset(epochs=5, batch_size=batch_size):

    train_imgset, gt_train = prepare.get_trainDataSet(train_path, train_gt_path)
    print(train_imgset[100], gt_train[100])

    # A vector of filenames for trainset.
    images_input_train = tf.constant(train_imgset)
    images_gt_train = gt_train

    dataset_train = tf.data.Dataset.from_tensor_slices((images_input_train, images_gt_train))

    # At time of this writing Tensorflow doesn't support a mixture of user defined python function with tensorflow operations.
    # So we can't use one py_func to process data using tenosrflow operation and nontensorflow operation.

    Batched_dataset_train = dataset_train.map(
        lambda img, gt: tf.py_func(prepare.read_npy_file, [img, gt], [img.dtype, tf.float32]))

    Batched_dataset_train = Batched_dataset_train \
        .shuffle(buffer_size=4000) \
        .map(prepare._parse_function) \
        .batch(batch_size=batch_size) \
        .repeat(epochs)

    return Batched_dataset_train


def core_model(input_image):

    mcnn_model = model.MCNN(input_image)
    predicted_density_map = mcnn_model.final_layer_output
    return predicted_density_map

def training_model(input_fn):
    inputs = input_fn()
    image = inputs[0]
    gt  = inputs[1]
    predicted_density_map = core_model(image)
    cost = tf.losses.mean_squared_error(gt, predicted_density_map)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    return cost

def do_training(update_op, loss):

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            step = 0
            while True:
                _, loss_value = sess.run((update_op, loss))
                if step % 100 == 0:
                    print('Step {} with loss {}'.format(step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def create_parallel_optimization(model_fn, input_fn, optimizer, controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                loss = model_fn(input_fn)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss


def parallel_training(model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    update_op, loss = create_parallel_optimization(model_fn,
                                                   input_fn,
                                                   optimizer)

    do_training(update_op, loss)


if __name__=="__main__":

    tf.reset_default_graph()
    parallel_training(training_model, training_dataset(epochs=2))