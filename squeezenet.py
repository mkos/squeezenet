import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

# define squeeze module
def squeeze(input, channels, layer_num):
    """
    Defines squeezed block for fire module.

    :param input: input tensor
    :param channels: number of output channels
    :param layer_num: layer number for naming purposes
    :return: output tensor convoluted with squeeze layer
    """
    layer_name = 'squeeze_' + str(layer_num)
    input_channels = input.get_shape().as_list()[3]

    with tf.name_scope(layer_name):
        weights = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels]))
        biases = tf.Variable(tf.zeros([1, 1, 1, channels]), name='biases')
        onebyone = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1), padding='VALID') + biases
        A = tf.nn.relu(onebyone)

    return A

# define expand module
def expand(input, channels_1by1, channels_3by3, layer_num):
    """
    Defines expand block for fire module.
    :param input: input tensor
    :param channels_1by1: number of output channels in 1x1 layers
    :param channels_3by3: number of output channels in 3x3 layers
    :param layer_num: layer number for naming purposes
    :return: output tensor convoluted with expand layer
    """

    layer_name = 'squeeze_' + str(layer_num)
    input_channels = input.get_shape().as_list()[3]

    with tf.name_scope(layer_name):
        weights1x1 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels_1by1]))
        biases1x1 = tf.Variable(tf.zeros([1, 1, 1, channels_1by1]), name='biases')
        onebyone = tf.nn.conv2d(input, weights1x1, strides=(1, 1, 1, 1), padding='VALID') + biases1x1
        A_1x1 = tf.nn.relu(onebyone)

        weights3x3 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, input_channels, channels_3by3]))
        biases3x3 = tf.Variable(tf.zeros([1, 1, 1, channels_3by3]), name='biases')
        threebythree = tf.nn.conv2d(input, weights3x3, strides=(1, 1, 1, 1), padding='SAME') + biases3x3
        A_3x3 = tf.nn.relu(threebythree)

    return tf.concat([A_1x1, A_3x3], axis=3)


# define fire module
def fire_module(input, squeeze_channels, expand_channels_1by1, expand_channels_3by3, layer_num):
    """
    Train fire module. Fire module does not change input height and width, only depth.
    :param input: input tensor
    :param squeeze_channels: number of channels for 1x1 squeeze layer
    :param expand_channels_1by1: number of channels for 1x1 expand layer
    :param expand_channels_3by3: number of channels for 3x3 expand layer
    :param layer_num: number of layer for naming purposes only
    :return: a tensor of shape [input_height x input_width x expand_channels_1by1 * expand_channels_3by3]
    """
    with tf.name_scope('fire_' + str(layer_num)):
        squeeze_output = squeeze(input, squeeze_channels, layer_num)
        return expand(squeeze_output, expand_channels_1by1, expand_channels_3by3, layer_num)


def model(input_height, input_width, input_channels, output_classes, pooling_size=(1, 3, 3, 1)):
    """
    Define tensorflow graph.
    :param input_height: input image height
    :param input_width: input image width
    :param input_channels: input image channels
    :param output_classes: number of output classes
    :param pooling_size: size of the pooling
    :return: list of input placeholders and output operations
    """
    with tf.Graph().as_default() as graph:
    # define placeholders
        input_image = tf.placeholder(tf.float32,
                                     shape=[None, input_height, input_width, input_channels],
                                     name='input_image')
        labels = tf.placeholder(tf.int32, shape=[None, 1])
        in_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.placeholder(tf.float32, shape=())

    # define structure of the net
    # layer 1 - conv 1
        with tf.name_scope('conv_1'):
            W_conv1 = tf.Variable(tf.contrib.layers.xavier_initializer()([7, 7, 3, 96]))
            b_conv1 = tf.Variable(tf.zeros([1, 1, 1, 96]))
            X_1 = tf.nn.conv2d(input_image, W_conv1, strides=(1, 2, 2, 1), padding='VALID') + b_conv1
            A_1 = tf.nn.relu(X_1)

        # layer 2 - maxpool
        # TODO: smaller size maxpool for small networks
        maxpool_1 = tf.nn.max_pool(A_1, ksize=pooling_size, strides=(1, 2, 2, 1), padding='VALID', name='maxpool_1')

        # layer 3-5 - fire modules
        fire_2 = fire_module(maxpool_1, 16, 64, 64, layer_num=2)
        fire_3 = fire_module(fire_2, 16, 64, 64, layer_num=3)
        fire_4 = fire_module(fire_3, 32, 128, 128, layer_num=4)

        # layer 6 - maxpool
        # TODO: smaller size maxpool for small networks
        maxpool_4 = tf.nn.max_pool(fire_4, ksize=pooling_size, strides=(1, 2, 2, 1), padding='VALID', name='maxpool_4')

        # layer 7-10 - fire modules
        fire_5 = fire_module(maxpool_4, 32, 128, 128, layer_num=5)
        fire_6 = fire_module(fire_5, 48, 192, 192, layer_num=6)
        fire_7 = fire_module(fire_6, 48, 192, 192, layer_num=7)
        fire_8 = fire_module(fire_7, 64, 256, 256, layer_num=8)

        # layer 11 - maxpool
        # TODO: smaller size maxpool for small networks
        maxpool_8 = tf.nn.max_pool(fire_8, ksize=pooling_size, strides=(1, 2, 2, 1), padding='VALID', name='maxpool_8')

        # layer 12 - fire 9 + dropout
        fire_9 = fire_module(maxpool_8, 64, 256, 256, layer_num=9)

        dropout_9 = tf.cond(in_training,
                            lambda: tf.nn.dropout(fire_9, keep_prob=0.5),
                            lambda: fire_9)

        # layer 13 - final
        with tf.name_scope('final'):
            W_conv10 = tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 512, output_classes]))
            b_conv10 = tf.Variable(tf.zeros([]))
            conv_10 = tf.nn.conv2d(dropout_9, W_conv10, strides=(1, 1, 1,1), padding='VALID') + b_conv10
            A_conv_10 = tf.nn.relu(conv_10)

        # avg pooling to get [1 x 1 x num_classes] must average over entire window oh H x W from input layer
        _, H_last, W_last, _ = A_conv_10.get_shape().as_list()
        pooled = tf.nn.avg_pool(A_conv_10, ksize=(1, H_last, W_last, 1), strides=(1, 1, 1, 1), padding='VALID')
        logits = tf.squeeze(pooled, axis=[2])

        # loss + optimizer
        one_hot_labels = tf.one_hot(labels, output_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # accuracy
        logits_class_num = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(logits_class_num, labels), dtype=tf.float32))

    return graph, input_image, labels, in_training, learning_rate, loss, accuracy, optimizer

def run(iterations, minibatch_size):
    # ImageNet
    # input_height = input_width = 227
    # input_channels = 3
    # output_classes = 10
    # CIFAR10
    input_height = input_width = 32
    input_channels = 3
    output_classes = 10

    # We expect input data in NHWC format, but keras returns it in NCHW, so we need to move dimensions around:
    # x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32)
    # y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
    # TODO: do it in tensorflow
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = np.transpose(x_train_nchw, [0, 3, 1, 2])
    # x_test = np.transpose(x_test_nchw, [0, 3, 1, 2])

    train_samples = x_train.shape[0]

    graph, input_batch, labels, in_training, learning_rate, loss, accuracy, optimizer = \
        model(input_height, input_width, input_channels, output_classes, (1, 2, 2, 1))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # pick random minibatch
            mb_start = np.random.randint(0, train_samples - minibatch_size)
            mb_end = mb_start + minibatch_size
            mb_data = x_train[mb_start:mb_end, :, :, :]
            mb_labels = y_train[mb_start:mb_end]

            feed_dict = {
                input_batch: mb_data,
                labels: mb_labels,
                in_training: True,
                learning_rate: 0.05
            }

            collectibles = [loss, accuracy, optimizer]

            loss_val, accurracy_val, _ = sess.run(collectibles, feed_dict=feed_dict)
            if i % 100 == 0:
                print('Iteration: {}\t, loss: {:.3f}\t, accuracy: {:.3f}'.format(i, loss_val, accurracy_val))

run(2000, 128)


# define session
# for n steps
#   create minibatch
#   run training step
#   run validation step
#   write results to output

# run testing summary
# print results
# bonus: save trained model

