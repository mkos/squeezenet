import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10


def model(input_height, input_width, input_channels, output_classes):
    """
    Define tensorflow graph.
    :param input_height: input image height
    :param input_width: input image width
    :param input_channels: input image channels
    :param output_classes: number of output classes
    :return: list of input placeholders and output operations
    """
    with tf.Graph().as_default() as graph:
    # define placeholders
        input_image = tf.placeholder(tf.float32,
                                     shape=[None, input_height, input_width, input_channels],
                                     name='input_image')
        labels = tf.placeholder(tf.int64, shape=[None, 1])
        in_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.placeholder(tf.float32, shape=())
        reg = tf.placeholder(tf.float32, shape=())
        tf.summary.image('input image', input_image)

    # define structure of the net
    # layer 1 - conv 1
        with tf.name_scope('conv_1'):
            W_conv1 = tf.Variable(tf.contrib.layers.xavier_initializer()([5, 5, 3, 16]))
            b_conv1 = tf.Variable(tf.zeros([1, 1, 1, 16]))
            log_conv1 = tf.nn.conv2d(input_image, W_conv1, strides=(1, 1, 1, 1), padding='SAME') + b_conv1
            act_conv1 = tf.nn.relu(log_conv1)
            tf.summary.histogram('conv1 weights', W_conv1)
            tf.summary.histogram('conv1 biases', b_conv1)
            tf.summary.histogram('conv1 logits', log_conv1)
            tf.summary.histogram('conv1 activations', act_conv1)

        # layer 6 - maxpool
        maxpool_1 = tf.nn.max_pool(act_conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='maxpool_1')
        # [16 x 16]

        with tf.name_scope('conv_2'):
            W_conv2 = tf.Variable(tf.contrib.layers.xavier_initializer()([3, 3, 16, 32]))
            b_conv2 = tf.Variable(tf.zeros([1, 1, 1, 32]))
            log_conv2 = tf.nn.conv2d(maxpool_1, W_conv2, strides=(1, 1, 1, 1), padding='SAME') + b_conv2
            act_conv2 = tf.nn.relu(log_conv2)
            tf.summary.histogram('conv2 weights', W_conv2)
            tf.summary.histogram('conv2 biases', b_conv2)
            tf.summary.histogram('conv2 logits', log_conv2)
            tf.summary.histogram('conv2 activations', act_conv2)

        # layer 6 - maxpool
        maxpool_2 = tf.nn.max_pool(act_conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='maxpool_1')
        # [8 x 8]

        # layer 13 - final
        with tf.name_scope('final'):
            W_conv10 = tf.Variable(tf.contrib.layers.xavier_initializer()([8, 8, 32, output_classes]))
            b_conv10 = tf.Variable(tf.zeros([1, 1, 1, output_classes]))
            conv_10 = tf.nn.conv2d(maxpool_2, W_conv10, strides=(1, 1, 1, 1), padding='VALID') + b_conv10
            A_conv_10 = tf.nn.relu(conv_10)

            tf.summary.histogram('conv10 weights', W_conv10)
            tf.summary.histogram('conv10 biases', b_conv10)
            tf.summary.histogram('conv10 logits', conv_10)
            tf.summary.histogram('conv10 activations', A_conv_10)


        # avg pooling to get [1 x 1 x num_classes] must average over entire window oh H x W from input layer
        # _, H_last, W_last, _ = A_conv_10.get_shape().as_list()
        # pooled = tf.nn.avg_pool(A_conv_10, ksize=(1, H_last, W_last, 1), strides=(1, 1, 1, 1), padding='VALID')
        logits = tf.reshape(conv_10, [-1, output_classes])#tf.squeeze(conv_10, axis=[1, 2])
        #tf.summary.histogram('avg_pool', pooled)
        tf.summary.histogram('logits', logits)

        # loss + optimizer
        one_hot_labels = tf.one_hot(labels, output_classes, name='one_hot_encoding')
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits), name='loss')
        tf.summary.scalar('loss', loss)
    


        # accuracy
        predictions = tf.reshape(tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int64), [-1, 1])
        correct_predictions = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))
        #accuracy = tf.metrics.accuracy(labels, predictions)
        tf.summary.scalar('accuracy', accuracy)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        summaries = tf.summary.merge_all()

    return graph, input_image, labels, in_training, learning_rate, reg, loss, accuracy, summaries, optimizer

def next_experiment_dir(top_dir):
    """We need directory with consecutive subdirectories to store results of consecutive trainings. """
    dirs = [int(dirname) for dirname in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, dirname))]
    if len(dirs) > 0:
        return os.path.join(top_dir, str(max(dirs) + 1))
    else:
        return os.path.join(top_dir, '1')

def prepare_input(data, mu=None, sigma=None):

    #do mean normaization across all samples
    if mu is None:
        mu = np.mean(data)
        #mu = mu.reshape(1,-1)
    if sigma is None:
        sigma = np.std(data)
        #sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    return data, mu, sigma

def run(iterations, minibatch_size):
    # ImageNet
    # input_height = input_width = 227
    # input_channels = 3
    # output_classes = 10
    # CIFAR10
    input_height = input_width = 32
    input_channels = 3
    output_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, mu_train, sigma_train = prepare_input(x_train)
    x_test, _, _ = prepare_input(x_test, mu_train, sigma_train)
    train_samples = x_train.shape[0]

    graph, input_batch, labels, in_training, learning_rate, reg, loss, accuracy, summaries, optimizer = \
        model(input_height, input_width, input_channels, output_classes)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        experiment_dir = next_experiment_dir('/tmp/squeezenet')
        print("Creating output dir:", experiment_dir)
        train_writer = tf.summary.FileWriter(experiment_dir, sess.graph)

        for i in range(iterations):
            # pick random minibatch
            mb_start = np.random.randint(0, train_samples - minibatch_size)
            mb_end = mb_start + minibatch_size
            mb_data = x_train[mb_start:mb_end, :, :, :]
            mb_labels = y_train[mb_start:mb_end, :]

            feed_dict = {
                input_batch: mb_data,
                labels: mb_labels,
                in_training: True,
                learning_rate: 0.001,
                reg: 0.2
            }

            collectibles = [loss, accuracy, summaries, optimizer]

            loss_val, accuracy_val, s, _ = sess.run(collectibles, feed_dict=feed_dict)

            train_writer.add_summary(s, i)

            if i % 100 == 0:
                feed_dict = {
                    input_batch: x_test,
                    labels: y_test,
                    in_training: False,
                    learning_rate: 0.0004,
                    reg: 0.5
                }
                test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                print('Iteration: {}\t loss: {:.3f}\t accuracy: {:.3f}\t test accuracy: {:.3f}'.format(
                    i, loss_val, accuracy_val, test_accuracy))

run(10000, 128)


# define session
# for n steps
#   create minibatch
#   run training step
#   run validation step
#   write results to output

# run testing summary
# print results
# bonus: save trained model

