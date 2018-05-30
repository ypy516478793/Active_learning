# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
X_train = data['X_train'][:]
Y_train = data['Y_train'][:]
X_valid = data['X_valid'][:]
Y_valid = data['Y_valid'][:]
data.close()

X_train = np.reshape(X_train, (-1, X_train.shape[1]*X_train.shape[2]))
X_valid = np.reshape(X_valid, (-1, X_valid.shape[1]*X_valid.shape[2]))

mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
X_train = (X_train - mean_train) / std_train

mean_valid = np.mean(X_valid, axis=0)
std_valid = np.std(X_valid, axis=0)
X_valid = (X_valid - mean_valid) / std_valid

# # Visualize decoder setting
# # Parameters
# learning_rate = 0.01
# training_epochs = 5
# batch_size = 256
# display_step = 1
# examples_to_show = 10
#
# # Network Parameters
# n_input = 1024  # MNIST data input (img shape: 28*28)
#
# # tf Graph input (only pictures)
# X = tf.placeholder("float", [None, n_input])
#
# # hidden layer settings
# n_hidden_1 = 256 # 1st layer num features
# n_hidden_2 = 128 # 2nd layer num features
# weights = {
#     'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
#     'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b2': tf.Variable(tf.random_normal([n_input])),
# }
#
# # Building the encoder
# def encoder(x):
#     # Encoder Hidden layer with sigmoid activation #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
#                                    biases['encoder_b1']))
#     # Decoder Hidden layer with sigmoid activation #2
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
#                                    biases['encoder_b2']))
#     return layer_2
#
#
# # Building the decoder
# def decoder(x):
#     # Encoder Hidden layer with sigmoid activation #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
#                                    biases['decoder_b1']))
#     # Decoder Hidden layer with sigmoid activation #2
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#                                    biases['decoder_b2']))
#     return layer_2


# """

# Visualize encoder setting
# Parameters
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 30
batch_size = 256
display_step = 1

# Network Parameters
# n_input = 784  # MNIST data input (img shape: 28*28)
n_input = 1024

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 512
n_hidden_2 = 256
n_hidden_3 = 160
n_hidden_4 = 128

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4
# """

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = int(len(X_train)/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X_train[i * batch_size: (i+1) * batch_size]
            batch_ys = Y_train[i * batch_size: (i+1) * batch_size]
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # # Applying encode and decode over test set
    # encode_decode = sess.run(
    #     # y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    #     y_pred, feed_dict = {X: X_valid[:examples_to_show]})
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     # a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[0][i].imshow(np.reshape(X_valid[i], (32, 32)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (32, 32)))
    # plt.show()

    from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-
    encoder_result = sess.run(encoder_op, feed_dict={X: X_valid})
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(encoder_result[:, 0], encoder_result[:, 1], encoder_result[:, 2], c=Y_valid)

    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=Y_valid)
    # plt.colorbar()
    plt.show()

    encoder_features = sess.run(encoder_op, feed_dict={X: X_train})
    import pickle
    file = open('train_features.pickle', 'wb')
    pickle.dump(encoder_features, file)
    file.close()

    file = open('valid_features.pickle', 'wb')
    pickle.dump(encoder_result, file)
    file.close()


    print('')

