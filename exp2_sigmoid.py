import h5py
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
X_train = data['X_train'][:]
Y_train = data['Y_train'][:]
X_valid = data['X_valid'][:]
Y_valid = data['Y_valid'][:]
data.close()

def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[10,10])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('$x_{%d}$' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def onehot(label):
    class_num = np.max(label).astype(int) + 1
    new_label = np.zeros((label.shape[0], class_num))
    new_label[np.arange(label.shape[0]), label.astype(int)] = 1
    return new_label

# Y_train = onehot(Y_train)
# Y_valid = onehot(Y_valid)
y_0 = [i for i,y in enumerate(Y_valid[:36]) if y == 0]
y_1 = [i for i,y in enumerate(Y_valid[:36]) if y == 1]
print('y == 0: ', y_0)
print('y == 1: ', y_1)

Y_train = Y_train[:, np.newaxis]
Y_valid = Y_valid[:, np.newaxis]


print("Image Train Data Shape", X_train.shape)
print("Label Train Data Shape", Y_train.shape)
print("Image Valid Data Shape", X_valid.shape)
print("Label Valid Data Shape", Y_valid.shape)

# sample_stack(X_train)


X_train = np.reshape(X_train, (-1, X_train.shape[1]*X_train.shape[2]))
X_valid = np.reshape(X_valid, (-1, X_valid.shape[1]*X_valid.shape[2]))

mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
X_train = (X_train - mean_train) / std_train

mean_valid = np.mean(X_valid, axis=0)
std_valid = np.std(X_valid, axis=0)
X_valid = (X_valid - mean_valid) / std_valid


# X_train = X_train[:100]
# Y_train = Y_train[:100]

#build network
x = tf.placeholder(tf.float32, [None, 32 * 32], name='input')
y = tf.placeholder(tf.float32, [None, 1], name='label')
input_reshape = tf.reshape(x, [-1, 32, 32, 1])
conv1 = tf.layers.conv2d(
    inputs=input_reshape,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    name='conv1')
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    name='conv2')
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64], name='flatten')
dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense1')
dropout1 = tf.layers.dropout(inputs=dense1, rate=1, name='dropout1')
# dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
# dropout2 = tf.layers.dropout(inputs=dense2, rate=0)
logits = tf.layers.dense(inputs=dropout1, units=1, name='logits')

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)

predictions = tf.round(tf.nn.sigmoid(logits))
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

accuracies, steps = [], []
start_time = time.time()
for t in range(1000):
    # training
    batch_index = np.random.randint(len(X_train), size=32)
    _, acc_, pred_, loss_ = sess.run([opt, accuracy, predictions, loss], {x: X_train[batch_index], y: Y_train[batch_index]})
    if t % 10 == 0:
        print("Step: %i" % t, "| Accurate: %.2f" % acc_, "| Loss: %.2f" % loss_, )

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, predictions, loss], {x: X_valid, y: Y_valid})
        accuracies.append(acc_)
        steps.append(t)
        print("Validation: Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)
        print('')
end_time = time.time()
print('spent: %.4fs' % (end_time-start_time))



# start_time = time.time()
# logisticRegr = LogisticRegression(solver = 'lbfgs')
# logisticRegr.fit(X_train, Y_train)
# predictions = logisticRegr.predict(X_valid)
# score = logisticRegr.score(X_valid, Y_valid)
# print(score)
# end_time = time.time()
# print('spent: %.4fs' % (end_time-start_time))
#
# cm = metrics.confusion_matrix(Y_valid, predictions)
# print(cm)
#
# # plot confusion matrix
# plt.figure(figsize=(9,9))
# plt.imshow(cm, interpolation='nearest', cmap='Blues')
# plt.title('Confusion matrix', size = 15)
# plt.colorbar()
# tick_marks = np.arange(2)
# plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
# plt.yticks(tick_marks, ["0", "1"], size = 10)
# plt.tight_layout()
# plt.ylabel('Actual label', size = 15)
# plt.xlabel('Predicted label', size = 15)
# width, height = cm.shape
# for x in list(range(width)):
#     for y in list(range(height)):
#         plt.annotate(str(cm[x][y]), xy=(y, x),
#                      horizontalalignment='center',
#                      verticalalignment='center')

print('')