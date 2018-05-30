import h5py
import numpy as np
import matplotlib.pyplot as plt
# from classifier import Classifier
from CNN_classifier import Classifier


data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
X_train = data['X_train'][:]
Y_train = data['Y_train'][:]
X_valid = data['X_valid'][:]
Y_valid = data['Y_valid'][:]
data.close()

X_train = np.reshape(X_train, (-1, X_train.shape[1] * X_train.shape[2]))
X_valid = np.reshape(X_valid, (-1, X_valid.shape[1] * X_valid.shape[2]))

mean_train = np.mean(X_train, axis=0)
std_train = np.std(X_train, axis=0)
X_train = (X_train - mean_train) / std_train

mean_valid = np.mean(X_valid, axis=0)
std_valid = np.std(X_valid, axis=0)
X_valid = (X_valid - mean_valid) / std_valid

X_train_features = X_train
X_valid_features = X_valid

# clf = Classifier('logistic')
clf = Classifier()
# clf.predict(X_valid[0], X_train, Y_train)
seed_id = np.arange(10)
unlabeled_id = np.arange(10, len(X_train))
seed_x = X_train_features[seed_id]
seed_y = Y_train[seed_id]
unlabeled_x = X_train_features[unlabeled_id]
unlabeled_y = Y_train[unlabeled_id]

score_loop = []

for loop in range(10):
    score_his = []
    score = clf.get_performance(seed_x, seed_y, X_valid_features, Y_valid)
    score_his.append(score)
    for i in range(0, 990, 10):
        np.random.shuffle(unlabeled_id)
        unlabeled_x = X_train_features[unlabeled_id]
        unlabeled_y = Y_train[unlabeled_id]
        # print(unlabeled_id[:10])
        queried_set_x = []
        queried_set_y = []
        for j in range(i+1):
            queried_set_x.append(unlabeled_x[j])
            queried_set_y.append(unlabeled_y[j])
        labeled_x = np.concatenate((seed_x, np.vstack(queried_set_x)))
        labeled_y = np.concatenate((seed_y, np.hstack(queried_set_y)))
        score = clf.get_performance(labeled_x, labeled_y, X_valid_features, Y_valid)
        score_his.append(score)
    score_loop.append(score_his)

av = np.average(score_loop, axis=0)
plt.plot(av)
plt.show()
plt.savefig()
print('')