import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

class Classifier(object):

    def __init__(self, type):
        self.name = type
        print('Classifier type: {}'.format(type))
        self.score = 0
        if type == 'logistic':
            self.clf = LogisticRegression(solver='lbfgs')
        elif type == 'SVM':
            self.clf = SVC


    def predict(self,x_sample, X_train=None, Y_train=None):
        x_sample = x_sample.reshape(1, -1)
        start_time = time.time()
        if X_train is not None:
            self.clf.fit(X_train, Y_train)
        try:
            prob = np.squeeze(self.clf.predict_proba(x_sample))
        except:
            print("")
        # print('prediction probabilities: ', prob)
        end_time = time.time()
        # print('spent: %.4fs' % (end_time - start_time))
        return prob

    def get_performance(self, X_train, Y_train, X_test, Y_test):
        self.clf.fit(X_train, Y_train)
        predictions = self.clf.predict(X_test)
        score = self.clf.score(X_test, Y_test)
        print('classification accuracy:', score)
        cm = metrics.confusion_matrix(Y_test, predictions)
        print(cm)
        return score


if __name__ == '__main__':
    import h5py
    import scipy.io

    '''

    train_mat = scipy.io.loadmat('train_features.mat')
    valid_mat = scipy.io.loadmat('valid_features.mat')

    X_train = np.transpose(train_mat['a'])
    X_valid = np.transpose(valid_mat['a'])

    data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
    # X_train = data['X_train'][:]
    Y_train = data['Y_train'][:]
    # X_valid = data['X_valid'][:]
    Y_valid = data['Y_valid'][:]
    data.close()

    '''

    '''
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
    '''

    # '''
    import pickle
    file = open('train_features.pickle', 'rb')
    X_train = pickle.load(file)
    file.close()

    file = open('valid_features.pickle', 'rb')
    X_valid = pickle.load(file)
    file.close()

    data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
    # X_train = data['X_train'][:]
    Y_train = data['Y_train'][:]
    # X_valid = data['X_valid'][:]
    Y_valid = data['Y_valid'][:]
    data.close()
    # '''


    clf = Classifier('logistic')
    # clf.predict(X_valid[0], X_train, Y_train)
    clf.get_performance(X_train, Y_train, X_valid, Y_valid)

    print('')
