import h5py
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics


data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
X_train = data['X_train'][:]
Y_train = data['Y_train'][:]
X_valid = data['X_valid'][:]
Y_valid = data['Y_valid'][:]
data.close()

# h5f_train = h5py.File('/home/cougarnet.uh.edu/pyuan2/Documents/Data/chest256_train_801010_no_normal.h5', 'r')
# x_train = h5f_train['X_train'][:]
# y_train = h5f_train['Y_train'][:]
# h5f_train.close()

def sample_stack(stack, rows=6, cols=6, start_with=0, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[10,10])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('$x_{%d}$' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

print("Image Train Data Shape", X_train.shape)
print("Label Train Data Shape", Y_train.shape)
print("Image Valid Data Shape", X_valid.shape)
print("Label Valid Data Shape", Y_valid.shape)

# print(Y_train[:36])
# sample_stack(X_train)

X_train = np.reshape(X_train, (-1, X_train.shape[1]*X_train.shape[2]))
X_valid = np.reshape(X_valid, (-1, X_valid.shape[1]*X_valid.shape[2]))

# X_train = X_train[:1000]
# Y_train = Y_train[:1000]

start_time = time.time()
# clf = LogisticRegression(solver = 'lbfgs')
clf = SGDClassifier(random_state=0)
# clf = LinearSVC(random_state=0)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_valid)
score = clf.score(X_valid, Y_valid)
print(score)
end_time = time.time()
print('spent: %.4fs' % (end_time-start_time))

cm = metrics.confusion_matrix(Y_valid, predictions)
print(cm)

# plot confusion matrix
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in list(range(width)):
    for y in list(range(height)):
        plt.annotate(str(cm[x][y]), xy=(y, x),
                     horizontalalignment='center',
                     verticalalignment='center')

print('')