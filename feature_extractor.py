import h5py
# import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import os
# from scatwave.scattering import Scattering

data = h5py.File('/home/cougarnet.uh.edu/pyuan2/Downloads/Lung_Nodule_2d.h5', 'r')
X_train = data['X_train'][:]
Y_train = data['Y_train'][:]
X_valid = data['X_valid'][:]
Y_valid = data['Y_valid'][:]
data.close()

# scat = Scattering(M=32, N=32, J=2).cuda()
# x = torch.randn(1, 3, 32, 32).cuda()
#
# print(scat(x).size())


# from PIL import Image
# im = Image.fromarray(X_train[1])
# im.save('./data/0.png')

import scipy.misc
for i in range(len(X_train)):
    if not os.path.exists('./data/train/'):
        os.makedirs('./data/train/')
    scipy.misc.imsave('./data/train/{}.png'.format(i), X_train[i])

for i in range(len(X_valid)):
    if not os.path.exists('./data/valid/'):
        os.makedirs('./data/valid/')
    scipy.misc.imsave('./data/valid/{}.png'.format(i), X_valid[i])


# scipy.misc.imsave('./data/1.png', X_train[0])

# import matplotlib
#
# matplotlib.image.imsave('./data/1.png', X_train[0], cmap='gray')
# image_file = './data/1.png'
# image = plt.imread(image_file)
# plt.imshow(image)



print("")