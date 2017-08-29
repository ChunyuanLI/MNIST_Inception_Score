import gzip
import os

from scipy import ndimage
from six.moves import urllib
import numpy as np

import math
import sys
import scipy.io
import matplotlib.pyplot as plt

import pdb



R1 = scipy.io.loadmat('ali_inception_50.mat')
icp1 = R1['icp']
n_p = icp1.shape[0]
x = np.linspace(0, n_p, num=n_p, endpoint=False)

fig = plt.figure()

plt.xlabel("Epoch")
plt.ylabel("Iception Score")

plt.grid()

plt.fill_between(x, icp1[:,0] - icp1[:,1],
                 icp1[:,0] + icp1[:,1], alpha=0.1,
                 color="r")
plt.plot(x, icp1[:,0], 'o-', color="r",
         label="ALI")

plt.legend(loc="best")
# plt.show()

fig.savefig('icp_plot.pdf')



