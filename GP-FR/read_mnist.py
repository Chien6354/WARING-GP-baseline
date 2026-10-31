import numpy as np
from PIL import Image
from skimage.io import imsave
import sys
# dataSetName = str(sys.argv[1])
dataSetName = 'Rectangle'
x_train = np.load(dataSetName + '_train_data.npy')
y_train = np.load(dataSetName + '_train_label.npy')

num_class = len(np.unique(y_train))
print(num_class)
for i in range(num_class):
    class_index, = np.where(y_train == i)
    print(class_index)
    x_image = x_train[class_index[2]+8]
    print(x_image.shape)
    imsave(str(i)+dataSetName+"22.jpg", x_image)

