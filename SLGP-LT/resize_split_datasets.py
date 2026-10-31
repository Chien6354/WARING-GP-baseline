import os
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pylab as plt
import collections


def load_Img(folder, class_label, width, height):
    imgs = []
    labels = []
    images = os.listdir(folder)
    if '.directory' in images:
        images.remove('.directory')
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    if '._.DS_Store' in images:
        images.remove('._.DS_Store')
    #print(len(images))
    for j in images:
        img = Image.open(folder +  '/' + j).convert('L')
        img = img.resize((width, height), Image.BILINEAR)
        imgs.append(np.asarray((img), dtype='float32'))
        labels.append(class_label)
    return np.asarray(imgs), np.asarray(labels)


def read_images(dataset_name, num_instance, path,width, height):
    randomseeds = 2324
    if path is not None:
        folders_path = path+dataset_name
    else: folders_path = dataset_name
    files = os.listdir(folders_path)
    x_train=[]
    y_train = []
    x_test = []
    y_test = []
    prop = []
    prop.append('Number of Classes ' + str(len(files)))
    for i in range(0,len(files)):
        prop.append(files[i]+' ' + str(i))
        x_data, y_data = load_Img(folders_path+'/'+files[i], i,width, height)
        print(x_data.shape, y_data.shape)
        #if len(data)<150: test_n = split_p
        #else: test_n = len(data)-150
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, train_size=num_instance, random_state=12)
        # trainD, testD, trainL, testL = train_test_split(x_data,y_data, train_size= 30, random_state=randomseeds)
        print(x_train1.shape, x_test1.shape)
        #if len(trainD)<35:print('less', len(trainD), len(testD))
        x_train.append(x_train1)
        x_test.append(x_test1)
        y_train.append(y_train1)
        y_test.append(y_test1)
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), prop

if __name__ == "__main__":
    dataset = ['yale']
    data_sets = ['grimace', 'jaffe', 'kth',  'yale','scene']
    data_sets=['jaffe']
    data_path = '/local/scratch/projects/datasets/mdatasets_o/'
    # data_path = '/local/scratch/ImageData/Outex-TC-00010/'
    # data_path = '/local/scratch/ImageData/Original_images/'
    wid =64
    hei =64
    num_ins =10
    for i in data_sets:
        x1, y1, x2, y2, prop = read_images(dataset_name=i, path=data_path, num_instance=num_ins, width=wid, height=hei)
        x1 = np.concatenate((x1), axis=0)
        x2 = np.concatenate((x2), axis=0)
        y1 = np.concatenate((y1), axis=0)
        y2 = np.concatenate((y2), axis=0)
        print(x1.shape, x2.shape)
        np.save(i + '_train_data', x1)
        np.save(i + '_train_label', y1)
        np.save(i + '_test_data', x2)
        np.save(i + '_test_label', y2)
        prop.append('Instances for training '+ str(x1.shape[0]))
        prop.append('Instances for testing ' + str(x2.shape[0]))
        prop.append('Image Size ' + str(x2.shape[1])+'  '+ str(x2.shape[2]))
        file = open(i+'_properties.txt','w')
        file.writelines(["%s\n" % item for item in prop])
        file.writelines('Train ' + str(collections.Counter(y1)) + '\n')
        file.writelines('Test ' + str(collections.Counter(y2)))
        file.close()
