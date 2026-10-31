from skimage.feature import hog
import numpy
from PIL import Image
import os

def HoGFeatures(image):
    img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                transform_sqrt=False, feature_vector=True)
    return realImage

def hog_features_patches(image,patch_size,moving_size):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = numpy.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    #print(hog_features, hog_features.shape)
    return hog_features

def load_Img(imgFoldName,label01,width,height):
    images = os.listdir(imgFoldName)
    if '.directory' in images:
        images.remove('.directory')
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    #print(type(images))
    #print(images)
    imgNum = len(images)
    #print(imgNum)
    data = numpy.empty((imgNum,144))
    label = numpy.empty((imgNum,))
    for i in range(imgNum):
        img = Image.open(imgFoldName+'/'+images[i])
        arr = numpy.asarray(img,dtype='float32')
        arr =(arr-arr.min())*(1/arr.max()-arr.min())
        feature=hog_features_patches(arr,patch_size=20,moving_size=10)
        #print(arr)
        #print(i, data.shape)
        data[i,:] = feature
        label[i] = label01
    return data,label

def imageData(dataSetName):
    craterDir0 = os.pardir + '/' + 'data/' + str(dataSetName) + '/'
    # craterDir0='data/'+str(dataSetName)+'/'
    foldName0 = craterDir0 + "training/positive"
    foldName1 = craterDir0 + "training/negative"
    validationName0 = craterDir0 + "validation/positive"
    validationName1 = craterDir0 + "validation/negative"
    testName0 = craterDir0 + "testing/positive"
    testName1 = craterDir0 + "testing/negative"
    reSizeW=128
    reSizeH=128
    if dataSetName=='uiuc':
        reSizeW = 100
        reSizeH = 40
    trainData0, label0 = load_Img(foldName0,0,reSizeW,reSizeH)
    trainData1, label1 = load_Img(foldName1,1,reSizeW,reSizeH)
    trainData=numpy.concatenate((trainData0,trainData1))
    trainLabel=numpy.concatenate((label0,label1))
    #print(trainLabel)
    #print(trainData)

    validationData0, validationLabel0 = load_Img(validationName0,0,reSizeW,reSizeH)
    validationData1, validationLabel1 = load_Img(validationName1,1,reSizeW,reSizeH)
    validationData=numpy.concatenate((validationData0,validationData1))
    validationLabel=numpy.concatenate((validationLabel0, validationLabel1))
    #print(validationData)
    #print(validationLabel)

    testData0, testLabel0 = load_Img(testName0,0,reSizeW,reSizeH)
    testData1, testLabel1 = load_Img(testName1,1,reSizeW,reSizeH)
    testData=numpy.concatenate((testData0,testData1))
    testLabel=numpy.concatenate((testLabel0,testLabel1))
    #print(testLabel)
    #print(testData)
    return trainData,trainLabel,testData,testLabel,validationData,validationLabel
