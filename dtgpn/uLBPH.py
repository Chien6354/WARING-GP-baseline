import os
import numpy
from skimage.feature import local_binary_pattern
from PIL import Image

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
    data = numpy.empty((imgNum,59))
    label = numpy.empty((imgNum,))
    for i in range(imgNum):
        img = Image.open(imgFoldName+'/'+images[i])
        arr = numpy.asarray(img,dtype='float32')
        arr =(arr-arr.min())*(1/arr.max()-arr.min())
        feature=histLBP(arr,radius=1.5, n_points=8)
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
    if dataSetName=='UIUC':
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

def uniform_LBP(image,radius,n_points):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    return lbp

def histLBP(image,radius,n_points):
    #uniform_LBP
    lbp=uniform_LBP(image,radius,n_points)
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,59])
    return hist

def lbp_ps(image,radius,n_points, method='nri_uniform'):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method)
    n_bins = 59
    if method =='uniform':
        n_bins=10
    elif method =='ror':
        n_bins=256
    hist,ax=numpy.histogram(lbp,n_bins,[0,n_bins])
    return hist


