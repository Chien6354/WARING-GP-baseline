import os
import numpy
import sys
from PIL import Image
import skimage

def featureMeanStd(region):
    #print(region)
    mean=numpy.mean(region)
    std=numpy.std(region)
    #print(mean,std)
    return mean,std

def feaEx_flexible(image, size1, size2):
    try:
        current_std = skimage.measure.block_reduce(image,(size1,size2),numpy.std)
        current_mean = skimage.measure.block_reduce(image,(size1,size2),numpy.mean)
        current_std = numpy.concatenate((current_std))
        current_mean = numpy.concatenate((current_mean))
        current = numpy.concatenate((current_std, current_mean))
    except:
        current=numpy.concatenate((image))
    return current

def fetureEx(image):
    feature=numpy.zeros((20))
    width,height=image.shape
    width1=int(width/2)
    height1=int(height/2)
    width2=int(width/4)
    height2=int(height/4)
    #A1B1C1D1
    feature[0],feature[1]=featureMeanStd(image)
    #A1E1OG1
    feature[2],feature[3]=featureMeanStd(image[0:width1,0:height1])
    #E1B1H1O
    feature[4],feature[5]=featureMeanStd(image[0:width1,height1:height])
    #G1OF1D1
    feature[6],feature[7]=featureMeanStd(image[width1:width,0:height1])
    #OH1C1F1
    feature[8],feature[9]=featureMeanStd(image[width1:width,height1:height])
    #A2B2C2D2
    feature[10],feature[11]=featureMeanStd(image[width2:(width2+width1),height2:(height1+height2)])
    #G1H1
    feature[12],feature[13]=featureMeanStd(image[width1,:])
    #E1F1
    feature[14],feature[15]=featureMeanStd(image[:,height1])
    #G2H2
    feature[16],feature[17]=featureMeanStd(image[width1,height2:(height1+height2)])
    #E2F2
    feature[18],feature[19]=featureMeanStd(image[width2:(width2+width1),height1])
    return feature

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
    data = numpy.empty((imgNum,20))
    label = numpy.empty((imgNum,))
    for i in range(imgNum):
        img = Image.open(imgFoldName+'/'+images[i])
        arr = numpy.asarray(img,dtype='float32')
        arr =arr/255
        feature=fetureEx(arr)
        #print(feature)
        #print(i, data.shape)
        data[i,:] = feature
        label[i] = label01
    return data,label

def imageData(dataSetName):
    craterDir0 = os.pardir + '/' + 'data/' + str(dataSetName) + '/'
    #craterDir0='data/'+str(dataSetName)+'/'
    foldName0 = craterDir0+"training/positive"
    foldName1 = craterDir0+"training/negative"
    validationName0 = craterDir0+"validation/positive"
    validationName1 = craterDir0+"validation/negative"
    testName0 =craterDir0+ "testing/positive"
    testName1 = craterDir0+"testing/negative"
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
