import sift_features
import numpy
from pylab import *
from scipy import ndimage
from skimage.filters import sobel
from skimage.filters import gabor
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import random


def conVector(input_data):
    if isinstance(input_data, list):  # 判断是否为 list 类型
        if len(input_data) > 0:  # 如果 list 长度大于 1
            # 统计不同形状的 ndarray
            shape_dict = {}  # 用于存储不同形状的 ndarray
            for item in input_data:
                if isinstance(item, np.ndarray):  # 判断是否为 ndarray
                    shape = item.shape  # 获取形状
                    if shape not in shape_dict:  # 如果形状未记录
                        shape_dict[shape] = item  # 记录一个该形状的 ndarray
                else:
                    raise ValueError(f"list 中包含非 ndarray 类型的元素，元素类型为: {type(item)}")

            # 将每种形状的 ndarray 转化为 1 维并拼接
            flattened_arrays = []
            for shape, array in shape_dict.items():
                flattened = array.reshape(-1)  # 转化为 1 维
                flattened_arrays.append(flattened)
            return np.concatenate(flattened_arrays)
        else:
            raise ValueError("list 的长度必须大于 1。")
    elif isinstance(input_data, np.ndarray):  # 判断是否为 ndarray 类型
        if input_data.ndim == 1:  # 如果维度为 1，直接返回
            return input_data
        else:  # 否则转化为 1 维
            return input_data.reshape(-1)
    else:
        print(f"输入的类型为: {type(input_data)}")
        raise TypeError("输入参数必须是 list 或 ndarray 类型。")

import numpy as np
# def print_shapes_or_lengths(data_tuple):
#     for i, item in enumerate(data_tuple):
#         if isinstance(item, np.ndarray):  # 判断是否为 ndarray
#             print(f"元素 {i} 是 ndarray，形状为: {item.shape}")
#         elif isinstance(item, list):  # 判断是否为 list
#             print(f"元素 {i} 是 list，长度为: {len(item)}")
#             if len(item) > 0:  # 如果 list 长度大于 1
#                 print(f"  list 中每个元素的形状为:")
#                 for j, sub_item in enumerate(item):
#                     if isinstance(sub_item, np.ndarray):  # 判断子元素是否为 ndarray
#                         print(f"    元素 {j} 的形状为: {sub_item.shape}")
#                     else:
#                         print(f"    元素 {j} 不是 ndarray 类型，无法获取形状。")
#         else:
#             print(f"元素 {i} 是其他类型: {type(item)}")


def feature_con(*args):
    # 遍历 args，对每个元素调用 conVector 并拼接结果
    result_arrays = []
    for arg in args:
        try:
            converted = conVector(arg)  # 调用 conVector 函数
            result_arrays.append(converted)
        except (ValueError, TypeError) as e:
            print(f"处理参数时出错: {e}")
            continue  # 跳过错误参数，继续处理下一个
    # 拼接所有结果
    if result_arrays:
        return np.concatenate(result_arrays)
    else:
        raise ValueError("没有有效的参数可以拼接。")

def feature_con2(vector1, vector2):
    vector1 = conVector(vector1)
    feature_vector = numpy.concatenate((vector1, vector2), axis=0)
    return feature_vector


# square
def regionS(left, x, y, windowSize):
    width, height = left.shape
    # region_result = []
    if (x + windowSize) < width and (y + windowSize) < height:
        region = left[x:(x + windowSize), y:(y + windowSize)]
    elif (x + windowSize) > width and (y + windowSize) < height:
        region = left[x:width, y:(y + windowSize)]
    elif (x + windowSize) < width and (y + windowSize) > height:
        region = left[x:(x + windowSize), y:height]
    else:
        region = left[x:width, y:height]
    # print('square', region.shape)
    # region_result.append(numpy.array(region))
    return region


# rectangle
def regionR(left, x, y, window_size1, window_size2):
    width, height = left.shape
    # region_result = []
    if (x + window_size1) < width and (y + window_size2) < height:
        region = left[x:(x + window_size1), y:(y + window_size2)]
    elif (x + window_size1) > width and (y + window_size2) < height:
        region = left[x:width, y:(y + window_size2)]
    elif (x + window_size1) < width and (y + window_size2) > height:
        region = left[x:(x + window_size1), y:height]
    else:
        region = left[x:width, y:height]
    # print('rectangle', region.shape)
    # region_result.append(numpy.array(region))
    return region


def featureMeanStd(region):
    #print(region)
    mean=np.mean(region)
    std=np.std(region)
    #print(mean,std)
    return mean,std


def featureEx_raw(image):
    features = []
    feature = np.zeros(20)
    width, height = image.shape
    width1 = int(width/2)
    height1 = int(height/2)
    width2 = int(width/4)
    height2 = int(height/4)
    #A1B1C1D1
    feature[0], feature[1] = featureMeanStd(image)
    #A1E1OG1
    feature[2], feature[3] = featureMeanStd(image[0:width1, 0:height1])
    #E1B1H1O
    feature[4], feature[5] = featureMeanStd(image[0:width1, height1:height])
    #G1OF1D1
    feature[6], feature[7] = featureMeanStd(image[width1:width, 0:height1])
    #OH1C1F1
    feature[8], feature[9] = featureMeanStd(image[width1:width, height1:height])
    #A2B2C2D2
    feature[10], feature[11] = featureMeanStd(image[width2:(width2+width1), height2:(height1+height2)])
    #G1H1
    feature[12], feature[13] = featureMeanStd(image[width1, :])
    #E1F1
    feature[14], feature[15] = featureMeanStd(image[:, height1])
    #G2H2
    feature[16], feature[17] = featureMeanStd(image[width1, height2:(height1+height2)])
    #E2F2
    feature[18], feature[19] = featureMeanStd(image[width2:(width2+width1), height1])
    features.append(numpy.array(feature))
    return feature


def featureEx_fun(image):
    feature=np.zeros((20))
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


def featureEx(img2):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    features = []
    if feature_number == 1:
        feature = featureEx_fun(x[0])
        features.append(numpy.array(feature))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            return features
        else:
            return features
    else:
        feature = featureEx_fun(x[0])
        features.append(numpy.array(feature))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            feature = featureEx_fun(x[i])
            features.append(numpy.array(feature))
        return features


def hist_base(image):
    n_bins = 32
    hist, ax = numpy.histogram(image, n_bins, [0,1])
    return hist
def hist_raw(image):
    n_bins = 32
    a = numpy.min(image)
    b = numpy.max(image)
    hist, ax = numpy.histogram(image, n_bins, [0, 1])
    features = []
    x = hist
    features.append(numpy.array(x))
    return features


def hist(img2):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    features = []
    n_bins = 32
    if feature_number == 1:
        hist, ax = numpy.histogram(x[0], n_bins, [0, 1])
        features.append(numpy.array(hist))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            return features
        else:
            return features
    else:
        hist, ax = numpy.histogram(x[0], n_bins, [0, 1])
        features.append(numpy.array(hist))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            hist, ax = numpy.histogram(x[i], n_bins, [0, 1])
            features.append(numpy.array(hist))
        return features


def uniform_lbp_base(image):
    # uniform_LBP
    lbp = local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist, ax = numpy.histogram(lbp, n_bins, [0, 59])
    return hist


def uniform_lbp_raw(image):
    # uniform_LBP
    lbp = local_binary_pattern(image, P=8, R=1.5, method='nri_uniform')
    n_bins = 59
    hist, ax = numpy.histogram(lbp, n_bins, [0, 59])
    features = []
    x = hist
    features.append(numpy.array(x))
    return features


def uniform_lbp(img2):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    features = []
    if feature_number == 1:
        lbp = local_binary_pattern(x[0], P=8, R=1.5, method='nri_uniform')
        n_bins = 59
        hist, ax = numpy.histogram(lbp, n_bins, [0, 59])
        features.append(numpy.array(hist))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            return features
        else:
            return features
    else:
        lbp = local_binary_pattern(x[0], P=8, R=1.5, method='nri_uniform')
        n_bins = 59
        hist, ax = numpy.histogram(lbp, n_bins, [0, 59])
        features.append(numpy.array(hist))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            lbp = local_binary_pattern(x[i], P=8, R=1.5, method='nri_uniform')
            n_bins = 59
            hist, ax = numpy.histogram(lbp, n_bins, [0, 59])
            features.append(numpy.array(hist))
        return features


def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image


def hog_features_patches(image,patch_size,moving_size):
    img = numpy.asarray(image)
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
    return hog_features


def global_hog_base(image):
    feature_vector = hog_features_patches(image, 4, 4)
    return feature_vector
def global_hog_raw(image):
    features = []
    x = hog_features_patches(image, 4, 4)
    features.append(numpy.array(x))
    # features = numpy.array(features)
    return features


def global_hog(img2):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    features = []
    if feature_number == 1:
        features.append(numpy.array(hog_features_patches(x[0], 4, 4)))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            return features
        else:
            return features
    else:
        features.append(numpy.array(hog_features_patches(x[0], 4, 4)))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            features.append(numpy.array(hog_features_patches(x[i], 4, 4)))
        return features


def all_sift_base(image):
    width, height = image.shape
    min_length = numpy.min((width, height))
    img = numpy.asarray(image[0:width, 0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
    # dimension 128 for all images
    w, h = feaArrSingle.shape
    feature_vector = numpy.reshape(feaArrSingle, (h,))
    return feature_vector


def all_sift_raw(image):
    width, height = image.shape
    min_length = numpy.min((width, height))
    img = numpy.asarray(image[0:width, 0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
    # dimension 128 for all images
    w, h = feaArrSingle.shape
    features = []
    x = numpy.reshape(feaArrSingle, (h,))
    features.append(numpy.array(x))
    return features


def all_sift(img2):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    features = []
    if feature_number == 1:
        width, height = x[0].shape
        min_length = numpy.min((width, height))
        img = numpy.asarray(x[0][0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        features.append(numpy.array(feature_vector))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            return features
        else:
            return features
    else:
        width, height = x[0].shape
        min_length = numpy.min((width, height))
        img = numpy.asarray(x[0][0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        w, h = feaArrSingle.shape
        feature_vector = numpy.reshape(feaArrSingle, (h,))
        features.append(numpy.array(feature_vector))
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            width, height = x[i].shape
            min_length = numpy.min((width, height))
            img = numpy.asarray(x[i][0:width, 0:height])
            extractor = sift_features.SingleSiftExtractor(min_length)
            feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
            w, h = feaArrSingle.shape
            feature_vector = numpy.reshape(feaArrSingle, (h,))
            features.append(numpy.array(feature_vector))
        return features


def gau_original(left, si):
    return gaussian(left, sigma=si)


def gau_raw(left, si, s):
    features = []
    x = gaussian(left, sigma=si)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def gau(img2, si, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(gaussian(x[0],sigma=si)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(gaussian(x[0],sigma=si)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(gaussian(x[i],sigma=si)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def gauD_original(left, si, or1, or2):
    return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])


def gauD_raw(left, si, or1, or2, s):
    features = []
    x = ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def gauD(img2, si, or1, or2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.gaussian_filter(x[0],sigma=si, order=[or1,or2])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.gaussian_filter(x[0],sigma=si, order=[or1,or2])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.gaussian_filter(x[i],sigma=si, order=[or1,or2])))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def laplace_original(left):
    return ndimage.laplace(left)


def laplace_raw(left, s):
    features = []
    x = ndimage.laplace(left)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def laplace(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.laplace(x[0])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.laplace(x[0])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.laplace(x[i])))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def gaussian_Laplace1_original(left):
    return ndimage.gaussian_laplace(left,sigma=1)


def gaussian_Laplace1_raw(left, s):
    features = []
    x = ndimage.gaussian_laplace(left, sigma=1)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def gaussian_Laplace1(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.gaussian_laplace(x[0], sigma=1)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.gaussian_laplace(x[0], sigma=1)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.gaussian_laplace(x[i], sigma=1)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def gaussian_Laplace2_original(left):
    return ndimage.gaussian_laplace(left,sigma=2)


def gaussian_Laplace2_raw(left, s):
    features = []
    x = ndimage.gaussian_laplace(left, sigma=2)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def gaussian_Laplace2(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.gaussian_laplace(x[0], sigma=2)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.gaussian_laplace(x[0], sigma=2)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.gaussian_laplace(x[i], sigma=2)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


# def sobelxy(left):
#     left = sobel(left)
#     feature_vector = conVector(left)
#     return feature_vector


def sobelxy_original(image):
    image = sobel(image)
    return image


def sobelxy_raw(image, s):
    features = []
    x = sobel(image)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def sobelxy(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(sobel(x[0])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(sobel(x[0])))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(sobel(x[i])))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def sobelx_original(image):
    image = ndimage.sobel(image, axis=0)
    return image


def sobelx_raw(image, s):
    features = []
    x = ndimage.sobel(image, axis=0)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def sobelx(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(numpy.array(ndimage.sobel(x[0], axis=0))))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.sobel(x[0], axis=0)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.sobel(x[i], axis=0)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def sobely_original(image):
    image = ndimage.sobel(image, axis=1)
    return image


def sobely_raw(image, s):
    features = []
    x = ndimage.sobel(image, axis=1)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def sobely(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(numpy.array(ndimage.sobel(x[0], axis=1))))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.sobel(x[0], axis=1)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.sobel(x[i], axis=1)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


# max filter
def maxf_original(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size = 3
    x = ndimage.maximum_filter(x, size)
    return x


def max_raw(x, s):
    size = 3
    features = []
    x = ndimage.maximum_filter(x, size)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def maxf(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    size = 3
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.maximum_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.maximum_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.maximum_filter(x[i], size)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


# median_filter
def medianf_original(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.median_filter(x,size)
    return x


def median_raw(x, s):
    size = 3
    features = []
    x = ndimage.median_filter(x, size)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def medianf(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    size = 3
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.median_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.median_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.median_filter(x[i], size)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


# mean_filter
def meanf_original(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.convolve(x, numpy.full((3, 3), 1 / (size * size)))
    return x


def mean_raw(x, s):
    size = 3
    features = []
    x = ndimage.convolve(x, numpy.full((3, 3), 1 / (size * size)))
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def meanf(img2, s):
    x = img2[0]
    s2 = img2[1]
    result = []
    feature_number = len(x)
    size = 3
    features = []
    signal_f = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.convolve(x[0], numpy.full((3, 3), 1 / (size * size)))))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.convolve(x[0], numpy.full((3, 3), 1 / (size * size)))))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.convolve(x[i], numpy.full((3, 3), 1 / (size * size)))))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


# minimum_filter
def minf_original(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x=ndimage.minimum_filter(x,size)
    return x


def min_raw(x, s):
    """
    :type args: arguments and filter size
    """
    size = 3
    features = []
    x = ndimage.minimum_filter(x, size)
    features.append(numpy.array(x))
    signal_f = []
    signal_f.append(s)
    result = []
    result.append(features)
    result.append(signal_f)
    return result


def minf(img2, s):
    x = img2[0]
    s2 = img2[1]
    feature_number = len(x)
    size = 3
    features = []
    signal_f = []
    result = []
    if feature_number == 1:
        features.append(numpy.array(ndimage.minimum_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
            result.append(features)
            result.append(signal_f)
            return result
        else:
            result.append(features)
            result.append(signal_f)
            return result
    else:
        features.append(numpy.array(ndimage.minimum_filter(x[0], size)))
        signal_f.append(s)
        if s2[0] == 1:
            features.append(numpy.array(x[0]))
            signal_f.append(s2[0])
        for i in range(1, feature_number):
            features.append(numpy.array(x[i]))
            signal_f.append(1)
            features.append(numpy.array(ndimage.minimum_filter(x[i], size)))
            signal_f.append(1)
        result.append(features)
        result.append(signal_f)
        return result


def classifier_selection(features, classification_number):
    final_features = []
    final_features.append(features)
    a = classification_number
    final_features.append(a)
    return final_features

