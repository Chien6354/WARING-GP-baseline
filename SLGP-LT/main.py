# python packages
import argparse
import math
import random
import os
import time
from collections import Counter
from rand_augment import apply_random_policy_augment
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
import numpy
from deap import base, creator, tools, gp

DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from strongGPDataType import Int1, Int2, Int3, Int4, Float1, Float2, Float4, Img0, Img, Vector, Flag, Vector_C, \
    Img1, Img2, X, Y, windowSize, width, height, s, ClassLabel
import fgp_functions_new as fe_fs
import multiprocessing
import hashlib
import pickle
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
from data.autoaugment import data_augmentation
from functools import wraps
from threading import Lock
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
import saveFile
import pickle
import sys
import operator
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_file", default="log/main.log", help="log file name", type=str
)
parser.add_argument(
    "--pkl_file", help="pkl file name", default="pkl/2215.pkl"
)
parser.add_argument(
    "--dataset",
    help="Dataset name (choose from the provided options)",
    type=str,
    choices=["mnist", "cifar10", "fashion_mnist", "STL", "MBI", "MRB", "MRD", "SVHN", "EuroSAT"],  # 限制可选值
    default="cifar10"  # 设置默认值
)


parser.add_argument(
    "--randomSeeds",default= 100
)

opt = parser.parse_args()

augmentation_policies = [
    (0.1, "invert", 7, 0.2, "contrast", 6),
    (0.7, "rotate", 2, 0.3, "translateX", 9),
    (0.8, "sharpness", 1, 0.9, "sharpness", 3),
    (0.5, "shearY", 8, 0.7, "translateY", 9),
    (0.5, "autocontrast", 8, 0.9, "equalize", 2),
    (0.2, "shearY", 7, 0.3, "posterize", 7),
    (0.4, "color", 3, 0.6, "brightness", 7),
    (0.3, "sharpness", 9, 0.7, "brightness", 9),
    (0.6, "equalize", 5, 0.5, "equalize", 1),
    (0.6, "contrast", 7, 0.6, "sharpness", 5),
    (0.7, "color", 7, 0.5, "translateX", 8),
    (0.3, "equalize", 7, 0.4, "autocontrast", 8),
    (0.4, "translateY", 3, 0.2, "sharpness", 6),
    (0.9, "brightness", 6, 0.2, "color", 8),
    (0.5, "solarize", 2, 0.0, "invert", 3),
    (0.2, "equalize", 0, 0.6, "autocontrast", 0),
    (0.2, "equalize", 8, 0.6, "equalize", 4),
    (0.9, "color", 9, 0.6, "equalize", 6),
    (0.8, "autocontrast", 4, 0.2, "solarize", 8),
    (0.1, "brightness", 3, 0.7, "color", 0),
    (0.4, "solarize", 5, 0.9, "autocontrast", 3),
    (0.9, "translateY", 9, 0.7, "translateY", 9),
    (0.9, "autocontrast", 2, 0.8, "solarize", 3),
    (0.8, "equalize", 8, 0.1, "invert", 3),
    (0.7, "translateY", 9, 0.9, "autocontrast", 1)
]


def apply_augmentation_sequence(img, policy, Flag):
    """
    应用增强策略序列（两个操作）

    参数:
        img: 输入图像（numpy数组格式）
        policy: 元组 (prob1, op1_name, mag1, prob2, op2_name, mag2, fillcolor)
        例如: (0.7, "shearX", 7, 0.5, "translateX", 8, fillcolor)

    返回:
        增强后的图像（numpy数组格式，待change处理）
    """
    if not Flag:
        img_pil = fe_fs.array_to_pil_channel(img)
        return fe_fs.change(img_pil)

    prob1, op1_name, mag1, prob2, op2_name, mag2 = policy

    # 将输入转为PIL格式
    img_pil = fe_fs.array_to_pil_channel(img)

    # 应用第一个操作
    if op1_name == "shearX":
        img_pil = fe_fs.shear_x(img_pil, mag1, prob1, True)
    elif op1_name == "shearY":
        img_pil = fe_fs.shear_y(img_pil, mag1, prob1, True)
    elif op1_name == "translateX":
        img_pil = fe_fs.translate_x(img_pil, mag1, prob1, True)
    elif op1_name == "translateY":
        img_pil = fe_fs.translate_y(img_pil, mag1, prob1, True)
    elif op1_name == "rotate":
        img_pil = fe_fs.rotate(img_pil, mag1, prob1, True)
    elif op1_name == "contrast":
        img_pil = fe_fs.contrast(img_pil, mag1, prob1, True)
    elif op1_name == "brightness":
        img_pil = fe_fs.brightness(img_pil, mag1, prob1, True)
    elif op1_name == "color":
        img_pil = fe_fs.color(img_pil, mag1, prob1, True)
    elif op1_name == "posterize":
        img_pil = fe_fs.posterize(img_pil, mag1, prob1, True)
    elif op1_name == "autocontrast":
        img_pil = fe_fs.auto_contrast(img_pil, mag1, prob1, True)
    elif op1_name == "equalize":
        img_pil = fe_fs.equalize(img_pil, mag1, prob1, True)
    elif op1_name == "invert":
        img_pil = fe_fs.invert(img_pil, mag1, prob1, True)
    elif op1_name == "sharpness":
        img_pil = fe_fs.sharpness(img_pil, mag1, prob1, True)
    elif op1_name == "solarize":
        img_pil = fe_fs.solarize(img_pil, mag1, prob1, True)

    # 应用第二个操作
    if op2_name == "shearX":
        img_pil = fe_fs.shear_x(img_pil, mag2, prob2, True)
    elif op2_name == "shearY":
        img_pil = fe_fs.shear_y(img_pil, mag2, prob2, True)
    elif op2_name == "translateX":
        img_pil = fe_fs.translate_x(img_pil, mag2, prob2, True)
    elif op2_name == "translateY":
        img_pil = fe_fs.translate_y(img_pil, mag2, prob2, True)
    elif op2_name == "rotate":
        img_pil = fe_fs.rotate(img_pil, mag2, prob2, True)
    elif op2_name == "contrast":
        img_pil = fe_fs.contrast(img_pil, mag2, prob2, True)
    elif op2_name == "brightness":
        img_pil = fe_fs.brightness(img_pil, mag2, prob2, True)
    elif op2_name == "color":
        img_pil = fe_fs.color(img_pil, mag2, prob2, True)
    elif op2_name == "posterize":
        img_pil = fe_fs.posterize(img_pil, mag2, prob2, True)
    elif op2_name == "autocontrast":
        img_pil = fe_fs.auto_contrast(img_pil, mag2, prob2, True)
    elif op2_name == "equalize":
        img_pil = fe_fs.equalize(img_pil, mag2, prob2, True)
    elif op2_name == "invert":
        img_pil = fe_fs.invert(img_pil, mag2, prob2, True)
    elif op2_name == "sharpness":
        img_pil = fe_fs.sharpness(img_pil, mag2, prob2, True)
    elif op2_name == "solarize":
        img_pil = fe_fs.solarize(img_pil, mag2, prob2, True)

    feat_aug = fe_fs.change(img_pil)

    return feat_aug


class Logger(object):
    def __init__(self, filename='default.log', stream=None, add_flag=True,
                 encoding='utf-8', keep_open=True):
        """
        filename   : 日志文件路径
        stream     : 原始输出流（通常是 sys.stdout）
        add_flag   : True->追加 (a)，False->覆盖 (w)
        encoding   : 写入文件时使用的编码（Windows 记事本请用 'utf-8-sig' 或 'gbk'）
        keep_open  : True->程序运行期间保持文件打开（性能好），False->每次写入打开/关闭
        """
        self.terminal = stream if stream is not None else sys.stdout
        self.filename = filename
        self.add_flag = bool(add_flag)
        self.encoding = encoding
        self.mode = 'a' if self.add_flag else 'w'
        self.keep_open = bool(keep_open)
        self._log = None
        if self.keep_open:
            # 打开文件（若目录不存在会抛错，按需先创建目录）
            self._log = open(self.filename, self.mode, encoding=self.encoding)

    def write(self, message):
        # 避免 None 或空字符串导致问题
        if message is None or message == '':
            return
        # 先写到终端（保持原有行为）
        try:
            self.terminal.write(message)
        except Exception:
            # 如果终端写入失败，也不要阻塞日志写文件
            pass
        # 再写到文件
        try:
            if self.keep_open:
                self._log.write(message)
                # 保证关键日志及时落盘
                self._log.flush()
            else:
                with open(self.filename, self.mode, encoding=self.encoding) as log:
                    log.write(message)
        except Exception:
            # 忽略文件写入的异常（或根据需求抛出/记录）
            pass

    def flush(self):
        # 支持 flush，某些库会调用
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            if self._log:
                self._log.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    def close(self):
        try:
            if self._log:
                self._log.close()
                self._log = None
        except Exception:
            pass

    def __del__(self):
        self.close()


def create_dataset(dataset):
    """
    返回:
    x_train, y_train,
    x_test_balanced, y_test_balanced,
    x_test_imbalanced, y_test_imbalanced,
    Is_RGB
    """

    # 初始化变量以避免未定义错误
    train_dataset = None
    test_dataset_bal = None
    test_dataset_imb = None
    x_train, y_train, x_test_bal, y_test_bal, x_test_imb, y_test_imb = None, None, None, None, None, None

    if dataset == "cifar10":
        # 实例化三个数据集
        train_dataset = IMBALANCECIFAR10("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'CIFAR-10'))
        test_dataset_bal = IMBALANCECIFAR10("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'CIFAR-10'))
        test_dataset_imb = IMBALANCECIFAR10("test_Imbalanced", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'CIFAR-10'))
        Is_RGB = True
        Flag = 1

    elif dataset == "STL":
        # 实例化三个数据集 (STL-10 需要 download=True)
        train_dataset = IMBALANCESTL10(phase="train", imbalance_ratio=0.01, root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True)
        test_dataset_bal = IMBALANCESTL10(phase="test", imbalance_ratio=0.01, root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True)
        test_dataset_imb = IMBALANCESTL10(phase="test_Imbalanced", imbalance_ratio=0.01, root=os.path.join(DATASETS_ROOT, 'STL-10'),
                                          download=True)
        Is_RGB = True
        Flag = 2

    elif dataset == "fashion_mnist":
        train_dataset = IMBALANCEFMNIST("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'Fashion_minist'))
        test_dataset_bal = IMBALANCEFMNIST("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'Fashion_minist'))
        test_dataset_imb = IMBALANCEFMNIST("test_Imbalanced", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'Fashion_minist'))
        Is_RGB = False
        Flag = 1

    elif dataset == "SVHN" or dataset == "EuroSAT":
        # Numpy 加载路径
        x_train_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_data.npy')) / 255.0
        y_train_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_label.npy'))
        x_test_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_data.npy')) / 255.0
        y_test_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_label.npy'))

        Is_RGB = True
        Flag = 4

    else:
        if dataset == "mnist":
            train_dataset = IMBALANCEMNIST("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            test_dataset_bal = IMBALANCEMNIST("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            test_dataset_imb = IMBALANCEMNIST("test_Imbalanced", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            Flag = 1
        else:
            # 其他自定义 Numpy 数据集
            x_train_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_data.npy')) / 255.0
            y_train_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_label.npy'))
            x_test_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_data.npy')) / 255.0
            y_test_raw = numpy.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_label.npy'))
            Flag = 3
        Is_RGB = False

    # --- 数据处理逻辑 ---

    if Flag == 1:
        # CIFAR10, FMNIST, MNIST (Torch Dataset 路径)
        # 提取训练集
        x_train = train_dataset.data
        y_train = numpy.array(train_dataset.labels)

        # 提取平衡测试集
        x_test_bal = test_dataset_bal.data
        y_test_bal = numpy.array(test_dataset_bal.labels)

        # 提取不平衡测试集
        x_test_imb = test_dataset_imb.data
        y_test_imb = numpy.array(test_dataset_imb.labels)

        # 类型转换与形状调整
        if not Is_RGB:
            if hasattr(x_test_bal, 'numpy'): x_test_bal = x_test_bal.numpy()
            if hasattr(x_test_imb, 'numpy'): x_test_imb = x_test_imb.numpy()

        x_train = x_train.astype(numpy.float32)
        x_test_bal = x_test_bal.astype(numpy.float32)
        x_test_imb = x_test_imb.astype(numpy.float32)

        dim = x_train.ndim
        if dim == 4:
            shape = x_train.shape
            # 如果通道在最后 (N, H, W, C)，转到中间 (N, C, H, W)
            if shape[1] != 3 and shape[-1] in [1, 3]:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test_bal = x_test_bal.transpose(dims)
                x_test_imb = x_test_imb.transpose(dims)

    elif Flag == 3:
        # 自定义 Numpy 数据集
        x_train_raw = x_train_raw.astype(numpy.float32)
        x_test_raw = x_test_raw.astype(numpy.float32)
        y_train_raw = y_train_raw.astype(numpy.int64)
        y_test_raw = y_test_raw.astype(numpy.int64)

        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        # 调用更新后的函数，返回 6 个值
        x_train, y_train, x_test_bal, y_test_bal, x_test_imb, y_test_imb = imbalance_MRBand(
            x_train_raw, y_train_raw, x_test_raw, y_test_raw,
            img_num_per_cls,
            imb_test_factor=0.1  # 设置不平衡因子，例如 0.1 (10:1)
        )

    elif Flag == 4:
        # SVHN / EuroSAT
        x_train_raw = x_train_raw.astype(numpy.float32)
        x_test_raw = x_test_raw.astype(numpy.float32)
        y_train_raw = y_train_raw.astype(numpy.int64)
        y_test_raw = y_test_raw.astype(numpy.int64)

        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        x_train, y_train, x_test_bal, y_test_bal, x_test_imb, y_test_imb = imbalance_MRBand(
            x_train_raw, y_train_raw, x_test_raw, y_test_raw,
            img_num_per_cls,
            imb_test_factor=0.1
        )

        dim = x_train.ndim
        if dim == 4:
            shape = x_train.shape
            if shape[1] != 3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test_bal = x_test_bal.transpose(dims)
                x_test_imb = x_test_imb.transpose(dims)
                print("jiaohuan")

    else:
        # Flag == 2 (STL-10)
        # 提取数据
        x_train = train_dataset.data
        y_train = numpy.array(train_dataset.labels)

        x_test_bal = test_dataset_bal.data
        y_test_bal = numpy.array(test_dataset_bal.labels)

        x_test_imb = test_dataset_imb.data
        y_test_imb = numpy.array(test_dataset_imb.labels)

        x_train = x_train.astype(numpy.float32)
        x_test_bal = x_test_bal.astype(numpy.float32)
        x_test_imb = x_test_imb.astype(numpy.float32)

        dim = x_train.ndim
        if dim == 4:
            shape = x_train.shape
            if shape[1] != 3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test_bal = x_test_bal.transpose(dims)
                x_test_imb = x_test_imb.transpose(dims)

        # STL-10 特有的下采样 (96x96 -> 48x48)
        zoom_factors = (1, 1, 0.5, 0.5)
        try:
            x_train = zoom(x_train, zoom_factors, order=1)
            x_test_bal = zoom(x_test_bal, zoom_factors, order=1)
            x_test_imb = zoom(x_test_imb, zoom_factors, order=1)
        except Exception as e:
            print(f"Warning: Zoom failed for {dataset}. Error: {e}")

    # 返回 7 个值
    return x_train, y_train, x_test_bal, y_test_bal, x_test_imb, y_test_imb, Is_RGB


group2_indices = [4, 5, 6]  # 中间三类
group3_indices = [7, 8, 9]  # 尾部三类

x_train,y_train,x_test,y_test, x_test_imb, y_test_imb,Is_RGB = create_dataset(opt.dataset)


print(f"Training Set Shape: {x_train.shape}")
print(f"Balanced Test Set Shape: {x_test.shape}")       # 原始均衡测试集 (例如 10000 张)
print(f"Imbalanced Test Set Shape: {x_test_imb.shape}")     # 长尾测试集 (例如 ~5000 张)
print(f"Imbalanced Test Set Shape: {y_test_imb.shape}")

unique_labels = numpy.unique(y_train)
num_unique_labels = len(unique_labels)

def make_Singma():
    return random.randint(1, 4)

def make_Order():
    return random.randint(0, 3)

def make_Theta():
    return random.randint(0, 8)

def make_Frequency():
    return random.randint(0, 8)

def make_classificationN():
    return random.randint(1, 4)

def make_coordsX():
    return random.randint(0, h-3)

def make_coordsY():
    return random.randint(0, w-3)

def make_windowSize():
    return random.randint(3, w)

def make_width():
    return random.randint(3, w)

def make_height():
    return random.randint(3, h)

def make_reuse():
    return random.randint(0, 1)

def make_Magnitude():
    return random.randint(1, 10)

def make_Probability():
    return random.randint(1, 10) /10.0
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# parameters:
population = 100
generation = 25
generation_1 = 25
steepness=30
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 4 #4
augMinDepth = 1#3
augMaxDepth = 6#6
initialMaxDepth = 10 #10
maxDepth = 13
if Is_RGB:
    h = x_train[0].shape[1]
    w = x_train[0].shape[2]
else:
    h = x_train[0].shape[0]
    w = x_train[0].shape[1]
print('h', h)
print('w', w)

# GP
if Is_RGB:
    pset = gp.PrimitiveSetTyped('MAIN', [Img, Img, Img], ClassLabel, prefix='Image')
else:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], ClassLabel, prefix='Image')

pset.addPrimitive(fe_fs.classifier_selection, [Vector_C, Int3], ClassLabel, name='classifier')

# high level features--feature construction layer
pset.addPrimitive(fe_fs.feature_con, [Vector, Vector], Vector_C, name='Features2C')
pset.addPrimitive(fe_fs.feature_con2, [Vector, Vector_C], Vector_C, name='FeaturesC_dif')
pset.addPrimitive(fe_fs.feature_con, [Vector, Vector, Vector], Vector_C, name='Features3C')
pset.addPrimitive(fe_fs.feature_con, [Vector, Vector, Vector, Vector], Vector_C, name='Features4C')



# pset.addPrimitive(fe_fs.feature_con, [Vector_raw, Vector], Vector_C, name='Features')


# input is the whole raw image
pset.addPrimitive(fe_fs.uniform_lbp_raw, [Img], Vector, name='uLBP_G')
pset.addPrimitive(fe_fs.all_sift_raw, [Img], Vector, name='SIFT_G')
pset.addPrimitive(fe_fs.global_hog_raw, [Img], Vector, name='HOG_G')
pset.addPrimitive(fe_fs.hist_raw, [Img], Vector, name='Hist_G')
pset.addPrimitive(fe_fs.featureEx_raw, [Img], Vector, name='featureEx_G')

pset.addPrimitive(fe_fs.medianf_original, [Img], Img, name='Median_G')
pset.addPrimitive(fe_fs.meanf_original, [Img], Img, name='Mean_G')
pset.addPrimitive(fe_fs.minf_original, [Img], Img, name='Min_G')
pset.addPrimitive(fe_fs.maxf_original, [Img], Img, name='Max_G')
pset.addPrimitive(fe_fs.gau_original, [Img, Int1], Img, name='Gau_G')
pset.addPrimitive(fe_fs.gauD_original, [Img, Int1, Int2, Int2], Img, name='GauD_G')
pset.addPrimitive(fe_fs.laplace_original, [Img], Img, name='Lap_G')
pset.addPrimitive(fe_fs.gaussian_Laplace1_original, [Img], Img, name='LoG1_G')
pset.addPrimitive(fe_fs.gaussian_Laplace2_original, [Img], Img, name='LoG2_G')
pset.addPrimitive(fe_fs.sobelxy_original, [Img], Img, name='Sobel_G')
pset.addPrimitive(fe_fs.sobelx_original, [Img], Img, name='SobelX_G')
pset.addPrimitive(fe_fs.sobely_original, [Img], Img, name='SobelY_G')

# mid level features--feature extraction layer
pset.addPrimitive(fe_fs.uniform_lbp, [Img2], Vector, name='uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img2], Vector, name='SIFT')
pset.addPrimitive(fe_fs.global_hog, [Img2], Vector, name='HOG')
pset.addPrimitive(fe_fs.hist, [Img2], Vector, name='Hist')
pset.addPrimitive(fe_fs.featureEx, [Img2], Vector, name='featureEx')

# the input is detected image region

pset.addPrimitive(fe_fs.medianf, [Img2, s], Img2, name='Median2')
pset.addPrimitive(fe_fs.meanf, [Img2, s], Img2, name='Mean2')
pset.addPrimitive(fe_fs.minf, [Img2, s], Img2, name='Min2')
pset.addPrimitive(fe_fs.maxf, [Img2, s], Img2, name='Max2')
pset.addPrimitive(fe_fs.gau, [Img2, Int1, s], Img2, name='Gau2')
pset.addPrimitive(fe_fs.gauD, [Img2, Int1, Int2, Int2, s], Img2, name='GauD2')
pset.addPrimitive(fe_fs.laplace, [Img2, s], Img2, name='Lap2')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img2, s], Img2, name='LoG12')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img2, s], Img2, name='LoG22')
pset.addPrimitive(fe_fs.sobelxy, [Img2, s], Img2, name='Sobel2')
pset.addPrimitive(fe_fs.sobelx, [Img2, s], Img2, name='SobelX2')
pset.addPrimitive(fe_fs.sobely, [Img2, s], Img2, name='SobelY2')

pset.addPrimitive(fe_fs.median_raw, [Img1, s], Img2, name='Median')
pset.addPrimitive(fe_fs.mean_raw, [Img1, s], Img2, name='Mean')
pset.addPrimitive(fe_fs.min_raw, [Img1, s], Img2, name='Min')
pset.addPrimitive(fe_fs.max_raw, [Img1, s], Img2, name='Max')
pset.addPrimitive(fe_fs.gau_raw, [Img1, Int1, s], Img2, name='Gau')
pset.addPrimitive(fe_fs.gauD_raw, [Img1, Int1, Int2, Int2, s], Img2, name='GauD')
pset.addPrimitive(fe_fs.laplace_raw, [Img1, s], Img2, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1_raw, [Img1, s], Img2, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2_raw, [Img1, s], Img2, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy_raw, [Img1, s], Img2, name='Sobel')
pset.addPrimitive(fe_fs.sobelx_raw, [Img1, s], Img2, name='SobelX')
pset.addPrimitive(fe_fs.sobely_raw, [Img1, s], Img2, name='SobelY')

# region detection
pset.addPrimitive(fe_fs.regionR, [Img, X, Y, width, height], Img1, name='RegionR')
pset.addPrimitive(fe_fs.regionS, [Img, X, Y, windowSize], Img1, name='RegionS')


# Terminals
pset.renameArguments(ARG0='Image')


pset.addEphemeralConstant('Singma', make_Singma, Int1)
pset.addEphemeralConstant('Order', make_Order, Int2)
pset.addEphemeralConstant('Theta', make_Theta, Float1)
pset.addEphemeralConstant('Frequency', make_Frequency, Float2)
pset.addEphemeralConstant('classificationN', make_classificationN, Int3)
pset.addEphemeralConstant('coordsX', make_coordsX, X)
pset.addEphemeralConstant('coordsY', make_coordsY, Y)
pset.addEphemeralConstant('windowSize', make_windowSize, windowSize)
pset.addEphemeralConstant('width', make_width, width)
pset.addEphemeralConstant('height', make_height, height)
pset.addEphemeralConstant('reuse', make_reuse, s)


pset_1 = gp.PrimitiveSetTyped('MAIN', [Img0, Flag], Img, prefix='Image')

# Change
pset_1.addPrimitive(fe_fs.change, [Img0], Img, name='Change')

#Data augment for Img
pset_1.addPrimitive(fe_fs.shear_x, [Img0, Int4, Float4, Flag], Img0, name='ShearX')
pset_1.addPrimitive(fe_fs.shear_y, [Img0, Int4, Float4, Flag], Img0, name='ShearY')
pset_1.addPrimitive(fe_fs.translate_x, [Img0, Int4, Float4, Flag], Img0, name='TranslateX')
pset_1.addPrimitive(fe_fs.translate_y, [Img0, Int4, Float4, Flag], Img0, name='TranslateY')
pset_1.addPrimitive(fe_fs.rotate, [Img0, Int4, Float4, Flag], Img0, name='Rotate')
pset_1.addPrimitive(fe_fs.color, [Img0, Int4, Float4, Flag], Img0, name='Color')
pset_1.addPrimitive(fe_fs.auto_contrast, [Img0, Int4, Float4, Flag], Img0, name='AutoContrast')
pset_1.addPrimitive(fe_fs.sharpness, [Img0, Int4, Float4, Flag], Img0, name='Sharpness')
pset_1.addPrimitive(fe_fs.brightness, [Img0, Int4, Float4, Flag], Img0, name='Brightness')
pset_1.addPrimitive(fe_fs.posterize, [Img0, Int4, Float4, Flag], Img0, name='Posterize')
pset_1.addPrimitive(fe_fs.solarize, [Img0, Int4, Float4, Flag], Img0, name='Solarize')
pset_1.addPrimitive(fe_fs.equalize, [Img0, Int4, Float4, Flag], Img0, name='Equalize')
pset_1.addPrimitive(fe_fs.invert, [Img0, Int4, Float4, Flag], Img0, name='Invert')
pset_1.addPrimitive(fe_fs.cutout, [Img0, Int4, Float4, Flag], Img0, name='Cutout')
pset_1.addPrimitive(fe_fs.contrast, [Img0, Int4, Float4, Flag], Img0, name='Contrast')
pset_1.addEphemeralConstant('Probability', make_Probability, Float4)
pset_1.addEphemeralConstant('Magnitude', make_Magnitude, Int4)
pset_1.renameArguments(ARG0='Image')
pset_1.renameArguments(Image1='Flag')
##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox1 = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox1.register("expr", gp_restrict.genConstrained, pset=pset_1, min_=augMinDepth, max_=augMaxDepth)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox1.register("individual", tools.initIterate, creator.Individual, toolbox1.expr)

toolbox.register("compile", gp.compile, pset=pset)
toolbox1.register("compile", gp.compile, pset=pset_1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox1.register("population", tools.initRepeat, list, toolbox1.individual)
toolbox.register("mapp", map)




def balanced_sample_plan(y, random_state=None):
    """
    计算每个原始样本在平衡过采样中应被使用的次数。

    对于每个类别：
      - 若其数量 < 最大类数量，则通过有放回采样补足；
      - 每次采样对应一个样本的“使用次数+1”；
      - 最终每个样本的使用次数 = 1（原始） + 被重复选中的次数。

    Args:
        y: 标签数组 (num_samples,)
        random_state: 随机种子

    Returns:
        sample_plan: np.array of shape (num_samples,), dtype=int
                     sample_plan[i] = 样本 i 在平衡后应被使用的总次数（≥1）
    """
    import numpy as np
    from collections import Counter
    from sklearn.utils import resample

    y = np.asarray(y)
    N = len(y)
    sample_plan = np.zeros(N, dtype=int)

    class_counts = Counter(y)
    if not class_counts:
        return sample_plan

    max_class_count = max(class_counts.values())

    sample_plan = np.zeros(N, dtype=int)

    for cls in class_counts:
        idx = np.where(y == cls)[0]
        current_count = len(idx)
        target = max_class_count

        if current_count >= target:
            # 取前 target 个，每个用1次
            selected = idx[:target]
            for i in selected:
                sample_plan[i] = 1
        else:
            # 先保证每个样本至少1次
            for i in idx:
                sample_plan[i] = 1

            # 还需补充 extra = target - current_count 次
            extra = target - current_count
            if extra > 0:
                additional_idx = resample(
                    idx,
                    replace=True,
                    n_samples=extra,
                    random_state=random_state
                )
                # 统计额外选中的次数
                unique_add, add_counts = np.unique(additional_idx, return_counts=True)
                for i, cnt in zip(unique_add, add_counts):
                    sample_plan[i] += cnt

    return sample_plan

import numpy as numpy
from collections import Counter




def get_feature_requirements(target_count, current_gen, ngen=50, steepness=30.0):
    """
    根据 Sigmoid 函数计算当前代数对应的特征需求值：
      - 若 current_gen == 0，返回 0；
      - 否则，计算 Sigmoid 值，乘以 target_count，并向上取整。

    参数:
        target_count (float): 目标值（y 轴上限）
        current_gen (float): 当前代数（标量，0 <= current_gen <= ngen）
        ngen (int, optional): 总代数，默认为 50
        steepness (float, optional): Sigmoid 曲线陡峭度，默认为 10.0

    返回:
        int: 向上取整后的 y 值，但 gen=0 时为 0
    """
    if current_gen == 0:
        return 0

    x = steepness * (current_gen - ngen / 2) / ngen
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    y = target_count * sigmoid
    return math.ceil(y)

def compute_global_sample_plan(fold_splits, y_train, current_gen=1, ngen=50, steepness=30):
    """
    计算每张图像在 K 折 CV 中，在任意一折训练集中所需出现的总样本数（原始+增强）的最大值，
    并返回每折每类的最终目标数量（原始数量 + 当前代数下需要的增强数量）。

    参数:
        fold_splits: list of tuples, [(train_idx, val_idx), ...], 来自 KFold.split()
        y_train: np.array, shape=(N,), 原始训练标签
        current_gen, ngen, growth_factor: 传入以计算指数增长的增强数量

    返回:
        sample_plan: np.array, shape=(N,),
                     sample_plan[i] = 图像 i 在所有折中某折所需的最大总使用次数（含原图）
        fold_targets_list: list 与 fold_splits 对应的列表，每项为 dict {cls: target_total}
                           target_total = original_count_in_fold + need（need 来自 get_feature_requirements）
    """
    N = len(y_train)
    sample_plan = numpy.zeros(N, dtype=int)
    fold_targets_list = []

    for train_idx, _ in fold_splits:
        y_train_fold = y_train[train_idx]
        class_counts = Counter(y_train_fold)
        if not class_counts:
            # 空折，记录空映射并继续
            fold_targets_list.append({})
            continue

        max_count = max(class_counts.values())
        fold_usage = numpy.zeros(N, dtype=int)
        fold_target_map = {}

        # 对每个类别进行规划
        for cls, count in class_counts.items():
            # 在该折训练集中的全局索引
            cls_in_train = [idx for idx in train_idx if y_train[idx] == cls]

            if count >= max_count:
                # 多数类：每张图像只需用一次（原始样本）
                for img_id in cls_in_train:
                    fold_usage[img_id] = 1
                fold_target_map[cls] = count  # 无需增强
            else:
                gap = max_count - count  # 最终目标与当前类样本数的差（需要增强的总数）
                need = gap
                #need = get_feature_requirements(gap, current_gen=current_gen, ngen=ngen, steepness=steepness)
                #print(need)
                target_total = count + need
                fold_target_map[cls] = target_total

                if need <= 0:
                    # 当前代不需要增强，所有样本至少一次
                    for img_id in cls_in_train:
                        fold_usage[img_id] = 1
                else:
                    # 随机有放回地选择样本进行增强（返回的是全局索引）
                    chosen = numpy.random.choice(cls_in_train, size=need, replace=True)
                    chosen_indices, counts = numpy.unique(chosen, return_counts=True)

                    # 每张图像的总使用次数 = 原始1次 + 被选中增强的次数
                    for img_id, n in zip(chosen_indices, counts):
                        fold_usage[img_id] = 1 + int(n)

                    # 确保未被抽中的该类图像也至少用一次
                    all_cls_in_train_set = set(cls_in_train)
                    chosen_set = set(chosen_indices)
                    for img_id in all_cls_in_train_set - chosen_set:
                        fold_usage[img_id] = 1

        # 更新全局计划：取最大值（跨折的最大使用需求）
        sample_plan = numpy.maximum(sample_plan, fold_usage)
        fold_targets_list.append(fold_target_map)
    #print(fold_targets_list)
    return sample_plan, fold_targets_list


def evalTrain(pair, current_gen, x_train, y_train, ngen, steepness):
    """
    评估函数，基于 compute_global_sample_plan 的结果预生成增强池，并在每折中无放回采样构建平衡训练集。

    参数:
        pair: (individual, individual_1)（遗传表达式）
        x_train, y_train: 训练数据与标签
        current_gen, ngen, growth_factor: 用于 get_feature_requirements 的参数（传入到 compute_global_sample_plan）
    返回:
        (accuracy,) 作为 DEAP 的适应度元组
    """
    try:
        individual, individual_1 = pair
        classifier_idx = individual[len(individual) - 1].value
        confusion_matrices = []

        scaler = MinMaxScaler()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_splits = list(kfold.split(x_train, y_train))

        # 由 compute_global_sample_plan 返回（sample_plan, fold_targets_list）
        sample_plan, fold_targets = compute_global_sample_plan(
            fold_splits, y_train,
            current_gen=current_gen, ngen=ngen, steepness=steepness
        )
        #print(fold_targets)
        func = toolbox.compile(expr=individual)
        func_1 = toolbox1.compile(expr=individual_1)
        N = len(x_train)
        aug_pool = [[] for _ in range(N)]

        # ========== 预生成增强池：第0个是原始（Flag=False），其余是增强（Flag=True） ==========
        for i in range(N):
            num_needed = int(sample_plan[i])
            if num_needed <= 0:
                continue

            aug_samples = []
            # 第一次：原始特征（Flag=False）
            feat0 = func_1(x_train[i], False)
            if Is_RGB:
                output0 = func(feat0[0], feat0[1], feat0[2])[0]
            else:
                output0 = func(feat0[0])[0]
            aug_samples.append(output0)

            # 后续：增强特征（Flag=True）
            for _ in range(1, num_needed):

                # policy = random.choice(augmentation_policies)
                # feat_aug = apply_augmentation_sequence(x_train[i],  policy, True)
                feat_aug = apply_random_policy_augment(x_train[i], True)
                # feat_aug = func_1(x_train[i], True)
                if Is_RGB:
                    output_aug = func(feat_aug[0], feat_aug[1], feat_aug[2])[0]
                else:
                    output_aug = func(feat_aug[0])[0]
                aug_samples.append(output_aug)

            aug_pool[i] = aug_samples

        # 选择分类器
        if classifier_idx == 1:
            classifier = LinearSVC()
        elif classifier_idx == 2:
            classifier = RandomForestClassifier()
        elif classifier_idx == 3:
            classifier = LogisticRegression()
        else:
            classifier = ExtraTreesClassifier()

        # ========== K 折交叉验证 ==========
        for fold_idx, (train_index, val_index) in enumerate(fold_splits):
            X_train_sub = []
            y_train_sub = []

            y_train_fold = y_train[train_index]
            class_counts = Counter(y_train_fold)
            fold_target_map = fold_targets[fold_idx] if fold_idx < len(fold_targets) else {}

            # ========== 构建平衡训练集（无放回，因 aug_pool 已满足数量） ==========
            for cls in class_counts:
                cls_train_indices = [idx for idx in train_index if y_train[idx] == cls]

                # 收集该类所有预生成的特征（原始 + 增强）
                available_features = []
                for idx in cls_train_indices:
                    available_features.extend(aug_pool[idx])

                # 从 precomputed fold_targets 中取出 target_count（若不存在则退回到 max_count 行为）
                target_count = fold_target_map.get(cls, None)
                if target_count is None:
                    # 退回兼容逻辑：使用该折的最大类数作为目标
                    target_count = max(class_counts.values()) if class_counts else 0

                # 理论上 len(available_features) >= target_count
                assert len(available_features) >= target_count, \
                    f"Class {cls}: available={len(available_features)} < target={target_count}"

                # 无放回随机采样 target_count 个
                selected_idx = numpy.random.choice(
                    len(available_features),
                    size=target_count,
                    replace=False
                )
                for j in selected_idx:
                    X_train_sub.append(available_features[j])
                    y_train_sub.append(cls)

            X_train_sub = numpy.array(X_train_sub)
            y_train_sub = numpy.array(y_train_sub)
            #print(f"{len(X_train_sub)}!!!!!!!!!!!!!!!!!!!!!!!")
            # ========== 构建验证集：固定使用原始特征（aug_pool[i][0]） ==========
            X_test_sub = []
            y_test_sub = []
            for i in val_index:
                assert len(aug_pool[i]) > 0, f"aug_pool[{i}] is empty!"
                X_test_sub.append(aug_pool[i][0])
                y_test_sub.append(y_train[i])

            X_test_sub = numpy.array(X_test_sub)
            y_test_sub = numpy.array(y_test_sub)

            # ========== 训练与评估 ==========
            train_norm = scaler.fit_transform(X_train_sub)
            test_norm = scaler.transform(X_test_sub)

            classifier.fit(train_norm, y_train_sub)
            y_pred = classifier.predict(test_norm)

            cm = confusion_matrix(y_test_sub, y_pred, labels=unique_labels)
            confusion_matrices.append(cm)

        final_cm = numpy.sum(confusion_matrices, axis=0)

        # for i in range(len(unique_labels)):
        #     total_samples = final_cm[i].sum()  # 第i类的总样本数（所有预测列的和）
        #     correct = final_cm[i, i]  # 第i类正确分类的数量
        #     print(
        #         f"类别 {unique_labels[i]}: 总样本={total_samples}, 正确={correct}, 准确率={correct / total_samples * 100:.2f}%")

        accuracy = final_cm.diagonal().sum() / final_cm.sum()
        accuracy = round(100 * accuracy, 2)
    except Exception as e:
        # 如需调试可临时打印 e
        # print("evalTrain exception:", e)
        accuracy = 0

    print(accuracy)
    return (accuracy,)



def evalTrain_0(individual, x_train, y_train):
    try:
        func = toolbox.compile(expr=individual)
        classifier_idx = individual[len(individual) - 1].value
        train_tf = []
        if Is_RGB:
            for i in range(0, len(y_train)):
                train_tf.append(numpy.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])[0]))
        else:
            for i in range(0, len(y_train)):
                train_tf.append(numpy.asarray(func(x_train[i][ :, :])[0]))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        if classifier_idx == 1:
            lsvm = LinearSVC()
            accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
        elif classifier_idx == 2:
            rf = RandomForestClassifier()
            accuracy = round(100 * cross_val_score(rf, train_norm, y_train, cv=5).mean(), 2)
        elif classifier_idx == 3:
            lr = LogisticRegression()
            accuracy = round(100 * cross_val_score(lr, train_norm, y_train, cv=5).mean(), 2)
        else:
            erf = ExtraTreesClassifier()
            accuracy = round(100 * cross_val_score(erf, train_norm, y_train, cv=5).mean(), 2)
    except:
        accuracy=0
    print(accuracy)
    return accuracy,


def parallel_evalTrain(pair, current_gen, x_train, y_train, ngen, steepness):
    return evalTrain(pair, current_gen =current_gen, x_train = x_train, y_train = y_train, ngen=ngen, steepness=steepness )

def parallel_evalTrain_0(individual, x_train, y_train):
    return evalTrain_0(individual, x_train = x_train, y_train = y_train)

# Register the evaluation function with caching
toolbox.register("evaluate_0", parallel_evalTrain_0, x_train = x_train, y_train = y_train)
toolbox.register("evaluate", parallel_evalTrain, current_gen = 1, x_train = x_train, y_train = y_train, ngen=generation_1, steepness=steepness)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


# Register the evaluation function with caching
toolbox1.register("evaluate", parallel_evalTrain, x_train = x_train, y_train = y_train)
toolbox1.register("select", tools.selTournament, tournsize=5)
toolbox1.register("selectElitism", tools.selBest)
def cxConstrainedHeight_individual(ind1, ind2, min_h=2, max_h=5, max_trials=20):
    # 调用原始交叉（接受 PrimitiveTree，返回 PrimitiveTree）
    new_tree1, new_tree2 = gp_restrict.cxConstrainedHeight(ind1, ind2, min_h, max_h, max_trials)
    # 包装为 Individual
    return creator.Individual(new_tree1), creator.Individual(new_tree2)

# 注册到 toolbox1
toolbox1.register("mate", cxConstrainedHeight_individual, min_h=augMinDepth, max_h=augMaxDepth, max_trials=20)
toolbox1.register("mutate", gp_restrict.mutUniformHeight, pset= pset_1, min_h=augMinDepth, max_h=augMaxDepth )


def GPMain(pkl_file, randomSeeds):
    import os
    import pickle
    import random
    import multiprocessing
    from deap import tools

    random.seed(randomSeeds)

    pop = None
    pop_1 = None
    start_gen_2 = 1  # 第二阶段的起始代数
    run_stage_1 = True  # 是否需要运行第一阶段

    # 1. 尝试从 checkpoint 恢复
    try:
        with open(pkl_file, "rb") as cp_file:
            checkpoint_data = pickle.load(cp_file)
            loaded_pop = checkpoint_data.get('population', None)
            loaded_pop_1 = checkpoint_data.get('population_1', None)
            loaded_gen = int(checkpoint_data.get('generation', 0))

            if loaded_pop is not None and loaded_pop_1 is not None:
                print(f"检测到第二阶段 Checkpoint，将跳过第一阶段，从第二阶段第 {loaded_gen + 1} 代恢复。")
                pop = loaded_pop
                pop_1 = loaded_pop_1
                start_gen_2 = loaded_gen + 1
                run_stage_1 = False  # 成功加载第二阶段，跳过第一阶段
            else:
                print("检测到第一阶段 Checkpoint。因第一阶段极快，忽略此 Checkpoint，将重新开始。")
                # 保持 pop, pop_1 为 None，触发后续的重新初始化
    except FileNotFoundError:
        print("未找到 Checkpoint，将初始化全新种群。")
    except Exception as e:
        print("加载 Checkpoint 失败 (将重新初始化):", e)

    # 2. 若未恢复第二阶段，则初始化全新种群
    if pop is None:
        pop = toolbox.population(population)  # 注意：此处的 population 应该是种群大小变量
    if pop_1 is None:
        pop_1 = toolbox1.population(population)

    num_processes = 10  # min(10, multiprocessing.cpu_count())

    # 3. 第一阶段执行逻辑
    if run_stage_1 and generation > 0:
        print("--- 第一阶段启动 ---")
        with multiprocessing.Pool(processes=num_processes) as pool:
            old_map = getattr(toolbox, "map", None)
            toolbox.map = pool.map
            try:
                # 第一阶段永远从第 1 代开始
                pop = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, 1, pkl_file)
            finally:
                if old_map is None:
                    delattr(toolbox, "map")
                else:
                    toolbox.map = old_map
        print("--- 第一阶段结束 ---")
    elif run_stage_1 and generation <= 0:
        print("第一阶段未启用，直接进入第二阶段")

    # 4. 第二阶段执行逻辑
    if start_gen_2 <= generation_1:
        print(f"--- 第二阶段启动 (当前代数: {start_gen_2}, 目标代数: {generation_1}) ---")
        with multiprocessing.Pool(processes=num_processes) as pool:
            old_map = getattr(toolbox, "map", None)
            toolbox.map = pool.map
            try:
                best_a, best_b = evalGP.eaSimple_1(pop, pop_1, toolbox, toolbox1, cxProb, mutProb, elitismProb,
                                                   generation_1, start_gen_2, pkl_file, pset_1, x_train, y_train,
                                                   steepness)
            finally:
                if old_map is None:
                    delattr(toolbox, "map")
                else:
                    toolbox.map = old_map
        print("--- 第二阶段结束 ---")
    else:
        print("第二阶段已满足目标代数，直接提取最优个体。")

        best_a = tools.selBest(pop, k=1)[0]
        best_b = tools.selBest(pop_1, k=1)[0]

    return best_a, best_b


def evalTest(toolbox, individual, individual_1, trainData, trainLabel,
             test_bal, testL_bal,
             test_imb, testL_imb):
    """
    在平衡测试集和不平衡测试集上分别评估模型。

    参数:
        test_bal, testL_bal: 平衡测试集数据和标签
        test_imb, testL_imb: 不平衡测试集数据和标签

    返回:
        (bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
         imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1)
    """
    print("开始 test (双种群协同 - 双测试集评估)")
    classifier_idx = individual[len(individual) - 1].value
    print(f"Classifier Index: {classifier_idx}")

    func = toolbox.compile(expr=individual)
    func_1 = toolbox1.compile(expr=individual_1)  # 修正：原代码用了 toolbox1，这里假设是同一个 toolbox 或已定义

    # ========== 1. 训练阶段 (只执行一次) ==========
    sample_plan = balanced_sample_plan(trainLabel, random_state=42)
    train_tf = []
    trainLabel_balanced = []

    # RGB 分支
    if Is_RGB:
        for i in range(len(trainData)):
            total_needed = sample_plan[i]
            for k in range(total_needed):
                Flag = False if k == 0 else True

                # policy = random.choice(augmentation_policies)
                # enhanced_img = apply_augmentation_sequence(trainData[i], policy, Flag)
                enhanced_img = apply_random_policy_augment(trainData[i], Flag)

                # enhanced_img = func_1(trainData[i], Flag)
                r, g, b = enhanced_img[0, :, :], enhanced_img[1, :, :], enhanced_img[2, :, :]
                final_feat = func(r, g, b)[0]
                train_tf.append(numpy.asarray(final_feat))
                trainLabel_balanced.append(trainLabel[i])
    # 灰度分支
    else:
        for i in range(len(trainData)):
            total_needed = sample_plan[i]
            for k in range(total_needed):
                Flag = False if k == 0 else True

                # policy = random.choice(augmentation_policies)
                # enhanced_img = apply_augmentation_sequence(trainData[i], policy, Flag)

                enhanced_img = apply_random_policy_augment(trainData[i], Flag)

                # enhanced_img = func_1(trainData[i], Flag)
                r = enhanced_img[0, :, :]
                final_feat = func(r)[0]
                train_tf.append(numpy.asarray(final_feat))
                trainLabel_balanced.append(trainLabel[i])

    # 训练分类器
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)

    if classifier_idx == 1:
        classifier = LinearSVC()
        classifier.fit(train_norm, trainLabel_balanced)
    elif classifier_idx == 2:
        classifier = RandomForestClassifier()
        classifier.fit(train_norm, trainLabel_balanced)
    elif classifier_idx == 3:
        classifier = LogisticRegression()
        classifier.fit(train_norm, trainLabel_balanced)
    else:
        classifier = ExtraTreesClassifier()
        classifier.fit(train_norm, trainLabel_balanced)

    # ========== 2. 定义内部评估函数 ==========
    def _run_single_evaluation(test_data, test_labels, set_name="Test"):
        print(f"正在评估 {set_name} ...")
        test_tf = []

        # 特征提取 (针对当前测试集)
        if Is_RGB:
            for j in range(len(test_data)):
                enhanced_test = func_1(test_data[j], False)
                r, g, b = enhanced_test[0, :, :], enhanced_test[1, :, :], enhanced_test[2, :, :]
                final_test = func(r, g, b)[0]
                test_tf.append(numpy.asarray(final_test))
        else:
            for j in range(len(test_data)):
                enhanced_test = func_1(test_data[j], False)
                r = enhanced_test[0, :, :]
                final_test = func(r)[0]
                test_tf.append(numpy.asarray(final_test))

        if len(test_tf) == 0:
            # 防止空测试集报错
            return None, "No samples", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 归一化 (使用训练集的 scaler)
        test_norm = min_max_scaler.transform(test_tf)

        # 预测
        if classifier_idx == 1:
            probabilities = classifier.decision_function(test_norm)
            # LinearSVC 的 decision_function 输出形状可能是 (n_samples,) 或 (n_samples, n_classes)
            if probabilities.ndim == 1:
                # 二分类或多分类 One-vs-Rest 可能需要特殊处理，这里简单 reshape 以便后续 argmax
                # 注意：LinearSVC decision_function 对于多类通常是 (n_samples, n_classes)
                # 如果是 (n_samples,) 说明是二分类，需要转换
                if len(numpy.unique(trainLabel_balanced)) > 2:
                    # 这种情况通常不会发生，除非 SVC 配置特殊
                    probabilities = probabilities.reshape(-1, 1)
            else:
                # 如果已经是 (N, C)，不需要 reshape
                pass
        else:
            probabilities = classifier.predict_proba(test_norm)

        # 确保 probabilities 是 2D
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        # 行归一化 (Min-Max Normalization per sample)
        # 注意：如果某行最大值等于最小值（例如所有概率相等），分母为0，需处理
        min_vals = numpy.min(probabilities, axis=1, keepdims=True)
        max_vals = numpy.max(probabilities, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        # 避免除以零
        range_vals[range_vals == 0] = 1.0
        probabilities = (probabilities - min_vals) / range_vals

        final_predictions = numpy.argmax(probabilities, axis=1)

        unique_labels = numpy.unique(numpy.concatenate((test_labels, final_predictions)))
        print(f"\n{'=' * 50}")
        print(f"{set_name} 各类别统计结果:")
        print(f"{'=' * 50}")

        for label in sorted(unique_labels):
            # 该类别的真实样本数
            total_samples = numpy.sum(test_labels == label)
            # 该类别的正确预测数
            correct_samples = numpy.sum((test_labels == label) & (final_predictions == label))
            accuracy_class = (correct_samples / total_samples * 100) if total_samples > 0 else 0

            print(
                f"类别 {label}: 总样本数 = {total_samples:4d}, 正确数 = {correct_samples:4d}, 准确率 = {accuracy_class:6.2f}%")

        # 可选：打印混淆矩阵的关键信息（每类被预测成什么）
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, final_predictions, labels=sorted(unique_labels))
        print(f"\n混淆矩阵（行=真实，列=预测）:")
        print("     ", "  ".join([f"C{c:2d}" for c in sorted(unique_labels)]))
        for i, label in enumerate(sorted(unique_labels)):
            print(f"C{label:2d}: ", "  ".join([f"{cm[i, j]:4d}" for j in range(len(sorted(unique_labels)))]))
        print(f"{'=' * 50}\n")
        # ========== 统计结束 ==========

        # 计算指标
        top1_accuracy = numpy.mean(final_predictions == test_labels)

        if probabilities.shape[1] >= 5:
            top5_indices = numpy.argsort(probabilities, axis=1)[:, -5:]
            top5_matches = numpy.any(top5_indices == test_labels[:, None], axis=1)
            top5_accuracy = numpy.mean(top5_matches)
        else:
            top5_accuracy = top1_accuracy

        # 分组准确率 (假设类别索引 0-9，分组逻辑保持不变)
        group1_indices = [0, 1, 2, 3]
        group2_indices = [4, 5, 6]
        group3_indices = [7, 8, 9]

        def calculate_group_accuracy(preds, labels, group):
            mask = numpy.isin(labels, group)
            if not numpy.any(mask):
                return 0.0
            return numpy.mean(preds[mask] == labels[mask])

        acc_g1 = calculate_group_accuracy(final_predictions, test_labels, group1_indices)
        acc_g2 = calculate_group_accuracy(final_predictions, test_labels, group2_indices)
        acc_g3 = calculate_group_accuracy(final_predictions, test_labels, group3_indices)

        macro_f1 = f1_score(test_labels, final_predictions, average='macro', zero_division=0)

        # 计算 Mean Per Class Accuracy (Macro Accuracy)
        unique_classes = numpy.unique(numpy.concatenate((trainLabel, test_labels)))
        num_classes = int(numpy.max(unique_classes)) + 1
        per_class_accs = []
        for c in range(num_classes):
            mask = (test_labels == c)
            if numpy.any(mask):
                acc = numpy.mean(final_predictions[mask] == test_labels[mask])
                per_class_accs.append(acc)
            else:
                per_class_accs.append(0.0)
        mean_per_class_acc = numpy.mean(per_class_accs) if per_class_accs else 0.0

        final_report = classification_report(test_labels, final_predictions, zero_division=0)

        # 转百分比并保留两位小数
        top1_accuracy = round(100 * top1_accuracy, 2)
        top5_accuracy = round(100 * top5_accuracy, 2)
        acc_g1 = round(100 * acc_g1, 2)
        acc_g2 = round(100 * acc_g2, 2)
        acc_g3 = round(100 * acc_g3, 2)
        macro_f1 = round(100 * macro_f1, 2)
        mean_per_class_acc = round(100 * mean_per_class_acc, 2)

        return (
            final_predictions,
            final_report,
            top1_accuracy,
            top5_accuracy,
            acc_g1,
            acc_g2,
            acc_g3,
            mean_per_class_acc,
            macro_f1
        )

    # ========== 3. 执行两次评估 ==========

    # A. 平衡测试集
    res_bal = _run_single_evaluation(test_bal, testL_bal, "Balanced Test Set")

    # B. 不平衡测试集
    res_imb = _run_single_evaluation(test_imb, testL_imb, "Imbalanced Test Set")

    # 解包结果
    (bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1) = res_bal
    (imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1) = res_imb

    # 打印简要对比
    print(f"[Result] Balanced  -> Top1: {bal_top1}%, Macro-F1: {bal_macro_f1}%")
    print(f"[Result] Imbalanced-> Top1: {imb_top1}%, Macro-F1: {imb_macro_f1}%")

    # ========== 4. 返回所有结果 (共 18 个值) ==========
    return (
        bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
        imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1
    )


if __name__ == "__main__":
    #sys.stdout = Logger(opt.log_file, sys.stdout)
    sys.stdout = Logger(opt.log_file, stream=sys.stdout, add_flag=True, encoding='utf-8')
    beginTime = time.perf_counter()
    pop, pop_1 = GPMain(opt.pkl_file,opt.randomSeeds)


    print("\n" + "=" * 50)
    print(">>> Best Feature Tree (Pop) Stats:")
    print(pop)
    print(f"    Nodes (Size): {len(pop)}")
    print(f"    Depth:        {pop.height}")

    # 可选：打印树的结构字符串
    # print(f"    Structure:    {best_ind_feat}")

    print("\n>>> Best Augment Tree (Pop_1) Stats:")
    print(pop_1)
    print(f"    Nodes (Size): {len(pop_1)}")
    print(f"    Depth:        {pop_1.height}")
    # print(f"    Structure:    {best_ind_aug}")
    print("=" * 50 + "\n")
    endTime = time.perf_counter()
    trainTime = endTime - beginTime

    results = evalTest(toolbox, pop, pop_1, x_train, y_train, x_test, y_test, x_test_imb, y_test_imb)

    testTime = time.perf_counter() - endTime

    (
        bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
        imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1
    ) = results

    # 3. 打印第一组：平衡测试集 (Balanced Test Set)
    print("=" * 50)
    print(">>> Evaluation on BALANCED Test Set")
    print("=" * 50)
    print('Final Classification Report:\n', bal_report)
    print(f'Top1 Accuracy: {bal_top1:.2f}')
    print(f'Top5 Accuracy: {bal_top5:.2f}')
    print(f'accuracy_group1: {bal_g1:.2f}')
    print(f'accuracy_group2: {bal_g2:.2f}')
    print(f'accuracy_group3: {bal_g3:.2f}')
    # 注意：原代码中的 Macro_Recall 对应的是 mean_per_class_acc (Macro Accuracy)
    print(f'Macro_Recall: {bal_macro_acc:.2f}')
    print(f'macro_f1: {bal_macro_f1:.2f}')

    # 4. 打印第二组：不平衡测试集 (Imbalanced/Long-tail Test Set)
    print("\n" + "=" * 50)
    print(">>> Evaluation on IMBALANCED Test Set (Long-tail)")
    print("=" * 50)
    print('Final Classification Report:\n', imb_report)
    print(f'Top1 Accuracy: {imb_top1:.2f}')
    print(f'Top5 Accuracy: {imb_top5:.2f}')
    print(f'accuracy_group1: {imb_g1:.2f}')
    print(f'accuracy_group2: {imb_g2:.2f}')
    print(f'accuracy_group3: {imb_g3:.2f}')
    print(f'Macro_Recall: {imb_macro_acc:.2f}')
    print(f'macro_f1: {imb_macro_f1:.2f}')
    print("Process Finish")
