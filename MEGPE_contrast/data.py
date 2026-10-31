import argparse
import random
import time
import operator
import os

from keras.src.ops import shape
from sympy import transpose

import evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import numpy as np
import select_nsga
from deap import base, creator, tools, gp

DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('..')
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10
import warnings
import multiprocessing
import hashlib
import pickle
from data.autoaugment import data_augmentation
from functools import wraps
from threading import Lock
from sklearn.metrics import accuracy_score

warnings.simplefilter("ignore")
'FLGP_MEGPE'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_file", default="log/main.log", help="log file name", type=str
)
parser.add_argument(
    "--pkl_file", help="pkl file name", default="log/latest_population.pkl"
)
parser.add_argument(
    "--dataset",
    help="Dataset name (choose from the provided options)",
    type=str,
    choices=["mnist", "cifar10", "fashion_minist", "STL"],  # 限制可选值
    default="mnist"  # 设置默认值
)
parser.add_argument(
    "--augment", help="Is data augment", default=True
)

opt = parser.parse_args()


# Logger class remains unchanged
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

train_dataset = IMBALANCESTL10(root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True)
test_dataset = IMBALANCESTL10(root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True, train=False)

