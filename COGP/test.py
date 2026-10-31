
import argparse
import random
import sys
import time
import operator
import os
#for using multiple CPU cores in fitness evaluation
#from scoop import futures
import multiprocessing
import hashlib
import pickle
from functools import wraps
from threading import Lock
import numpy as np
from scipy.ndimage import zoom

# deap package
import evalGP
import gp_restrict
import gp_tree
from sympy import transpose
from deap import base, creator, tools, gp

DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10
# fitness function and function for testing
from FEVal_norm_fast import evalTest, feature_length
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#COGP data types
from STGPdataType import Img, Img2, Vector, Int, Double, Filter # defined by author
import functionSet as fs
from data.ImbalanceCIFAR import imbalance_MRBand
import warnings
from data.autoaugment import data_augmentation

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

def create_dataset(dataset):
    if dataset=="cifar10":
        train_dataset = IMBALANCECIFAR10("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'CIFAR-10'))
        test_dataset = IMBALANCECIFAR10("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'CIFAR-10'))
        Is_RGB=True
        Flag=1
    elif dataset=="STL":
        train_dataset = IMBALANCESTL10(root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True)
        test_dataset = IMBALANCESTL10(root=os.path.join(DATASETS_ROOT, 'STL-10'), download=True, train=False)
        Is_RGB = True
        Flag =2
    elif dataset=="fashion_mnist":
        train_dataset = IMBALANCEFMNIST("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'Fashion_minist'))
        test_dataset = IMBALANCEFMNIST("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'Fashion_minist'))
        Is_RGB = False
        Flag =1
    else:
        if dataset=="mnist":
            train_dataset = IMBALANCEMNIST("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            test_dataset = IMBALANCEMNIST("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            Flag =1
        else:
            x_train = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_train_data.npy')) /255.0
            y_train = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_train_label.npy'))
            x_test = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_test_data.npy')) /255.0
            y_test = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_test_label.npy'))
            Flag = 3
        Is_RGB = False
    if Flag==1:
        x_train = train_dataset.data
        y_train = np.array(train_dataset.labels)
        x_test = test_dataset.data
        if Is_RGB==False:
            x_test = x_test.numpy()
        y_test = np.array(test_dataset.labels)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        dim=x_train.ndim
        if dim==4:
            shape=x_train.shape
            if shape[1] !=3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test = x_test.transpose(dims)
    elif Flag==3:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        img_num_per_cls=[200, 155, 120, 93, 72, 56, 43, 33, 26, 20]
        x_train,y_train,x_test,y_test=imbalance_MRBand(x_train,y_train,x_test,y_test,img_num_per_cls)
    else:
        x_train = train_dataset.data
        y_train = np.array(train_dataset.labels)
        x_test = test_dataset.data
        y_test = np.array(test_dataset.labels)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        dim = x_train.ndim
        if dim == 4:
            shape = x_train.shape
            if shape[1] != 3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test = x_test.transpose(dims)
        zoom_factors = (1, 1, 0.5, 0.5)
        x_train = zoom(x_train, zoom_factors, order=1)
        x_test = zoom(x_test, zoom_factors, order=1)
    return x_train,y_train,x_test,y_test,Is_RGB


group2_indices = [3, 4, 5]  # 中间三类
group3_indices = [6, 7, 8, 9]  # 尾部三类

dataSetName = 'Cifar_10'
population=100
generation=50
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=6
maxDepth=8


def make_randomD():
    return round(random.random(),3)

def make_filters3():
    return list(fs.random_filters(3))

def make_filters5():
    return list(fs.random_filters(5))

def make_filters7():
    return list(fs.random_filters(7))

def make_KernelSize():
    return random.randrange(2,5,2)


def create_ind(Is_RGB):
    if Is_RGB:
        pset = gp_tree.PrimitiveSetTyped('MAIN', [Img, Img, Img], Vector, prefix='Image')
    else:
        pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
    # Functions at Concatenation layer
    pset.addPrimitive(fs.root_conVector2, [Vector, Vector], Vector, name='Root1')
    pset.addPrimitive(fs.root_conVector2, [Img2, Img2], Vector, name='Root2')
    pset.addPrimitive(fs.root_conVector3, [Img2, Img2, Img2], Vector, name='Root3')
    pset.addPrimitive(fs.root_conVector4, [Img2, Img2, Img2, Img2], Vector, name='Root4')
    # Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
    pset.addPrimitive(fs.ZeromaxP, [Img2, Int, Int], Img2, name='ZMaxPF')
    # Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
    pset.addPrimitive(fs.mixconadd, [Img2, Double, Img2, Double], Img2, name='AddF')
    pset.addPrimitive(fs.mixconsub, [Img2, Double, Img2, Double], Img2, name='SubF')
    pset.addPrimitive(np.abs, [Img2], Img2, name='AbsF')
    pset.addPrimitive(fs.sqrt, [Img2], Img2, name='SqrtF')
    pset.addPrimitive(fs.relu, [Img2], Img2, name='ReluF')
    pset.addPrimitive(fs.conv_filters, [Img2, Filter], Img2, name='ConvF')  # convolution operator
    # Pooling functions at the Pooling layer.
    pset.addPrimitive(fs.maxP, [Img2, Int, Int], Img2, name='MaxPF')  # max-pooling operator
    pset.addPrimitive(fs.maxP, [Img, Int, Int], Img2, name='MaxP')  # max-pooling operator
    # Filteing functions at the Filtering layer
    pset.addPrimitive(fs.mixconadd, [Img, Double, Img, Double], Img, name='Add')
    pset.addPrimitive(fs.mixconsub, [Img, Double, Img, Double], Img, name='Sub')
    pset.addPrimitive(np.abs, [Img], Img, name='Abs')
    pset.addPrimitive(fs.sqrt, [Img], Img, name='Sqrt')
    pset.addPrimitive(fs.relu, [Img], Img, name='Relu')
    pset.addPrimitive(fs.conv_filters, [Img, Filter], Img, name='Conv')  # convolution operator
    # Terminals
    pset.renameArguments(ARG0='grey')  # the input image
    pset.addEphemeralConstant('randomD', make_randomD, Double)  # parameters for the Add, Sub, AddF and SubF functions
    pset.addEphemeralConstant('filters3', make_filters3, Filter)  # 3 * 3 filters
    pset.addEphemeralConstant('filters5', make_filters5, Filter)  # 5 * 5 filters
    pset.addEphemeralConstant('filters7', make_filters7, Filter)  # 7 * 7 filters
    pset.addEphemeralConstant('kernelSize', make_KernelSize, Int)  # kernel size for the max-pooling functions

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # toolbox.register("mapp", futures.map)


# Register the evaluation function with caching




def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        if Is_RGB:
            for i in range(0, len(y_train)):
                train_tf.append(np.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
            #train_tf.append(np.asarray(func(x_train[i, :, :])))
        else:
            for i in range(len(y_train)):
                train_tf.append(np.asarray(func(x_train[i][:, :])))
        #train_tf = np.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm= LinearSVC()
        accuracy = round(100*cross_val_score(lsvm, train_norm, y_train, cv=5).mean(),2)
    except:
        accuracy=0
    print(accuracy)
    return accuracy,

def parallel_evalTrain(individual):
    return evalTrain(individual)





toolbox.register("evaluate", parallel_evalTrain)
toolbox.register("select", tools.selTournament,tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))









if __name__ == '__main__':
    pkl_root = r"D:\Code\longtail_multi_fewshot_GP\MEGPE\消融实验单目标\MEGPE_contrast\pkl"
    log_root = r"D:\Code\longtail_multi_fewshot_GP\MEGPE\消融实验单目标\MEGPE_contrast\log"
    os.makedirs(log_root, exist_ok=True)

    i = 0  # 用于命名日志文件（可选）

    for method in os.listdir(pkl_root):
        method_pkl_path = os.path.join(pkl_root, method)

        if not os.path.isdir(method_pkl_path):
            continue  # 跳过文件

        # 在 log 目录下创建同名 method 文件夹
        method_log_path = os.path.join(log_root, method)
        os.makedirs(method_log_path, exist_ok=True)
        print(f"Processing method: {method}")

        for dataset in os.listdir(method_pkl_path):
            dataset_pkl_path = os.path.join(method_pkl_path, dataset)

            if not os.path.isdir(dataset_pkl_path):
                continue

            # 在 log/method/ 下创建同名 dataset 文件夹
            dataset_log_path = os.path.join(method_log_path, dataset)
            os.makedirs(dataset_log_path, exist_ok=True)
            print(f"  Processing dataset: {dataset}")

            # 遍历所有 .pkl 文件
            for pkl_file in os.listdir(dataset_pkl_path):
                if not pkl_file.endswith('.pkl'):
                    continue

                pkl_path = os.path.join(dataset_pkl_path, pkl_file)
                log_file_name = f"{pkl_file.replace('.pkl', '')}_{i}.log"
                log_file_path = os.path.join(dataset_log_path, log_file_name)

                print(f"    Processing: {pkl_file} -> {log_file_name}")

                try:
                    with open(pkl_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                        pop = checkpoint_data['population']
                        hof = tools.HallOfFame(1)
                        hof.update(pop)
                        x_train, y_train, x_test, y_test, Is_RGB = create_dataset(method)
                        create_ind(Is_RGB)
                        # 重定向输出到日志文件
                        sys.stdout = Logger(log_file_path, sys.stdout)
                        # 执行评估
                        beginTime = time.perf_counter()
                        pop, hof = GPMain(opt.pkl_file, opt.randomSeeds)
                        endTime = time.perf_counter()
                        trainTime = endTime - beginTime
                        final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3, Macro_Recall, macro_f1 = evalTest(
                            hof[0])
                        endTime1 = time.perf_counter()
                        testTime = endTime1 - endTime
                        # saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log, f1)
                        print('Final Classification Report:\n', final_report)
                        print(f'Top1 Accuracy: {top1_accuracy:.2f}')
                        print(f'Top5 Accuracy: {top5_accuracy:.2f}')
                        print(f'accuracy_group1: {accuracy_group1:.2f}')
                        print(f'accuracy_group2: {accuracy_group2:.2f}')
                        print(f'accuracy_group3: {accuracy_group3:.2f}')
                        print(f'Macro_Recall: {Macro_Recall:.2f}')
                        print(f'macro_f1: {macro_f1:.2f}')
                        print('trainTime:', trainTime)
                        print('testTime:', testTime)
                        print('End')


                        i += 1  # 日志编号递增

                        # 恢复标准输出（重要！）
                        sys.stdout = sys.stdout.terminal

                except Exception as e:
                    # 如果出错，也要确保恢复 stdout
                    sys.stdout = sys.__stdout__
                    error_log = os.path.join(dataset_log_path, f"ERROR_{pkl_file}.log")
                    with open(error_log, 'w') as ef:
                        ef.write(f"Error processing {pkl_file}:\n{str(e)}")
                    print(f"    Error processing {pkl_file}: {e}")

    # 最后确保 stdout 恢复
    sys.stdout = sys.__stdout__
    print("All pkl files processed.")