# python packages
import argparse
import random
import time

from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, classification_report, f1_score

import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
import numpy
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Img, Vector, ClassLabel, Vector_C, \
    Img1, Img2, X, Y, windowSize, width, height, s
import fgp_functions_new as fe_fs
import multiprocessing
import hashlib
import pickle
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
from data.autoaugment import data_augmentation
from functools import wraps
from threading import Lock
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
    default="SVHN"  # 设置默认值
)


parser.add_argument(
    "--randomSeeds",default=1
)

parser.add_argument(
    "--lt_mode",
    help="CIFAR-10 长尾构造模式: fixed=固定数组[200..20]; exp=按指数构造(头400/尾20/不平衡比例20)。数据均从上级 data 目录加载",
    type=str,
    choices=["fixed", "exp"],
    default="fixed"
)

parser.add_argument(
    "--disable_imb_test",
    help="关闭最终 test 阶段在不平衡测试集(x_test_imb)上的评估 (默认不关闭, 即仍会评估不平衡测试集)。本项目无不平衡测试集评估, 参数仅为接口一致",
    action="store_true"
)

parser.add_argument("--population", help="GP 种群数量 (默认 100)", type=int, default=100)
parser.add_argument("--generation", help="GP 进化代数 (默认 50)", type=int, default=50)
parser.add_argument("--num_processes", help="并行评估进程数 (默认 10)", type=int, default=10)

opt = parser.parse_args()

# 共享数据集根目录: 项目目录上一级的 data 文件夹 (按 main.py 位置相对定位, 与盘符无关)
import os
DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))


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

def create_dataset(dataset):
    if dataset=="cifar10":
        cifar_root = os.path.join(DATASETS_ROOT, 'CIFAR-10')
        train_dataset = IMBALANCECIFAR10("train", imbalance_ratio=100, root=cifar_root, lt_mode=opt.lt_mode)
        test_dataset = IMBALANCECIFAR10("test", imbalance_ratio=100, root=cifar_root, lt_mode=opt.lt_mode)
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
    elif dataset == "SVHN" or dataset == "EuroSAT":
        npy_root = DATASETS_ROOT
        x_train = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_data.npy') / 255.0
        y_train = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_label.npy')
        x_test = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_data.npy') / 255.0
        y_test = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_label.npy')
        Is_RGB=True
        Flag=4
    else:
        if dataset=="mnist":
            train_dataset = IMBALANCEMNIST("train", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            test_dataset = IMBALANCEMNIST("test", imbalance_ratio=100, root=os.path.join(DATASETS_ROOT, 'MNIST'))
            Flag =1
        else:
            npy_root = DATASETS_ROOT
            x_train = numpy.load(npy_root + '/'+dataset+'/'+dataset+'_train_data.npy') / 255.0
            y_train = numpy.load(npy_root + '/'+dataset+'/'+dataset+'_train_label.npy')
            x_test = numpy.load(npy_root + '/'+dataset+'/'+dataset+'_test_data.npy') / 255.0
            y_test = numpy.load(npy_root + '/'+dataset+'/'+dataset+'_test_label.npy')
            Flag = 3
        Is_RGB = False
    if Flag==1:
        x_train = train_dataset.data
        y_train = numpy.array(train_dataset.labels)
        x_test = test_dataset.data
        if Is_RGB==False:
            x_test = x_test.numpy()
        y_test = numpy.array(test_dataset.labels)
        x_train = x_train.astype(numpy.float32)
        x_test = x_test.astype(numpy.float32)
        dim=x_train.ndim
        if dim==4:
            shape=x_train.shape
            if shape[1] !=3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test = x_test.transpose(dims)
    elif Flag==3:
        x_train = x_train.astype(numpy.float32)
        x_test = x_test.astype(numpy.float32)
        y_train = y_train.astype(numpy.int64)
        y_test = y_test.astype(numpy.int64)
        img_num_per_cls = ([400, 286, 205, 147, 105, 75, 54, 38, 27, 20]
                   if opt.lt_mode == "exp"
                   else [200, 158, 126, 100, 79, 63, 50, 40, 32, 20])
        x_train,y_train,x_test,y_test=imbalance_MRBand(x_train,y_train,x_test,y_test,img_num_per_cls)
    elif Flag == 4:
        x_train = x_train.astype(numpy.float32)
        x_test = x_test.astype(numpy.float32)
        y_train = y_train.astype(numpy.int64)
        y_test = y_test.astype(numpy.int64)
        img_num_per_cls = ([400, 286, 205, 147, 105, 75, 54, 38, 27, 20]
                   if opt.lt_mode == "exp"
                   else [200, 158, 126, 100, 79, 63, 50, 40, 32, 20])
        x_train,y_train,x_test,y_test=imbalance_MRBand(x_train,y_train,x_test,y_test,img_num_per_cls)
        dim = x_train.ndim
        if dim == 4:
            shape = x_train.shape
            if shape[1] != 3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test = x_test.transpose(dims)
                print("jiaohuan")
    else:
        x_train = train_dataset.data
        y_train = numpy.array(train_dataset.labels)
        x_test = test_dataset.data
        y_test = numpy.array(test_dataset.labels)
        x_train = x_train.astype(numpy.float32)
        x_test = x_test.astype(numpy.float32)
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


group2_indices = [4, 5, 6]  # 中间三类
group3_indices = [7, 8, 9]  # 尾部三类

x_train,y_train,x_test,y_test,Is_RGB = create_dataset(opt.dataset)



print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

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
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# parameters:
population = opt.population
generation = opt.generation
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 4
initialMaxDepth = 10
maxDepth = 10
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
    pset = gp.PrimitiveSetTyped('MAIN', [Img,Img,Img], ClassLabel, prefix='Image')
else:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], ClassLabel, prefix='Image')

# image classification layer
pset.addPrimitive(fe_fs.classifier_selection, [Vector_C, Int3], ClassLabel, name='classifier')

# high level features--feature construction layer
pset.addPrimitive(fe_fs.feature_con2, [Vector, Vector_C], Vector_C, name='FeaturesC_dif')
pset.addPrimitive(fe_fs.feature_con, [Vector, Vector], Vector_C, name='Features2C')
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

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)


def evalTrain(individual):
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
            recall = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5, scoring='recall_macro').mean(), 2)
        elif classifier_idx == 2:
            rf = RandomForestClassifier()
            recall = round(100 * cross_val_score(rf, train_norm, y_train, cv=5, scoring='recall_macro').mean(), 2)
        elif classifier_idx == 3:
            lr = LogisticRegression()
            recall = round(100 * cross_val_score(lr, train_norm, y_train, cv=5, scoring='recall_macro').mean(), 2)
        else:
            erf = ExtraTreesClassifier()
            recall = round(100 * cross_val_score(erf, train_norm, y_train, cv=5, scoring='recall_macro').mean(), 2)
    except:
        recall=0
    print(recall)
    return recall,

def parallel_evalTrain(individual):
    return evalTrain(individual)


# Register the evaluation function with caching
toolbox.register("evaluate", parallel_evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(pkl_file,randomSeeds):
    random.seed(randomSeeds)
    try:
        with open("qwq.pkl", "rb") as cp_file:
            checkpoint_data = pickle.load(cp_file)
            print("已读取")
            pop = checkpoint_data['population']
            start_gen = checkpoint_data['generation'] + 1
            print(f"Resuming from generation {start_gen}.")
    except FileNotFoundError:
        # 如果没有找到检查点，则初始化新种群
        pop = toolbox.population(population)
        start_gen = 1
    # print('the size of population', len(pop))
    hof = tools.HallOfFame(10)
    num_processes = opt.num_processes
    # pop, log, best_ind_gen = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
    #                                          stats=mstats, halloffame=hof, verbose=True)
    with multiprocessing.Pool(processes=num_processes) as pool:
        toolbox.register("map", pool.map)
        pop,  best_ind_gen = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,start_gen,pkl_file,
                                                  halloffame=hof, verbose=True)
        pool.close()
        pool.join()
    return pop, hof


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    classifier_idx=individual[len(individual)-1].value
    train_tf = []
    test_tf = []
    if Is_RGB:
        for i in range(0, len(trainLabel)):
            train_tf.append(numpy.asarray(func(trainData[i][0, :, :], trainData[i][1, :, :], trainData[i][2, :, :])[0]))
        for j in range(0, len(testL)):
            test_tf.append(numpy.asarray(func(test[j][0, :, :], test[j][1, :, :], test[j][2, :, :])[0]))
    else:
        for i in range(0, len(trainLabel)):
            train_tf.append(numpy.asarray(func(trainData[i][ :, :])[0]))
        for j in range(0, len(testL)):
            test_tf.append(numpy.asarray(func(test[j][ :, :])[0]))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    print('test', train_tf[0][1])
    if classifier_idx == 1:
        classifier = LinearSVC()
        classifier.fit(train_norm, trainLabel)
        probabilities= classifier.decision_function(test_norm)
    elif classifier_idx == 2:
        classifier = RandomForestClassifier()
        classifier.fit(train_norm, trainLabel)
        probabilities=classifier.predict_proba(test_norm)
    elif classifier_idx == 3:
        classifier = LogisticRegression()
        classifier.fit(train_norm, trainLabel)
        probabilities=classifier.predict_proba(test_norm)# 获取预测标签
    else:
        erf = ExtraTreesClassifier()
        erf.fit(train_norm, trainLabel)
        probabilities = erf.predict_proba(test_norm)  # 获取预测标签
    probabilities = (probabilities - numpy.min(probabilities)) / (numpy.max(probabilities) - numpy.min(probabilities))
    group1_indices = [0, 1, 2,3]  # 前四类
    group2_indices = [4, 5,6]  # 中间三类
    group3_indices = [7, 8, 9]  # 后三类
    def calculate_group_accuracy(predictions, true_labels, group_indices):
        # 找出属于该组的所有样本的索引
        mask = numpy.isin(true_labels, group_indices)
        # 过滤出属于该组的真实标签和预测标签
        filtered_true_labels = true_labels[mask]
        filtered_predictions = predictions[mask]
        accuracy = accuracy_score(filtered_true_labels, filtered_predictions)
        return accuracy

    final_predictions = numpy.argmax(probabilities, axis=1)
    top1_accuracy = numpy.mean(final_predictions == y_test)
    top5_indices = numpy.argsort(probabilities, axis=1)[:, -5:]
    top5_matches = numpy.any(top5_indices == y_test[:, None], axis=1)
    top5_accuracy = numpy.mean(top5_matches)
    final_report = classification_report(y_test, final_predictions)
    accuracy_group1 = calculate_group_accuracy(final_predictions, y_test, group1_indices)
    accuracy_group2 = calculate_group_accuracy(final_predictions, y_test, group2_indices)
    accuracy_group3 = calculate_group_accuracy(final_predictions, y_test, group3_indices)
    # 新增：宏 F1（macro f1）
    macro_f1 = f1_score(testL, final_predictions, average='macro')

    num_classes = int(numpy.max(numpy.concatenate((trainLabel, testL))) + 1)  # 或你可显式设置类别数
    per_class_accs = []
    for c in range(num_classes):
        mask = (testL == c)
        if numpy.sum(mask) == 0:
            acc_c = 0.0
        else:
            acc_c = numpy.mean(final_predictions[mask] == testL[mask])
        per_class_accs.append(acc_c)
    mean_accuracy_by_class = round(100 * numpy.mean(per_class_accs), 2)
    top1_accuracy = round(top1_accuracy * 100, 2)
    top5_accuracy = round(top5_accuracy * 100, 2)
    accuracy_group1 = round(accuracy_group1 * 100, 2)
    accuracy_group2 = round(accuracy_group2 * 100, 2)
    accuracy_group3 = round(accuracy_group3 * 100, 2)
    macro_f1 = round(macro_f1 * 100, 2)
    return final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3, mean_accuracy_by_class, macro_f1




if __name__ == "__main__":
    sys.stdout = Logger(opt.log_file, sys.stdout)
    beginTime = time.perf_counter()
    pop, hof = GPMain(opt.pkl_file,opt.randomSeeds)
    endTime = time.perf_counter()
    trainTime = endTime - beginTime

    final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3, Macro_Recall, macro_f1 = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)
    testTime = time.perf_counter() - endTime
    #saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, log)

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
