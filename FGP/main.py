# python packages
import argparse
import random
import time

from scipy.ndimage import zoom

import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
import numpy
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import fgp_functions as fe_fs
import multiprocessing
import hashlib
import pickle
from functools import wraps
from threading import Lock
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
from data.autoaugment import data_augmentation
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import sys
'FGP'
dataSetName = 'Cifar_10'



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
    choices=["mnist", "cifar10", "fashion_mnist", "STL", "MBI", "MRB", "MRD", "SVHN", "EuroSAT"],  # 限制可选值
    default="cifar10"  # 设置默认值
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
    help="关闭最终 test 阶段在不平衡测试集(x_test_imb)上的评估 (默认不关闭, 即仍会评估不平衡测试集)",
    action="store_true"
)

parser.add_argument("--population", help="GP 种群数量 (默认 100)", type=int, default=100)
parser.add_argument("--generation", help="GP 进化代数 (默认 50)", type=int, default=50)
parser.add_argument("--num_processes", help="并行评估进程数 (默认 10)", type=int, default=10)

opt = parser.parse_args()

# 共享数据集根目录: 项目目录上一级的 data 文件夹 (按 main.py 位置相对定位, 与盘符无关)
import os
DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))


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
        cifar_root = os.path.join(DATASETS_ROOT, 'CIFAR-10')
        train_dataset = IMBALANCECIFAR10("train", imbalance_ratio=100, root=cifar_root, lt_mode=opt.lt_mode)
        test_dataset_bal = IMBALANCECIFAR10("test", imbalance_ratio=100, root=cifar_root, lt_mode=opt.lt_mode)
        test_dataset_imb = IMBALANCECIFAR10("test_Imbalanced", imbalance_ratio=100, root=cifar_root, lt_mode=opt.lt_mode)
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
        # Numpy 加载路径 (统一从共享 data 文件夹加载)
        npy_root = DATASETS_ROOT
        x_train_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_data.npy') / 255.0
        y_train_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_label.npy')
        x_test_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_data.npy') / 255.0
        y_test_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_label.npy')

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

        img_num_per_cls = ([400, 286, 205, 147, 105, 75, 54, 38, 27, 20]
                   if opt.lt_mode == "exp"
                   else [200, 158, 126, 100, 79, 63, 50, 40, 32, 20])

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

        img_num_per_cls = ([400, 286, 205, 147, 105, 75, 54, 38, 27, 20]
                   if opt.lt_mode == "exp"
                   else [200, 158, 126, 100, 79, 63, 50, 40, 32, 20])

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
def make_Sigma():
    return random.randint(1, 4)


def make_Order():
    return random.randint(0, 3)


def make_Theta():
    return random.randint(0, 8)


def make_Frequency():
    return random.randint(0, 5)


def make_n():
    return round(random.random(), 3)


def make_KernelSize():
    return random.randrange(2, 5, 2)



# parameters:
population = opt.population
generation = opt.generation
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8
##GP
if Is_RGB:
    pset = gp.PrimitiveSetTyped('MAIN', [Img,Img,Img], Vector1, prefix='Image')
else:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
#feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')
##feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')
#filtering
pset.addPrimitive(fe_fs.gau, [Img1, Int1], Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab, [Img1, Float1, Float2], Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx, [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely, [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf, [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf, [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf, [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp, [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W_AddF')
pset.addPrimitive(fe_fs.mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W_SubF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')
# filtering
pset.addPrimitive(fe_fs.gau, [Img, Int1], Img, name='Gau')
pset.addPrimitive(fe_fs.gauD, [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab, [Img, Float1, Float2], Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx, [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely, [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf, [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf, [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf, [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp, [Img], Img, name='LBP_F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG_F')
pset.addPrimitive(fe_fs.mixconadd, [Img, Float3, Img, Float3], Img, name='W_Add')
pset.addPrimitive(fe_fs.mixconsub, [Img, Float3, Img, Float3], Img, name='W_Sub')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')
# Terminals
pset.renameArguments(ARG0='Image')
pset.addEphemeralConstant('Singma', make_Sigma, Int1)
pset.addEphemeralConstant('Order', make_Order, Int2)
pset.addEphemeralConstant('Theta', make_Theta, Float1)
pset.addEphemeralConstant('Frequency', make_Frequency, Float2)
pset.addEphemeralConstant('n', make_n, Float3)
pset.addEphemeralConstant('KernelSize', make_KernelSize, Int3)

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
        train_tf = []
        if Is_RGB:
            for i in range(0, len(y_train)):
                train_tf.append(numpy.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
        else:
            for i in range(0, len(y_train)):
                train_tf.append(numpy.asarray(func(x_train[i][ :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC()
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    except:
        accuracy=0
    print(accuracy)
    return accuracy,

def parallel_evalTrain(individual):
    return evalTrain(individual)


# Register the evaluation function with caching
toolbox.register("evaluate", parallel_evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

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
    hof = tools.HallOfFame(10)
    num_processes = opt.num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        toolbox.register("map", pool.map)
        pop= evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, start_gen, pkl_file,
                                 halloffame=hof, verbose=True)
        pool.close()
        pool.join()


    return pop, hof


def evalTest(toolbox, individual, x_train, trainLabel, x_test, testLabel, x_test_imb, y_test_imb, Is_RGB):
    """
    单树架构：在平衡测试集和不平衡测试集上分别评估模型。

    参数:
        x_test, testLabel: 平衡测试集数据和标签
        x_test_imb, y_test_imb: 不平衡测试集数据和标签
        Is_RGB: 是否为 RGB 图像的标志

    返回:
        18个指标 (平衡集 9 个 + 不平衡集 9 个)
    """
    print("开始 test (单种群 - 双测试集评估)")
    func = toolbox.compile(expr=individual)

    # ========== 1. 训练阶段 (只执行一次) ==========
    train_tf = []

    # --- 1.1 提取训练集特征 ---
    if Is_RGB:
        for i in range(len(trainLabel)):
            # 保留原代码中的切片方式 [0, :, :]
            feat = func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])
            train_tf.append(numpy.asarray(feat))
    else:
        for i in range(len(trainLabel)):
            feat = func(x_train[i][:, :])
            train_tf.append(numpy.asarray(feat))

    train_tf = numpy.asarray(train_tf, dtype=float)

    # --- 1.2 归一化训练集特征 ---
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)

    # --- 1.3 训练分类器 ---
    lsvm = LinearSVC()
    lsvm.fit(train_norm, trainLabel)  # 修复了原代码中 y_train 未定义的 Bug

    # ========== 2. 定义内部评估函数 ==========
    def _run_single_evaluation(test_data, test_labels, set_name="Test"):
        print(f"正在评估 {set_name} ...")
        test_tf = []

        # --- 2.1 提取测试集特征 ---
        if Is_RGB:
            for j in range(len(test_labels)):
                feat = func(test_data[j][0, :, :], test_data[j][1, :, :], test_data[j][2, :, :])
                test_tf.append(numpy.asarray(feat))
        else:
            for j in range(len(test_labels)):
                feat = func(test_data[j][:, :])
                test_tf.append(numpy.asarray(feat))

        if len(test_labels) == 0:
            return None, "No samples", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        test_tf = numpy.asarray(test_tf, dtype=float)

        # --- 2.2 归一化测试集特征 (使用训练集的 scaler) ---
        test_norm = min_max_scaler.transform(test_tf)

        # --- 2.3 预测与概率处理 ---
        probabilities = lsvm.decision_function(test_norm)

        # 修复 LinearSVC 在二分类时输出 1D 数组导致后续报错的问题
        if probabilities.ndim == 1:
            probabilities = numpy.vstack([-probabilities, probabilities]).T

        # 确保 probabilities 是 2D
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        # 行归一化 (Min-Max Normalization per sample) - 替换全局归一化，更健壮
        min_vals = numpy.min(probabilities, axis=1, keepdims=True)
        max_vals = numpy.max(probabilities, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除以零
        probabilities = (probabilities - min_vals) / range_vals

        final_predictions = numpy.argmax(probabilities, axis=1)
        test_labels_np = numpy.asarray(test_labels)  # 统一转换为 numpy 数组

        # --- 2.4 指标计算 ---
        # A. Top-1 Accuracy (修复了原代码中 y_test 未定义的 Bug)
        top1_accuracy = numpy.mean(final_predictions == test_labels_np)

        # B. Top-5 Accuracy
        if probabilities.shape[1] >= 5:
            top5_indices = numpy.argsort(probabilities, axis=1)[:, -5:]
            top5_matches = numpy.any(top5_indices == test_labels_np[:, None], axis=1)
            top5_accuracy = numpy.mean(top5_matches)
        else:
            top5_accuracy = top1_accuracy

        # C. Group Accuracy (分组准确率)
        group1_indices = [0, 1, 2]  # 前三类
        group2_indices = [3, 4, 5]  # 中间三类
        group3_indices = [6, 7, 8, 9]  # 后四类

        def calculate_group_accuracy(preds, labels, group):
            mask = numpy.isin(labels, group)
            if not numpy.any(mask):
                return 0.0
            return numpy.mean(preds[mask] == labels[mask])

        acc_g1 = calculate_group_accuracy(final_predictions, test_labels_np, group1_indices)
        acc_g2 = calculate_group_accuracy(final_predictions, test_labels_np, group2_indices)
        acc_g3 = calculate_group_accuracy(final_predictions, test_labels_np, group3_indices)

        # D. Macro F1
        macro_f1 = f1_score(test_labels_np, final_predictions, average='macro', zero_division=0)

        # E. Mean Accuracy by Class (平均类别准确率)
        unique_classes = numpy.unique(numpy.concatenate((trainLabel, test_labels_np)))
        num_classes = int(numpy.max(unique_classes)) + 1
        per_class_accs = []
        for c in range(num_classes):
            mask = (test_labels_np == c)
            if numpy.any(mask):
                acc = numpy.mean(final_predictions[mask] == test_labels_np[mask])
                per_class_accs.append(acc)
            else:
                per_class_accs.append(0.0)
        mean_per_class_acc = numpy.mean(per_class_accs) if per_class_accs else 0.0

        # 分类报告
        final_report = classification_report(test_labels_np, final_predictions, zero_division=0)

        return (
            final_predictions,
            final_report,
            round(100 * top1_accuracy, 2),
            round(100 * top5_accuracy, 2),
            round(100 * acc_g1, 2),
            round(100 * acc_g2, 2),
            round(100 * acc_g3, 2),
            round(100 * mean_per_class_acc, 2),
            round(100 * macro_f1, 2)
        )

    # ========== 3. 执行两次评估 ==========

    # A. 平衡测试集
    res_bal = _run_single_evaluation(x_test, testLabel, "Balanced Test Set")

    # B. 不平衡测试集 (可通过 --disable_imb_test 关闭)
    if opt.disable_imb_test:
        res_imb = (None, "Disabled by --disable_imb_test", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        res_imb = _run_single_evaluation(x_test_imb, y_test_imb, "Imbalanced Test Set")

    # 解包结果
    (bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1) = res_bal
    (imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1) = res_imb

    # ========== 4. 返回所有结果 (共 18 个值) ==========
    return (
        bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
        imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1
    )


if __name__ == "__main__":
    sys.stdout = Logger(opt.log_file, stream=sys.stdout, add_flag=True, encoding='utf-8')
    beginTime = time.perf_counter()
    pop, hof = GPMain(opt.pkl_file,opt.randomSeeds)


    print("\n" + "=" * 50)
    print(">>> Best Feature Tree (Pop) Stats:")
    print(hof[0])
    print(f"    Nodes (Size): {len(hof[0])}")
    print(f"    Depth:        {hof[0].height}")
    print("=" * 50 + "\n")
    endTime = time.perf_counter()
    trainTime = endTime - beginTime
    results = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test, x_test_imb, y_test_imb, Is_RGB)


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
    if not opt.disable_imb_test:
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
    else:
        print("\n[Info] Imbalanced test set evaluation disabled by --disable_imb_test")
    print("Process Finish")
