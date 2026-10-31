#python packages
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
warnings.filterwarnings('ignore')
'COGP'

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
    choices=["mnist", "cifar10", "fashion_mnist", "STL", "MRB", "MRD", "MBI"],  # 限制可选值
    default="cifar10"  # 设置默认值
)

parser.add_argument(
    "--randomSeeds",default=1
)

opt = parser.parse_args()

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
        x_train_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_data.npy')) / 255.0
        y_train_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_label.npy'))
        x_test_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_data.npy')) / 255.0
        y_test_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_label.npy'))

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
            x_train_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_data.npy')) / 255.0
            y_train_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_train_label.npy'))
            x_test_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_data.npy')) / 255.0
            y_test_raw = np.load(os.path.join(DATASETS_ROOT, dataset, dataset + '_test_label.npy'))
            Flag = 3
        Is_RGB = False

    # --- 数据处理逻辑 ---

    if Flag == 1:
        # CIFAR10, FMNIST, MNIST (Torch Dataset 路径)
        # 提取训练集
        x_train = train_dataset.data
        y_train = np.array(train_dataset.labels)

        # 提取平衡测试集
        x_test_bal = test_dataset_bal.data
        y_test_bal = np.array(test_dataset_bal.labels)

        # 提取不平衡测试集
        x_test_imb = test_dataset_imb.data
        y_test_imb = np.array(test_dataset_imb.labels)

        # 类型转换与形状调整
        if not Is_RGB:
            if hasattr(x_test_bal, 'numpy'): x_test_bal = x_test_bal.numpy()
            if hasattr(x_test_imb, 'numpy'): x_test_imb = x_test_imb.numpy()

        x_train = x_train.astype(np.float32)
        x_test_bal = x_test_bal.astype(np.float32)
        x_test_imb = x_test_imb.astype(np.float32)

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
        x_train_raw = x_train_raw.astype(np.float32)
        x_test_raw = x_test_raw.astype(np.float32)
        y_train_raw = y_train_raw.astype(np.int64)
        y_test_raw = y_test_raw.astype(np.int64)

        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        # 调用更新后的函数，返回 6 个值
        x_train, y_train, x_test_bal, y_test_bal, x_test_imb, y_test_imb = imbalance_MRBand(
            x_train_raw, y_train_raw, x_test_raw, y_test_raw,
            img_num_per_cls,
            imb_test_factor=0.1  # 设置不平衡因子，例如 0.1 (10:1)
        )

    elif Flag == 4:
        # SVHN / EuroSAT
        x_train_raw = x_train_raw.astype(np.float32)
        x_test_raw = x_test_raw.astype(np.float32)
        y_train_raw = y_train_raw.astype(np.int64)
        y_test_raw = y_test_raw.astype(np.int64)

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
        y_train = np.array(train_dataset.labels)

        x_test_bal = test_dataset_bal.data
        y_test_bal = np.array(test_dataset_bal.labels)

        x_test_imb = test_dataset_imb.data
        y_test_imb = np.array(test_dataset_imb.labels)

        x_train = x_train.astype(np.float32)
        x_test_bal = x_test_bal.astype(np.float32)
        x_test_imb = x_test_imb.astype(np.float32)

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

#COGP parameters:
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
##COGP tree structure, function set and terminal set
if Is_RGB:
    pset = gp_tree.PrimitiveSetTyped('MAIN',[Img,Img,Img], Vector, prefix='Image')
else:
    pset = gp_tree.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
#Functions at Concatenation layer
pset.addPrimitive(fs.root_conVector2,[Vector,Vector],Vector,name='Root1')
pset.addPrimitive(fs.root_conVector2,[Img2, Img2],Vector,name='Root2')
pset.addPrimitive(fs.root_conVector3,[Img2, Img2, Img2],Vector,name='Root3')
pset.addPrimitive(fs.root_conVector4,[Img2, Img2, Img2, Img2],Vector,name='Root4')
#Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
pset.addPrimitive(fs.ZeromaxP,[Img2, Int, Int],Img2,name='ZMaxPF')
#Filtering at a flexible layer. Use *F as the names of the functions to avoid the same names, which is not allowed in DEAP
pset.addPrimitive(fs.mixconadd, [Img2, Double, Img2, Double], Img2, name='AddF')
pset.addPrimitive(fs.mixconsub, [Img2, Double, Img2, Double], Img2, name='SubF')
pset.addPrimitive(np.abs, [Img2], Img2, name='AbsF')
pset.addPrimitive(fs.sqrt, [Img2], Img2, name='SqrtF')
pset.addPrimitive(fs.relu, [Img2], Img2, name='ReluF')
pset.addPrimitive(fs.conv_filters, [Img2, Filter], Img2, name='ConvF')  #convolution operator
#Pooling functions at the Pooling layer.
pset.addPrimitive(fs.maxP,[Img2, Int, Int], Img2,name='MaxPF') # max-pooling operator
pset.addPrimitive(fs.maxP,[Img, Int, Int], Img2,name='MaxP') #max-pooling operator
#Filteing functions at the Filtering layer
pset.addPrimitive(fs.mixconadd, [Img, Double, Img, Double], Img, name='Add')
pset.addPrimitive(fs.mixconsub, [Img, Double, Img, Double], Img, name='Sub')
pset.addPrimitive(np.abs, [Img], Img, name='Abs')
pset.addPrimitive(fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fs.relu, [Img], Img, name='Relu')
pset.addPrimitive(fs.conv_filters, [Img, Filter], Img, name='Conv') #convolution operator
#Terminals
pset.renameArguments(ARG0='grey')  #the input image
pset.addEphemeralConstant('randomD',make_randomD,Double) # parameters for the Add, Sub, AddF and SubF functions
pset.addEphemeralConstant('filters3',make_filters3, Filter)  #3 * 3 filters
pset.addEphemeralConstant('filters5',make_filters5, Filter) #5 * 5 filters
pset.addEphemeralConstant('filters7',make_filters7, Filter) #7 * 7 filters
pset.addEphemeralConstant('kernelSize',make_KernelSize, Int) # kernel size for the max-pooling functions

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
#toolbox.register("mapp", futures.map)


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


# Register the evaluation function with caching
toolbox.register("evaluate", parallel_evalTrain)
toolbox.register("select", tools.selTournament,tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')
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
    hof = tools.HallOfFame(10)
    num_processes = 10
    # pop = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation, start_gen, pkl_file, halloffame=hof,
    #                       verbose=True)
    with multiprocessing.Pool(processes=num_processes) as pool:
        toolbox.register("map", pool.map)
        pop = evalGP.eaSimple( pop, toolbox, cxProb, mutProb, elitismProb, generation,start_gen, pkl_file, halloffame=hof, verbose=True)
        pool.close()
        pool.join()
    return pop,hof

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
