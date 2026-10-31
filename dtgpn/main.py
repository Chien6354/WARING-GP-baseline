# defined by author
import argparse
import logging
import multiprocessing
import os
import pickle
import random
import sys
import time

# 在加载 numpy/BLAS 前限制线程，避免多进程×OpenMP 过订阅
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
import numpy as np
# deap package
from deap import base, creator, tools, gp
from scipy.ndimage import zoom
# fitness function
# from scoop import futures
from sklearn import preprocessing  ##
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC

import algo_mtflgp_old as evalGP
import feature_function_mt as fe_fs
import flgp_pset
# only for strongly typed GP
import gp_restrict
##plot tree and save
import saveFile as saveFile
##image Data
from strongGPDataType import indexType1, indexType2, windowSize, windowSize2
from eval_moflgp import evalTrain_mtflgp_new

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
    default="mnist"  # 设置默认值
)


parser.add_argument(
    "--randomSeeds",default= 100
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
    import numpy
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
            # 其他自定义 Numpy 数据集 (MRD 从共享 data 文件夹加载)
            npy_root = DATASETS_ROOT
            x_train_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_data.npy') / 255.0
            y_train_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_train_label.npy')
            x_test_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_data.npy') / 255.0
            y_test_raw = numpy.load(npy_root + '/' + dataset + '/' + dataset + '_test_label.npy')
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

x_train1, y_train1, x_test1, y_test1, x_test_imb, y_test_imb,Is_RGB = create_dataset(opt.dataset)


print(f"Training Set Shape: {x_train1.shape}")
print(f"Balanced Test Set Shape: {x_test1.shape}")       # 原始均衡测试集 (例如 10000 张)
print(f"Imbalanced Test Set Shape: {x_test_imb.shape}")     # 长尾测试集 (例如 ~5000 张)
print(f"Imbalanced Test Set Shape: {y_test_imb.shape}")


def make_index1():
    return random.randint(0, bound1 - 20)

def make_index12():
    return random.randint(0, bound2 - 20)

def make_windowSize1():
    return random.randint(20, 30)

def make_windowSize2():
    return random.randint(30, 51)

n_instances = 2
population = opt.population
generation = opt.generation
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8
if Is_RGB:
    bound1, bound2 = x_train1[1, 1,:, :].shape
else:
    bound1, bound2 = x_train1[1, :, :].shape

# print(bound1, bound2)
##GP##GP
pset = flgp_pset.all_pset(Is_RGB)
pset.renameArguments(ARG0='grey')
# pset.addEphemeralConstant('randomDouble',lambda:round(random.random(),2),float)
# pset.addEphemeralConstant('index1',lambda: random.randint(0,bound-40),indexType)


pset.addEphemeralConstant('index1', make_index1, indexType1)
pset.addEphemeralConstant('index12', make_index12, indexType2)
# pset.addEphemeralConstant('index3',lambda: random.randint(0,bound-20),indexType)
pset.addEphemeralConstant('windowSize1', make_windowSize1, windowSize)
pset.addEphemeralConstant('windowSize2', make_windowSize2, windowSize2)
# pset.addEphemeralConstant('windowSize3',lambda:random.randint(50,70),windowSize3)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
#toolbox.register("mapp", futures.map)
toolbox.register("mapp", map)

# global population
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

EVAL_TIMEOUT_SEC = 60.0


def map_evaluate_with_timeout(pool, timeout_sec=EVAL_TIMEOUT_SEC):
    """
    先批量 apply_async 再 get：进程池能同时跑满多个 worker；若先发再 get 的串行写法
    则任意时刻只会有一个任务在跑。超时仍用 get(timeout) + (0,0)。
    """
    def _map_func(func, iterable):
        items = list(iterable)
        asyncs = [pool.apply_async(func, (item,)) for item in items]
        out = []
        for r in asyncs:
            try:
                out.append(r.get(timeout=timeout_sec))
            except multiprocessing.TimeoutError:
                out.append((0.0, 0.0))
        return out
    return _map_func


def GPMain(pkl_file, randomSeeds):
    random.seed(randomSeeds)
    try:
        with open("qwq", "rb") as f:
            data = pickle.load(f)
        saved_pop = data['population']

        local_pop1 = [ind[0] for ind in saved_pop]  # 提取每个个体的第一部分
        global_pop = [ind[1] for ind in saved_pop]  # 提取每个个体的第二部分
        start_gen = data.get('generation', 0)

    except FileNotFoundError:
        global_pop = toolbox.population(population)
        local_pop1 = toolbox.population(population)
        start_gen = 1

    num_processes = opt.num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        toolbox.register("map", map_evaluate_with_timeout(pool, EVAL_TIMEOUT_SEC))
        hof =  evalGP.evalSimple(global_pop, local_pop1, toolbox, cxProb, mutProb, elitismProb, generation, start_gen, pkl_file)
        pool.close()
        pool.join()

    return hof


def get_decision_dist(decision_values, y_true, train_decision_values, y_train_fold):
    """
    实现论文公式 (3): 计算 Decision Space 中的距离度量 Dist
    """
    classes = np.unique(y_train_fold)
    centroids = {}

    # 1. 计算训练集在决策空间的类中心 (Equation 6)
    for c in classes:
        # 获取属于该类的样本在决策空间的值
        c_samples = train_decision_values[y_train_fold == c]
        if len(c_samples) > 0:
            centroids[c] = np.mean(c_samples, axis=0)
        else:
            centroids[c] = np.zeros(decision_values.shape[1])

    total_dist_ratio = 0.0
    epsilon = 1e-8  # 防止除以0

    # 2. 遍历测试集样本计算距离
    for i in range(len(decision_values)):
        z_i = decision_values[i]  # 当前测试样本的决策空间坐标
        label_i = y_true[i]  # 当前测试样本的真实标签

        # S_i: 到同类中心的距离 (分子)
        if label_i in centroids:
            s_i = np.sum((z_i - centroids[label_i]) ** 2)
        else:
            s_i = 0  # 理论上不应发生，除非训练集缺类

        # D_i: 到所有异类中心的距离之和 (分母)
        d_i = 0.0
        for c in classes:
            if c != label_i:
                d_i += np.sum((z_i - centroids[c]) ** 2)

        # 累加比率 (Equation 3 的求和部分)
        total_dist_ratio += s_i / (d_i + epsilon)

    return total_dist_ratio


def _eval_timed_out(t0: float) -> bool:
    return (time.perf_counter() - t0) > EVAL_TIMEOUT_SEC


def evaluate_dtgpn(individual):
    """
    根据论文 Algorithm 1 实现的评估函数
    """
    t0 = time.perf_counter()
    # 1. 编译双树 (Dual-Tree)
    # individual[0] 是左树, individual[1] 是右树
    func1 = toolbox.compile(expr=individual[0])
    func2 = toolbox.compile(expr=individual[1])

    features_list = []

    # 2. 特征提取与拼接 (Feature Extraction & Concatenation)
    # 遍历所有训练样本
    for i in range(len(y_train1)):
        if _eval_timed_out(t0):
            return 0.0, 0.0
        if Is_RGB:
            # 假设输入是 (3, H, W) 或类似结构，根据你的 evalTrain 参考代码调整
            img_input = (x_train1[i][0, :, :], x_train1[i][1, :, :], x_train1[i][2, :, :])
            f1 = np.asarray(func1(*img_input)).flatten()
            f2 = np.asarray(func2(*img_input)).flatten()
        else:
            # 灰度图
            img_input = x_train1[i][:, :]
            f1 = np.asarray(func1(img_input)).flatten()
            f2 = np.asarray(func2(img_input)).flatten()

        # 拼接两个树的特征
        features_list.append(np.concatenate([f1, f2]))

    features_array = np.asarray(features_list)

    # 处理特征提取失败或为空的情况
    if features_array.size == 0 or np.isnan(features_array).any():
        return 0.0, 0.0

    # 3. 归一化 (Normalization)
    min_max_scaler = MinMaxScaler()
    try:
        X_norm = min_max_scaler.fit_transform(features_array)
    except ValueError:
        return 0.0, 0.0

    # 4. K-Fold Cross Validation (K = number of shots)
    # 论文中 K 等于每类的样本数。例如 3-shot 任务，K=3。
    # StratifiedKFold 会自动根据 y_train 的分布进行分层抽样
    # 假设数据是平衡的，n_splits 设为每类的样本数即可
    n_shots = int(len(y_train1) / len(np.unique(y_train1)))
    # 保护措施：如果 shot 数太少（如1-shot），无法做标准CV，需特殊处理，这里按论文逻辑假设 K>=2
    K = max(2, n_shots)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    fitness_values = []

    for train_index, test_index in skf.split(X_norm, y_train1):
        if _eval_timed_out(t0):
            return 0.0, 0.0
        X_fold_train, X_fold_test = X_norm[train_index], X_norm[test_index]
        y_fold_train, y_fold_test = y_train1[train_index], y_train1[test_index]

        # 训练 SVM (LinearSVC)
        clf = LinearSVC(max_iter=2000, dual="auto")  # dual="auto" 或者是 False 以加速
        clf.fit(X_fold_train, y_fold_train)

        # 计算 Acc (准确率)
        y_pred = clf.predict(X_fold_test)
        acc_k = accuracy_score(y_fold_test, y_pred)

        # 计算 Dist (距离度量)
        # 获取决策空间的值 (Decision Space)
        # decision_function 返回样本到超平面的距离
        decision_test = clf.decision_function(X_fold_test)
        decision_train = clf.decision_function(X_fold_train)

        # 处理二分类情况 (decision_function 返回一维数组)
        if len(decision_test.shape) == 1:
            decision_test = decision_test.reshape(-1, 1)
            decision_train = decision_train.reshape(-1, 1)

        dist_k = get_decision_dist(decision_test, y_fold_test, decision_train, y_fold_train)

        # 论文公式 (1) 中的单次折叠适应度: Acc - Dist
        # 注意：Dist 是累加的距离比率，通常需要归一化或者直接减去
        # 论文 Algorithm 1 Line 20: Acc_k - Dist_k
        fitness_values.append(acc_k - dist_k)

    # 5. 计算最终适应度 (Equation 1)
    final_fitness = np.mean(fitness_values)
    print(final_fitness)
    return final_fitness, final_fitness,

def parallel_evalTrain(individual):
    return evaluate_dtgpn(individual)

toolbox.register("evaluate", parallel_evalTrain)

if __name__ == "__main__":
    sys.stdout = Logger(opt.log_file, stream=sys.stdout, add_flag=True, encoding='utf-8')
    beginTime = time.perf_counter()
    hof = GPMain(opt.pkl_file,opt.randomSeeds)

    print("\n" + "=" * 50)
    print(">>> Best Feature Tree (Pop) Stats:")
    print(hof[0][0])
    print(f"    Nodes (Size): {len(hof[0][0])}")
    print(f"    Depth:        {hof[0][0].height}")

    # 可选：打印树的结构字符串
    # print(f"    Structure:    {best_ind_feat}")

    print("\n>>> Best Augment Tree (Pop_1) Stats:")
    print(hof[0][1])
    print(f"    Nodes (Size): {len(hof[0][1])}")
    print(f"    Depth:        {hof[0][1].height}")
    # print(f"    Structure:    {best_ind_aug}")
    print("=" * 50 + "\n")
    endTime = time.perf_counter()
    trainTime = endTime - beginTime

    results = fe_fs.test_task1(hof[0][0], hof[0][1],toolbox, opt.dataset ,x_train1, y_train1, x_test1, y_test1, x_test_imb, y_test_imb, Is_RGB, disable_imb_test=opt.disable_imb_test)
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