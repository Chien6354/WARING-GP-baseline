import argparse
import os

import select_nsga
from main_all import *

DATASETS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
import argparse
from scipy.ndimage import zoom
import numpy as np
import select_nsga
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import sys
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
import pickle
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    help="Dataset name (choose from the provided options)",
    type=str,
    choices=["mnist", "cifar10", "fashion_mnist", "STL", "MBI", "MRB","MRD"],  # 限制可选值
    default="MBI"  # 设置默认值
)



opt = parser.parse_args()
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+',encoding="utf-8") as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w',encoding="utf-8") as log:
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
            x_train = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_train_data.npy')) / 255.0
            y_train = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_train_label.npy'))
            x_test = np.load(os.path.join(DATASETS_ROOT, dataset, dataset+'_test_data.npy')) / 255.0
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
        img_num_per_cls=[200, 158, 126, 100, 79, 63, 50, 40, 32, 20]
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
group2_indices = [4, 5, 6]  # 中间三类
group3_indices = [7, 8, 9]  # 尾部三类

x_train,y_train,x_test,y_test,Is_RGB = create_dataset(opt.dataset)

if Is_RGB:
    pset = gp.PrimitiveSetTyped('MAIN', [Img, Img, Img], Vector1, prefix='Image')
else:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
# image classification layer
# pset.addPrimitive(fe_fs.classifier_selection, [Vector1, Int4], ClassLabel, name='classifier')

# Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')

# Feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')

# Pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')

# Filtering (输入都是图片，输出也是图片，并且大小没有改变)
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
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')

# Pooling for Img
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')

# Filtering for Img
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

def test_individual_func(individual, toolbox, x_train, trainLabel, x_test, testLabel):
    func = toolbox.compile(expr=individual)
    print("Testing individual:", individual)
    train_tf = []
    test_tf = []
    if Is_RGB:
        for i in range(len(trainLabel)):
            train_tf.append(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :]))
        train_tf = np.asarray(train_tf, dtype=float)

        for j in range(len(testLabel)):
            test_tf.append(func(x_test[j][0, :, :], x_test[j][1, :, :], x_test[j][2, :, :]))
        test_tf = np.asarray(test_tf, dtype=float)
    else:
        for i in range(len(trainLabel)):
            train_tf.append(func(x_train[i][:, :]))
        train_tf = np.asarray(train_tf, dtype=float)

        for j in range(len(testLabel)):
            test_tf.append(func(x_test[j][:, :]))
        test_tf = np.asarray(test_tf, dtype=float)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)

    classifier = LinearSVC(max_iter=10000)
    classifier.fit(train_norm, trainLabel)
    probabilities = classifier.decision_function(test_norm)
    probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    return probabilities


def evalTest(toolbox, hof, x_train, trainLabel, x_test, testLabel):
    all_probabilities = np.zeros((len(testLabel), num_unique_labels))  # Assuming there are 10 classes

    def create_toolbox():
        local_toolbox = base.Toolbox()
        local_toolbox.register("compile", gp.compile, pset=pset)

        return local_toolbox

    with Pool(processes=6) as p:
        test_func = partial(test_individual_func,toolbox = create_toolbox(), x_train=x_train, trainLabel=trainLabel, x_test=x_test,
                            testLabel=testLabel)
        results = p.map(test_func, hof)

    for probabilities in results:
        all_probabilities += probabilities

    group1_indices = [0, 1, 2, 3]  # 前四类
    group2_indices = [4, 5, 6]  # 中间三类
    group3_indices = [7, 8, 9]  # 后三类

    def calculate_group_accuracy(predictions, true_labels, group_indices):
        # 找出属于该组的所有样本的索引
        mask = np.isin(true_labels, group_indices)
        # 过滤出属于该组的真实标签和预测标签
        filtered_true_labels = true_labels[mask]
        filtered_predictions = predictions[mask]
        accuracy=accuracy_score(filtered_true_labels,filtered_predictions)
        return accuracy

    final_predictions = np.argmax(all_probabilities, axis=1)
    top1_accuracy = np.mean(final_predictions == testLabel)
    top5_indices = np.argsort(all_probabilities, axis=1)[:, -5:]
    top5_matches = np.any(top5_indices == testLabel[:, None], axis=1)
    top5_accuracy = np.mean(top5_matches)
    final_report = classification_report(testLabel, final_predictions)

    accuracy_group1 = calculate_group_accuracy(final_predictions, testLabel, group1_indices)
    accuracy_group2 = calculate_group_accuracy(final_predictions, testLabel, group2_indices)
    accuracy_group3 = calculate_group_accuracy(final_predictions, testLabel, group3_indices)

    return final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3



print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

unique_labels = np.unique(y_train)
num_unique_labels = len(unique_labels)
total_instances = len(y_train)
per_class_num = [np.sum(y_train == label) for label in unique_labels]
class_ratios = [np.sum(y_train == label) / total_instances for label in unique_labels]

class_reciprocals = [1 / num for num in per_class_num]
sum_of_reciprocals = sum(class_reciprocals)

toolbox = base.Toolbox()
root_dir=r"D:\Code\loongtail_multi_fewshot_GP\消融实验单目标\MEGPE_contrast\MEGPE_contrast\pkl"
method_path = os.path.join(root_dir, opt.dataset)

if __name__ == '__main__':
    i=1
    for pkl_file in os.listdir(method_path):
        pkl = os.path.join(method_path, pkl_file)
        with open(pkl, 'rb') as f:
            log_dir = r"D:\Code\loongtail_multi_fewshot_GP\消融实验单目标\MEGPE_contrast\MEGPE_contrast\log"
            log_file = os.path.join(log_dir, f"{opt.dataset}_{i}.log")
            #sys.stdout = Logger(log_file, sys.stdout)
            checkpoint_data = pickle.load(f)
            pop = checkpoint_data['population']
            #nsga2 = select_nsga.selNSGA2(pop, 10, nd='standard', select_first_front=True)
            print(f"第{i}个run：")
            print("______________________________________________________________________________________________")
            #print(f"非支配解个数：{len(nsga2)}")
            for p in pop:
                print(p)
                print(p.fitness.values)
            i = i + 1

    # 你的多进程代码
    pass