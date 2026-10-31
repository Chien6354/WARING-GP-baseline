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
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn import preprocessing
import sys
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
import pickle
from sklearn.metrics import accuracy_score


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
    if dataset=="cifar":
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



def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    if Is_RGB:
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
        for j in range(0, len(y_test)):
            test_tf.append(np.asarray(func(x_test[j][0, :, :], x_test[j][1, :, :], x_test[j][2, :, :])))
    else:
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i][ :, :])))
        for j in range(0, len(y_test)):
            test_tf.append(np.asarray(func(x_test[j][ :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()


    lsvm.fit(train_norm, y_train)

    probabilities = lsvm.decision_function(test_norm)
    probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    group1_indices = [0, 1, 2, 3]  # 前四类
    group2_indices = [ 4, 5, 6]  # 中间三类
    group3_indices = [7, 8, 9]  # 后三类
    def calculate_group_accuracy(predictions, true_labels, group_indices):
        # 找出属于该组的所有样本的索引
        mask = np.isin(true_labels, group_indices)
        # 过滤出属于该组的真实标签和预测标签
        filtered_true_labels = true_labels[mask]
        filtered_predictions = predictions[mask]
        accuracy=accuracy_score(filtered_true_labels,filtered_predictions)
        return accuracy

    final_predictions = np.argmax(probabilities, axis=1)
    top1_accuracy = np.mean(final_predictions == y_test)
    top5_indices = np.argsort(probabilities, axis=1)[:, -5:]
    top5_matches = np.any(top5_indices == y_test[:, None], axis=1)
    top5_accuracy = np.mean(top5_matches)
    final_report = classification_report(y_test, final_predictions)
    accuracy_group1 = calculate_group_accuracy(final_predictions, y_test, group1_indices)
    accuracy_group2 = calculate_group_accuracy(final_predictions, y_test, group2_indices)
    accuracy_group3 = calculate_group_accuracy(final_predictions, y_test, group3_indices)

    # 新增：宏 F1（macro f1）
    macro_f1 = f1_score(y_test, final_predictions, average='macro')

    num_classes = int(np.max(np.concatenate((y_train, y_test))) + 1)  # 或你可显式设置类别数
    per_class_accs = []
    for c in range(num_classes):
        mask = (y_test == c)
        if np.sum(mask) == 0:
            acc_c = 0.0
        else:
            acc_c = np.mean(final_predictions[mask] == y_test[mask])
        per_class_accs.append(acc_c)
    Macro_Recall = round(100 * np.mean(per_class_accs), 2)
    top1_accuracy = round(top1_accuracy * 100, 2)
    top5_accuracy = round(top5_accuracy * 100, 2)
    accuracy_group1 = round(accuracy_group1 * 100, 2)
    accuracy_group2 = round(accuracy_group2 * 100, 2)
    accuracy_group3 = round(accuracy_group3 * 100, 2)
    macro_f1 = round(macro_f1 * 100, 2)
    return final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3, Macro_Recall, macro_f1



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