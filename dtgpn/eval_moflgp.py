import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import confusion_matrix
import logging
from collections import Counter
from sklearn.model_selection import StratifiedKFold

def feature_length(ind, instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances))
    except: feature_len=0
    return feature_len,

def data_Transform(toolbox, individual, x_train):
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(x_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    return train_norm


def evalTest(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(trainLabel)):
        train_tf.append(np.asarray(func(trainData[i, :, :])))
    for j in range(0, len(testL)):
        test_tf.append(np.asarray(func(test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC(random_state=5)
    lsvm.fit(train_norm, trainLabel)
##    print(np.asarray(train_tf).shape, np.asarray(test_tf).shape)
    accuracy = round(100*lsvm.score(test_norm, testL),2)
    y_predict = lsvm.predict(test_norm)
    cm = confusion_matrix(y_predict, testL)
    # y_true = pd.Series(testL, name="Actual")
    # y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    # df_confusion = pd.crosstab(y_true, y_pred)
    return train_tf.shape[1], accuracy, y_predict

def evalTest_multitree(toolbox, individual, trainData, trainLabel, test, testL):
    func1 = toolbox.compile(expr=individual[0])
    func2= toolbox.compile(expr=individual[1])
    train_tf1 = []
    train_t2 = []
    test_tf1 = []
    test_tf2 = []
    for i in range(0, len(trainLabel)):
        train_tf1.append(np.asarray(func1(trainData[i, :, :])))
    for i in range(0, len(trainLabel)):
        train_t2.append(np.asarray(func2(trainData[i, :, :])))
    for j in range(0, len(testL)):
        test_tf1.append(np.asarray(func1(test[j, :, :])))
    for j in range(0, len(testL)):
        test_tf2.append(np.asarray(func2(test[j, :, :])))
    train_tf1= np.asarray(train_tf1)
    train_t2= np.asarray(train_t2)
    test_tf1= np.asarray(test_tf1)
    test_tf2= np.asarray(test_tf2)
    train_tf = np.concatenate((train_tf1,train_t2), axis=1)
    test_tf = np.concatenate((test_tf1,test_tf2), axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC(random_state=5)
    lsvm.fit(train_norm, trainLabel)
##    print(np.asarray(train_tf).shape, np.asarray(test_tf).shape)
    accuracy = round(100*lsvm.score(test_norm, testL),2)
    y_predict = lsvm.predict(test_norm)
    cm = confusion_matrix(y_predict, testL)
    # y_true = pd.Series(testL, name="Actual")
    # y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    # df_confusion = pd.crosstab(y_true, y_pred)
    return train_tf.shape[1], accuracy, y_predict


def mltitree_transform(toolbox, individual, x_train, y_train):
    train_norm1 = data_Transform(toolbox, individual[0], x_train)
    train_norm2 = data_Transform(toolbox, individual[1], x_train)
    # print(train_norm1.shape, train_norm2.shape)
    if len(train_norm2.shape)==2 and len(train_norm1.shape)==2:
        train_norm = np.concatenate((train_norm1, train_norm2), axis=1)
        lsvm = LinearSVC(max_iter=100, random_state=5)
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
        distances = evalTrain_dis(train_norm, y_train)
    else:
        accuracy=0
        distances=0
    return accuracy, distances

def evalTrain_mtflgp_new(toolbox, individual, x_train, y_train):
    train_norm1 = data_Transform(toolbox, individual[0], x_train)
    train_norm2 = data_Transform(toolbox, individual[1], x_train)
    if len(train_norm2.shape)==2 and len(train_norm1.shape)==2:
        train_norm = np.concatenate((train_norm1, train_norm2), axis=1)
        acc_list = 0
        dis_k_fold = 0
        n_instances = len(y_train[y_train==0])
        n_class = len(np.unique(y_train))
        # print(n_instances)
        skf = StratifiedKFold(n_splits=n_instances, random_state=5)
        for train_index, test_index in skf.split(train_norm, y_train):
            # print("TRAIN:", train_index, "TEST:", test_index)
            x_train_new, x_test_new = train_norm[train_index], train_norm[test_index]
            y_train_new, y_test_new = y_train[train_index], y_train[test_index]
            lsvm = LinearSVC(max_iter=100, random_state=5)
            lsvm.fit(x_train_new, y_train_new)
            acc_single = round(100*lsvm.score(x_test_new, y_test_new), 2)
            acc_list += acc_single
            coef = lsvm.coef_
            b = lsvm.intercept_
            x_transform = np.matmul(coef, train_norm.T)
            x_transform = (x_transform.T + b)
            x_trans_train_new, x_trans_test_new = x_transform[train_index], x_transform[test_index]
            # print(x_trans_train_new.shape)
            # print(x_trans_test_new.shape)
            wc = 0
            global_mean = np.mean(train_norm, axis=0)
            # print(transformed_data.shape)
            class_mean_all = []
            for i in range(n_class):
                if x_trans_train_new.shape[1] == 1:
                    one_class = x_trans_train_new[y_train_new == i, :]
                    class_mean = np.mean(one_class)
                    class_mean_all.append(class_mean)
                else:
                    one_class = x_trans_train_new[y_train_new==i, :]
                    class_mean = np.mean(one_class, axis = 0)
                    class_mean_all.append(class_mean)
            class_mean_all = np.asarray(class_mean_all)
            # print(class_mean_all.shape)
            prob_all = 0
            for i in range(len(y_test_new)):
                distance_list = []
                for j in range(len(class_mean_all)):
                    dis = np.sum((class_mean_all[j]-x_trans_test_new[i])**2, axis=0)
                    distance_list.append(dis)
                within_dis = distance_list[y_test_new[i]]
                prob_all += round(100*within_dis/(np.sum(np.asarray(distance_list))-within_dis),2)
                # prob_all = within_dis
            dis_k_fold+=prob_all
        dis_average = round(np.asarray(dis_k_fold)/n_instances, 2)
        acc_average = round(np.asarray(acc_list)/n_instances, 2)
        fitness = acc_average-dis_average
    else:
        fitness=-100
    return fitness,
