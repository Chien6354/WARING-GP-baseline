from feEXFeatures import fetureEx, feaEx_flexible
from hog_features_patch import hog_features_patches as hog_features
from uLBPH import histLBP, lbp_ps
import sift_features
import numpy as np
import skimage
from PHOG import PHogFeatures
from sklearn.svm import LinearSVC
from sklearn import preprocessing  ##
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging
from sklearn import metrics


def phog(image):
    # print(image.shape)
    phogggg = PHogFeatures()
    features = phogggg.get_features(image)
    # print(features)
    return features

def root_con_number3(v1, v2, v3):
    return np.asarray([v1, v2, v3])

def root_con_number2(v1, v2):
    return np.asarray([v1, v2])

def root_con2(v1, v2):
    v11 = np.concatenate((v1))
    v12 = np.concatenate((v2))
    feature_vector=np.concatenate((v11, v12),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_con3(v1, v2, v3):
    v11 = np.concatenate((v1))
    v12 = np.concatenate((v2))
    v13 = np.concatenate((v3))
    feature_vector=np.concatenate((v11, v12, v13),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_con(*args):
    feature_vector=np.concatenate((args),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_one(input):
    feature_vector=input
    #print(feature_vector.shape)
    return feature_vector

def  local_conca(image):
    return np.concatenate((image))

def all_feex(image, size1=0.25, size2=0.25):
    #global and local
    width = int(image.shape[0]*size1)
    height = int(image.shape[1]*size2)
    feature_vector = feaEx_flexible(image, width, height)
    #unknown dimension for all type images
    return feature_vector

def all_histogram(image, bins=32):
    # global and local
    n_bins = bins
    hist, ax = np.histogram(image, n_bins, [0, 1])
    # dimension 24 for all type images
    return hist

def  global_hog(image, size1=20, size2=10):
    try:
        feature_vector=hog_features(image,size1,size2)
    except: feature_vector = np.concatenate(image)
    #dimension don't know
    return feature_vector

def  local_hog(image, size1=10, size2=10):
    try:
        feature_vector=hog_features(image,size1,size2)
    except: feature_vector = np.concatenate(image)
    #dimension don't know
    return feature_vector

def all_lbp(image, rad=1.5, number=8):
    # global and local
    feature_vector = histLBP(image, rad, number)
    # dimension 59 for all images
    return feature_vector

def all_lbp_ip(image, rad=1, method='nri_uniform'):
    # global and local
    number = 8*rad
    feature_vector = lbp_ps(image, rad, number, method)
    # dimension 59 for all images
    return feature_vector

def all_sift(image):
    # global and local
    width,height=image.shape
    min_length=np.min((width,height))
    img=np.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w,h=feaArrSingle.shape
    feature_vector=np.reshape(feaArrSingle, (h,))
    return feature_vector

def regionS(left,x,y,windowSize):
    width,height=left.shape
    xx = min(width, x+windowSize)
    yy = min(height, y+windowSize)
    slice = left[x:xx, y:yy]
    return slice

#rectangle
def regionR(left, x, y, windowSize1,windowSize2):
    width, height = left.shape
    xx = min(width, x + windowSize1)
    yy = min(height, y + windowSize2)
    slice = left[x:xx, y:yy]
    return slice

def averaging(image, factor):
    try:
        current = skimage.measure.block_reduce(image,(factor,factor),np.mean)
    except ValueError:
        current=image
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def maxP(image, factor):
    try:
        current = skimage.measure.block_reduce(image,(factor,factor),np.max)
    except ValueError:
        current=image
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def test_task1(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test):
    func_global = toolbox.compile(expr=ind0)
    func_local2 = toolbox.compile(expr=ind1)
    train_g_tf = []
    train_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func_local2(x_train[i, :, :])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_tf)), axis=1)
    test_g_tf = []
    test_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
    for i in range(0, len(y_test)):
        test_tf.append(np.asarray(func_local2(x_test[i, :, :])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_tf)), axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 *lsvm.score(test_norm, y_test), 2)
    y_predict = lsvm.predict(test_norm)
    logging.info(dataSet +' test results ' + str(acc))
    num_features = train_tf.shape[1]
    logging.info(dataSet +' Num Features ' + str(num_features))
    cm = confusion_matrix(y_predict, y_test)
    y_true = pd.Series(y_test, name="Actual")
    y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    df_confusion = pd.crosstab(y_true, y_pred)
    f1 = metrics.f1_score(y_test, y_predict, average='macro')  ##### f1-measure
    del lsvm
    return train_norm.shape[1], acc, df_confusion, f1


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import logging


def test_task1(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test, x_test_imb, y_test_imb, Is_RGB, disable_imb_test=False):
    """
    双树架构 (Global + Local 特征拼接)：在平衡测试集和不平衡测试集上分别评估模型。

    返回:
        18个指标 (平衡集 9 个 + 不平衡集 9 个)
    """
    print(f"开始 test_task1 (双树拼接 - 双测试集评估) - Is_RGB: {Is_RGB}")

    # 编译两个个体（Global 和 Local）
    func_global = toolbox.compile(expr=ind0)
    func_local2 = toolbox.compile(expr=ind1)

    # ========== 1. 训练阶段 (只执行一次) ==========
    train_g_tf = []
    train_l_tf = []

    # --- 1.1 提取训练集特征 ---
    if Is_RGB:
        # RGB 模式：传入三个通道
        for i in range(len(y_train)):
            train_g_tf.append(np.asarray(func_global(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
            train_l_tf.append(np.asarray(func_local2(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
    else:
        # 灰度模式：兼容原始代码逻辑
        for i in range(len(y_train)):
            img_data = x_train[i] if isinstance(x_train, list) else x_train[i, :, :]
            if img_data.ndim == 3 and img_data.shape[0] == 1:
                img_data = img_data[0, :, :]  # Handle (1, H, W)
            train_g_tf.append(np.asarray(func_global(img_data)))
            train_l_tf.append(np.asarray(func_local2(img_data)))

    # --- 1.2 特征拼接与归一化 ---
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_l_tf)), axis=1)
    train_tf = np.asarray(train_tf, dtype=float)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)

    num_features = train_norm.shape[1]
    logging.info(f"{dataSet} Num Features {num_features}")

    # --- 1.3 SVM 训练 ---
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)

    # ========== 2. 定义内部评估函数 ==========
    def _run_single_evaluation(test_data, test_labels, set_name="Test"):
        print(f"正在评估 {set_name} ...")
        test_g_tf = []
        test_l_tf = []

        # --- 2.1 提取测试集特征 ---
        if Is_RGB:
            for i in range(len(test_labels)):
                test_g_tf.append(
                    np.asarray(func_global(test_data[i][0, :, :], test_data[i][1, :, :], test_data[i][2, :, :])))
                test_l_tf.append(
                    np.asarray(func_local2(test_data[i][0, :, :], test_data[i][1, :, :], test_data[i][2, :, :])))
        else:
            for i in range(len(test_labels)):
                img_data = test_data[i] if isinstance(test_data, list) else test_data[i, :, :]
                if img_data.ndim == 3 and img_data.shape[0] == 1:
                    img_data = img_data[0, :, :]
                test_g_tf.append(np.asarray(func_global(img_data)))
                test_l_tf.append(np.asarray(func_local2(img_data)))

        if len(test_labels) == 0:
            return None, "No samples", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # --- 2.2 特征拼接与归一化 ---
        test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_l_tf)), axis=1)
        test_tf = np.asarray(test_tf, dtype=float)
        test_norm = min_max_scaler.transform(test_tf)

        # --- 2.3 预测与概率处理 ---
        probabilities = lsvm.decision_function(test_norm)

        # 处理二分类情况 (LinearSVC decision_function 返回一维数组)
        if probabilities.ndim == 1:
            probabilities = np.vstack([-probabilities, probabilities]).T

        # 行归一化 (Min-Max Normalization per sample) - 替换全局归一化，更健壮
        min_vals = np.min(probabilities, axis=1, keepdims=True)
        max_vals = np.max(probabilities, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除以零
        probabilities = (probabilities - min_vals) / range_vals

        final_predictions = np.argmax(probabilities, axis=1)
        test_labels_np = np.asarray(test_labels)

        # --- 2.4 指标计算 ---
        # A. Top-1 Accuracy
        top1_accuracy = np.mean(final_predictions == test_labels_np)

        # B. Top-5 Accuracy
        if probabilities.shape[1] >= 5:
            top5_indices = np.argsort(probabilities, axis=1)[:, -5:]
            top5_matches = np.any(top5_indices == test_labels_np[:, None], axis=1)
            top5_accuracy = np.mean(top5_matches)
        else:
            top5_accuracy = top1_accuracy

        # C. Group Accuracy (分组准确率)
        group1_indices = [0, 1, 2]  # 前三类
        group2_indices = [3, 4, 5]  # 中间三类
        group3_indices = [6, 7, 8, 9]  # 后四类

        def calculate_group_accuracy(preds, labels, group):
            mask = np.isin(labels, group)
            if not np.any(mask): return 0.0
            return np.mean(preds[mask] == labels[mask])

        acc_g1 = calculate_group_accuracy(final_predictions, test_labels_np, group1_indices)
        acc_g2 = calculate_group_accuracy(final_predictions, test_labels_np, group2_indices)
        acc_g3 = calculate_group_accuracy(final_predictions, test_labels_np, group3_indices)

        # D. Macro F1
        macro_f1 = f1_score(test_labels_np, final_predictions, average='macro', zero_division=0)

        # E. Mean Accuracy by Class (平均类别准确率)
        unique_classes = np.unique(np.concatenate((y_train, test_labels_np)))
        num_classes = int(np.max(unique_classes)) + 1
        per_class_accs = []
        for c in range(num_classes):
            mask = (test_labels_np == c)
            if np.any(mask):
                acc = np.mean(final_predictions[mask] == test_labels_np[mask])
                per_class_accs.append(acc)
            else:
                per_class_accs.append(0.0)
        mean_per_class_acc = np.mean(per_class_accs) if per_class_accs else 0.0

        # 分类报告
        final_report = classification_report(test_labels_np, final_predictions, zero_division=0)

        # 混淆矩阵日志记录 (保留你原代码的 pandas 逻辑)
        y_true_pd = pd.Series(test_labels_np, name="Actual")
        y_pred_pd = pd.Series(final_predictions, name="Predicted")
        df_confusion = pd.crosstab(y_true_pd, y_pred_pd)
        logging.info(f"\nConfusion Matrix for {set_name}:\n{df_confusion}")

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
    res_bal = _run_single_evaluation(x_test, y_test, "Balanced Test Set")

    # B. 不平衡测试集 (可通过 --disable_imb_test 关闭)
    if disable_imb_test:
        res_imb = (None, "Disabled by --disable_imb_test", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        res_imb = _run_single_evaluation(x_test_imb, y_test_imb, "Imbalanced Test Set")

    # 解包结果
    (bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1) = res_bal
    (imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1) = res_imb

    # 打印简要对比
    print(f"[Result] Balanced  -> Top1: {bal_top1}%, Macro-F1: {bal_macro_f1}%")
    print(f"[Result] Imbalanced-> Top1: {imb_top1}%, Macro-F1: {imb_macro_f1}%")

    # 释放内存
    del lsvm

    # ========== 4. 返回所有结果 (共 18 个值) ==========
    return (
        bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
        imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1
    )


def test_task_coloar(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test):
    func_global = toolbox.compile(expr=ind0)
    func_local2 = toolbox.compile(expr=ind1)
    train_g_tf = []
    train_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :, 0],x_train[i, :, :, 1], x_train[i, :, :, 2], x_train[i, :, :, 3])))
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func_local2(x_train[i, :, :, 0],x_train[i, :, :, 1], x_train[i, :, :, 2], x_train[i, :, :, 3])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_tf)), axis=1)
    test_g_tf = []
    test_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :, 0], x_test[i, :, :, 1], x_test[i, :, :, 2], x_test[i, :, :, 3])))
    for i in range(0, len(y_test)):
        test_tf.append(np.asarray(func_local2(x_test[i, :, :, 0], x_test[i, :, :, 1], x_test[i, :, :, 2], x_test[i, :, :, 3])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_tf)), axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 *lsvm.score(test_norm, y_test), 2)
    y_predict = lsvm.predict(test_norm)
    logging.info(dataSet +' test results ' + str(acc))
    num_features = train_tf.shape[1]
    logging.info(dataSet +' Num Features ' + str(num_features))
    cm = confusion_matrix(y_predict, y_test)
    y_true = pd.Series(y_test, name="Actual")
    y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    df_confusion = pd.crosstab(y_true, y_pred)
    del lsvm
    return train_norm.shape[1], acc, df_confusion


def test_task_transform(ind0, ind1, toolbox, x_train, y_train, x_test, y_test):
    func_global = toolbox.compile(expr=ind0)
    func_local2 = toolbox.compile(expr=ind1)
    train_g_tf = []
    train_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func_local2(x_train[i, :, :])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_tf)), axis=1)
    test_g_tf = []
    test_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
    for i in range(0, len(y_test)):
        test_tf.append(np.asarray(func_local2(x_test[i, :, :])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_tf)), axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    return train_norm, test_norm


def test_task2(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test):
    func_global = toolbox.compile(expr=ind0)
    func_local2 = toolbox.compile(expr=ind1)
    train_g_tf = []
    train_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func_local2(x_train[i, :, :])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_tf)), axis=1)
    test_g_tf = []
    test_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
    for i in range(0, len(y_test)):
        test_tf.append(np.asarray(func_local2(x_test[i, :, :])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_tf)), axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 *lsvm.score(test_norm, y_test), 2)
    y_predict = lsvm.predict(test_norm)
    logging.info(dataSet +' test results ' + str(acc))
    num_features = train_tf.shape[1]
    logging.info(dataSet +' Num Features ' + str(num_features))
    cm = confusion_matrix(y_predict, y_test)
    y_true = pd.Series(y_test, name="Actual")
    y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    df_confusion = pd.crosstab(y_true, y_pred)
    del lsvm
    return train_norm.shape[1], acc, df_confusion

# def test_task1_select(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test):
#     n_class_task = len(np.unique(y_train))
#     func_global = toolbox.compile_global(expr=ind0)
#     func_local2 = toolbox.compile_local1(expr=ind1)
#     train_g_tf = []
#     train_tf = []
#     # print(individual[0], individual[1])
#     for i in range(0, len(y_train)):
#         train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
#     for i in range(0, len(y_train)):
#         train_tf.append(np.asarray(func_local2(x_train[i, :, :])))
#     train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_tf)), axis=1)
#     test_g_tf = []
#     test_tf = []
#     # print(individual[0], individual[1])
#     for i in range(0, len(y_test)):
#         test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
#     for i in range(0, len(y_test)):
#         test_tf.append(np.asarray(func_local2(x_test[i, :, :])))
#     test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_tf)), axis=1)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     train_norm = min_max_scaler.fit_transform(train_tf)
#     test_norm = min_max_scaler.transform(test_tf)
#     num_features_select1 = int(0.5 * train_norm.shape[1])
#     fs = ReliefF(n_neighbors=n_class_task, n_features_to_keep=num_features_select1)
#     train_norm_select = fs.fit_transform(train_norm, y_train)
#     test_norm_select = fs.transform(test_norm)
#     lsvm = LinearSVC()
#     lsvm.fit(train_norm_select, y_train)
#     acc = round(100 *lsvm.score(test_norm_select, y_test), 2)
#     y_predict = lsvm.predict(test_norm_select)
#     logging.info(dataSet +' test results ' + str(acc))
#     num_features = train_tf.shape[1]
#     logging.info(dataSet +' Num Features ' + str(num_features))
#     cm = confusion_matrix(y_predict, y_test)
#     y_true = pd.Series(y_test, name="Actual")
#     y_pred = pd.Series(y_predict, name="Predicted")
#     logging.info(cm)
#     df_confusion = pd.crosstab(y_true, y_pred)
#     del lsvm
#     return train_norm_select.shape[1], acc, df_confusion

def acc_lsvm(x_train, x_test, y_train, y_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(x_train))
    test_norm = min_max_scaler.transform(np.asarray(x_test))
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 * lsvm.score(test_norm, y_test), 2)
    return acc

def test_task_all(ind0, ind1, toolbox, dataSet, x_train, y_train, x_test, y_test):
    func_global = toolbox.compile_global(expr=ind0)
    func_local2 = toolbox.compile_local1(expr=ind1)
    train_g_tf = []
    train_l_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
    for i in range(0, len(y_train)):
        train_l_tf.append(np.asarray(func_local2(x_train[i, :, :])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_l_tf)), axis=1)
    test_g_tf = []
    test_l_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
    for i in range(0, len(y_test)):
        test_l_tf.append(np.asarray(func_local2(x_test[i, :, :])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_l_tf)), axis=1)
    acc_global = acc_lsvm(train_g_tf, test_g_tf, y_train, y_test)
    acc_local = acc_lsvm(train_l_tf, test_l_tf, y_train, y_test)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 *lsvm.score(test_norm, y_test), 2)
    y_predict = lsvm.predict(test_norm)
    logging.info(dataSet +' test results ' + str(acc))
    num_features = train_tf.shape[1]
    logging.info(dataSet +' Num Features ' + str(num_features))
    cm = confusion_matrix(y_predict, y_test)
    y_true = pd.Series(y_test, name="Actual")
    y_pred = pd.Series(y_predict, name="Predicted")
    logging.info(cm)
    df_confusion = pd.crosstab(y_true, y_pred)
    del lsvm
    return train_norm.shape[1], acc, df_confusion, acc_global, acc_local