import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def feature_length(ind, instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances))
    except: feature_len=0
    return feature_len,


def evalTest(toolbox, individual, trainData, trainLabel, test, testL, x_test_imb, y_test_imb, Is_RGB):
    """
    单树架构 (指定 Is_RGB)：在平衡测试集和不平衡测试集上分别评估模型。

    参数:
        test, testL: 平衡测试集数据和标签
        x_test_imb, y_test_imb: 不平衡测试集数据和标签
        Is_RGB: 是否为 RGB 图像的标志

    返回:
        18个指标 (平衡集 9 个 + 不平衡集 9 个)
    """
    print("开始 test (单种群 - 双测试集评估 - Is_RGB传入)")
    func = toolbox.compile(expr=individual)

    # ========== 1. 训练阶段 (只执行一次) ==========
    train_tf = []

    # 提取训练集特征
    if Is_RGB:
        for i in range(len(trainLabel)):
            # 注意：保留了你原代码中通道在最后的切片方式 [:, :, 0]
            feat = func(trainData[i][:, :, 0], trainData[i][:, :, 1], trainData[i][:, :, 2])
            train_tf.append(np.asarray(feat))
    else:
        for i in range(len(trainLabel)):
            feat = func(trainData[i][:, :])
            train_tf.append(np.asarray(feat))

    # 归一化训练集特征
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))

    # 训练分类器 (原代码中固定为 LinearSVC)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, trainLabel)

    # ========== 2. 定义内部评估函数 ==========
    def _run_single_evaluation(test_data, test_labels, set_name="Test"):
        print(f"正在评估 {set_name} ...")
        test_tf = []

        # 提取测试集特征
        if Is_RGB:
            for j in range(len(test_labels)):
                feat = func(test_data[j][:, :, 0], test_data[j][:, :, 1], test_data[j][:, :, 2])
                test_tf.append(np.asarray(feat))
        else:
            for j in range(len(test_labels)):
                feat = func(test_data[j][:, :])
                test_tf.append(np.asarray(feat))

        if len(test_tf) == 0:
            return None, "No samples", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 归一化测试集特征 (使用训练集的 scaler)
        test_norm = min_max_scaler.transform(np.asarray(test_tf))

        # 预测决策分数
        probabilities = lsvm.decision_function(test_norm)

        # 修复 LinearSVC 在二分类时输出 1D 数组导致后续报错的问题
        if probabilities.ndim == 1:
            probabilities = np.vstack([-probabilities, probabilities]).T

        # 确保 probabilities 是 2D
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        # 行归一化 (Min-Max Normalization per sample) - 替换全局归一化，更健壮
        min_vals = np.min(probabilities, axis=1, keepdims=True)
        max_vals = np.max(probabilities, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除以零
        probabilities = (probabilities - min_vals) / range_vals

        final_predictions = np.argmax(probabilities, axis=1)

        # 计算 Top-1 和 Top-5
        top1_accuracy = np.mean(final_predictions == test_labels)

        if probabilities.shape[1] >= 5:
            top5_indices = np.argsort(probabilities, axis=1)[:, -5:]
            top5_matches = np.any(top5_indices == test_labels[:, None], axis=1)
            top5_accuracy = np.mean(top5_matches)
        else:
            top5_accuracy = top1_accuracy

        group1_indices = [0, 1, 2, 3]
        group2_indices = [4, 5, 6]
        group3_indices = [7, 8, 9]

        def calculate_group_accuracy(preds, labels, group):
            mask = np.isin(labels, group)
            if not np.any(mask):
                return 0.0
            return np.mean(preds[mask] == labels[mask])

        acc_g1 = calculate_group_accuracy(final_predictions, test_labels, group1_indices)
        acc_g2 = calculate_group_accuracy(final_predictions, test_labels, group2_indices)
        acc_g3 = calculate_group_accuracy(final_predictions, test_labels, group3_indices)

        # 宏 F1
        macro_f1 = f1_score(test_labels, final_predictions, average='macro', zero_division=0)

        # 计算 Mean Per Class Accuracy (Macro Accuracy)
        unique_classes = np.unique(np.concatenate((trainLabel, test_labels)))
        num_classes = int(np.max(unique_classes)) + 1
        per_class_accs = []
        for c in range(num_classes):
            mask = (test_labels == c)
            if np.any(mask):
                acc = np.mean(final_predictions[mask] == test_labels[mask])
                per_class_accs.append(acc)
            else:
                per_class_accs.append(0.0)
        mean_per_class_acc = np.mean(per_class_accs) if per_class_accs else 0.0

        # 分类报告
        final_report = classification_report(test_labels, final_predictions, zero_division=0)

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
    res_bal = _run_single_evaluation(test, testL, "Balanced Test Set")

    # B. 不平衡测试集
    res_imb = _run_single_evaluation(x_test_imb, y_test_imb, "Imbalanced Test Set")

    # 解包结果
    (bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1) = res_bal
    (imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1) = res_imb

    # 打印简要对比
    print(f"[Result] Balanced  -> Top1: {bal_top1}%, Macro-F1: {bal_macro_f1}%")
    print(f"[Result] Imbalanced-> Top1: {imb_top1}%, Macro-F1: {imb_macro_f1}%")

    # ========== 4. 返回所有结果 (共 18 个值) ==========
    return (
        bal_preds, bal_report, bal_top1, bal_top5, bal_g1, bal_g2, bal_g3, bal_macro_acc, bal_macro_f1,
        imb_preds, imb_report, imb_top1, imb_top5, imb_g1, imb_g2, imb_g3, imb_macro_acc, imb_macro_f1
    )





