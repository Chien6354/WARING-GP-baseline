"""
Adopted from https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image



class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    dataset_name = 'CIFAR-10-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp', lt_mode='fixed'):
        # 只要 phase 不是 "train"，父类就会加载测试集 (10000张)
        train = True if phase == "train" else False
        np.random.seed(0)
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)

        self.train = train
        self.phase = phase

        # 1. 训练集不平衡处理
        if self.phase == "train":
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio, lt_mode)
            self.gen_imbalanced_data(self.img_num_per_cls)

        # 2. 测试集不平衡处理 (新增 test_Imbalanced 情况)
        elif self.phase == "test_Imbalanced":
            # 强制使用指定的参数：类别数10，类型'exp'，不平衡率10
            self.img_num_per_cls = self.get_img_num_per_cls_test(self.cls_num, imb_type='exp', imb_factor=10)
            self.gen_imbalanced_data(self.img_num_per_cls)

        # 3. 如果 phase == "test" (或其他情况)，则保持原始测试集不变 (均衡的10000张)

        # 统一设置 transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, lt_mode='fixed'):
        if lt_mode == 'exp':
            # exp 模式：按指数衰减构造长尾，头部类别 400 张，不平衡比例 20，尾部类别 20 张
            imb_factor = 20
            img_max = 400
            gamma = 1. / imb_factor
            img_num_per_cls = []
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            gamma = 1. / imb_factor
            img_max = len(self.data) / cls_num
            img_num_per_cls = []
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_cls.append(int(img_max * gamma))
            else:
                img_num_per_cls.extend([int(img_max)] * cls_num)

            # 保留原代码中的硬编码覆盖
            img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

        return img_num_per_cls

    def get_img_num_per_cls_test(self, cls_num, imb_type, imb_factor):
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCEFMNIST(torchvision.datasets.FashionMNIST):
    cls_num = 10
    dataset_name = 'FMNIST-LT'  # 修正了名称，区分于 MNIST

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp'):
        np.random.seed(0)
        train = True if phase == "train" else False
        # 加载原始数据
        super(IMBALANCEFMNIST, self).__init__(root, train, transform=None, target_transform=None, download=True)

        self.train = train
        self.phase = phase

        # 1. 训练集不平衡处理 (使用固定的硬编码分布)
        if self.phase == "train":
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))  # 修正为 FashionMNIST 的归一化参数
            ])

        # 2. 测试集不平衡处理 (新增 test_Imbalanced 情况)
        elif self.phase == "test_Imbalanced":
            # 强制使用指定的参数：类别数10，类型'exp'，不平衡率10
            # 注意：这里调用的是新的 test 专用函数，不会应用硬编码覆盖
            self.img_num_per_cls = self.get_img_num_per_cls_test(self.cls_num, imb_type='exp', imb_factor=10)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

        # 3. 标准测试集 (phase == "test" 或其他)
        else:
            # 保持原始测试集不变 (均衡的 10000 张)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """
        用于训练集：包含硬编码覆盖，强制固定分布
        """
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # ⚠️ 固定的硬编码分布 (训练集专用)
        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

        return img_num_per_cls

    def get_img_num_per_cls_test(self, cls_num, imb_type, imb_factor):
        """
        用于测试集 (test_Imbalanced)：动态计算，无硬编码覆盖
        """
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # 注意：这里没有硬编码覆盖，返回的是根据 imb_factor=10 计算出的真实分布
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)

        if len(new_data) > 0:
            new_data = np.vstack(new_data)
            self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10
    dataset_name = 'MNIST-LT'  # 修正了拼写错误 (MINIST -> MNIST)

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp'):
        np.random.seed(0)
        train = True if phase == "train" else False
        # 加载原始数据
        super(IMBALANCEMNIST, self).__init__(root, train, transform=None, target_transform=None, download=True)

        self.train = train
        self.phase = phase

        # 1. 训练集不平衡处理 (使用固定的硬编码分布)
        if self.phase == "train":
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准参数
            ])

        # 2. 测试集不平衡处理 (新增 test_Imbalanced 情况)
        elif self.phase == "test_Imbalanced":
            # 强制使用指定的参数：类别数10，类型'exp'，不平衡率10
            # 注意：这里调用的是新的 test 专用函数，不会应用硬编码覆盖
            self.img_num_per_cls = self.get_img_num_per_cls_test(self.cls_num, imb_type='exp', imb_factor=10)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # 3. 标准测试集 (phase == "test" 或其他)
        else:
            # 保持原始测试集不变 (均衡的 10000 张)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """
        用于训练集：包含硬编码覆盖，强制固定分布
        """
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # ⚠️ 固定的硬编码分布 (训练集专用)
        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

        return img_num_per_cls

    def get_img_num_per_cls_test(self, cls_num, imb_type, imb_factor):
        """
        用于测试集 (test_Imbalanced)：动态计算，无硬编码覆盖
        """
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # 注意：这里没有硬编码覆盖，返回的是根据 imb_factor=10 计算出的真实分布
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)

            # 关键修复：确保要取的数量不超过实际存在的数量
            actual_num = min(the_img_num, len(idx))
            self.num_per_cls_dict[the_class] = actual_num

            # 使用 actual_num 进行切片和生成标签
            selec_idx = idx[:actual_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * actual_num)

        if len(new_data) > 0:
            new_data = np.vstack(new_data)
            self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCESTL10(torchvision.datasets.STL10):
    cls_num = 10
    dataset_name = 'STL10-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp', transform=None, target_transform=None,
                 download=False):
        """
        Args:
            phase (str): "train", "test_Imbalanced", or "test".
            imbalance_ratio (float): The imbalance ratio (e.g., 0.01, 0.1).
                                     Note: In training mode, this is ignored due to hardcoding.
                                     In test_Imbalanced mode, this is used (or forced to 0.01 depending on your need).
            root (string): Root directory of dataset.
            imb_type (str): Type of imbalance ('exp', 'step').
            transform, target_transform: Transforms for input and target.
            download (bool): If True, downloads the dataset.
        """
        # 确定加载的数据split
        # STL10: split='train' (5000), split='test' (8000), split='unlabeled'
        # 如果 phase 是 train，加载 train split
        # 如果 phase 是 test 或 test_Imbalanced，加载 test split
        split_flag = 'train' if phase == "train" else 'test'

        super(IMBALANCESTL10, self).__init__(root, split=split_flag,
                                             transform=None,  # 先在内部处理 transform
                                             target_transform=target_transform,
                                             download=download)

        self.phase = phase
        self.train = (phase == "train")
        np.random.seed(0)

        # 1. 训练集不平衡处理 (使用固定的硬编码分布)
        if self.phase == "train":
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)

            # STL10 通常需要强增强，这里假设传入的 transform 是一个列表或组合
            # 如果用户传入的是单个 transform，这里可能需要调整，但为了兼容原代码逻辑：
            self.transform = transform

            # 2. 测试集不平衡处理 (新增 test_Imbalanced 情况)
        elif self.phase == "test_Imbalanced":
            # 强制使用指定的参数：类型'exp'，不平衡率 0.01 (对应之前的硬编码比例 ~10:1 到 50:1 的极端情况)
            # 注意：原代码 imb_factor=0.01 意味着最大类是最小类的 100 倍 (1/0.01)
            # 这里我们调用 test 专用函数，不进行硬编码覆盖
            self.img_num_per_cls = self.get_img_num_per_cls_test(self.cls_num, imb_type='exp', imb_factor=0.01)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transform

        # 3. 标准测试集 (phase == "test" 或其他)
        else:
            # 保持原始测试集不变 (均衡的 8000 张)
            self.transform = transform

        self.labels = self.labels  # STL10 父类中属性名为 labels

        # 打印信息
        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """
        用于训练集：包含硬编码覆盖，强制固定分布
        """
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            # 注意：原代码公式是 imb_factor ** (...)，通常 imb_factor < 1
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # ⚠️ 固定的硬编码分布 (训练集专用)
        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]

        return img_num_per_cls

    def get_img_num_per_cls_test(self, cls_num, imb_type, imb_factor):
        """
        用于测试集 (test_Imbalanced)：动态计算，无硬编码覆盖
        """
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        # 注意：这里没有硬编码覆盖
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)

        if len(new_data) > 0:
            new_data = np.vstack(new_data)
            self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
            If train and transform is a list/tuple of 3, returns ([img1, img2, img3], target)
        """
        img, target = self.data[index], self.labels[index]

        # STL10 data shape is (N, 3, 96, 96), need to transpose to (96, 96, 3) for PIL
        if img.shape == (3, 96, 96):
            img = np.transpose(img, (1, 2, 0))

        # Convert image array to a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            # 兼容原代码逻辑：如果 transform 是包含3个变换的列表/元组，且处于训练模式，返回3个样本
            if self.train and isinstance(self.transform, (list, tuple)) and len(self.transform) == 3:
                sample1 = self.transform[0](img)
                sample2 = self.transform[1](img)
                sample3 = self.transform[2](img)
                return [sample1, sample2, sample3], target
            else:
                # 普通情况：直接应用 transform (可以是单个 transform 或Compose)
                # 如果 transform 是 Compose，直接调用；如果是函数，直接调用
                if callable(self.transform):
                    return self.transform(img), target
                else:
                    #  fallback
                    return img, target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)


def imbalance_MRBand(x_train, y_train, x_test, y_test, img_num_per_cls, imb_test_factor=0.1):
    """
    生成不平衡训练集、平衡测试集 和 不平衡测试集。

    Args:
        x_train, y_train: 原始训练集数据和标签。
        x_test, y_test: 原始测试集数据和标签（必须是平衡的，作为基准）。
        img_num_per_cls: 列表，指定训练集每个类别的样本数量。
        imb_test_factor: float (例如 0.1)。
                         用于控制不平衡测试集的长尾程度。
                         最小类样本数 = 最大类样本数 * imb_test_factor。
                         如果设为 None 或 0，则不平衡测试集将退化为空或报错（建议始终传入有效浮点数）。

    Returns:
        tuple: (
            new_x_train, new_y_train,              # 长尾训练集
            new_x_test_bal, new_y_test_bal,        # 原始平衡测试集
            new_x_test_imb, new_y_test_imb         # 新生成的长尾测试集
        )
    """
    np.random.seed(0)
    classes = np.unique(y_train)
    num_classes = len(classes)

    assert len(img_num_per_cls) == num_classes, "img_num_per_cls 的长度必须与类别数一致！"

    # ================= 1. 处理训练集 (长尾) =================
    new_x_train = []
    new_y_train = []

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(y_train == the_class)[0]
        actual_num = min(the_img_num, len(idx))
        np.random.shuffle(idx)
        selec_idx = idx[:actual_num]
        new_x_train.append(x_train[selec_idx, ...])
        new_y_train.extend([the_class] * actual_num)

    if len(new_x_train) > 0:
        new_x_train = np.vstack(new_x_train)
    else:
        new_x_train = np.array([])
    new_y_train = np.array(new_y_train, dtype=np.int64)

    # ================= 2. 处理平衡测试集 (直接复制) =================
    # 保持原始测试集不变，作为标准评估基准
    new_x_test_bal = x_test
    new_y_test_bal = y_test

    # ================= 3. 处理不平衡测试集 (动态生成) =================
    new_x_test_imb = None
    new_y_test_imb = None

    if imb_test_factor is not None and imb_test_factor > 0:
        test_img_num_per_cls = []

        # 获取原始测试集各类别的数量（通常所有类都相等）
        original_test_counts = []
        for c in classes:
            count = np.sum(y_test == c)
            original_test_counts.append(count)

        # 以原始测试集中数量最多的类为基准（Max）
        max_count = max(original_test_counts)
        gamma = imb_test_factor

        # 计算每个类别在不平衡测试集中应保留的数量
        for cls_idx in range(num_classes):
            num = max_count * (gamma ** (cls_idx / (num_classes - 1.0)))
            limit = original_test_counts[cls_idx]
            final_num = int(min(num, limit))
            test_img_num_per_cls.append(final_num)

        # 执行采样
        new_x_test_list = []
        new_y_test_list = []

        for the_class, the_img_num in zip(classes, test_img_num_per_cls):
            idx = np.where(y_test == the_class)[0]
            if len(idx) == 0 or the_img_num == 0:
                continue
            actual_num = min(the_img_num, len(idx))
            np.random.shuffle(idx)
            selec_idx = idx[:actual_num]
            new_x_test_list.append(x_test[selec_idx, ...])
            new_y_test_list.extend([the_class] * actual_num)

        if len(new_x_test_list) > 0:
            new_x_test_imb = np.vstack(new_x_test_list)
            new_y_test_imb = np.array(new_y_test_list, dtype=np.int64)
        else:
            # 防止空数据报错，返回空数组
            new_x_test_imb = np.array([])
            new_y_test_imb = np.array([], dtype=np.int64)

        print(
            f"Test Set Imbalanced Generated: Factor={imb_test_factor}, Total={len(new_y_test_imb)}, Dist={test_img_num_per_cls}")
    else:
        # 如果没有提供因子，不平衡测试集设为 None 或与平衡集相同（视需求而定，这里设为None以示区分）
        new_x_test_imb = x_test
        new_y_test_imb = y_test

    return new_x_train, new_y_train, new_x_test_bal, new_y_test_bal, new_x_test_imb, new_y_test_imb