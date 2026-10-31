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
        train = True if phase == "train" else False
        np.random.seed(0)
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio, lt_mode)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
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
            img_num_per_cls= [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]#[200, 126, 80, 50, 32, 20, 13, 8, 5, 2]#[500, 336, 224, 149, 99, 66, 44, 29, 19, 13],[1000, 600, 360, 215, 129, 77, 46, 27, 16, 10]
        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

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
    dataset_name = 'MINIST-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp'):
        np.random.seed(0)
        train = True if phase == "train" else False
        super(IMBALANCEFMNIST, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
        else:
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
        img_num_per_cls= [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]
        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

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

class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10
    dataset_name = 'MINIST-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp'):
        np.random.seed(0)
        train = True if phase == "train" else False
        super(IMBALANCEMNIST, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
            self.transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
        else:
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
        img_num_per_cls= [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]
        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

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

class IMBALANCESTL10(torchvision.datasets.STL10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCESTL10, self).__init__(root, split='train' if train else 'test',
                                              transform=transform, target_transform=target_transform, download=download)
        self.train = train
        np.random.seed(0)

        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
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
        img_num_per_cls = [200, 158, 126, 100, 79, 63, 50, 40, 32, 20]
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
        """
        img, target = self.data[index], self.labels[index]
        if img.shape == (3, 96, 96):
            img = np.transpose(img, (1, 2, 0))

        # Convert image array to a PIL Image to maintain compatibility with transforms
        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](img)
                sample2 = self.transform[1](img)
                sample3 = self.transform[2](img)
                return [sample1, sample2, sample3], target
            else:
                return self.transform(img), target

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def imbalance_MRBand(x_train, y_train, x_test, y_test, img_num_per_cls):
    np.random.seed(0)
    classes = np.unique(y_train)
    assert len(classes) == len(img_num_per_cls), "img_num_per_cls 的长度必须与类别数一致！"

    # 处理训练集
    new_x_train = []
    new_y_train = []
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(y_train == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_x_train.append(x_train[selec_idx, ...])
        new_y_train.extend([the_class] * the_img_num)

    # 合并训练集数据
    new_x_train = np.vstack(new_x_train)
    new_y_train = np.array(new_y_train, dtype=np.int64)

    # 测试集保持不变（如果需要对测试集也进行处理，可以类似地操作）
    new_x_test = x_test
    new_y_test = y_test

    return new_x_train, new_y_train, new_x_test, new_y_test