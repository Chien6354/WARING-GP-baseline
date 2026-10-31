import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import random

class BDD100KLongTailDataset:
    def __init__(self, root, imb_factor=10, img_max=250, imb_type='exp', image_size=(224, 224), seed=42):
        """
        构建 BDD100K 18 类长尾数据集
        :param root: BDD100K 根目录
        :param imb_factor: 不平衡因子 (e.g., 10 → 最常见类是最罕见类的10倍)
        :param img_max: 最常见类的样本数（即 N1）
        :param imb_type: 'exp', 'step', or 'none'
        :param image_size: 图像 resize 尺寸
        """
        self.root = root
        self.imb_factor = imb_factor
        self.img_max = img_max
        self.imb_type = imb_type
        self.image_size = image_size
        random.seed(seed)
        np.random.seed(seed)

        # 定义 18 个类别（scene_time 组合）
        scenes = ["city street", "highway", "residential", "parking lot", "tunnel", "gas station"]
        times = ["daytime", "dawn/dusk", "night"]
        self.class_names = [f"{s}_{t}" for s in scenes for t in times]  # m=0 最常见 → m=17 最罕见
        self.cls_num = len(self.class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # 加载原始 train 数据路径
        self._load_data()

        # 计算每类应采样数量
        self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, self.imb_type, self.imb_factor)

        # 按数量采样，构建最终数据集
        self.x_data, self.y_data = self._build_longtail_dataset()

    def _load_data(self):
        """从 BDD100K train split 加载所有符合条件的图像路径"""
        json_path = os.path.join(self.root, "labels", "bdd100k_labels_images_train.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Label file not found: {json_path}")

        with open(json_path, 'r') as f:
            anns = json.load(f)

        image_dir = os.path.join(self.root, "images", "100k", "train")
        self.all_paths = defaultdict(list)

        valid_scenes = set(["city street", "highway", "residential", "parking lot", "tunnel", "gas station"])
        valid_times = set(["daytime", "dawn/dusk", "night"])

        for ann in anns:
            name = ann["name"]
            attr = ann.get("attributes", {})
            scene = attr.get("scene")
            timeofday = attr.get("timeofday")
            if not scene or not timeofday:
                continue
            if scene not in valid_scenes or timeofday not in valid_times:
                continue
            key = f"{scene}_{timeofday}"
            if key in self.class_to_idx:
                path = os.path.join(image_dir, name)
                if os.path.exists(path):
                    self.all_paths[key].append(path)

        # 检查每类是否有足够样本
        for cls_name in self.class_names:
            available = len(self.all_paths[cls_name])
            needed = self.img_max  # 最常见类需要 img_max 张
            if available < needed:
                print(f"Warning: {cls_name} has only {available} samples, but need at least {needed} for img_max={self.img_max}")

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        """
        你提供的函数，但 img_max 不再依赖 len(self.data)，而是使用 self.img_max
        """
        gamma = 1. / imb_factor
        img_max = self.img_max  # 关键修改：使用指定的 img_max，而非 len(self.data)/cls_num
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

    def _build_longtail_dataset(self):
        images = []
        labels = []

        for cls_idx, cls_name in enumerate(self.class_names):
            paths = self.all_paths[cls_name]
            target_num = self.img_num_per_cls[cls_idx]

            if target_num == 0:
                continue

            if len(paths) < target_num:
                raise ValueError(f"Not enough images for {cls_name}: have {len(paths)}, need {target_num}")

            selected = random.sample(paths, target_num)
            for p in selected:
                try:
                    img = Image.open(p).convert('RGB')
                    img = img.resize(self.image_size)
                    img_np = np.array(img)  # (H, W, 3), uint8
                    images.append(img_np)
                    labels.append(cls_idx)
                except Exception as e:
                    print(f"Skip {p}: {e}")

        x_data = np.array(images, dtype=np.float32)
        y_data = np.array(labels, dtype=np.int64)
        return x_data, y_data

    def save(self, save_path="BDD100K_longtail.npz"):
        np.savez(save_path, x_train=self.x_data, y_train=self.y_data)
        print(f"Saved long-tailed dataset to {save_path}")
        print(f"x_train shape: {self.x_data.shape}, dtype: {self.x_data.dtype}")
        print(f"y_train shape: {self.y_data.shape}, dtype: {self.y_data.dtype}")
        print("Per-class sample counts:")
        for i, count in enumerate(self.img_num_per_cls):
            print(f"  Class {i:2d} ({self.class_names[i]}): {count}")


# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    # 配置参数
    BDD_ROOT = "./bdd100k"          # 修改为你的 BDD100K 路径
    IMB_FACTOR = 10                 # 不平衡因子 γ = 10
    IMG_MAX = 200                   # 最常见类样本数（N1）
    IMB_TYPE = 'exp'                # 长尾类型

    # 构建数据集
    dataset = BDD100KLongTailDataset(
        root=BDD_ROOT,
        imb_factor=IMB_FACTOR,
        img_max=IMG_MAX,
        imb_type=IMB_TYPE,
        image_size=(224, 224),
        seed=42
    )

    # 保存
    dataset.save("BDD100K_longtail_18class_imb10_Nmax250.npz")