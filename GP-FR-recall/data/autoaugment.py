import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize, AffineTransform, warp


def data_augmentation(x_train, y_train, group2_indices, group3_indices, ratio=0.5):
    """
    改进版数据增强：使用基础图像增强策略，兼容灰度/RGB图像
    """
    augmented_data = []
    target_indices = group2_indices + group3_indices
    head_count = max([np.sum(y_train == i) for i in set(y_train) - set(target_indices)])

    for class_idx in target_indices:
        class_mask = y_train == class_idx
        x_class = x_train[class_mask]
        y_class = y_train[class_mask]
        current_count = len(x_class)

        n = int((head_count - current_count) * ratio)
        if n <= 0:
            continue

        selected_indices = np.random.choice(len(x_class), size=n, replace=True)

        for i in range(n):
            x_original = x_class[selected_indices[i]]
            y_original = y_class[selected_indices[i]]

            method = np.random.choice(['flip', 'rotate', 'crop', 'scale', 'shift'])

            try:
                if method == 'flip':
                    x_new = random_flip(x_original)
                elif method == 'rotate':
                    x_new = random_rotate(x_original)
                elif method == 'crop':
                    x_new = random_crop(x_original)
                elif method == 'scale':
                    x_new = random_scale(x_original)
                elif method == 'shift':
                    x_new = random_shift(x_original)

                augmented_data.append((x_new, y_original))
            except ValueError as e:
                #print(f"Skipping augmentation for {method}: {str(e)}")
                continue  # 跳过当前增强，继续下一个样本

    # 合并并打乱数据
    combined = list(zip(x_train, y_train)) + augmented_data
    np.random.shuffle(combined)

    x_train_new = np.stack([x for x, y in combined])
    y_train_new = np.array([y for x, y in combined])

    return x_train_new, y_train_new


def random_flip(img):
    """随机水平/垂直翻转"""
    if np.random.rand() < 0.5:
        return img[:, ::-1] if img.ndim == 2 else img[:, ::-1, :]  # 水平翻转
    return img[::-1, :] if img.ndim == 2 else img[::-1, :, :]  # 垂直翻转


def random_rotate(img):
    """随机旋转（-45°~45°）"""
    angle = np.random.uniform(-45, 45)
    return rotate(img, angle, reshape=False, mode='reflect')



def random_crop(img):
    """随机裁剪并恢复尺寸"""
    h, w = img.shape[:2]
    crop_h = int(h * np.random.uniform(0.7, 0.9))
    crop_w = int(w * np.random.uniform(0.7, 0.9))

    start_h = np.random.randint(0, h - crop_h)
    start_w = np.random.randint(0, w - crop_w)

    cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
    return resize(cropped, (h, w), preserve_range=True, anti_aliasing=True)

def random_scale(img, min_scale=0.7, max_scale=1.5):
    """限制最小缩放比例，防止图像过小"""
    scale = np.random.uniform(min_scale, max_scale)
    h, w = img.shape[:2]

    scaled = resize(img, (max(1, int(h * scale)), max(1, int(w * scale))),
                   preserve_range=True, anti_aliasing=True)
    return resize(scaled, (h, w), preserve_range=True, anti_aliasing=True)


def random_shift(img, max_shift=0.2):
    """随机平移（最大偏移20%）"""
    h, w = img.shape[:2]
    max_x = int(w * max_shift)
    max_y = int(h * max_shift)

    shift_x = np.random.randint(-max_x, max_x)
    shift_y = np.random.randint(-max_y, max_y)

    transform = AffineTransform(translation=(shift_x, shift_y))
    return warp(img, transform, mode='reflect', preserve_range=True)


# 辅助函数：保持数据格式一致
def safe_resize(img, target_shape):
    """保持数值范围和数据类型不变的resize"""
    original_dtype = img.dtype
    resized = resize(img, target_shape,
                     preserve_range=True,
                     anti_aliasing=True)
    return resized.astype(original_dtype)