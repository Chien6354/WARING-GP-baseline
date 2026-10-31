"""
单张图像 RandAugment 风格随机策略增强。
输入: numpy.ndarray, shape (3, H, W)；输出同 shape。
依赖: numpy, Pillow
"""

from __future__ import annotations

import random
from typing import Callable, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# 与题目一致的组合策略列表
AUGMENTATION_POLICIES = [
    (0.1, "invert", 7, 0.2, "contrast", 6),
    (0.7, "rotate", 2, 0.3, "translateX", 9),
    (0.8, "sharpness", 1, 0.9, "sharpness", 3),
    (0.5, "shearY", 8, 0.7, "translateY", 9),
    (0.5, "autocontrast", 8, 0.9, "equalize", 2),
    (0.2, "shearY", 7, 0.3, "posterize", 7),
    (0.4, "color", 3, 0.6, "brightness", 7),
    (0.3, "sharpness", 9, 0.7, "brightness", 9),
    (0.6, "equalize", 5, 0.5, "equalize", 1),
    (0.6, "contrast", 7, 0.6, "sharpness", 5),
    (0.7, "color", 7, 0.5, "translateX", 8),
    (0.3, "equalize", 7, 0.4, "autocontrast", 8),
    (0.4, "translateY", 3, 0.2, "sharpness", 6),
    (0.9, "brightness", 6, 0.2, "color", 8),
    (0.5, "solarize", 2, 0.0, "invert", 3),
    (0.2, "equalize", 0, 0.6, "autocontrast", 0),
    (0.2, "equalize", 8, 0.6, "equalize", 4),
    (0.9, "color", 9, 0.6, "equalize", 6),
    (0.8, "autocontrast", 4, 0.2, "solarize", 8),
    (0.1, "brightness", 3, 0.7, "color", 0),
    (0.4, "solarize", 5, 0.9, "autocontrast", 3),
    (0.9, "translateY", 9, 0.7, "translateY", 9),
    (0.9, "autocontrast", 2, 0.8, "solarize", 3),
    (0.8, "equalize", 8, 0.1, "invert", 3),
    (0.7, "translateY", 9, 0.9, "autocontrast", 1),
]


def _mag_to_value(magnitude: int, low: float, high: float) -> float:
    """强度 0–9 线性映射到 [low, high]；0 对应 low，9 对应 high。"""
    m = float(np.clip(int(magnitude), 0, 9))
    return low + (high - low) * (m / 9.0)


def _chw_numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"期望 shape (3, H, W)，得到 {arr.shape}")
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        hwc = np.transpose(x, (1, 2, 0))
        return Image.fromarray(hwc, mode="RGB")
    # 浮点按 [0,1] 或任意范围先裁剪再转 uint8
    xf = x.astype(np.float64)
    if xf.max() <= 1.0 + 1e-6:
        hwc = np.transpose(np.clip(xf, 0.0, 1.0), (1, 2, 0))
        return Image.fromarray((hwc * 255.0).round().astype(np.uint8), mode="RGB")
    hwc = np.transpose(np.clip(xf, 0.0, 255.0), (1, 2, 0))
    return Image.fromarray(hwc.round().astype(np.uint8), mode="RGB")


def _pil_rgb_to_chw_numpy(pil_img: Image.Image, out_dtype: np.dtype) -> np.ndarray:
    hwc = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    chw = np.transpose(hwc, (2, 0, 1))
    if out_dtype == np.uint8:
        return chw
    if np.issubdtype(out_dtype, np.floating):
        return (chw.astype(np.float32) / 255.0).astype(out_dtype)
    return chw.astype(out_dtype)


def _affine_pil(img: Image.Image, coeffs: Tuple[float, float, float, float, float, float]) -> Image.Image:
    """PIL AFFINE 逆采样系数 (a,b,c,d,e,f)：src_x = a*dst_x + b*dst_y + c。"""
    return img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BILINEAR)


def augment_shear_x(img: Image.Image, magnitude: int) -> Image.Image:
    # 水平剪切：x' = x + s*y -> 采样 src(x,y) = dst(x - s*y, y)
    s = _mag_to_value(magnitude, -0.3, 0.3)
    return _affine_pil(img, (1.0, -s, 0.0, 0.0, 1.0, 0.0))


def augment_shear_y(img: Image.Image, magnitude: int) -> Image.Image:
    # 垂直剪切：y' = y + s*x -> src(x,y) = dst(x, y - s*x)
    s = _mag_to_value(magnitude, -0.3, 0.3)
    return _affine_pil(img, (1.0, 0.0, 0.0, -s, 1.0, 0.0))


# 水平/垂直平移：强度 0–9 线性映射到 [-20%*边长, +20%*边长]（相对当前图宽高）
_TRANSLATE_MAX_FRAC = 0.2


def augment_translate_x(img: Image.Image, magnitude: int) -> Image.Image:
    w, _h = img.size
    lim = _TRANSLATE_MAX_FRAC * float(w)
    tx = _mag_to_value(magnitude, -lim, lim)
    return _affine_pil(img, (1.0, 0.0, -tx, 0.0, 1.0, 0.0))


def augment_translate_y(img: Image.Image, magnitude: int) -> Image.Image:
    _w, h = img.size
    lim = _TRANSLATE_MAX_FRAC * float(h)
    ty = _mag_to_value(magnitude, -lim, lim)
    return _affine_pil(img, (1.0, 0.0, 0.0, 0.0, 1.0, -ty))


def augment_rotate(img: Image.Image, magnitude: int) -> Image.Image:
    deg = _mag_to_value(magnitude, -30.0, 30.0)
    return img.rotate(
        float(deg),
        resample=Image.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def augment_autocontrast(img: Image.Image, magnitude: int) -> Image.Image:
    _ = magnitude
    return ImageOps.autocontrast(img)


def augment_invert(img: Image.Image, magnitude: int) -> Image.Image:
    _ = magnitude
    return ImageOps.invert(img)


def augment_equalize(img: Image.Image, magnitude: int) -> Image.Image:
    _ = magnitude
    return ImageOps.equalize(img)


def augment_solarize(img: Image.Image, magnitude: int) -> Image.Image:
    thr = int(round(_mag_to_value(magnitude, 0.0, 256.0)))
    thr = int(np.clip(thr, 0, 256))
    return ImageOps.solarize(img, threshold=thr)


def augment_posterize(img: Image.Image, magnitude: int) -> Image.Image:
    bits = int(round(_mag_to_value(magnitude, 4.0, 8.0)))
    bits = int(np.clip(bits, 4, 8))
    return ImageOps.posterize(img, bits)


def augment_contrast(img: Image.Image, magnitude: int) -> Image.Image:
    factor = _mag_to_value(magnitude, 0.1, 1.9)
    return ImageEnhance.Contrast(img).enhance(factor)


def augment_color(img: Image.Image, magnitude: int) -> Image.Image:
    factor = _mag_to_value(magnitude, 0.1, 1.9)
    return ImageEnhance.Color(img).enhance(factor)


def augment_brightness(img: Image.Image, magnitude: int) -> Image.Image:
    factor = _mag_to_value(magnitude, 0.1, 1.9)
    return ImageEnhance.Brightness(img).enhance(factor)


def augment_sharpness(img: Image.Image, magnitude: int) -> Image.Image:
    factor = _mag_to_value(magnitude, 0.1, 1.9)
    return ImageEnhance.Sharpness(img).enhance(factor)


_NAME_TO_OP: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "shearx": augment_shear_x,
    "sheary": augment_shear_y,
    "translatex": augment_translate_x,
    "translatey": augment_translate_y,
    "rotate": augment_rotate,
    "autocontrast": augment_autocontrast,
    "invert": augment_invert,
    "equalize": augment_equalize,
    "solarize": augment_solarize,
    "posterize": augment_posterize,
    "contrast": augment_contrast,
    "color": augment_color,
    "brightness": augment_brightness,
    "sharpness": augment_sharpness,
}


def apply_random_policy_augment(image: np.ndarray,flag: bool = False) -> np.ndarray:
    """
    从 AUGMENTATION_POLICIES 中随机选一条子策略，对图像依次尝试两次增强（各带自身概率）。

    参数
    ------
    image : ndarray
        shape (3, H, W)，dtype 建议 uint8 或 float32（float 按 [0,1] 或 [0,255] 自动处理）。

    返回
    ------
    ndarray
        与输入相同的 shape 与 dtype（在可表示范围内）。
    """
    if not flag:
        return image

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"期望 shape (3, H, W)，得到 {getattr(image, 'shape', None)}")

    in_dtype = image.dtype
    pil = _chw_numpy_to_pil_rgb(image)

    p1, name1, m1, p2, name2, m2 = random.choice(AUGMENTATION_POLICIES)
    # print("___________________________________________")
    # print(p1, name1, m1, p2, name2, m2)
    # print("___________________________________________")
    def maybe_apply(prob: float, op_name: str, mag: int) -> None:
        nonlocal pil
        if random.random() > prob:
            return
        key = str(op_name).lower()
        fn = _NAME_TO_OP.get(key)
        if fn is None:
            raise KeyError(f"未知增强: {op_name}")
        pil = fn(pil, int(mag))

    maybe_apply(float(p1), name1, int(m1))
    maybe_apply(float(p2), name2, int(m2))

    return _pil_rgb_to_chw_numpy(pil, in_dtype)


__all__ = [
    "AUGMENTATION_POLICIES",
    "apply_random_policy_augment",
]
