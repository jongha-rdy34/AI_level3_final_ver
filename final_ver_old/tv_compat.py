"""torchvision이 없는 PC에서만 쓰는 연습용 대체 모듈.

시험 환경에는 torchvision이 설치되어 있을 가능성이 높습니다. 노트북은 먼저
`from torchvision.transforms import v2 as T`를 시도하고, 실패할 때만 이 모듈을 씁니다.
이름·인자·동작을 torchvision.transforms.v2의 자주 쓰는 부분과 같게 맞췄으므로
여기서 연습한 코드를 시험에서 그대로 쓸 수 있습니다. (공식 구현이 아닙니다.)

지원: Compose, ToImage(numpy·PIL), ToDtype, Resize, CenterCrop, RandomResizedCrop, RandomAffine(fill),
      RandomHorizontalFlip, RandomVerticalFlip, ColorJitter(brightness, contrast), Normalize,
      RandomApply, Lambda
"""
import math
import random

import numpy as np
import torch


class _Namespace:
    pass


def _as_tensor_image(img):
    if not isinstance(img, (np.ndarray, torch.Tensor)) and hasattr(img, "mode"):   # PIL 이미지
        img = np.asarray(img.convert("RGB") if img.mode not in ("RGB", "L") else img)
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[..., None]
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # HWC → CHW
    return img


class Compose:
    """여러 변환을 순서대로 적용합니다. Compose([t1, t2, ...])"""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        inner = "\n".join(f"    {t!r}" for t in self.transforms)
        return f"Compose(\n{inner}\n)"


class ToImage:
    """numpy (H,W,C) 배열 또는 텐서 → (C,H,W) 텐서. 자료형은 바꾸지 않습니다(uint8 그대로)."""

    def __call__(self, img):
        return _as_tensor_image(img)

    def __repr__(self):
        return "ToImage()"


class ToDtype:
    """자료형 변환. scale=True이고 정수→실수이면 [0, 255] → [0, 1]로 나눕니다."""

    def __init__(self, dtype, scale=False):
        self.dtype, self.scale = dtype, scale

    def __call__(self, img):
        img = _as_tensor_image(img)
        if self.scale and not img.is_floating_point() and self.dtype.is_floating_point:
            return img.to(self.dtype) / 255.0
        return img.to(self.dtype)

    def __repr__(self):
        return f"ToDtype(dtype={self.dtype}, scale={self.scale})"


class RandomHorizontalFlip:
    """확률 p로 좌우 반전(마지막 축)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _as_tensor_image(img)
        return img.flip(-1) if torch.rand(1).item() < self.p else img

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """확률 p로 상하 반전(끝에서 두 번째 축)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _as_tensor_image(img)
        return img.flip(-2) if torch.rand(1).item() < self.p else img

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


def _range(v, name):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v < 0:
            raise ValueError(f"{name}은 0 이상이어야 합니다.")
        return (max(0.0, 1 - v), 1 + v)
    lo, hi = v
    return (float(lo), float(hi))


class ColorJitter:
    """밝기·대비를 무작위로 바꿉니다(이 대체 모듈은 saturation·hue를 지원하지 않음).

    brightness: float b → 배율 U[max(0,1-b), 1+b], 또는 (min, max) 튜플.
    실수 텐서는 [0,1], uint8 텐서는 [0,255]로 자릅니다.
    """

    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        if saturation is not None or hue is not None:
            raise NotImplementedError("tv_compat은 saturation·hue를 지원하지 않습니다(torchvision에서는 지원).")
        self.brightness, self.contrast = _range(brightness, "brightness"), _range(contrast, "contrast")

    def __call__(self, img):
        img = _as_tensor_image(img)
        is_float = img.is_floating_point()
        x = img.float() if not is_float else img
        top = 1.0 if is_float else 255.0
        order = [k for k in ("brightness", "contrast") if getattr(self, k) is not None]
        random.shuffle(order)
        for k in order:
            lo, hi = getattr(self, k)
            f = float(torch.empty(1).uniform_(lo, hi))
            if k == "brightness":
                x = (x * f).clamp(0, top)
            else:
                gray_mean = x.mean()
                x = ((x - gray_mean) * f + gray_mean).clamp(0, top)
        return x if is_float else x.round().to(img.dtype)

    def __repr__(self):
        return f"ColorJitter(brightness={self.brightness}, contrast={self.contrast})"


class Normalize:
    """채널별 (x - mean) / std. 입력은 실수 텐서 (C,H,W) 또는 (B,C,H,W)."""

    def __init__(self, mean, std):
        self.mean, self.std = list(mean), list(std)

    def __call__(self, img):
        if not img.is_floating_point():
            raise TypeError("Normalize는 실수 텐서에만 적용합니다. 먼저 ToDtype(torch.float32, scale=True)를 쓰세요.")
        shape = (-1, 1, 1)
        m = torch.tensor(self.mean, dtype=img.dtype).view(shape)
        s = torch.tensor(self.std, dtype=img.dtype).view(shape)
        return (img - m) / s

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class RandomApply:
    """확률 p로 transforms 목록 전체를 적용합니다."""

    def __init__(self, transforms, p=0.5):
        self.transforms, self.p = list(transforms), p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            for t in self.transforms:
                img = t(img)
        return img


class Lambda:
    """임의의 함수를 변환으로 씁니다."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, img):
        return self.fn(img)


def _resize(x, size, mode="bilinear"):
    xf = x.float().unsqueeze(0)
    out = torch.nn.functional.interpolate(xf, size=size, mode=mode, align_corners=False, antialias=mode in ("bilinear", "bicubic"))[0]
    if not x.is_floating_point():
        return out.round().clamp(0, 255).to(x.dtype)
    return out


class Resize:
    """(h, w) 또는 정수(짧은 변 기준이 아니라 이 대체 모듈에서는 정사각형)로 크기 변경."""

    def __init__(self, size, interpolation="bilinear", antialias=True):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.mode = interpolation if isinstance(interpolation, str) else "bilinear"

    def __call__(self, img):
        return _resize(_as_tensor_image(img), self.size, self.mode)

    def __repr__(self):
        return f"Resize(size={self.size})"


class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        img = _as_tensor_image(img)
        H, W = img.shape[-2:]
        h, w = self.size
        t, l = (H - h) // 2, (W - w) // 2
        return img[..., t:t + h, l:l + w]


class RandomResizedCrop:
    """원본 넓이의 scale 비율, 가로세로비 ratio 범위로 무작위 영역을 잘라 size로 키움."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), antialias=True):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale, self.ratio = scale, ratio

    def __call__(self, img):
        img = _as_tensor_image(img)
        H, W = img.shape[-2:]
        for _ in range(10):
            area = H * W * float(torch.empty(1).uniform_(*self.scale))
            logr = torch.empty(1).uniform_(math.log(self.ratio[0]), math.log(self.ratio[1]))
            ar = math.exp(float(logr))
            w, h = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
            if 0 < w <= W and 0 < h <= H:
                t = int(torch.randint(0, H - h + 1, (1,))); l = int(torch.randint(0, W - w + 1, (1,)))
                return _resize(img[..., t:t + h, l:l + w], self.size)
        return _resize(img, self.size)

    def __repr__(self):
        return f"RandomResizedCrop(size={self.size}, scale={self.scale})"


class RandomAffine:
    """회전(degrees)·이동(translate 비율)·배율(scale 범위). 빈 곳은 fill 값으로 채움.

    상품 사진처럼 배경이 흰색이면 fill=255(정수 이미지) 또는 1.0(실수 이미지)을 줍니다. 기본 0은 검정.
    """

    def __init__(self, degrees=0, translate=None, scale=None, fill=0):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else tuple(degrees)
        self.translate, self.scale, self.fill = translate, scale, fill

    def __call__(self, img):
        img = _as_tensor_image(img)
        is_float = img.is_floating_point()
        x = img.float().unsqueeze(0)
        H, W = x.shape[-2:]
        ang = math.radians(float(torch.empty(1).uniform_(*self.degrees)))
        s = float(torch.empty(1).uniform_(*self.scale)) if self.scale else 1.0
        tx = float(torch.empty(1).uniform_(-self.translate[0], self.translate[0])) * 2 if self.translate else 0.0
        ty = float(torch.empty(1).uniform_(-self.translate[1], self.translate[1])) * 2 if self.translate else 0.0
        c, si = math.cos(ang) / s, math.sin(ang) / s
        theta = torch.tensor([[c, -si, -(c * tx - si * ty)], [si, c, -(si * tx + c * ty)]], dtype=torch.float32).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        out = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        inside = torch.nn.functional.grid_sample(torch.ones_like(x[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out = out * inside + float(self.fill) * (1 - inside)
        out = out[0]
        return out if is_float else out.round().clamp(0, 255).to(img.dtype)

    def __repr__(self):
        return f"RandomAffine(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, fill={self.fill})"


transforms_v2 = _Namespace()
for _name in ["Compose", "ToImage", "ToDtype", "Resize", "CenterCrop", "RandomResizedCrop", "RandomAffine",
              "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Normalize", "RandomApply", "Lambda"]:
    setattr(transforms_v2, _name, globals()[_name])
