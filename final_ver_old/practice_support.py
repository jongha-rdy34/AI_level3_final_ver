"""[claude] Lv3 모의 v3 제공 도우미 (창작 연습용, 공식 도우미 아님).

- 데이터 경로·표 읽기, 육안 검사 격자, attempt, Timer
- 지정 모델: mini_tinyvit.create_model / resolve_data_config (timm과 같은 이름)
- Q5 모의 응답(mock_llm): 지정 모델은 이미지 모델이라 채팅이 안 되므로 미리 만든 응답을 씁니다
- 제출 형식 점검(check_submission) — submission_check.py (torch 없이 동작)
- ready_*: 앞 소문항을 못 풀었을 때만 쓰는 진행용 대체물(해당 소문항 평가와 무관)
"""
from pathlib import Path
import hashlib
import json
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mini_tinyvit import create_model, resolve_data_config  # noqa: F401  (timm.create_model 역할)
from submission_check import GUIDE, check_submission  # noqa: F401

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SUBMIT = ROOT / "제출물"
CLASSES = ["cap", "beanie", "tshirt", "shirt", "hoodie", "jacket", "dress", "skirt", "pants", "shorts",
           "sneakers", "boots", "sandals", "backpack", "handbag", "wallet", "watch", "sunglasses", "socks", "scarf"]
CLASSES_KO = ["볼캡", "비니", "반팔티", "셔츠", "후드티", "자켓", "원피스", "스커트", "긴바지", "반바지",
              "스니커즈", "부츠", "샌들", "백팩", "핸드백", "지갑", "시계", "선글라스", "양말", "머플러"]


def get_transforms_module():
    """torchvision.transforms.v2가 있으면 그것을, 없으면 같은 이름의 연습용 대체를 돌려줍니다."""
    try:
        from torchvision.transforms import v2 as T
        return T
    except ImportError:
        from tv_compat import transforms_v2 as T
        return T


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)


def attempt(name, fn, *args, **kwargs):
    """fn을 실행하고, 실패하면 오류만 보여 준 뒤 None을 돌려줍니다."""
    try:
        out = fn(*args, **kwargs)
        print(f"[{name}] 실행 완료")
        return out
    except Exception as exc:  # 연습용: 다음 문항으로 넘어갈 수 있게 모든 오류를 잡음
        print(f"[{name}] 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return None


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"[{self.name}] {time.time() - self.t0:.1f}초")


def read_table(name):
    return pd.read_csv(DATA / name, dtype={"id": str})


def show_images(images, titles=None, ncols=8, size=1.3):
    """육안 검사용 격자. images: PIL 이미지 또는 (H,W,3) uint8 배열의 리스트"""
    import matplotlib.pyplot as plt
    n = len(images)
    if n == 0:
        print("표시할 이미지가 없습니다.")
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            im = images[i]
            ax.imshow(im if not hasattr(im, "mode") or im.mode in ("RGB", "RGBA") else im.convert("RGB"))
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=7)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 진행용 대체물
def ready_normalize_label(s):
    if not isinstance(s, str):
        return None
    key = s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return key if key in CLASSES else None


def ready_load_rgb(path):
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            return bg
        return im.convert("RGB")


def ready_pad_to_square(img, fill=(255, 255, 255)):
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def ready_clean_table():
    """Q1 정답과 같은 기준으로 정제한 학습 표 (id, file, label)."""
    return read_table("_ready_clean.csv")


def ready_transforms(train, cfg):
    T = get_transforms_module()
    steps = [T.ToImage()]
    if train:
        steps += [T.RandomHorizontalFlip(0.5), T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.55, 1.0), fill=255)]
    steps += [T.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [T.ColorJitter(brightness=0.2, contrast=0.2)]
    steps += [T.Normalize(mean=list(cfg["mean"]), std=list(cfg["std"]))]
    return T.Compose(steps)


def preload(files, size, load_fn=ready_load_rgb, pad_fn=ready_pad_to_square):
    """파일 목록 → (N, size, size, 3) uint8 (정사각형으로 채운 뒤 크기 변경)."""
    return np.stack([np.asarray(pad_fn(load_fn(DATA / f)).resize((size, size), Image.BICUBIC)) for f in files])


class ReadyProductDataset(Dataset):
    def __init__(self, files, labels, transform, size, images=None):
        self.images = preload(files, size) if images is None else images
        self.transform = transform
        self.y = None if labels is None else [CLASSES.index(c) for c in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        y = -1 if self.y is None else self.y[i]
        return self.transform(self.images[i]), torch.tensor(y, dtype=torch.long)


def ready_train_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    counts = pd.Series(labels).value_counts()
    w = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
    return DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(w, len(w), True, generator=g))


def zoom_out(images, scale=0.5, fill=255):
    """(N,S,S,3) uint8 → 물체를 scale배로 줄여 가운데 둔 사진. test의 '작게 찍힌 사진'을 흉내 낸 검증용."""
    out = np.full_like(images, fill)
    S = images.shape[1]
    s = int(round(S * scale)); o = (S - s) // 2
    for i, im in enumerate(images):
        out[i, o:o + s, o:o + s] = np.asarray(Image.fromarray(im).resize((s, s), Image.BICUBIC))
    return out


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- Q5 모의 응답
_MOCK = None


def mock_llm(text_id):
    """**모의 응답**(미리 만든 문자열). 실제 언어모델 호출이 아니며 프롬프트와 무관합니다.

    FAQ: 지정 모델은 이미지 모델이라 채팅이 안 됩니다. 그래서 '프롬프트 작성 + 응답 파싱·정규화·검증'을
    연습하고, 응답은 실제 LLM에서 흔한 형식 오류를 섞어 미리 만들었습니다.
    """
    global _MOCK
    if _MOCK is None:
        _MOCK = json.loads((DATA / "texts" / "mock_llm_outputs.json").read_text(encoding="utf-8"))
    return _MOCK[text_id]


def check_prompt(prompt, text, schema_keys):
    """프롬프트에 필수 요소가 들어 있는지만 봅니다(품질 채점이 아님). 빠진 항목 리스트를 반환."""
    missing = []
    if text not in prompt:
        missing.append("상품 설명 원문 포함")
    for k in schema_keys:
        if k not in prompt:
            missing.append(f"필드 '{k}'")
    for word, label in [("JSON", "JSON 출력 지시"), ("null", "미확정 정보 → null 규칙"), ("원", "가격 단위(원) 규칙")]:
        if word not in prompt:
            missing.append(label)
    if not any(c in prompt for c in CLASSES):
        missing.append("category 허용값 목록")
    return missing


def save_jsonl(records, name, folder=None):
    folder = SUBMIT if folder is None else Path(folder)
    folder.mkdir(exist_ok=True)
    path = folder / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("저장:", path.relative_to(ROOT), len(records), "줄")
    return path
