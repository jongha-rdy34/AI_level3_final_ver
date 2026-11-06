"""Kaggle 원본 zip -> 공식 샘플 노트북이 읽는 형식으로 변환합니다.

입력 (setup.cmd가 받음)
  raw/fashion.zip  : paramaggarwal/fashion-product-images-small (60x80 상품 사진 + styles.csv)
  raw/utkface.zip  : jangedoo/utkface-new (UTKFace 200x200 얼굴, 파일명 맨 앞 숫자가 나이)

출력
  data/fashion_imbalance/prepared.npz, classes.json     (P1_1: 6종, 일부 클래스만 아주 적게)
  data/fashion_finetune/prepared.npz, classes.json,     (P2_1, P2_2: 20종)
                        p2_2_train_indices.npy, test/*.png
  test.csv                                              (P2_2: id,answer / answer 비어 있음)
  data/utkface/train.csv, val.csv, train/, val/         (P1_2: 112x112, train 정면 / val 카메라 각도 변형)
  _grading/p2_2_test_truth.csv                          (로컬 채점용 정답 - 노트북에서 읽지 않음)

공식 시험의 정확한 클래스 구성·분할은 공개되지 않았습니다. 형식과 '함정'만 재현한 버전입니다.
실행:  .venv\\Scripts\\python prepare_data.py
"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
DATA = ROOT / 'data'
SEED = 42
H, W = 80, 60

# P1_1: 다수 4종 + 소수 2종 (소수 클래스가 다수 클래스와 헷갈리는 쌍이 되도록 고름)
P11_CLASSES = ['Tshirts', 'Shirts', 'Casual Shoes', 'Sports Shoes', 'Handbags', 'Watches']
P11_MINORITY = {'Shirts': 40, 'Sports Shoes': 40}
P11_MAJORITY = 1200
P11_VAL = 200

N_CLASSES = 20
VAL_PER_CLASS = 125      # 20 x 125 = 2,500 (P2_2 안내와 같음)
TEST_PER_CLASS = 125
TRAIN_CAP = 1000         # P2_1 학습 상한 (클래스당)
P22_PER_CLASS = 600      # P2_2: 클래스당 600, 부족하면 반복

UTK_TRAIN, UTK_VAL = 6000, 1500
UTK_SIZE = 112


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- fashion
def open_fashion():
    z = zipfile.ZipFile(RAW / 'fashion.zip')
    names = z.namelist()
    styles = [n for n in names if n.endswith('styles.csv')]
    styles.sort(key=len)
    text = z.read(styles[0]).decode('utf-8', errors='replace')
    rows = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    idx_id, idx_type = header.index('id'), header.index('articleType')
    for r in reader:
        if len(r) < len(header):
            continue
        rows.append((r[idx_id].strip(), r[idx_type].strip()))
    images = {}
    for n in names:
        m = re.search(r'(?:^|/)images/(\d+)\.jpg$', n)
        if m and m.group(1) not in images:
            images[m.group(1)] = n
    return z, rows, images


def load_rgb(z, member):
    with Image.open(io.BytesIO(z.read(member))) as im:
        im = im.convert('RGB')
        if im.size != (W, H):
            im = im.resize((W, H), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def build_fashion():
    z, rows, images = open_fashion()
    by_type = {}
    for pid, t in rows:
        if pid in images:
            by_type.setdefault(t, []).append(pid)
    counts = sorted(by_type.items(), key=lambda kv: -len(kv[1]))
    classes20 = [t for t, _ in counts[:N_CLASSES]]
    log('20 classes:', [(t, len(by_type[t])) for t in classes20])
    rng = np.random.default_rng(SEED)
    cache = {}

    def img(pid):
        if pid not in cache:
            cache[pid] = load_rgb(z, images[pid])
        return cache[pid]

    # ---- P2: fashion_finetune
    out = DATA / 'fashion_finetune'
    (out / 'test').mkdir(parents=True, exist_ok=True)
    tr_x, tr_y, va_x, va_y, te = [], [], [], [], []
    used = set()
    for label, t in enumerate(classes20):
        ids = list(by_type[t])
        rng.shuffle(ids)
        va, tst, rest = ids[:VAL_PER_CLASS], ids[VAL_PER_CLASS:VAL_PER_CLASS + TEST_PER_CLASS], ids[VAL_PER_CLASS + TEST_PER_CLASS:]
        rest = rest[:TRAIN_CAP]
        tr_x += [img(p) for p in rest]; tr_y += [label] * len(rest)
        va_x += [img(p) for p in va]; va_y += [label] * len(va)
        te += [(p, label) for p in tst]
        used.update(ids)
    tr_x, tr_y = np.stack(tr_x), np.asarray(tr_y, dtype=np.int64)
    np.savez_compressed(out / 'prepared.npz', train_images=tr_x, train_labels=tr_y,
                        val_images=np.stack(va_x), val_labels=np.asarray(va_y, dtype=np.int64))
    (out / 'classes.json').write_text(json.dumps(classes20, ensure_ascii=False, indent=1), encoding='utf-8')
    p22 = []
    for label in range(N_CLASSES):
        pos = np.flatnonzero(tr_y == label)
        p22.append(rng.choice(pos, P22_PER_CLASS, replace=len(pos) < P22_PER_CLASS))
    p22 = np.concatenate(p22)
    rng.shuffle(p22)
    np.save(out / 'p2_2_train_indices.npy', p22.astype(np.int64))
    order = rng.permutation(len(te))
    (ROOT / '_grading').mkdir(exist_ok=True)
    with open(ROOT / 'test.csv', 'w', encoding='utf-8', newline='') as f1, \
         open(ROOT / '_grading' / 'p2_2_test_truth.csv', 'w', encoding='utf-8', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['id', 'answer']); w2.writerow(['id', 'answer'])
        for k, j in enumerate(order):
            pid, label = te[j]
            name = f'{k:05d}.png'
            Image.fromarray(img(pid)).save(out / 'test' / name)
            w1.writerow([name, '']); w2.writerow([name, label])
    log(f'fashion_finetune: train {len(tr_y)}, val {len(va_y)}, test {len(te)}, p2_2 idx {len(p22)}')

    # ---- P1_1: fashion_imbalance (P2와 겹치지 않아도 되지만 val은 따로 뽑음)
    out = DATA / 'fashion_imbalance'
    out.mkdir(parents=True, exist_ok=True)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for label, t in enumerate(P11_CLASSES):
        ids = list(by_type[t])
        rng.shuffle(ids)
        va = ids[:P11_VAL]
        n = P11_MINORITY.get(t, P11_MAJORITY)
        tr = ids[P11_VAL:P11_VAL + n]
        tr_x += [img(p) for p in tr]; tr_y += [label] * len(tr)
        va_x += [img(p) for p in va]; va_y += [label] * len(va)
    perm = rng.permutation(len(tr_y))
    np.savez_compressed(out / 'prepared.npz', train_images=np.stack(tr_x)[perm],
                        train_labels=np.asarray(tr_y, dtype=np.int64)[perm],
                        val_images=np.stack(va_x), val_labels=np.asarray(va_y, dtype=np.int64))
    (out / 'classes.json').write_text(json.dumps(P11_CLASSES, ensure_ascii=False, indent=1), encoding='utf-8')
    log('fashion_imbalance train counts:', np.bincount(tr_y).tolist(), 'val', len(va_y))


# ---------------------------------------------------------------- utkface
def camera_like(im, rng):
    """정면 사진 -> 설치 카메라처럼: 기울기, 위치 이동, 크기 변화, 좌우, 밝기."""
    from PIL import ImageEnhance
    angle = float(rng.uniform(-30, 30))
    im = im.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(124, 116, 104))
    s = im.size[0]
    scale = float(rng.uniform(0.78, 1.0))
    crop = int(s * scale)
    max_off = int(s * 0.12)
    cx = s // 2 + int(rng.integers(-max_off, max_off + 1))
    cy = s // 2 + int(rng.integers(-max_off, max_off + 1))
    box = (cx - crop // 2, cy - crop // 2, cx - crop // 2 + crop, cy - crop // 2 + crop)
    im = im.crop(box)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.7, 1.3)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.8, 1.2)))
    return im.resize((UTK_SIZE, UTK_SIZE), Image.BILINEAR)


def build_utkface():
    z = zipfile.ZipFile(RAW / 'utkface.zip')
    members = []
    for n in z.namelist():
        if not n.startswith('UTKFace/') or not n.lower().endswith('.jpg'):
            continue
        m = re.match(r'(\d+)_(\d)_(\d)_\d+', Path(n).name)
        if m and 1 <= int(m.group(1)) <= 100:
            members.append((n, int(m.group(1))))
    members.sort()
    rng = np.random.default_rng(SEED)
    rng.shuffle(members)
    log('UTKFace usable:', len(members))
    out = DATA / 'utkface'
    for split, part in (('train', members[:UTK_TRAIN]), ('val', members[UTK_TRAIN:UTK_TRAIN + UTK_VAL])):
        (out / split).mkdir(parents=True, exist_ok=True)
        with open(out / f'{split}.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'label'])
            for k, (n, age) in enumerate(part):
                with Image.open(io.BytesIO(z.read(n))) as im:
                    im = im.convert('RGB')
                    if split == 'train':
                        im = im.resize((UTK_SIZE, UTK_SIZE), Image.BILINEAR)
                    else:
                        im = camera_like(im, rng)
                    name = f'{split}/{k:06d}.png'
                    im.save(out / name)
                w.writerow([name, age])
        log(f'utkface {split}: {len(part)}')


if __name__ == '__main__':
    which = sys.argv[1:] or ['fashion', 'utkface']
    if 'fashion' in which:
        build_fashion()
    if 'utkface' in which:
        build_utkface()
    log('done')
