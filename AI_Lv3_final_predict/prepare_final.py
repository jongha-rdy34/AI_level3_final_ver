"""[예측 문제] 데이터 생성: ../AI_Lv3_sample_solution/raw/fashion.zip 을 읽어 만듭니다.

  data/f1_1_noisy/    prepared.npz, classes.json   10종, 라벨 오류(중복 사본에 다른 라벨) + 손상 이미지 + 같은 이미지 반복
  data/f1_2_shift/    prepared.npz, classes.json   20종, train 깨끗 / val 은 매장·앱 사진처럼 축소·이동·회전·조명·흑백
  data/f2_1_longtail/ prepared.npz, classes.json   20종, 긴 꼬리 불균형(800 -> 30), val 균형
  data/f2_2_kaggle/   prepared.npz, classes.json, test/*.png   20종 400장씩, test 절반은 분포 이동
  test.csv, _grading/f2_2_test_truth.csv
실행: .venv\\Scripts\\python prepare_final.py
"""
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
RAW_CANDIDATES = [ROOT / 'raw' / 'fashion.zip', ROOT.parent / 'AI_Lv3_sample_solution' / 'raw' / 'fashion.zip']
DATA = ROOT / 'data'
H, W = 80, 60
SEED = 7


def log(*a):
    print(*a, flush=True)


def open_fashion():
    path = next(p for p in RAW_CANDIDATES if p.exists())
    z = zipfile.ZipFile(path)
    names = z.namelist()
    text = z.read(sorted([n for n in names if n.endswith('styles.csv')], key=len)[0]).decode('utf-8', errors='replace')
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    i_id, i_t = header.index('id'), header.index('articleType')
    rows = [(r[i_id].strip(), r[i_t].strip()) for r in reader if len(r) >= len(header)]
    images = {}
    for n in names:
        m = re.search(r'(?:^|/)images/(\d+)\.jpg$', n)
        if m and m.group(1) not in images:
            images[m.group(1)] = n
    by_type = {}
    for pid, t in rows:
        if pid in images:
            by_type.setdefault(t, []).append(pid)
    return z, images, by_type


def to_array(im):
    im = im.convert('RGB')
    if im.size != (W, H):
        im = im.resize((W, H), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def shift_like_store(arr, rng):
    """깨끗한 상품컷 -> 앱 업로드/매장 사진처럼: 작게, 한쪽으로, 기울게, 조명·흑백."""
    im = Image.fromarray(arr)
    scale = float(rng.uniform(0.5, 0.8))
    small = im.resize((max(8, int(W * scale)), max(8, int(H * scale))), Image.BICUBIC)
    small = small.rotate(float(rng.uniform(-15, 15)), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    small.thumbnail((W, H))
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    x = int(rng.integers(0, W - small.size[0] + 1))
    y = int(rng.integers(0, H - small.size[1] + 1))
    canvas.paste(small, (x, y))
    canvas = ImageEnhance.Brightness(canvas).enhance(float(rng.uniform(0.65, 1.15)))
    canvas = ImageEnhance.Contrast(canvas).enhance(float(rng.uniform(0.7, 1.1)))
    if rng.random() < 0.25:
        canvas = ImageOps.grayscale(canvas).convert('RGB')
    return np.asarray(canvas, dtype=np.uint8)


def save_npz(out, classes, tr_x, tr_y, va_x, va_y):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'prepared.npz', train_images=np.asarray(tr_x, dtype=np.uint8),
                        train_labels=np.asarray(tr_y, dtype=np.int64),
                        val_images=np.asarray(va_x, dtype=np.uint8), val_labels=np.asarray(va_y, dtype=np.int64))
    (out / 'classes.json').write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    z, images, by_type = open_fashion()
    classes20 = [t for t, _ in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:20]]
    rng = np.random.default_rng(SEED)
    pools = {}
    for t in classes20:
        ids = list(by_type[t]); rng.shuffle(ids); pools[t] = ids
    cache = {}

    def img(pid):
        if pid not in cache:
            with Image.open(io.BytesIO(z.read(images[pid]))) as im:
                cache[pid] = to_array(im)
        return cache[pid]

    # 각 클래스 풀을 문항별로 겹치지 않게 나눔: [0:100] val공통, [100:200] test, 나머지 train용
    VAL, TEST = 100, 100

    # ---------------- F1_1 noisy (10종)
    cls10 = ['Tshirts', 'Shirts', 'Casual Shoes', 'Sports Shoes', 'Watches', 'Handbags', 'Heels', 'Sandals', 'Flip Flops', 'Kurtas']
    tr_x, tr_y, va_x, va_y = [], [], [], []
    clean_ids = {}
    for label, t in enumerate(cls10):
        ids = pools[t]
        va = ids[:VAL]
        tr = ids[TEST + VAL: TEST + VAL + 500]
        clean_ids[t] = tr
        tr_x += [img(p) for p in tr]; tr_y += [label] * len(tr)
        va_x += [img(p) for p in va]; va_y += [label] * len(va)
    # (a) 협력사 업로드 오류: 신발·상의 사진의 '사본'이 엉뚱한 라벨로 들어감 (각 원본은 정상 라벨로도 존재)
    wrong_map = {'Sandals': 'Flip Flops', 'Flip Flops': 'Sandals', 'Shirts': 'Tshirts', 'Sports Shoes': 'Casual Shoes',
                 'Heels': 'Sandals'}
    n_wrong = 0
    for src, dst in wrong_map.items():
        picks = rng.choice(len(clean_ids[src]), size=180, replace=False)
        for k in picks:
            for _ in range(int(rng.integers(1, 3))):          # 사본 1~2개
                tr_x.append(img(clean_ids[src][k])); tr_y.append(cls10.index(dst)); n_wrong += 1
    # (b) 손상 이미지: 빈 흰 화면, 검은 화면, 잡음 (라벨은 아무거나)
    n_bad = 0
    for kind in ('white', 'black', 'noise'):
        for _ in range(60):
            if kind == 'white':
                a = np.full((H, W, 3), 255, np.uint8)
            elif kind == 'black':
                a = np.zeros((H, W, 3), np.uint8) + rng.integers(0, 6)
            else:
                a = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            tr_x.append(a.astype(np.uint8)); tr_y.append(int(rng.integers(0, 10))); n_bad += 1
    # (c) 같은 이미지 과다 반복 (정상 라벨) - 특정 상품 30장이 각 15번씩
    n_rep = 0
    for t in ('Watches', 'Handbags'):
        for k in range(15):
            for _ in range(15):
                tr_x.append(img(clean_ids[t][k])); tr_y.append(cls10.index(t)); n_rep += 1
    perm = rng.permutation(len(tr_y))
    save_npz(DATA / 'f1_1_noisy', cls10, np.asarray(tr_x)[perm], np.asarray(tr_y)[perm], va_x, va_y)
    log(f'f1_1: train {len(tr_y)} (wrong copies {n_wrong}, broken {n_bad}, repeats {n_rep}), val {len(va_y)}')

    # ---------------- F1_2 shift (20종)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for label, t in enumerate(classes20):
        ids = pools[t]
        tr = ids[TEST + VAL + 500: TEST + VAL + 500 + 250]
        if len(tr) < 250:
            tr = ids[TEST + VAL: TEST + VAL + 250]
        va = ids[:VAL]
        tr_x += [img(p) for p in tr]; tr_y += [label] * len(tr)
        va_x += [shift_like_store(img(p), rng) for p in va]; va_y += [label] * len(va)
    save_npz(DATA / 'f1_2_shift', classes20, tr_x, tr_y, va_x, va_y)
    log(f'f1_2: train {len(tr_y)}, val(shift) {len(va_y)}')

    # ---------------- F2_1 long-tail (20종)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    counts = np.round(800 * (30 / 800) ** (np.arange(20) / 19)).astype(int)    # 800 -> 30 지수 감소
    order = rng.permutation(20)                                                 # 어떤 클래스가 소수가 될지 섞음
    for label, t in enumerate(classes20):
        ids = pools[t]
        n = int(counts[order[label]])
        tr = ids[TEST + VAL: TEST + VAL + n]
        va = ids[:VAL]
        tr_x += [img(p) for p in tr]; tr_y += [label] * len(tr)
        va_x += [img(p) for p in va]; va_y += [label] * len(va)
    save_npz(DATA / 'f2_1_longtail', classes20, tr_x, tr_y, va_x, va_y)
    log('f2_1 train counts:', np.bincount(tr_y).tolist())

    # ---------------- F2_2 kaggle (20종, 400장씩, test 절반 이동)
    out = DATA / 'f2_2_kaggle'
    (out / 'test').mkdir(parents=True, exist_ok=True)
    tr_x, tr_y, va_x, va_y, te = [], [], [], [], []
    for label, t in enumerate(classes20):
        ids = pools[t]
        rest = ids[TEST + VAL:]
        tr = [rest[i % len(rest)] for i in range(400)]
        va = ids[:VAL]
        tr_x += [img(p) for p in tr]; tr_y += [label] * len(tr)
        va_x += [img(p) if i % 2 == 0 else shift_like_store(img(p), rng) for i, p in enumerate(va)]; va_y += [label] * len(va)
        te += [(p, label) for p in ids[VAL:VAL + TEST]]
    save_npz(out, classes20, tr_x, tr_y, va_x, va_y)
    order = rng.permutation(len(te))
    (ROOT / '_grading').mkdir(exist_ok=True)
    with open(ROOT / 'test.csv', 'w', encoding='utf-8', newline='') as f1, \
         open(ROOT / '_grading' / 'f2_2_test_truth.csv', 'w', encoding='utf-8', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['id', 'answer']); w2.writerow(['id', 'answer'])
        for k, j in enumerate(order):
            pid, label = te[j]
            arr = img(pid) if k % 2 == 0 else shift_like_store(img(pid), rng)
            name = f'{k:05d}.png'
            Image.fromarray(arr).save(out / 'test' / name)
            w1.writerow([name, '']); w2.writerow([name, label])
    log(f'f2_2: train {len(tr_y)}, val {len(va_y)} (half shifted), test {len(te)} (half shifted)')


if __name__ == '__main__':
    main()
