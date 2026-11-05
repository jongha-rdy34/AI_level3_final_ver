"""v3 데이터·가중치 생성기 (모두 합성 연습 자료).

실행: python make_data_v3.py [images|pretrain|text ...]   (인자 없으면 전부)
- images  : 상품 사진 20종 (train 이미지 파일 + 품질 문제, test, sample_submission, 정답)
- pretrain: 도형 16종으로 연습용 TinyViT 사전학습 → assets/tiny_vit_practice.pt
- text    : Q5 상품 설명 문장, 정답 레코드, 모의 응답
"""
import hashlib
import io
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRUTH = ROOT / "정답_채점용"
sys.path.insert(0, str(ROOT))

CLASSES = ["cap", "beanie", "tshirt", "shirt", "hoodie", "jacket", "dress", "skirt", "pants", "shorts",
           "sneakers", "boots", "sandals", "backpack", "handbag", "wallet", "watch", "sunglasses", "socks", "scarf"]
PALETTE = [(30, 30, 30), (235, 235, 235), (128, 128, 128), (200, 40, 40), (40, 80, 200), (40, 150, 70),
           (220, 200, 160), (130, 80, 40), (30, 40, 90), (230, 130, 170), (240, 180, 40), (110, 60, 160)]


# ======================================================================  그리기 도구
def _jit(rng, c, s=25):
    return tuple(int(np.clip(v + rng.integers(-s, s + 1), 0, 255)) for v in c)


def _shade(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


def _colors(rng):
    c1 = _jit(rng, PALETTE[rng.integers(len(PALETTE))])
    c2 = _shade(c1, 0.55) if sum(c1) > 200 else _shade(c1, 1.8) if sum(c1) < 150 else _shade(c1, 0.6)
    if sum(c1) < 120:
        c2 = (110, 110, 110)
    return c1 + (255,), c2 + (255,)


def _j(rng, pts, s=6):
    """다각형 꼭짓점을 조금씩 흔들어 개체마다 모양이 다르게."""
    return [(x + rng.uniform(-s, s), y + rng.uniform(-s, s)) for x, y in pts]


def draw_product(cls, rng):
    """256×256 투명 캔버스(RGBA)에 상품 실루엣을 그린다."""
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c1, c2 = _colors(rng)
    J = lambda pts, s=6: _j(rng, pts, s)
    if cls == "cap":
        d.pieslice([40, 60, 200, 230], 180, 360, fill=c1)
        d.ellipse([150, 128, 250, 160], fill=c2)
        d.ellipse([112, 52, 128, 68], fill=c2)
    elif cls == "beanie":
        d.pieslice([60, 40, 196, 250], 180, 360, fill=c1)
        d.rectangle([54, 138, 202, 185], fill=c2)
        for x in range(64, 200, 14):
            d.line([x, 140, x, 183], fill=_shade(c2, 0.8), width=3)
        d.ellipse([110, 18, 146, 54], fill=c2)
    elif cls == "tshirt":
        d.polygon(J([(90, 45), (166, 45), (218, 88), (196, 118), (170, 98), (170, 222), (86, 222), (86, 98), (60, 118), (38, 88)]), fill=c1)
        d.arc([106, 30, 150, 66], 0, 180, fill=c2, width=6)
    elif cls == "shirt":
        d.polygon(J([(90, 45), (166, 45), (222, 185), (200, 194), (170, 105), (170, 222), (86, 222), (86, 105), (56, 194), (34, 185)]), fill=c1)
        d.polygon([(106, 42), (128, 72), (118, 34)], fill=c2)
        d.polygon([(150, 42), (128, 72), (138, 34)], fill=c2)
        for y in range(85, 215, 24):
            d.ellipse([124, y, 132, y + 8], fill=c2)
    elif cls == "hoodie":
        d.polygon(J([(82, 55), (174, 55), (226, 190), (204, 198), (176, 110), (176, 224), (80, 224), (80, 110), (52, 198), (30, 190)]), fill=c1)
        d.ellipse([94, 12, 162, 78], fill=c1)
        d.ellipse([110, 26, 146, 70], fill=c2)
        d.rectangle([100, 158, 156, 196], fill=c2)
        d.line([118, 70, 116, 110], fill=c2, width=3); d.line([138, 70, 140, 110], fill=c2, width=3)
    elif cls == "jacket":
        d.polygon(J([(76, 42), (180, 42), (230, 196), (206, 204), (182, 112), (182, 226), (74, 226), (74, 112), (50, 204), (26, 196)]), fill=c1)
        d.line([128, 48, 128, 226], fill=c2, width=6)
        d.polygon([(100, 42), (128, 92), (112, 42)], fill=c2); d.polygon([(156, 42), (128, 92), (144, 42)], fill=c2)
        d.rectangle([84, 170, 114, 182], fill=c2); d.rectangle([142, 170, 172, 182], fill=c2)
    elif cls == "dress":
        d.polygon(J([(104, 52), (152, 52), (158, 112), (204, 228), (52, 228), (98, 112)]), fill=c1)
        d.line([108, 52, 104, 24], fill=c1, width=6); d.line([148, 52, 152, 24], fill=c1, width=6)
        d.rectangle([98, 106, 158, 116], fill=c2)
    elif cls == "skirt":
        d.rectangle([86, 62, 170, 82], fill=c2)
        d.polygon(J([(86, 82), (170, 82), (208, 204), (48, 204)]), fill=c1)
        for x in (104, 128, 152):
            d.line([x, 86, x + (x - 128) * 0.5, 200], fill=_shade(c1, 0.85), width=3)
    elif cls == "pants":
        d.rectangle([80, 26, 176, 46], fill=c2)
        d.polygon(J([(80, 46), (176, 46), (182, 234), (140, 234), (128, 104), (116, 234), (74, 234)], 4), fill=c1)
    elif cls == "shorts":
        d.rectangle([74, 72, 182, 92], fill=c2)
        d.polygon(J([(74, 92), (182, 92), (194, 176), (142, 176), (128, 126), (114, 176), (62, 176)], 4), fill=c1)
    elif cls == "sneakers":
        d.polygon(J([(36, 150), (58, 108), (118, 98), (160, 120), (218, 140), (222, 168), (36, 168)]), fill=c1)
        d.rectangle([30, 166, 228, 190], fill=(240, 240, 240, 255))
        d.line([30, 190, 228, 190], fill=(150, 150, 150, 255), width=3)
        for x in range(84, 140, 14):
            d.ellipse([x, 112, x + 7, 119], fill=c2)
    elif cls == "boots":
        d.rectangle([70, 24, 140, 160], fill=c1)
        d.polygon(J([(70, 150), (140, 150), (212, 176), (216, 202), (70, 202)]), fill=c1)
        d.rectangle([64, 200, 220, 216], fill=c2)
        d.rectangle([70, 24, 140, 40], fill=c2)
    elif cls == "sandals":
        d.rounded_rectangle([30, 186, 226, 204], radius=8, fill=c2)
        d.arc([78, 118, 142, 214], 180, 360, fill=c1, width=12)
        d.arc([132, 128, 196, 214], 180, 360, fill=c1, width=12)
    elif cls == "backpack":
        d.arc([98, 14, 158, 74], 180, 360, fill=c2, width=9)
        d.rounded_rectangle([60, 44, 196, 226], radius=36, fill=c1)
        d.rounded_rectangle([86, 140, 170, 208], radius=14, fill=c2)
        d.line([86, 70, 86, 132], fill=c2, width=5); d.line([170, 70, 170, 132], fill=c2, width=5)
    elif cls == "handbag":
        d.arc([80, 40, 176, 160], 180, 360, fill=c2, width=11)
        d.polygon(J([(64, 106), (192, 106), (214, 214), (42, 214)]), fill=c1)
        d.rectangle([116, 112, 140, 128], fill=c2)
    elif cls == "wallet":
        d.rectangle([40, 92, 216, 190], fill=c1)
        d.polygon([(40, 92), (216, 92), (216, 128), (128, 150), (40, 128)], fill=c2)
        d.ellipse([120, 136, 136, 152], fill=(220, 200, 80, 255))
    elif cls == "watch":
        d.rectangle([108, 14, 148, 96], fill=c2); d.rectangle([108, 160, 148, 242], fill=c2)
        d.ellipse([74, 74, 182, 182], fill=c1)
        d.ellipse([88, 88, 168, 168], fill=(245, 245, 245, 255))
        d.line([128, 128, 128, 98], fill=(20, 20, 20, 255), width=4); d.line([128, 128, 150, 138], fill=(20, 20, 20, 255), width=4)
    elif cls == "sunglasses":
        d.line([14, 112, 44, 118], fill=c1, width=6); d.line([212, 118, 242, 112], fill=c1, width=6)
        d.ellipse([36, 96, 122, 162], fill=(25, 25, 35, 255), outline=c1, width=6)
        d.ellipse([134, 96, 220, 162], fill=(25, 25, 35, 255), outline=c1, width=6)
        d.arc([110, 100, 146, 130], 180, 360, fill=c1, width=6)
    elif cls == "socks":
        d.polygon(J([(100, 22), (150, 22), (150, 158), (204, 178), (210, 216), (112, 216), (100, 192)], 4), fill=c1)
        d.rectangle([100, 22, 150, 48], fill=c2)
        d.ellipse([186, 182, 214, 216], fill=c2)
    elif cls == "scarf":
        d.polygon([(42, 64), (72, 40), (214, 190), (186, 216)], fill=c1)
        for t in np.linspace(0.15, 0.85, 5):
            x, y = 57 + t * 143, 52 + t * 151
            d.line([x - 16, y + 14, x + 16, y - 14], fill=c2, width=5)
        for k in range(5):
            d.line([44 + k * 6, 66 - k * 6, 30 + k * 6, 52 - k * 6], fill=c1, width=2)
            d.line([186 + k * 6, 214 - k * 6, 196 + k * 6, 228 - k * 6], fill=c1, width=2)
    else:
        raise KeyError(cls)
    return im


SHAPES = ["circle", "square", "triangle", "star", "cross", "ring", "crescent", "hexagon", "diamond", "arrow",
          "heart", "lshape", "tshape", "ushape", "ellipse", "trapezoid"]


def draw_shape(name, rng):
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c1, c2 = _colors(rng)
    J = lambda pts, s=8: _j(rng, pts, s)
    if name == "circle": d.ellipse([48, 48, 208, 208], fill=c1)
    elif name == "square": d.rectangle([56, 56, 200, 200], fill=c1)
    elif name == "triangle": d.polygon(J([(128, 36), (224, 210), (32, 210)]), fill=c1)
    elif name == "star":
        pts = [(128 + (90 if i % 2 == 0 else 38) * math.sin(i * math.pi / 5), 132 - (90 if i % 2 == 0 else 38) * math.cos(i * math.pi / 5)) for i in range(10)]
        d.polygon(pts, fill=c1)
    elif name == "cross": d.rectangle([104, 36, 152, 220], fill=c1); d.rectangle([36, 104, 220, 152], fill=c1)
    elif name == "ring": d.ellipse([40, 40, 216, 216], fill=c1); d.ellipse([84, 84, 172, 172], fill=(0, 0, 0, 0))
    elif name == "crescent": d.ellipse([40, 40, 216, 216], fill=c1); d.ellipse([90, 26, 246, 182], fill=(0, 0, 0, 0))
    elif name == "hexagon": d.polygon([(128 + 90 * math.cos(a), 128 + 90 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1]], fill=c1)
    elif name == "diamond": d.polygon(J([(128, 28), (200, 128), (128, 228), (56, 128)]), fill=c1)
    elif name == "arrow": d.polygon(J([(30, 104), (150, 104), (150, 56), (226, 128), (150, 200), (150, 152), (30, 152)]), fill=c1)
    elif name == "heart": d.ellipse([44, 50, 136, 142], fill=c1); d.ellipse([120, 50, 212, 142], fill=c1); d.polygon([(50, 116), (206, 116), (128, 220)], fill=c1)
    elif name == "lshape": d.polygon(J([(64, 32), (112, 32), (112, 176), (200, 176), (200, 224), (64, 224)]), fill=c1)
    elif name == "tshape": d.polygon(J([(36, 40), (220, 40), (220, 88), (152, 88), (152, 224), (104, 224), (104, 88), (36, 88)]), fill=c1)
    elif name == "ushape": d.polygon(J([(48, 40), (96, 40), (96, 176), (160, 176), (160, 40), (208, 40), (208, 224), (48, 224)]), fill=c1)
    elif name == "ellipse": d.ellipse([28, 80, 228, 176], fill=c1)
    elif name == "trapezoid": d.polygon(J([(88, 60), (168, 60), (220, 200), (36, 200)]), fill=c1)
    if rng.random() < 0.5:                                  # 무늬 하나: 모양 자체를 배우게
        d.line([60, 128, 196, 128], fill=c2, width=6)
    return im


def compose(obj, rng, W, H, scale, offset):
    """투명 배경 물체 → (흰 배경 RGB, 투명 배경 RGBA) 두 버전."""
    if rng.random() < 0.5:
        obj = obj.transpose(Image.FLIP_LEFT_RIGHT)
    obj = obj.rotate(rng.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
    obj = obj.crop(obj.getbbox())
    s = scale * min(W, H) / max(obj.size)
    obj = obj.resize((max(4, int(obj.width * s)), max(4, int(obj.height * s))), Image.BICUBIC)
    cx = W / 2 + offset * rng.uniform(-1, 1) * (W - obj.width) / 2
    cy = H / 2 + offset * rng.uniform(-1, 1) * (H - obj.height) / 2
    pos = (int(cx - obj.width / 2), int(cy - obj.height / 2))
    bg = int(rng.integers(244, 256))
    rgb = Image.new("RGB", (W, H), (bg, bg, bg))
    if rng.random() < 0.4:                                  # 바닥 그림자
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse([pos[0], pos[1] + obj.height - 6, pos[0] + obj.width, pos[1] + obj.height + 6], fill=(0, 0, 0, 40))
        rgb.paste(sh.filter(ImageFilter.GaussianBlur(4)), (0, 0), sh.filter(ImageFilter.GaussianBlur(4)))
    rgb.paste(obj, pos, obj)
    arr = np.asarray(rgb).astype(np.float32) * rng.uniform(0.92, 1.04) + rng.normal(0, 3, (H, W, 3))
    rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0))          # 투명 픽셀의 RGB는 (0,0,0) → 단순 convert면 검은 배경
    rgba.paste(obj, pos, obj)
    return rgb, rgba


def save_image(rgb, rgba, path, mode, rng):
    if mode == "RGBA":
        rgba.save(path.with_suffix(".png"))
        return path.with_suffix(".png")
    if mode == "L":
        rgb.convert("L").save(path.with_suffix(".png"))
        return path.with_suffix(".png")
    if mode == "CMYK":
        rgb.convert("CMYK").save(path.with_suffix(".jpg"), quality=92)
        return path.with_suffix(".jpg")
    if rng.random() < 0.5:
        rgb.save(path.with_suffix(".jpg"), quality=int(rng.integers(80, 96)))
        return path.with_suffix(".jpg")
    rgb.save(path.with_suffix(".png"))
    return path.with_suffix(".png")


def pick_mode(rng):
    r = rng.random()
    return "RGBA" if r < 0.08 else "L" if r < 0.13 else "CMYK" if r < 0.16 else "RGB"


# ======================================================================  상품 사진 데이터
def make_images():
    rng = np.random.default_rng(20261107)
    out = DATA / "images"
    if out.exists():
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True); (out / "test").mkdir(parents=True)
    counts = [int(round(260 * 0.915 ** k)) for k in range(len(CLASSES))]
    order = rng.permutation(len(CLASSES))                   # 클래스별 표본 수를 섞어서 배정
    n_of = {CLASSES[order[k]]: counts[k] for k in range(len(CLASSES))}
    rows, nid = [], 0

    def new_id():
        nonlocal nid
        nid += 1
        return f"P{nid:05d}"

    # --- train: 대부분 크게(0.6~0.9), 일부만 작게·치우치게
    for cls in CLASSES:
        for _ in range(n_of[cls]):
            W, H = int(rng.integers(80, 161)), int(rng.integers(80, 161))
            small = rng.random() < 0.10
            scale = rng.uniform(0.30, 0.50) if small else rng.uniform(0.62, 0.92)
            rgb, rgba = compose(draw_product(cls, rng), rng, W, H, scale, 0.8 if small else 0.15)
            pid = new_id()
            path = save_image(rgb, rgba, out / "train" / pid, pick_mode(rng), rng)
            rows.append({"id": pid, "file": f"images/train/{path.name}", "label": cls})
    train = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)

    # --- 품질 문제 심기
    issues = {}
    idx = rng.permutation(len(train))
    broken = idx[:12]; issues["broken"] = train.loc[broken, "id"].tolist()
    for k, i in enumerate(broken):
        p = DATA / train.loc[i, "file"]
        data = p.read_bytes()
        p.write_bytes(b"" if k % 2 == 0 else data[: len(data) // 3])     # 빈 파일 / 잘린 파일
    typo_map = {"tshirt": ["T-Shirt", "t shirt", "Tshirt "], "sneakers": ["Sneakers ", "sneaker s"], "sunglasses": ["sun glasses", "Sun-Glasses"],
                "backpack": ["back pack", "Back_Pack"], "handbag": ["Hand Bag", "HANDBAG"], "hoodie": [" Hoodie", "hoo-die"], "cap": ["CAP", " Cap "]}
    typo_rows = [i for i in idx[12:] if train.loc[i, "label"] in typo_map][:30]
    for i in typo_rows:
        train.loc[i, "label"] = rng.choice(typo_map[train.loc[i, "label"]])
    unknown_rows = idx[200:206]
    train.loc[unknown_rows, "label"] = ["etc", "기타", "unknown", "etc", "misc", "기타"]
    issues["unknown_label"] = train.loc[unknown_rows, "id"].tolist()
    # 중복: 같은 파일 바이트를 새 id로 복사. 8개는 레이블이 엇갈림
    dup_src = [i for i in idx[220:] if i not in set(broken)][:25]
    extra = []
    for k, i in enumerate(dup_src):
        src = DATA / train.loc[i, "file"]
        pid = new_id()
        dst = src.with_name(pid + src.suffix)
        shutil.copyfile(src, dst)
        lab = train.loc[i, "label"].strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if k < 8:
            lab = CLASSES[(CLASSES.index(lab) + 1 + int(rng.integers(0, 3))) % len(CLASSES)] if lab in CLASSES else lab
        extra.append({"id": pid, "file": f"images/train/{dst.name}", "label": lab})
    issues["dup_conflict_src"] = train.loc[dup_src[:8], "id"].tolist()
    # 파일이 없는 행
    missing = [{"id": new_id(), "file": f"images/train/missing_{k}.png", "label": rng.choice(CLASSES)} for k in range(10)]
    train = pd.concat([train, pd.DataFrame(extra + missing)]).sample(frac=1, random_state=2).reset_index(drop=True)
    train.to_csv(DATA / "train.csv", index=False)

    # --- test: 균형 30장씩, 45%는 작고 치우친 사진(분포 차이)
    trows = []
    for cls in CLASSES:
        for _ in range(30):
            W, H = int(rng.integers(80, 161)), int(rng.integers(80, 161))
            small = rng.random() < 0.45
            scale = rng.uniform(0.30, 0.50) if small else rng.uniform(0.62, 0.92)
            rgb, rgba = compose(draw_product(cls, rng), rng, W, H, scale, 0.8 if small else 0.15)
            pid = f"T{len(trows) + 1:05d}"
            path = save_image(rgb, rgba, out / "test" / pid, pick_mode(rng), rng)
            trows.append({"id": pid, "file": f"images/test/{path.name}", "label": cls})
    test = pd.DataFrame(trows).sample(frac=1, random_state=3).reset_index(drop=True)
    test[["id", "file"]].to_csv(DATA / "test.csv", index=False)
    sample = test[["id"]].sample(frac=1, random_state=4).reset_index(drop=True)      # test.csv와 순서가 다름
    sample["label"] = "cap"
    sample.to_csv(DATA / "sample_submission.csv", index=False)
    test[["id", "label"]].to_csv(TRUTH / "test_truth.csv", index=False)
    write_ready_clean(train)
    (TRUTH / "_build" / "planted_issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=1), encoding="utf-8")
    print("train rows", len(train), "| class counts", sorted(n_of.values(), reverse=True)[:3], "...", min(n_of.values()), "| test", len(test))


def write_ready_clean(train):
    """Q1 정답 규칙(1-4)과 같은 기준의 정제 표 → data/_ready_clean.csv (진행용 대체물)"""
    import practice_support as ps
    t = train.copy()
    t["label"] = t["label"].map(ps.ready_normalize_label)
    ok, md5 = [], []
    for f in t["file"]:
        path = DATA / f
        try:
            ps.ready_load_rgb(path); ok.append(True); md5.append(ps.file_md5(path))
        except Exception:
            ok.append(False); md5.append(None)
    t["ok"], t["md5"] = ok, md5
    t = t[t["ok"] & t["label"].notna()].copy()
    drop = set()
    for _, g in t.groupby("md5"):
        if len(g) > 1:
            drop |= set(g.index) if g["label"].nunique() > 1 else set(g.index[1:])
    t = t.drop(index=list(drop))[["id", "file", "label"]].reset_index(drop=True)
    t.to_csv(DATA / "_ready_clean.csv", index=False)
    print("ready clean rows", len(t))


# ======================================================================  사전학습
def preprocess_pil(im, size=64):
    from practice_support import ready_pad_to_square   # 정답과 같은 전처리
    return np.asarray(ready_pad_to_square(im).resize((size, size), Image.BICUBIC))


def make_pretrain_set(n_per, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, name in enumerate(SHAPES):
        for _ in range(n_per):
            W, H = int(rng.integers(64, 129)), int(rng.integers(64, 129))
            small = rng.random() < 0.3
            rgb, _ = compose(draw_shape(name, rng), rng, W, H, rng.uniform(0.3, 0.5) if small else rng.uniform(0.6, 0.92), 0.8 if small else 0.2)
            X.append(preprocess_pil(rgb)); y.append(k)
    return np.stack(X), np.array(y)


def pretrain(epochs=14):
    import torch
    from torch import nn
    import mini_tinyvit as mtv
    torch.manual_seed(0); torch.set_num_threads(8)
    Xtr, ytr = make_pretrain_set(420, 11)
    Xva, yva = make_pretrain_set(60, 12)
    m = torch.tensor(mtv.IMAGENET_MEAN).view(1, 3, 1, 1); s = torch.tensor(mtv.IMAGENET_STD).view(1, 3, 1, 1)
    to_t = lambda a: (torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255 - m) / s
    xtr, xva, Ytr = to_t(Xtr), to_t(Xva), torch.tensor(ytr)
    model = mtv.TinyVit(mtv.PRETRAIN_CLASSES, drop_path_rate=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    steps = epochs * math.ceil(len(xtr) / 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    print("params", sum(p.numel() for p in model.parameters()))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = xtr[b]
            f = torch.rand(len(b)) < 0.5
            xb[f] = xb[f].flip(3)
            loss = nn.functional.cross_entropy(model(xb), Ytr[b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xva).argmax(1).numpy() == yva).mean()
        print(f"epoch {ep + 1} loss {loss.item():.3f} val acc {acc:.3f}", flush=True)
    (ROOT / "assets").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "assets" / "tiny_vit_practice.pt")
    (TRUTH / "_build" / "pretrain_log.json").write_text(json.dumps({"epochs": epochs, "val_acc": float(acc)}), encoding="utf-8")


# ======================================================================  Q5 상품 설명 문장
KO_CAT = {"cap": ["볼캡", "야구모자"], "beanie": ["비니"], "tshirt": ["반팔 티셔츠", "반팔티"], "shirt": ["셔츠", "남방"],
          "hoodie": ["후드티", "후드 티셔츠"], "jacket": ["자켓", "재킷"], "dress": ["원피스"], "skirt": ["스커트", "치마"],
          "pants": ["슬랙스", "바지"], "shorts": ["반바지", "숏팬츠"], "sneakers": ["스니커즈", "운동화"], "boots": ["부츠", "워커"],
          "sandals": ["샌들", "슬리퍼"], "backpack": ["백팩", "배낭"], "handbag": ["핸드백", "토트백"], "wallet": ["반지갑", "지갑"],
          "watch": ["손목시계", "시계"], "sunglasses": ["선글라스"], "socks": ["양말"], "scarf": ["머플러", "스카프"]}
KO_COLOR = {"black": ["블랙", "검정"], "white": ["화이트", "흰색"], "gray": ["그레이", "회색"], "red": ["레드", "빨강"], "blue": ["블루", "파랑"],
            "green": ["그린", "초록"], "beige": ["베이지"], "brown": ["브라운", "갈색"], "navy": ["네이비", "남색"], "pink": ["핑크", "분홍"]}
KO_MAT = {"cotton": ["면 100%", "코튼"], "polyester": ["폴리에스터"], "wool": ["울", "양모"], "leather": ["천연가죽", "가죽"],
          "denim": ["데님"], "nylon": ["나일론"]}
BRANDS = ["노바핏", "루미에르", "어반키드", "모노웨이", "SOLID", "하루살이", "그레이스톤", "NOMAD"]
CLOTH = {"tshirt", "shirt", "hoodie", "jacket", "dress", "skirt", "pants", "shorts"}
SHOES = {"sneakers", "boots", "sandals"}
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]


def won(n, rng):
    style = rng.integers(0, 4)
    if style == 0:
        return f"{n:,}원"
    if style == 1:
        return f"₩{n:,}"
    if style == 2:
        return f"{n}원"
    man, rest = divmod(n, 10000)
    chun = rest // 1000
    return f"{man}만" + (f" {chun}천" if chun else "") + "원"


def make_text_record(rng):
    cat = CLASSES[rng.integers(len(CLASSES))]
    rec = {"category": cat, "brand": None, "price_krw": None, "colors": [], "sizes": [], "material": None,
           "in_stock": None, "free_shipping": None}
    parts = []
    brand = BRANDS[rng.integers(len(BRANDS))] if rng.random() < 0.75 else None
    rec["brand"] = brand
    name = rng.choice(KO_CAT[cat])
    parts.append((f"[{brand}] " if brand else "") + rng.choice(["", "남성 ", "여성 ", "남녀공용 "]) + rng.choice(["오버핏 ", "베이직 ", "데일리 ", ""]) + name)
    r = rng.random()
    if r < 0.62:
        p = int(rng.integers(9, 200)) * 1000 + (500 if rng.random() < 0.1 else 0)
        rec["price_krw"] = p
        parts.append("가격 " + won(p, rng))
    elif r < 0.78:
        p0 = int(rng.integers(40, 200)) * 1000; p = int(p0 * rng.choice([0.6, 0.7, 0.8]) // 1000 * 1000)
        rec["price_krw"] = p
        parts.append(f"정가 {p0:,}원 → 할인가 {won(p, rng)}")
    elif r < 0.9:
        parts.append(rng.choice(["가격 문의", "가격은 문의 바랍니다"]))
    else:
        parts.append(f"${int(rng.integers(15, 90))}")       # 외화 → null
    cols = list(rng.choice(list(KO_COLOR), size=int(rng.integers(1, 4)), replace=False)) if rng.random() < 0.85 else []
    rec["colors"] = sorted(cols)
    if cols:
        parts.append("색상: " + rng.choice([", ", "/", " · "]).join(rng.choice(KO_COLOR[c]) for c in cols))
    if cat in CLOTH and rng.random() < 0.9:
        if rng.random() < 0.6:
            a = int(rng.integers(0, 3)); b = int(rng.integers(a + 1, 6))
            rec["sizes"] = SIZE_ORDER[a:b + 1]
            parts.append(f"사이즈 {SIZE_ORDER[a]}~{SIZE_ORDER[b]}")
        elif rng.random() < 0.5:
            rec["sizes"] = ["FREE"]; parts.append(rng.choice(["사이즈 FREE", "프리사이즈"]))
        else:
            sz = sorted(rng.choice(SIZE_ORDER, size=2, replace=False), key=SIZE_ORDER.index)
            rec["sizes"] = list(sz); parts.append("사이즈 " + ", ".join(sz))
    elif cat in SHOES and rng.random() < 0.9:
        start = int(rng.integers(22, 26)) * 10
        sz = [str(start + 10 * k) for k in range(int(rng.integers(2, 5)))]
        rec["sizes"] = sz; parts.append("사이즈 " + "/".join(sz))
    rm = rng.random()
    if rm < 0.6:
        mat = rng.choice(list(KO_MAT)); rec["material"] = mat; parts.append("소재 " + rng.choice(KO_MAT[mat]))
    elif rm < 0.75:
        parts.append("소재 " + rng.choice(["면/폴리 혼방", "울 혼방"]))         # 혼방 → null
    rs = rng.random()
    if rs < 0.25:
        rec["in_stock"] = False; parts.append(rng.choice(["품절", "일시품절", "재입고 예정"]))
    elif rs < 0.6:
        rec["in_stock"] = True; parts.append(rng.choice(["재고 있음", "즉시 출고"]))
    rf = rng.random()
    if rf < 0.35:
        rec["free_shipping"] = True; parts.append("무료배송")
    elif rf < 0.6:
        rec["free_shipping"] = False; parts.append(f"배송비 {rng.choice([2500, 3000, 3500]):,}원")
    elif rf < 0.75:
        parts.append("5만원 이상 무료배송")                    # 조건부 → null
    head, rest = parts[0], parts[1:]
    rng.shuffle(rest)
    sep = rng.choice([" / ", ". ", ", ", "\n"])
    return head + sep + sep.join(rest), rec


def mock_response(rec, note, rng, k):
    """정답 레코드에 실제 응답에서 흔한 형식 오류를 섞은 '모의 응답' 문자열."""
    out = dict(rec)
    if rng.random() < 0.4:
        out["category"] = rng.choice(KO_CAT[rec["category"]])
    if out["price_krw"] is not None and rng.random() < 0.4:
        out["price_krw"] = rng.choice([f"{rec['price_krw']:,}원", str(rec["price_krw"]), won(rec["price_krw"], rng)])
    if rec["colors"] and rng.random() < 0.4:
        out["colors"] = [rng.choice(KO_COLOR[c]) for c in rec["colors"]]
    if rec["sizes"] and rng.random() < 0.4:
        s = rec["sizes"]
        out["sizes"] = f"{s[0]}~{s[-1]}" if s[0] in SIZE_ORDER and len(s) > 2 else "/".join(s)
    if rec["material"] and rng.random() < 0.35:
        out["material"] = KO_MAT[rec["material"]][0]
    if rec["in_stock"] is not None and rng.random() < 0.35:
        out["in_stock"] = {True: "재고 있음", False: "품절"}[rec["in_stock"]]
    if rec["free_shipping"] is not None and rng.random() < 0.3:
        out["free_shipping"] = {True: "Y", False: "N"}[rec["free_shipping"]]
    if rec["brand"] is None and rng.random() < 0.5:
        out["brand"] = rng.choice(["unknown", "없음", ""])
    if not rec["sizes"] and rng.random() < 0.3:
        out["sizes"] = None
    if rng.random() < 0.15:
        out["comment"] = "상품명에서 추출했습니다."
    out = json.loads(json.dumps(out, ensure_ascii=False))     # numpy 문자열·정수 → 파이썬 기본 자료형
    text = json.dumps(out, ensure_ascii=False, indent=2)
    style = k % 9
    if style == 1:
        text = "```json\n" + text + "\n```"
    elif style == 2:
        text = "추출 결과입니다.\n" + text + "\n확인 부탁드립니다."
    elif style == 3:
        text = text.replace("\n}", ",\n}")
    elif style == 4:
        text = repr({k2: v for k2, v in out.items()})
    elif style == 5 and rng.random() < 0.35:
        text = text[: int(len(text) * 0.6)]                  # 끊긴 응답
    return text


def make_text():
    rng = np.random.default_rng(777)
    rows, gold, mock = [], {}, {}
    for i in range(70):
        nid = f"D{i + 1:03d}"
        note, rec = make_text_record(rng)
        rows.append({"id": nid, "split": "dev" if i < 40 else "test", "text": note})
        gold[nid] = json.loads(json.dumps(rec, ensure_ascii=False))
        mock[nid] = mock_response(rec, note, rng, i)
    tdir = DATA / "texts"; tdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tdir / "products.csv", index=False)
    (tdir / "dev_labels.json").write_text(json.dumps({k: v for k, v in gold.items() if k <= "D040"}, ensure_ascii=False, indent=1), encoding="utf-8")
    (tdir / "mock_llm_outputs.json").write_text(json.dumps(mock, ensure_ascii=False, indent=1), encoding="utf-8")
    (TRUTH / "q5_truth.json").write_text(json.dumps({k: v for k, v in gold.items() if k > "D040"}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("texts", len(rows))


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["images", "pretrain", "text"]
    for j in jobs:
        {"images": make_images, "pretrain": pretrain, "text": make_text}[j]()
