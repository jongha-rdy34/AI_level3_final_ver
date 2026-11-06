"""v3 노트북 생성기 (TinyViT · 상품 20분류). 실행: python -I build_v3.py

작성 셀마다 세 버전: A(빈칸) / B(직접구현) / 정답(줄마다 주석)
"""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), None, None))


def run(code):
    CELLS.append(("run", "# [실행만 하세요]\n" + code.strip("\n"), None, None))


def write(blank, stub, answer):
    CELLS.append(("write", blank.strip("\n"), stub.strip("\n"), answer.strip("\n")))


# ======================================================================  머리말
md(r"""
# [claude] Lv3 모의평가 v3 — TinyViT-5M · 상품 이미지 20분류

**창작 모의문제입니다. 실제 시험 문제가 아닙니다.** 공식 정보 "**지정 모델 TinyViT-5M**, 과제 **상품 이미지 20개 분류**(모자·의류·신발 등)"를 반영했습니다. 먼저 `README.md`, `제출가이드.md`, `assets/모델카드.md`를 읽으세요.

| 문항 | 공지 분야 | 유형 | 시간 |
|---|---|---|---:|
| Q1 | 데이터 분석/전처리 (이미지) | 구현 — 표·파일 점검, 이미지 모드(RGBA·L·CMYK), 중복·깨짐, 분포 차이, 정제 | 35분 |
| Q2 | 데이터 분석/전처리 (이미지) | 구현 — 정사각형 패딩, 모델 설정으로 transforms, Dataset, 불균형 샘플러 | 30분 |
| Q3 | AI 모델링 및 최적화 (이미지) | 구현 + **성능 기준** — head 교체, 학습 범위, layer-wise lr decay, label smoothing·mixup, warmup+cosine, 체크포인트 | 65분 |
| Q4 | AI 모델링 및 최적화 (이미지) | **성능 기준** — 오분류 분석, TTA, 앙상블, 최종 제출 | 45분 |
| Q5 | 프롬프트 엔지니어링 (텍스트) | 구현 + 기준 — 상품 설명 → JSON (가격·색상·사이즈 정규화, 미확정 처리) | 50분 |

**성능 기준(이 모의의 가정):** Q3 test macro-F1 ≥ __Q3_TH__ · Q4 test macro-F1 ≥ __Q4_TH__ · Q5 dev 필드 정확도 ≥ 0.95 · test ≥ 0.93, 스키마 위반 0건

진행 규칙
- 필요한 import는 첫 셀에 있습니다. 막히면 `Reference/`(공식 문서 형식 요약)만 보고 푸세요. 인터넷·생성형 AI는 쓰지 않습니다.
- 이 PC에는 GPU·timm이 없어 **같은 이름·API의 축소판 TinyViT**(`mini_tinyvit`, 입력 64)를 씁니다. 시험의 `timm.create_model("tiny_vit_5m_224", ...)`(입력 224)와 코드가 같도록 맞췄습니다.
- `[작성하세요]` 셀만 고칩니다. 함수 이름·매개변수·반환 형식은 그대로. 앞 소문항을 못 풀면 확인 셀이 진행용 대체물로 이어 갑니다.
- **제출 형식이 틀리면 그 문항은 0점**(FAQ). test 정답은 쓰지 않습니다. `정답_채점용/`은 풀이가 끝날 때까지 열지 마세요.
""")

run(r"""
# 문제 풀이에 필요한 라이브러리 (시험처럼 미리 작성되어 있습니다)
from pathlib import Path
import sys, os, time, copy, json, re, ast, hashlib, itertools, random, math
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "practice_support.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("모의문제 폴더(또는 그 하위 폴더)에서 노트북을 여세요.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
try:
    from torchvision.transforms import v2 as T
    print("torchvision.transforms.v2 사용")
except ImportError:                        # 이 PC에 torchvision이 없을 때만: 같은 이름의 연습용 대체
    from tv_compat import transforms_v2 as T
    print("torchvision 없음 → tv_compat(같은 이름·인자) 사용")
# 시험에서는:  import timm;  from timm.data import resolve_data_config;  timm.create_model("tiny_vit_5m_224", ...)
from mini_tinyvit import create_model, resolve_data_config      # 연습용 축소판(같은 API)
import practice_support as ps
from practice_support import attempt, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASSES)
ps.seed_everything(42)
ps.SUBMIT.mkdir(exist_ok=True)
print("장치:", DEVICE, "| torch", torch.__version__, "| 클래스", NUM_CLASSES, "개")
""")

# ======================================================================  Q1
md(r"""
---
## Q1 데이터셋 구성·품질·분포 점검 (데이터 분석/전처리 · 구현)

온라인 쇼핑몰 상품 사진 20종입니다(합성 자료). 클래스: `cap(볼캡), beanie(비니), tshirt, shirt, hoodie, jacket, dress, skirt, pants, shorts, sneakers, boots, sandals, backpack, handbag, wallet, watch, sunglasses, socks, scarf`
- `data/train.csv`: `id, file, label` — `file`은 `data/` 기준 상대 경로(`images/train/P00001.png`)
- `data/test.csv`: `id, file` / `data/sample_submission.csv`: 제출 양식(**test.csv와 행 순서가 다릅니다**)
- 사진 크기·형식(PNG/JPG)·색 모드가 제각각입니다.

**품질 문제가 섞여 있습니다.** 무엇인지는 알려 주지 않습니다. 직접 찾으세요.
""")

run(r"""
train_df = ps.read_table("train.csv")
test_df = ps.read_table("test.csv")
sample_df = ps.read_table("sample_submission.csv")
print("train", train_df.shape, "| test", test_df.shape, "| sample", sample_df.shape)
train_df.head()
""")

md(r"""
### 1-1 표 점검과 레이블 정리
- `normalize_label(s)`: 문자열이 아니면 `None`. 앞뒤 공백 제거 → 소문자 → 공백·`-`·`_` 제거. 결과가 `CLASSES`에 있으면 그 값, 없으면 `None`
- `summarize_table(df)`: 다음 키의 dict
  - `n_rows`, `n_unique_ids`, `missing_files`(파일이 실제로 없는 행 수 — `ps.DATA / file`)
  - `n_raw_label_values`(정리 전 고유 레이블 수), `unknown_labels`(정리 결과가 None인 원래 레이블의 정렬된 고유값 리스트)
  - `class_counts`(정리 후 클래스별 개수 dict), `imbalance_ratio`(최다 / 최소)
""")

write(r"""
def normalize_label(s):
    if not isinstance(s, str):
        return None
    key = s.___().lower().replace(" ", "").replace("-", "").replace(___, "")
    return key if key in ___ else None

def summarize_table(df):
    info = {}
    info["n_rows"] = len(df)
    info["n_unique_ids"] = df["id"].___()
    info["missing_files"] = int((~df["file"].map(lambda f: (ps.DATA / f).___())).sum())
    norm = df["label"].map(___)
    info["n_raw_label_values"] = df["label"].nunique()
    info["unknown_labels"] = sorted(df.loc[norm.___(), "label"].unique())
    counts = norm.value_counts()
    info["class_counts"] = counts.to_dict()
    info["imbalance_ratio"] = float(___)
    return info
""", r"""
def normalize_label(s):
    '''흔들린 레이블 표기를 CLASSES의 이름으로. 모르면 None'''
    # 작성하세요

def summarize_table(df):
    '''위 7개 키를 가진 dict'''
    info = {}
    # 작성하세요
    return info
""", r"""
def normalize_label(s):
    if not isinstance(s, str):                     # NaN 등
        return None
    # 'T-Shirt', ' Sun-Glasses', 'back pack' → 'tshirt', 'sunglasses', 'backpack'
    key = s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return key if key in CLASSES else None         # 'etc', '기타' 같은 값은 버릴 대상

def summarize_table(df):
    info = {}
    info["n_rows"] = len(df)
    info["n_unique_ids"] = df["id"].nunique()
    # 표에는 있는데 파일이 없는 행 (학습 중에 FileNotFoundError로 멈추는 원인)
    info["missing_files"] = int((~df["file"].map(lambda f: (ps.DATA / f).exists())).sum())
    norm = df["label"].map(normalize_label)
    info["n_raw_label_values"] = df["label"].nunique()          # 20보다 크면 표기가 흔들린 것
    info["unknown_labels"] = sorted(df.loc[norm.isna(), "label"].unique())
    counts = norm.value_counts()                                 # None은 value_counts에서 빠짐
    info["class_counts"] = counts.to_dict()
    info["imbalance_ratio"] = float(counts.max() / counts.min())
    return info
""")

run(r"""
def _check_1_1():
    assert normalize_label("T-Shirt") == "tshirt" and normalize_label(" Sun-Glasses") == "sunglasses"
    assert normalize_label("Back_Pack") == "backpack" and normalize_label("기타") is None and normalize_label(None) is None
    info = summarize_table(train_df)
    for k, v in info.items():
        print(f"{k:20s}: {v}")
    return info
q1_info = attempt("1-1", _check_1_1)
""")

md(r"""
### 1-2 파일 점검: 형식·크기·깨짐·중복
`inspect_files(df)`: 행마다 다음 열을 가진 DataFrame(행 순서는 `df`와 같게)
- `id`, `exists`(파일 있음), `md5`(파일 바이트의 `hashlib.md5`; 없으면 None)
- `readable`: `Image.open(path)` 후 **`im.load()`까지** 성공하면 True (`open`만으로는 잘린 파일을 못 잡습니다)
- `mode`(`"RGB"`, `"RGBA"`, `"L"`, `"CMYK"` …), `width`, `height` (읽지 못하면 None)
""")

write(r"""
def inspect_files(df):
    rows = []
    for r in df.itertuples():
        path = ps.DATA / r.file
        info = {"id": r.id, "exists": path.exists(), "md5": None, "readable": False, "mode": None, "width": None, "height": None}
        if info["exists"]:
            info["md5"] = hashlib.md5(path.___()).hexdigest()
            try:
                with Image.open(path) as im:
                    im.___()                                   # 픽셀까지 실제로 읽기
                    info.update(readable=True, mode=im.___, width=im.width, height=im.height)
            except Exception:
                pass
        rows.append(info)
    return pd.___(rows)
""", r"""
def inspect_files(df):
    '''행마다 id, exists, md5, readable, mode, width, height 를 가진 DataFrame'''
    # 작성하세요
""", r"""
def inspect_files(df):
    rows = []
    for r in df.itertuples():
        path = ps.DATA / r.file
        info = {"id": r.id, "exists": path.exists(), "md5": None, "readable": False, "mode": None, "width": None, "height": None}
        if info["exists"]:
            info["md5"] = hashlib.md5(path.read_bytes()).hexdigest()   # 바이트가 같은 파일 = 같은 해시
            try:
                with Image.open(path) as im:
                    im.load()            # open은 머리만 읽는다. load에서 잘린·빈 파일이 오류를 낸다
                    info.update(readable=True, mode=im.mode, width=im.width, height=im.height)
            except Exception:            # 읽을 수 없는 파일은 readable=False로 남긴다
                pass
        rows.append(info)
    return pd.DataFrame(rows)
""")

run(r"""
def _check_1_2():
    fi = inspect_files(train_df)
    assert list(fi.columns) == ["id", "exists", "md5", "readable", "mode", "width", "height"] and len(fi) == len(train_df)
    print("파일 없음", int((~fi.exists).sum()), "| 못 읽음", int((fi.exists & ~fi.readable).sum()))
    print("색 모드:", fi["mode"].value_counts().to_dict())
    print("크기(가로) 범위:", fi.width.min(), "~", fi.width.max(), "| 세로:", fi.height.min(), "~", fi.height.max())
    ok = fi[fi.readable].merge(train_df[["id", "label"]], on="id")
    ok["norm"] = ok["label"].map(normalize_label) if "normalize_label" in globals() else ok["label"]
    groups = [g for _, g in ok.groupby("md5") if len(g) > 1]
    print(f"중복 묶음 {len(groups)}개 (레이블이 엇갈린 묶음 {sum(g.norm.nunique() > 1 for g in groups)}개)")
    return fi
q1_files = attempt("1-2", _check_1_2)
""")

md(r"""
### 1-3 육안 검사: 색 모드와 분포 차이
아래 셀은 **`Image.open(p).convert("RGB")`로 단순 변환한** RGBA·L·CMYK 사진과 클래스별 사진을 보여 줍니다. 이상한 점을 찾으세요.

그다음 작성하세요.
- `load_rgb(path)`: 사진을 **흰 배경의 RGB** PIL 이미지로. 모드가 `RGBA`·`LA`이거나 `P`인데 `"transparency"`가 있으면, `RGBA`로 바꾼 뒤 흰 RGB 캔버스에 **알파 채널을 마스크로** 붙입니다(`paste(im, mask=im.getchannel("A"))`). 그 외는 `convert("RGB")`
- `pad_to_square(img, fill=(255, 255, 255))`: 긴 변 길이의 정사각형 캔버스(fill 색) 가운데에 붙여 반환 — **비율을 지킨 채** 정사각형으로(늘이거나 자르지 않음)
- `object_ratio(arr)`: (H,W,3) uint8 배열에서 **세 채널 중 최솟값이 230 미만인 픽셀 비율**(= 흰 배경이 아닌 부분, 물체 크기 지표)
- 서술 `q1_note`: 육안 검사·분포 점검에서 찾은 점과 학습에 주는 의미(2~4문장)
""")

run(r"""
_fi = q1_files if q1_files is not None else None
rng = np.random.default_rng(0)
if _fi is not None:
    for mode in ["RGBA", "L", "CMYK"]:
        ids = _fi.loc[_fi["mode"] == mode, "id"].head(8)
        files = train_df.set_index("id").loc[ids, "file"]
        ps.show_images([Image.open(ps.DATA / f).convert("RGB") for f in files], [f"{mode}→convert"] * len(files))
clean_like = train_df[train_df["label"].isin(CLASSES)]
for c in CLASSES[:20:4]:
    files = clean_like.loc[clean_like.label == c, "file"].sample(6, random_state=0)
    ps.show_images([ps.ready_load_rgb(ps.DATA / f) for f in files], [c] * 6, ncols=6)
""")

write(r"""
def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.___):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel(___))                 # 알파 채널을 마스크로
            return bg
        return im.convert(___)

def pad_to_square(img, fill=(255, 255, 255)):
    w, h = img.size
    s = ___(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, ___))
    return out

def object_ratio(arr):
    return float((arr.___(axis=2) < 230).mean())

q1_note = ""  # 작성하세요 (2~4문장)
""", r"""
def load_rgb(path):
    '''투명 배경은 흰색으로 채운 RGB PIL 이미지'''
    # 작성하세요

def pad_to_square(img, fill=(255, 255, 255)):
    '''비율을 지킨 채 가운데 정렬로 정사각형 패딩'''
    # 작성하세요

def object_ratio(arr):
    '''세 채널 최솟값 < 230 인 픽셀 비율'''
    # 작성하세요

q1_note = ""  # 작성하세요 (2~4문장)
""", r"""
def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        # 투명 배경 PNG: 투명 픽셀의 RGB 값은 보통 (0,0,0)이라 convert("RGB")만 하면 배경이 검게 나온다
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))     # 상품 사진의 기본 배경색인 흰색
            bg.paste(im, mask=im.getchannel("A"))               # 불투명한 부분만 붙인다
            return bg
        return im.convert("RGB")                                 # L(흑백)·CMYK·RGB → RGB 3채널

def pad_to_square(img, fill=(255, 255, 255)):
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)                         # 배경과 같은 흰색으로 채워야 경계가 안 생김
    out.paste(img, ((s - w) // 2, (s - h) // 2))                 # 가운데 정렬. 늘이면 신발·바지 비율이 바뀐다
    return out

def object_ratio(arr):
    # 흰 배경(모든 채널이 밝음)이 아닌 픽셀 = 물체. 비율이 작을수록 작게 찍힌 사진
    return float((arr.min(axis=2) < 230).mean())

q1_note = (
    "RGBA 사진을 단순히 convert('RGB')하면 투명 배경이 검게 바뀌어, 흰 배경 위에 합성한 뒤 써야 한다. "
    "흑백(L)·CMYK 사진도 섞여 있어 모두 RGB 3채널로 통일하고, 크기와 비율이 제각각이라 비율을 지킨 정사각형 패딩 후 크기를 맞춘다. "
    "test는 물체가 작게 찍힌 사진의 비율이 train보다 훨씬 높아(물체 비율이 작음), 학습 때 축소·이동 증강을 넣고 작게 만든 검증으로도 확인해야 한다. "
    "클래스 수가 최대 5배 이상 차이 나므로 macro-F1로 평가하고 불균형 대응을 비교한다."
)
""")

run(r"""
def _check_1_3():
    fi = q1_files
    pick = fi.loc[fi["mode"] == "RGBA", "id"].head(4).tolist() + fi.loc[fi["mode"] == "L", "id"].head(2).tolist() + fi.loc[fi["mode"] == "CMYK", "id"].head(2).tolist()
    files = train_df.set_index("id").loc[pick, "file"]
    imgs = [load_rgb(ps.DATA / f) for f in files]
    assert all(im.mode == "RGB" for im in imgs)
    assert all(np.asarray(im)[0, 0].min() > 200 for im in imgs[:4]), "RGBA의 배경이 흰색이 아닙니다"
    sq = pad_to_square(Image.new("RGB", (30, 10), (0, 0, 0)))
    assert sq.size == (30, 30) and np.asarray(sq)[0, 0].tolist() == [255, 255, 255] and np.asarray(sq)[15, 15].tolist() == [0, 0, 0]
    ps.show_images(imgs, ["load_rgb"] * len(imgs))
    ratio = lambda files, n=300: np.array([object_ratio(np.asarray(pad_to_square(load_rgb(ps.DATA / f)).resize((64, 64)))) for f in files[:n]])
    ok_files = train_df.loc[train_df.id.isin(fi.loc[fi.readable, "id"]), "file"].tolist()
    r_tr, r_te = ratio(ok_files), ratio(test_df["file"].tolist())
    print(f"물체 비율 평균 | train {r_tr.mean():.3f} | test {r_te.mean():.3f}")
    print(f"물체 비율 < 0.12 (작게 찍힌 사진) | train {(r_tr < 0.12).mean():.2f} | test {(r_te < 0.12).mean():.2f}")
    print("서술 작성:", bool(q1_note.strip()))
attempt("1-3", _check_1_3)
""")

md(r"""
### 1-4 정제된 학습 표 만들기
`build_clean_table(df, file_info)` → `id, file, label` 세 열의 DataFrame(`reset_index(drop=True)`). 규칙:
1. 레이블은 `normalize_label`로 정리하고, `None`인 행은 제거
2. `file_info`(1-2 결과)에서 `readable`이 False인 행 제거(없는 파일·깨진 파일)
3. `md5`가 같은 중복 묶음: 레이블이 **모두 같으면 첫 행만** 남기고, **엇갈리면 묶음 전체 제거**
4. RGBA·L·CMYK 사진은 **버리지 않습니다**(읽을 때 `load_rgb`로 고침)
""")

write(r"""
def build_clean_table(df, file_info):
    t = df.merge(file_info[["id", "readable", "md5"]], on="id", how="left")
    t["label"] = t["label"].map(___)
    t = t[t["readable"].fillna(False) & t["label"].___()].copy()         # 규칙 1, 2
    drop = set()
    for _, g in t.groupby("md5"):
        if len(g) > 1:
            if g["label"].nunique() > 1:
                drop |= set(g.index)                                    # 엇갈림 → 전부
            else:
                drop |= set(g.index[___:])                              # 같음 → 첫 행만 남김
    return t.drop(index=list(drop))[["id", "file", "label"]].reset_index(drop=___)
""", r"""
def build_clean_table(df, file_info):
    '''위 규칙으로 정제한 id, file, label DataFrame'''
    # 작성하세요
""", r"""
def build_clean_table(df, file_info):
    # 1-2의 파일 점검 결과를 id로 붙인다 (행 순서가 아니라 id로 맞춤)
    t = df.merge(file_info[["id", "readable", "md5"]], on="id", how="left")
    t["label"] = t["label"].map(normalize_label)
    t = t[t["readable"].fillna(False) & t["label"].notna()].copy()        # 1: 모르는 레이블, 2: 없는·깨진 파일
    drop = set()
    for _, g in t.groupby("md5"):                                         # 3: 바이트가 같은 파일 묶음
        if len(g) > 1:
            if g["label"].nunique() > 1:
                drop |= set(g.index)        # 같은 사진에 다른 레이블 → 어느 쪽도 믿을 수 없음
            else:
                drop |= set(g.index[1:])    # 같은 레이블 → 한 장만 (중복은 그 사진의 가중치만 키움)
    return t.drop(index=list(drop))[["id", "file", "label"]].reset_index(drop=True)
""")

run(r"""
def _check_1_4():
    clean = build_clean_table(train_df, q1_files)
    assert list(clean.columns) == ["id", "file", "label"] and clean.id.is_unique and clean.label.isin(CLASSES).all()
    assert len(clean) == len(ps.ready_clean_table()), f"정제 결과 {len(clean)}행 (규칙을 다시 확인하세요)"
    print("정제 후", len(clean), "행 |", clean.label.value_counts().tail(5).to_dict(), "(가장 적은 5개)")
    return clean
clean_df = attempt("1-4", _check_1_4)
if clean_df is None:
    print("→ 이후 진행용 준비 표를 사용합니다(1-4 평가와 무관).")
    clean_df = ps.ready_clean_table()
""")

# ======================================================================  Q2
md(r"""
---
## Q2 전처리와 학습 데이터 재구성 (데이터 분석/전처리 · 구현)

지정 모델의 **전처리 설정은 모델이 알려 줍니다.** timm은 `resolve_data_config(model=model)`로 입력 크기·정규화 값·보간 방법을 돌려줍니다(`Reference/timm_tinyvit.md`). 숫자를 직접 쓰지 말고 이 설정에서 읽으세요.
""")

run(r"""
model0 = create_model("tiny_vit_5m_224", pretrained=True)        # 시험: timm.create_model("tiny_vit_5m_224", pretrained=True, ...)
DATA_CFG = resolve_data_config(model=model0)
IMG_SIZE = DATA_CFG["input_size"][-1]
print(DATA_CFG)
print("입력 크기:", IMG_SIZE, "(실제 TinyViT-5M은 224)")
""")

md(r"""
### 2-1 transforms
`build_transforms(train, cfg)`: 입력은 **이미 정사각형으로 패딩·크기 조정된** (S,S,3) uint8 배열입니다(2-2에서 미리 해 둠). `T.Compose`를 반환합니다.
- 학습용: `ToImage` → 좌우 반전(p=0.5) → `RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.55, 1.0), fill=255)` → `ToDtype(float32, scale=True)` → `ColorJitter(brightness=0.2, contrast=0.2)` → `Normalize(cfg의 mean, std)`
- 검증·test용: `ToImage` → `ToDtype(float32, scale=True)` → `Normalize`
""")

write(r"""
def build_transforms(train, cfg):
    steps = [T.ToImage()]
    if train:
        steps += [T.RandomHorizontalFlip(p=0.5),
                  T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.55, 1.0), fill=___)]   # 빈 곳 = 흰색
    steps += [T.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [T.ColorJitter(brightness=0.2, contrast=0.2)]
    steps += [T.Normalize(mean=list(cfg[___]), std=list(cfg["std"]))]
    return T.Compose(steps)
""", r"""
def build_transforms(train, cfg):
    '''학습용/검증용 T.Compose (정규화 값은 cfg에서)'''
    # 작성하세요
""", r"""
def build_transforms(train, cfg):
    steps = [T.ToImage()]                                       # (S,S,3) uint8 → (3,S,S) uint8
    if train:
        steps += [T.RandomHorizontalFlip(p=0.5),                # 상품은 좌우 어느 쪽을 봐도 같은 종류
                  # scale<1: 축소 → test의 '작게 찍힌 사진' 대비. fill=255: 생긴 빈 곳을 배경과 같은 흰색으로
                  T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.55, 1.0), fill=255)]
    steps += [T.ToDtype(torch.float32, scale=True)]             # [0,1] 실수
    if train:
        steps += [T.ColorJitter(brightness=0.2, contrast=0.2)]  # 조명 차이. 색은 레이블 근거가 아님
    steps += [T.Normalize(mean=list(cfg["mean"]), std=list(cfg["std"]))]   # 사전학습과 같은 정규화(ImageNet 값)
    return T.Compose(steps)
""")

run(r"""
def _check_2_1():
    tv, tt = build_transforms(False, DATA_CFG), build_transforms(True, DATA_CFG)
    white = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
    x = torch.as_tensor(tv(white))
    assert tuple(x.shape) == (3, IMG_SIZE, IMG_SIZE) and x.dtype == torch.float32
    want = (1 - np.array(DATA_CFG["mean"])) / np.array(DATA_CFG["std"])
    assert np.allclose(x[:, 0, 0].numpy(), want, atol=1e-4), "정규화 값(cfg의 mean/std)을 확인하세요"
    torch.manual_seed(0)
    y = torch.as_tensor(tt(white)) * torch.tensor(DATA_CFG["std"]).view(3, 1, 1) + torch.tensor(DATA_CFG["mean"]).view(3, 1, 1)
    assert float(y.min()) > 0.7, "학습 변환이 검은 빈 곳을 만듭니다(fill)"
    img = np.asarray(ps.ready_pad_to_square(ps.ready_load_rgb(ps.DATA / clean_df.file[0])).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
    show = [((torch.as_tensor(tt(img)) * torch.tensor(DATA_CFG["std"]).view(3, 1, 1) + torch.tensor(DATA_CFG["mean"]).view(3, 1, 1)).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8) for _ in range(8)]
    ps.show_images(show, ["train tf"] * 8)
    print(tt)
attempt("2-1", _check_2_1)
""")

md(r"""
### 2-2 Dataset (미리 읽어 두기)
`ProductDataset(files, labels, transform, size, images=None)`:
- `images`가 None이면 `files`의 사진을 **처음에 한 번만** 읽어 `load_rgb` → `pad_to_square` → `resize((size, size), Image.BICUBIC)` → uint8 배열로 쌓아 `self.images` (N,size,size,3)에 둡니다. `images`를 주면 그것을 그대로 씁니다(같은 사진을 두 번 읽지 않도록).
- `labels`: 문자열 리스트 또는 None(test → y = -1). `__getitem__(i)` → `(transform(self.images[i]), torch.long 스칼라)`
""")

write(r"""
class ProductDataset(Dataset):
    def __init__(self, files, labels, transform, size, images=None):
        if images is None:
            images = np.stack([np.asarray(pad_to_square(load_rgb(ps.DATA / f)).resize((size, size), Image.___))
                               for f in files])
        self.images = images
        self.transform = transform
        self.y = None if labels is None else [CLASSES.___(c) for c in labels]

    def __len__(self):
        return len(self.___)

    def __getitem__(self, i):
        y = -1 if self.y is None else self.y[i]
        return self.transform(self.images[i]), torch.tensor(y, dtype=torch.___)
""", r"""
class ProductDataset(Dataset):
    '''사진을 미리 읽어 두고 (transform(image), y) 를 돌려주는 Dataset'''
    def __init__(self, files, labels, transform, size, images=None):
        # 작성하세요
        pass
    # 작성하세요: __len__, __getitem__
""", r"""
class ProductDataset(Dataset):
    def __init__(self, files, labels, transform, size, images=None):
        if images is None:
            # 사진 해석(디코딩)·패딩·크기 조정은 결정적이므로 한 번만 해 두면 epoch마다 반복하지 않아도 된다
            images = np.stack([np.asarray(pad_to_square(load_rgb(ps.DATA / f)).resize((size, size), Image.BICUBIC))
                               for f in files])
        self.images = images                                    # (N, size, size, 3) uint8
        self.transform = transform                              # 무작위 증강은 꺼낼 때마다 새로
        self.y = None if labels is None else [CLASSES.index(c) for c in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        y = -1 if self.y is None else self.y[i]
        return self.transform(self.images[i]), torch.tensor(y, dtype=torch.long)
""")

run(r"""
_load = load_rgb if attempt("1-3 재확인", lambda: load_rgb(ps.DATA / clean_df.file[0]).mode) == "RGB" else ps.ready_load_rgb
_pad = pad_to_square if attempt("1-3 재확인", lambda: pad_to_square(Image.new("RGB", (4, 2))).size) == (4, 4) else ps.ready_pad_to_square
_tf = build_transforms if attempt("2-1 재확인", lambda: build_transforms(False, DATA_CFG)(np.zeros((8, 8, 3), np.uint8)).shape) else ps.ready_transforms

def _check_2_2():
    ds = ProductDataset(clean_df.file[:5].tolist(), clean_df.label[:5].tolist(), _tf(False, DATA_CFG), IMG_SIZE)
    x, y = ds[2]
    assert len(ds) == 5 and tuple(x.shape) == (3, IMG_SIZE, IMG_SIZE) and y.dtype == torch.long and int(y) == CLASSES.index(clean_df.label[2])
    assert ds.images.shape == (5, IMG_SIZE, IMG_SIZE, 3) and ds.images.dtype == np.uint8
    ds2 = ProductDataset(None, None, _tf(False, DATA_CFG), IMG_SIZE, images=ds.images)
    assert int(ds2[0][1]) == -1 and len(ds2) == 5
    print("Dataset 확인 통과")
attempt("2-2", _check_2_2)
""")

md(r"""
### 2-3 분할과 불균형 재구성
- `stratified_split(labels, val_ratio=0.2, seed=42)`: 층화 분할 `(train_idx, val_idx)`
- `make_loader(dataset, labels, batch_size=64, balanced=False, seed=42)`: `balanced=False`면 `shuffle=True`(+generator), `True`면 표본 가중치 `1 / (그 클래스 개수)`의 `WeightedRandomSampler(…, replacement=True, generator=…)`
""")

write(r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=___)
    return tr, va

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    counts = pd.Series(labels).value_counts()
    weights = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=___, generator=g)
    return DataLoader(dataset, batch_size=batch_size, ___=sampler)
""", r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    '''층화 분할 (train_idx, val_idx)'''
    # 작성하세요

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    '''학습용 DataLoader. balanced=True면 역빈도 WeightedRandomSampler'''
    # 작성하세요
""", r"""
def stratified_split(labels, val_ratio=0.2, seed=42):
    idx = np.arange(len(labels))
    # 층화: 48장뿐인 클래스도 train/val에 같은 비율로 → val에 적어도 9~10장
    tr, va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=labels)
    return tr, va

def make_loader(dataset, labels, batch_size=64, balanced=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    counts = pd.Series(labels).value_counts()
    weights = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)   # 표본마다 하나
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)               # shuffle과 함께 쓰지 않음
""")

run(r"""
def _check_2_3():
    labels = clean_df.label.to_numpy()
    tr, va = stratified_split(labels)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == len(labels)
    ds = ps.ReadyProductDataset(None, labels[tr], lambda a: torch.zeros(1), IMG_SIZE, images=np.zeros((len(tr), 1, 1, 3), np.uint8))
    seen = torch.cat([yb for _, yb in make_loader(ds, labels[tr], balanced=True)])
    share = torch.bincount(seen, minlength=NUM_CLASSES).float() / len(seen)
    print(f"균형 샘플러 1 epoch: 클래스 비율 최소 {share.min():.3f} / 최대 {share.max():.3f} (균등 = {1 / NUM_CLASSES:.3f})")
    assert (share - 1 / NUM_CLASSES).abs().max() < 0.03
    return tr, va
q2_split = attempt("2-3", _check_2_3)
""")

md(r"""
### 2-4 증강 판단 (서술)
`q2_aug_note`에 쓰세요(각 1문장).
1. `RandomAffine`의 `fill=255`가 필요한 이유 (기본값 0이면 어떻게 되는가)
2. `scale=(0.55, 1.0)`처럼 **축소**를 넣은 이유 (1-3의 발견과 연결)
3. 이 과제에서 위험한 증강 하나와 이유 (예: 크게 잘라 내는 `RandomResizedCrop(scale=(0.2, 1))`, 상하 반전)
""")

write(r"""
q2_aug_note = ""  # 작성하세요
""", r"""
q2_aug_note = ""  # 작성하세요
""", r"""
q2_aug_note = (
    "(1) 회전·이동·축소로 생긴 빈 곳을 기본값 0(검정)으로 채우면, 흰 배경 상품 사진에 없는 검은 테두리가 생겨 모델이 엉뚱한 단서를 배운다. "
    "(2) test에는 물체가 작게 찍힌 사진이 train보다 훨씬 많으므로(1-3), 학습 때 물체를 줄여 보여 주어 그 분포에 대비한다. "
    "(3) 크게 잘라 내는 증강은 긴바지의 아래쪽이나 부츠의 목을 잘라 반바지·스니커즈처럼 보이게 만들어 레이블과 맞지 않는 표본을 만들고, "
    "상하 반전은 실제 상품 사진에 없는 모습이라 도움이 되지 않는다."
)
""")

run(r"""
# Q3·Q4에서 쓸 공통 입력 (앞 소문항이 실패하면 진행용 대체물)
_DS = ProductDataset if attempt("2-2 재확인", lambda: ProductDataset(None, None, _tf(False, DATA_CFG), IMG_SIZE, images=np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.uint8))[0][0].shape) else ps.ReadyProductDataset
_loader = make_loader if attempt("2-3 재확인", lambda: len(make_loader(list(range(4)), ["cap", "cap", "boots", "boots"], 2, True))) else ps.ready_train_loader
labels_all = clean_df.label.to_numpy()
if q2_split is None:
    q2_split = train_test_split(np.arange(len(labels_all)), test_size=0.2, random_state=42, stratify=labels_all)
tr_idx, va_idx = q2_split
with ps.Timer("사진 미리 읽기 (train+test)"):
    all_img = ps.preload(clean_df.file, IMG_SIZE, load_fn=_load, pad_fn=_pad)
    test_img = ps.preload(test_df.file, IMG_SIZE, load_fn=_load, pad_fn=_pad)
y_va = np.array([CLASSES.index(c) for c in labels_all[va_idx]])
train_ds_aug = _DS(None, labels_all[tr_idx], _tf(True, DATA_CFG), IMG_SIZE, images=all_img[tr_idx])
train_ds_plain = _DS(None, labels_all[tr_idx], _tf(False, DATA_CFG), IMG_SIZE, images=all_img[tr_idx])

def eval_loader(images, labels=None):
    return DataLoader(_DS(None, labels, _tf(False, DATA_CFG), IMG_SIZE, images=images), batch_size=256, shuffle=False)

val_loader = eval_loader(all_img[va_idx], labels_all[va_idx])
# val_small: 같은 검증 사진의 물체를 0.5배로 줄인 것 → test의 '작게 찍힌 사진'을 흉내 낸 분포 점검용 검증
val_small_loader = eval_loader(ps.zoom_out(all_img[va_idx], 0.5), labels_all[va_idx])
test_loader = eval_loader(test_img)
print("train", len(tr_idx), "| val", len(va_idx), "| test", len(test_img))
""")

# ======================================================================  Q3
md(r"""
---
## Q3 TinyViT 전이학습 (AI 모델링 및 최적화 · 구현 + 성능 기준)

지정 모델 TinyViT는 **합성곱 단계(stage 0)와 attention 단계(stage 1~3)**를 섞은 작은 ViT입니다. 하위 모듈은 `patch_embed → stages[0..3] → head(global_pool → norm → fc)`입니다(`assets/모델카드.md`, `Reference/timm_tinyvit.md`).

- 지표: **macro-F1** (20개 클래스를 똑같은 무게로)
- **성능 기준(가정): test macro-F1 ≥ __Q3_TH__** → `제출물/q3_submission.csv`
""")

run(r"""
print("파라미터 수:", f"{sum(p.numel() for p in model0.parameters()):,}", "(실제 TinyViT-5M ≈ 5.4M)")
print([name for name, _ in model0.named_children()], "| stages:", len(model0.stages), "| head:", model0.head)
print("사전학습 head 출력:", model0.get_classifier())
""")

md(r"""
### 3-1 출력층 교체
`build_classifier(num_classes, drop_path_rate=0.0)`: `create_model("tiny_vit_5m_224", pretrained=True, drop_path_rate=drop_path_rate)`로 불러온 뒤 **출력층만** `num_classes`개로 바꿉니다. timm 모델은 `model.reset_classifier(num_classes)`를 지원합니다(또는 `model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)`). 나머지 가중치는 유지합니다.
""")

write(r"""
def build_classifier(num_classes, drop_path_rate=0.0):
    model = create_model("tiny_vit_5m_224", pretrained=___, drop_path_rate=drop_path_rate)
    model.___(num_classes)                         # head.fc만 새 Linear로
    return model
""", r"""
def build_classifier(num_classes, drop_path_rate=0.0):
    '''사전학습 TinyViT의 출력층만 num_classes개로 바꿔 반환'''
    # 작성하세요
""", r"""
def build_classifier(num_classes, drop_path_rate=0.0):
    # drop_path_rate: 블록의 잔차 가지를 무작위로 끄는 규제(stochastic depth). 미세조정 때 0.05~0.1을 흔히 씀
    model = create_model("tiny_vit_5m_224", pretrained=True, drop_path_rate=drop_path_rate)
    model.reset_classifier(num_classes)            # head.fc = Linear(num_features, num_classes), 나머지는 그대로
    return model
""")

run(r"""
def _check_3_1():
    m, ref = build_classifier(NUM_CLASSES), create_model("tiny_vit_5m_224", pretrained=True)
    assert m(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE)).shape == (2, NUM_CLASSES) and m.head.fc.out_features == NUM_CLASSES
    assert torch.equal(m.stages[3].blocks[0].attn.qkv.weight, ref.stages[3].blocks[0].attn.qkv.weight), "사전학습 가중치가 바뀌었습니다"
    print("head.fc:", m.head.fc)
attempt("3-1", _check_3_1)
""")

md(r"""
### 3-2 학습 범위와 layer-wise lr decay
- `set_trainable(model, scope)`: `scope`가 `"head"`면 이름이 `head`로 시작하는 파라미터만, `"last"`면 `stages.3`·`head`, `"all"`이면 전부 `requires_grad=True`. 나머지는 False. 학습할 파라미터 **리스트**를 반환
- `param_groups_llrd(model, base_lr, layer_decay=1.0, weight_decay=0.05)`: `requires_grad`인 파라미터만 optimizer 그룹으로 나눕니다.
  - 깊이 번호: `head` → 0, `stages.3` → 1, `stages.2` → 2, `stages.1` → 3, `stages.0` → 4, 그 외(`patch_embed`) → 5
  - 그룹 lr = `base_lr × layer_decay ** 깊이` — **입력에 가까울수록 작은 학습률**(사전학습 특징을 덜 흔듦)
  - 1차원 파라미터(bias, LayerNorm·BN 가중치)는 `weight_decay=0.0`, 나머지는 `weight_decay`
  - 반환: `{"params": [...], "lr": ..., "weight_decay": ...}` dict의 리스트(같은 깊이·같은 decay 여부끼리 한 그룹)
""")

write(r"""
SCOPES = {"head": ("head",), "last": ("stages.3", "head"), "all": ("",)}     # "" 는 모든 이름에 맞음

def set_trainable(model, scope):
    prefixes = SCOPES[scope]
    for name, p in model.named_parameters():
        p.requires_grad = name.___(prefixes)
    return [p for p in model.parameters() if p.___]

def layer_depth(name):
    if name.startswith("head"):
        return 0
    if name.startswith("stages."):
        return 4 - int(name.split(".")[___])            # stages.3 → 1, …, stages.0 → 4
    return 5

def param_groups_llrd(model, base_lr, layer_decay=1.0, weight_decay=0.05):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = layer_depth(name)
        no_wd = p.___ == 1                               # bias·정규화 가중치
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * layer_decay ** ___,
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())
""", r"""
SCOPES = {"head": ("head",), "last": ("stages.3", "head"), "all": ("",)}     # "" 는 모든 이름에 맞음

def set_trainable(model, scope):
    '''scope에 맞게 requires_grad 설정, 학습할 파라미터 리스트 반환'''
    # 작성하세요

def param_groups_llrd(model, base_lr, layer_decay=1.0, weight_decay=0.05):
    '''깊이별 학습률·weight decay 그룹 리스트'''
    # 작성하세요
""", r"""
SCOPES = {"head": ("head",), "last": ("stages.3", "head"), "all": ("",)}     # "" 는 모든 이름에 맞음

def set_trainable(model, scope):
    prefixes = SCOPES[scope]
    for name, p in model.named_parameters():       # 예: 'stages.3.blocks.0.attn.qkv.weight', 'head.fc.bias'
        p.requires_grad = name.startswith(prefixes)
    return [p for p in model.parameters() if p.requires_grad]

def layer_depth(name):
    # 출력에서 멀수록(입력에 가까울수록) 큰 번호
    if name.startswith("head"):
        return 0
    if name.startswith("stages."):
        return 4 - int(name.split(".")[1])          # 'stages.2.blocks…' → split → ['stages', '2', …]
    return 5                                        # patch_embed

def param_groups_llrd(model, base_lr, layer_decay=1.0, weight_decay=0.05):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:                     # 고정한 파라미터는 optimizer에 넣지 않음
            continue
        depth = layer_depth(name)
        no_wd = p.ndim == 1                         # bias·LayerNorm·BN 가중치에는 weight decay를 주지 않는 것이 관례
        key = (depth, no_wd)
        if key not in groups:
            # layer_decay=0.75면 head 1배, stages.3 0.75배, …, patch_embed 0.24배
            groups[key] = {"params": [], "lr": base_lr * layer_decay ** depth,
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())
""")

run(r"""
def _check_3_2():
    m = build_classifier(NUM_CLASSES)
    n_all = sum(p.numel() for p in m.parameters())
    for scope in ["head", "last", "all"]:
        params = set_trainable(m, scope)
        print(f"{scope:5s}: 학습 파라미터 {sum(p.numel() for p in params):>7,} / {n_all:,}")
    assert not m.patch_embed.conv1.conv.weight.requires_grad is False
    set_trainable(m, "last")
    assert m.stages[3].blocks[0].mlp.fc1.weight.requires_grad and not m.stages[2].blocks[0].mlp.fc1.weight.requires_grad
    set_trainable(m, "all")
    groups = param_groups_llrd(m, 1e-3, 0.5, 0.05)
    lrs = sorted({round(g["lr"], 8) for g in groups})
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters())), "빠진 파라미터가 있습니다"
    assert lrs[-1] == 1e-3 and abs(lrs[0] - 1e-3 * 0.5 ** 5) < 1e-12, lrs
    assert all(g["weight_decay"] == 0.0 for g in groups if all(p.ndim == 1 for p in g["params"]))
    print("그룹 수", len(groups), "| 학습률 종류", lrs)
attempt("3-2", _check_3_2)
""")

md(r"""
### 3-3 손실 함수: label smoothing과 mixup (직접 구현)
- `smooth_targets(y, num_classes, eps)`: 정수 정답 `(B,)` → `(1 − eps) × one-hot + eps / num_classes` 인 `(B, C)` 실수 텐서
- `soft_target_ce(logits, targets)`: 표본마다 `−Σ targets · log_softmax(logits)`의 평균 (`F.cross_entropy` 호출 금지)
- `mixup_batch(x, targets, alpha)`: `lam ~ Beta(alpha, alpha)`(`torch.distributions.Beta`), `perm = torch.randperm(B)`로 `x·lam + x[perm]·(1−lam)`, 정답도 같은 비율로 섞어 `(x_mix, t_mix, lam)` 반환

mixup은 두 상품 사진을 반투명하게 겹치고 정답도 그 비율로 섞어, 결정 경계를 부드럽게 만드는 규제입니다(데이터가 적고 과적합할 때).
""")

write(r"""
def smooth_targets(y, num_classes, eps):
    one_hot = F.one_hot(y, num_classes).___()
    return one_hot * (1 - eps) + eps / ___

def soft_target_ce(logits, targets):
    return -(targets * F.___(logits, dim=1)).sum(dim=1).___()

def mixup_batch(x, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mix = x * lam + x[perm] * (1 - ___)
    t_mix = targets * lam + targets[___] * (1 - lam)
    return x_mix, t_mix, lam
""", r"""
def smooth_targets(y, num_classes, eps):
    '''(B,) 정수 → (B,C) smoothing된 목표 분포'''
    # 작성하세요

def soft_target_ce(logits, targets):
    '''soft target cross entropy 평균'''
    # 작성하세요

def mixup_batch(x, targets, alpha):
    '''(x_mix, t_mix, lam)'''
    # 작성하세요
""", r"""
def smooth_targets(y, num_classes, eps):
    one_hot = F.one_hot(y, num_classes).float()                 # (B, C)
    # 정답 확률을 1 대신 1-eps+eps/C로 → 과신(overconfidence)을 줄이는 규제
    return one_hot * (1 - eps) + eps / num_classes

def soft_target_ce(logits, targets):
    # 목표가 확률 분포(soft label)여도 쓸 수 있는 CE. one-hot이면 일반 CE와 같다
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

def mixup_batch(x, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())   # alpha가 작을수록 lam이 0이나 1 근처(약하게 섞임)
    perm = torch.randperm(x.shape[0], device=x.device)            # 배치 안에서 짝을 무작위로
    x_mix = x * lam + x[perm] * (1 - lam)
    t_mix = targets * lam + targets[perm] * (1 - lam)             # 정답도 같은 비율로 섞어야 일관됨
    return x_mix, t_mix, lam
""")

run(r"""
def _check_3_3():
    torch.manual_seed(0)
    lg, y = torch.randn(16, NUM_CLASSES) * 3, torch.randint(0, NUM_CLASSES, (16,))
    t = smooth_targets(y, NUM_CLASSES, 0.1)
    assert t.shape == (16, NUM_CLASSES) and torch.allclose(t.sum(1), torch.ones(16))
    assert torch.allclose(soft_target_ce(lg, t), F.cross_entropy(lg, y, label_smoothing=0.1), atol=1e-5)
    assert torch.allclose(soft_target_ce(lg, smooth_targets(y, NUM_CLASSES, 0.0)), F.cross_entropy(lg, y), atol=1e-5)
    x = torch.randn(16, 3, 4, 4)
    xm, tm, lam = mixup_batch(x, t, 0.4)
    assert xm.shape == x.shape and torch.allclose(tm.sum(1), torch.ones(16)) and 0 <= lam <= 1
    print(f"label smoothing·mixup 확인 통과 (lam 예: {lam:.2f})")
attempt("3-3", _check_3_3)
""")

md(r"""
### 3-4 학습 루프: warmup + cosine, gradient clipping
- `warmup_cosine(optimizer, warmup_steps, total_steps)`: `torch.optim.lr_scheduler.LambdaLR`를 반환. step `s`의 배율 = `s < warmup`이면 `(s + 1) / warmup`, 아니면 `0.5 × (1 + cos(π × (s − warmup) / (total − warmup)))`. **배치마다** `scheduler.step()`
- `train_one_epoch(model, loader, optimizer, scheduler, cfg, device)`:
  1. `model.train()` 후, **학습하지 않는(파라미터가 모두 고정된) BatchNorm2d는 `eval()`**
  2. 배치마다: 장치 이동 → `smooth_targets(y, NUM_CLASSES, cfg["smoothing"])` → `cfg["mixup"] > 0`이고 `torch.rand(1) < 0.5`면 `mixup_batch(…, cfg["mixup"])` → `zero_grad` → `soft_target_ce` → `backward` → `clip_grad_norm_(학습 파라미터, 1.0)` → `optimizer.step()` → `scheduler.step()`
  3. 표본 가중 평균 손실(float) 반환
- `predict_logits(model, loader, device)`: `eval()` + `no_grad`, CPU 텐서 `(N, C)`
""")

write(r"""
def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.___(optimizer, factor)

def train_one_epoch(model, loader, optimizer, scheduler, cfg, device):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.___()
    params = [p for p in model.parameters() if p.requires_grad]
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        tb = smooth_targets(yb, NUM_CLASSES, cfg["smoothing"])
        if cfg["mixup"] > 0 and torch.rand(1).item() < 0.5:
            xb, tb, _ = mixup_batch(xb, tb, cfg["mixup"])
        optimizer.zero_grad()
        loss = soft_target_ce(model(xb), tb)
        loss.___()
        torch.nn.utils.clip_grad_norm_(params, ___)
        optimizer.step()
        scheduler.___()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)

def predict_logits(model, loader, device):
    model.eval()
    outs = []
    with torch.___():
        for xb, _ in loader:
            outs.append(model(xb.to(device)).cpu())
    return torch.cat(outs)
""", r"""
def warmup_cosine(optimizer, warmup_steps, total_steps):
    '''LambdaLR: 선형 warmup 후 cosine 감소'''
    # 작성하세요

def train_one_epoch(model, loader, optimizer, scheduler, cfg, device):
    '''한 epoch 학습, 표본 가중 평균 손실(float)'''
    # 작성하세요

def predict_logits(model, loader, device):
    '''eval + no_grad 로 logits (N,C) CPU 텐서'''
    # 작성하세요
""", r"""
def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps          # 처음엔 작게 → 무작위 초기화된 head가 큰 기울기로 몸통을 망치지 않게
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)   # 그룹마다 자기 lr에 같은 배율을 곱함(LLRD 유지)

def train_one_epoch(model, loader, optimizer, scheduler, cfg, device):
    model.train()
    for m in model.modules():                       # 고정한 단계의 BN이 running 통계를 바꾸지 않도록
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        tb = smooth_targets(yb, NUM_CLASSES, cfg["smoothing"])
        if cfg["mixup"] > 0 and torch.rand(1).item() < 0.5:   # 배치의 절반만 섞어 원본도 보게
            xb, tb, _ = mixup_batch(xb, tb, cfg["mixup"])
        optimizer.zero_grad()
        loss = soft_target_ce(model(xb), tb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)            # ViT 미세조정에서 흔한 안정화: 기울기 폭주 방지
        optimizer.step()
        scheduler.step()                                       # step 단위 스케줄
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)

def predict_logits(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            outs.append(model(xb.to(device)).cpu())
    return torch.cat(outs)
""")

run(r"""
def _check_3_4():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = warmup_cosine(opt, 4, 20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    assert abs(lrs[0] - 0.25) < 1e-9 and abs(lrs[3] - 1.0) < 1e-9 and lrs[-1] < 0.05 and lrs[4] > lrs[10]
    m = build_classifier(NUM_CLASSES).to(DEVICE)
    set_trainable(m, "last")
    opt = torch.optim.AdamW(param_groups_llrd(m, 1e-3, 0.8), lr=1e-3)
    small = DataLoader(torch.utils.data.Subset(train_ds_aug, range(128)), batch_size=64, shuffle=True)
    sch = warmup_cosine(opt, 1, len(small))
    before = m.patch_embed.conv1.bn.running_mean.clone()
    loss = train_one_epoch(m, small, opt, sch, dict(smoothing=0.1, mixup=0.4), DEVICE)
    assert torch.equal(before, m.patch_embed.conv1.bn.running_mean), "고정한 BN의 통계가 바뀌었습니다"
    out = predict_logits(m, eval_loader(all_img[va_idx[:50]]), DEVICE)
    assert isinstance(loss, float) and tuple(out.shape) == (50, NUM_CLASSES) and out.device.type == "cpu"
    print("warmup 처음 4 step 배율", [round(v, 2) for v in lrs[:5]], "| 짧은 학습 loss", round(loss, 3))
attempt("3-4", _check_3_4)
""")

md(r"""
### 3-5 최고 가중치 보관
`fit(model, train_loader, select_loader, y_select, optimizer, scheduler, cfg, device)`: epoch마다 `train_one_epoch` → `select_loader` 예측 → macro-F1. **최고일 때만** `copy.deepcopy(model.state_dict())` 보관, 끝에 복원. `(model, best_f1, history DataFrame)` 반환 — `history` 열: `epoch, train_loss, select_f1, lr`(그 epoch 끝의 첫 그룹 lr)
""")

write(r"""
def fit(model, train_loader, select_loader, y_select, optimizer, scheduler, cfg, device):
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        pred = predict_logits(model, select_loader, device).argmax(1).numpy()
        f1 = f1_score(y_select, pred, average=___)
        history.append({"epoch": ep, "train_loss": round(loss, 4), "select_f1": round(f1, 4),
                        "lr": optimizer.param_groups[0]["lr"]})
        if f1 > best_f1:
            best_f1, best_state = f1, copy.___(model.state_dict())
    model.load_state_dict(___)
    return model, best_f1, pd.DataFrame(history)
""", r"""
def fit(model, train_loader, select_loader, y_select, optimizer, scheduler, cfg, device):
    '''최고 macro-F1 가중치로 되돌린 (model, best_f1, history)'''
    # 작성하세요
""", r"""
def fit(model, train_loader, select_loader, y_select, optimizer, scheduler, cfg, device):
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        pred = predict_logits(model, select_loader, device).argmax(1).numpy()
        f1 = f1_score(y_select, pred, average="macro")
        history.append({"epoch": ep, "train_loss": round(loss, 4), "select_f1": round(f1, 4),
                        "lr": optimizer.param_groups[0]["lr"]})
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())   # 참조가 아니라 복사본
    model.load_state_dict(best_state)                                      # 마지막이 아니라 최고 epoch로
    return model, best_f1, pd.DataFrame(history)
""")

run(r"""
def _check_3_5():
    ps.seed_everything(0)
    m = build_classifier(NUM_CLASSES).to(DEVICE)
    set_trainable(m, "head")
    opt = torch.optim.AdamW(param_groups_llrd(m, 3e-3), lr=3e-3)
    small = DataLoader(torch.utils.data.Subset(train_ds_plain, range(256)), batch_size=64, shuffle=True)
    sch = warmup_cosine(opt, 2, 3 * len(small))
    m, best, hist = fit(m, small, val_loader, y_va, opt, sch, dict(epochs=3, smoothing=0.0, mixup=0.0), DEVICE)
    again = f1_score(y_va, predict_logits(m, val_loader, DEVICE).argmax(1).numpy(), average="macro")
    assert list(hist.columns) == ["epoch", "train_loss", "select_f1", "lr"] and abs(again - best) < 1e-9
    print(hist.to_string(index=False))
attempt("3-5", _check_3_5)
""")

md(r"""
### 3-6 실험과 선택 (성능 기준)
아래 셀이 먼저 **baseline**(head만, lr 1e-3, 4 epoch, 증강·smoothing·mixup 없음)을 학습합니다. 그다음 `Q3_CANDIDATES`에 설정 **2~3개**를, `Q3_SELECT_BY`에 고를 기준(`"val"` 또는 `"val_small"`)을 적으세요. 설정 키:

| 키 | 값 |
|---|---|
| `scope` | `"head"` / `"last"` / `"all"` |
| `lr`, `layer_decay` | 기본 학습률, 깊이별 감소율(1.0이면 모두 같은 lr) |
| `epochs` | epoch 수 (CPU 기준 epoch당 수 초) |
| `smoothing`, `mixup` | label smoothing eps, mixup alpha(0이면 끔) |
| `drop_path` | stochastic depth 비율 |
| `augment`, `balanced` | 2-1 학습 증강 사용, 균형 샘플러 사용 |

생각할 것: head만 학습하면 사전학습 특징(도형)을 상품 구분에 맞게 바꿀 수 없습니다. 전체를 학습할 때는 입력 쪽 학습률을 줄이세요(layer decay). test는 작게 찍힌 사진이 많습니다(1-3).
""")

run(r"""
def run_q3(cfg, seed=42, verbose=True):
    ps.seed_everything(seed)
    model = build_classifier(NUM_CLASSES, cfg["drop_path"]).to(DEVICE)
    set_trainable(model, cfg["scope"])
    opt = torch.optim.AdamW(param_groups_llrd(model, cfg["lr"], cfg["layer_decay"]), lr=cfg["lr"])
    ds = train_ds_aug if cfg["augment"] else train_ds_plain
    loader = _loader(ds, labels_all[tr_idx], batch_size=64, balanced=cfg["balanced"], seed=seed)
    total = cfg["epochs"] * len(loader)
    sched = warmup_cosine(opt, max(1, int(0.1 * total)), total)
    select_loader = val_small_loader if globals().get("Q3_SELECT_BY") == "val_small" else val_loader
    t0 = time.time()
    model, best, hist = fit(model, loader, select_loader, y_va, opt, sched, cfg, DEVICE)
    res = {k: v for k, v in cfg.items()}
    res["best_epoch"] = int(hist.loc[hist.select_f1.idxmax(), "epoch"])
    for name, ld in [("val", val_loader), ("val_small", val_small_loader)]:
        pred = predict_logits(model, ld, DEVICE).argmax(1).numpy()
        res[f"{name}_f1"] = round(f1_score(y_va, pred, average="macro"), 3)
        res[f"{name}_worst_recall"] = round(float(recall_score(y_va, pred, average=None, labels=range(NUM_CLASSES)).min()), 2)
    res["sec"] = round(time.time() - t0)
    if verbose:
        print({k: res[k] for k in ["scope", "lr", "epochs", "val_f1", "val_small_f1", "sec"]})
    return model, res

BASELINE = dict(scope="head", lr=1e-3, layer_decay=1.0, epochs=4, smoothing=0.0, mixup=0.0, drop_path=0.0, augment=False, balanced=False)
q3_base = attempt("3-6 baseline", run_q3, BASELINE)
""")

write(r"""
Q3_CANDIDATES = [
    dict(scope="last", lr=1e-3, layer_decay=1.0, epochs=12, smoothing=0.1, mixup=0.0, drop_path=0.0, augment=True, balanced=False),
    dict(scope=___, lr=___, layer_decay=___, epochs=___, smoothing=___, mixup=___, drop_path=___, augment=___, balanced=___),
]
Q3_SELECT_BY = ___   # "val" 또는 "val_small"
""", r"""
Q3_CANDIDATES = []   # 작성하세요: 설정 dict 2~3개 (키는 위 표)
Q3_SELECT_BY = ""    # 작성하세요: "val" 또는 "val_small"
""", r"""
Q3_CANDIDATES = [
    # A: 마지막 단계 + head만 — "얼마나 넓게 학습해야 하나"의 중간 지점
    dict(scope="last", lr=1e-3, layer_decay=1.0, epochs=12, smoothing=0.1, mixup=0.0, drop_path=0.0, augment=True, balanced=False),
    # B: 전체 미세조정, 모든 층 같은 lr (layer decay 없음)
    dict(scope="all", lr=2e-3, layer_decay=1.0, epochs=12, smoothing=0.1, mixup=0.0, drop_path=0.05, augment=True, balanced=False),
    # C: 전체 + layer decay 0.75 + mixup + 균형 샘플러 — ViT 미세조정의 흔한 묶음이 여기서도 나은지 확인
    dict(scope="all", lr=2e-3, layer_decay=0.75, epochs=12, smoothing=0.1, mixup=0.4, drop_path=0.05, augment=True, balanced=True),
]
Q3_SELECT_BY = "val_small"   # test에 작게 찍힌 사진이 많으므로(1-3) 그 조건을 흉내 낸 검증으로 고른다
""")

run(r"""
def _run_3_6():
    assert 1 <= len(Q3_CANDIDATES) <= 3 and Q3_SELECT_BY in ("val", "val_small"), "설정 2~3개와 선택 기준을 적으세요"
    runs = [run_q3(c) for c in Q3_CANDIDATES]
    rows = ([{"name": "baseline", **q3_base[1]}] if q3_base else []) + [{"name": f"후보{i}", **r} for i, (_, r) in enumerate(runs)]
    table = pd.DataFrame(rows).drop(columns=["best_epoch"])
    print(table.to_string(index=False))
    best_i = int(np.argmax([r[f"{Q3_SELECT_BY}_f1"] for _, r in runs]))
    print(f"선택 기준 {Q3_SELECT_BY}_f1 → 후보{best_i}")
    return runs[best_i][0], Q3_CANDIDATES[best_i], table

q3_out = attempt("3-6", _run_3_6)
if q3_out:
    q3_model, q3_cfg, q3_table = q3_out
    q3_test_pred = np.array(CLASSES)[predict_logits(q3_model, test_loader, DEVICE).argmax(1).numpy()]
    print("test 예측 분포(상위 5):", pd.Series(q3_test_pred).value_counts().head().to_dict())
""")

md(r"""
### 3-7 제출 파일 (제출가이드 준수)
- `make_submission(sample, ids, values)`: `sample`(sample_submission)의 **id 순서 그대로** `id, label` DataFrame. `values[k]`는 `ids[k]`의 예측 — **test.csv와 sample_submission의 행 순서가 다르므로 id로 맞춰야 합니다.**
- `q3_sub`를 만들어 `제출물/q3_submission.csv`로 저장(인덱스 없이)
""")

write(r"""
def make_submission(sample, ids, values):
    pred = pd.Series(values, index=___)
    sub = sample[["id"]].copy()
    sub["label"] = pred.___(sub["id"]).to_numpy()
    return sub

q3_sub = make_submission(sample_df, test_df["id"].to_numpy(), q3_test_pred)
q3_sub.to_csv(ps.SUBMIT / "q3_submission.csv", index=___)
""", r"""
def make_submission(sample, ids, values):
    '''sample의 id 순서로 [id, label] DataFrame'''
    # 작성하세요

# 작성하세요: q3_sub 만들고 제출물/q3_submission.csv 로 저장
""", r"""
def make_submission(sample, ids, values):
    pred = pd.Series(values, index=ids)              # test.csv 순서의 예측을 id로 찾을 수 있게
    sub = sample[["id"]].copy()                      # 행 순서의 기준은 sample_submission
    sub["label"] = pred.reindex(sub["id"]).to_numpy()   # 위치가 아니라 id로 맞춤
    return sub

q3_sub = make_submission(sample_df, test_df["id"].to_numpy(), q3_test_pred)
q3_sub.to_csv(ps.SUBMIT / "q3_submission.csv", index=False)
""")

run(r"""
print("q3 형식 점검:", ps.check_submission("q3") or "통과")
""")

md(r"""
### 3-8 분석 (서술)
`q3_analysis`: (1) baseline과 후보들의 `val`/`val_small` 결과 비교(수치 인용), (2) 선택 기준과 최종 선택의 이유, (3) 학습 범위·layer decay·증강 중 무엇이 효과가 컸는지와 그 이유
""")

write(r"""
q3_analysis = ""  # 작성하세요
""", r"""
q3_analysis = ""  # 작성하세요
""", r"""
q3_analysis = (
    "__Q3_ANALYSIS__"
)
""")

# ======================================================================  Q4
md(r"""
---
## Q4 오분류 분석과 성능 극대화 (AI 모델링 및 최적화 · 성능 기준)

3-6에서 고른 모델(`q3_model`, 설정 `q3_cfg`)을 분석하고, **학습을 거의 다시 하지 않는 방법(TTA)**과 **여러 모델을 합치는 방법(앙상블)**으로 점수를 올립니다.
- **성능 기준(가정): test macro-F1 ≥ __Q4_TH__** → `제출물/q4_submission.csv`
""")

md(r"""
### 4-1 어디서 틀리는가
`top_confusions(y_true, y_pred, k=6)`: 정수 레이블 두 배열로 혼동 행렬(`labels=range(NUM_CLASSES)`)을 만들고, **대각선을 뺀** 칸 중 개수가 큰 `k`개를 `실제, 예측, 개수` 열의 DataFrame으로(개수 내림차순, 클래스는 이름으로) 반환
""")

write(r"""
def top_confusions(y_true, y_pred, k=6):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    np.fill_diagonal(cm, ___)                                   # 맞힌 칸은 제외
    flat = np.argsort(cm, axis=None)[::-1][:k]                   # 큰 칸부터
    rows = [(CLASSES[i // NUM_CLASSES], CLASSES[i % ___], int(cm.flat[i])) for i in flat]
    return pd.DataFrame(rows, columns=["실제", "예측", "개수"])
""", r"""
def top_confusions(y_true, y_pred, k=6):
    '''대각선을 뺀 혼동 상위 k개 DataFrame (실제, 예측, 개수)'''
    # 작성하세요
""", r"""
def top_confusions(y_true, y_pred, k=6):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))   # 행 = 실제, 열 = 예측
    np.fill_diagonal(cm, 0)                                            # 맞힌 칸은 관심 밖
    flat = np.argsort(cm, axis=None)[::-1][:k]                          # 펼친 위치를 큰 값부터
    rows = [(CLASSES[i // NUM_CLASSES], CLASSES[i % NUM_CLASSES], int(cm.flat[i])) for i in flat]   # 위치 → (행, 열)
    return pd.DataFrame(rows, columns=["실제", "예측", "개수"])
""")

run(r"""
def _analyze_4_1():
    toy = top_confusions(np.array([0, 0, 1, 1, 1, 2]), np.array([1, 1, 1, 0, 1, 2]), k=2)
    assert toy.values.tolist() == [[CLASSES[0], CLASSES[1], 2], [CLASSES[1], CLASSES[0], 1]]
    for name, ld in [("val", val_loader), ("val_small", val_small_loader)]:
        pred = predict_logits(q3_model, ld, DEVICE).argmax(1).numpy()
        print(f"--- {name}: macro-F1 {f1_score(y_va, pred, average='macro'):.3f}")
        print(top_confusions(y_va, pred).to_string(index=False))
    pred = predict_logits(q3_model, val_small_loader, DEVICE).argmax(1).numpy()
    f1s = f1_score(y_va, pred, average=None, labels=range(NUM_CLASSES))
    print("val_small F1 낮은 클래스:", {CLASSES[i]: round(f1s[i], 2) for i in np.argsort(f1s)[:5]})
    top = top_confusions(y_va, pred, k=1).iloc[0]
    wrong = np.flatnonzero((y_va == CLASSES.index(top["실제"])) & (pred == CLASSES.index(top["예측"])))[:8]
    ps.show_images([ps.zoom_out(all_img[va_idx][[i]], 0.5)[0] for i in wrong], [f"{top['실제']}→{top['예측']}"] * len(wrong))
attempt("4-1", _analyze_4_1)
""")

md(r"""
### 4-2 TTA
`predict_proba_tta(model, loader, device)`: 배치마다 원본과 **좌우 반전**(`xb.flip(3)`) 두 입력의 `softmax` 확률을 평균해 CPU 텐서 `(N, C)`로 반환(`eval()` + `no_grad`). logits가 아니라 **확률**을 평균합니다.
""")

write(r"""
def predict_proba_tta(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            p = model(xb).softmax(1) + model(xb.flip(___)).softmax(1)
            outs.append((p / ___).cpu())
    return torch.cat(outs)
""", r"""
def predict_proba_tta(model, loader, device):
    '''원본 + 좌우 반전 확률 평균 (N,C)'''
    # 작성하세요
""", r"""
def predict_proba_tta(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            # 학습 때 쓴 변환(좌우 반전)만 TTA에 쓴다: 상품 종류가 바뀌지 않는 변형
            p = model(xb).softmax(1) + model(xb.flip(3)).softmax(1)
            outs.append((p / 2).cpu())
    return torch.cat(outs)
""")

run(r"""
def _check_4_2():
    p = predict_proba_tta(q3_model, val_loader, DEVICE)
    assert tuple(p.shape) == (len(va_idx), NUM_CLASSES) and torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5)
    for name, ld in [("val", val_loader), ("val_small", val_small_loader)]:
        single = predict_logits(q3_model, ld, DEVICE).argmax(1).numpy()
        tta = predict_proba_tta(q3_model, ld, DEVICE).argmax(1).numpy()
        print(f"{name:9s} 단일 {f1_score(y_va, single, average='macro'):.3f} → TTA {f1_score(y_va, tta, average='macro'):.3f}")
attempt("4-2", _check_4_2)
""")

md(r"""
### 4-3 앙상블
`ensemble_proba(models, loader, device, tta=True)`: 모델마다 확률(`tta=True`면 `predict_proba_tta`, 아니면 `predict_logits(...).softmax(1)`)을 구해 **평균**한 `(N, C)`

아래 셀은 3-6에서 고른 설정(`q3_cfg`)을 seed만 바꿔 2개 더 학습하고(`Q4_EXTRA_SEEDS`), 단일·TTA·앙상블을 비교합니다. 시간이 있으면 **서로 다른 설정**(예: 다른 후보)을 섞는 편이 다양성이 큽니다.
""")

write(r"""
def ensemble_proba(models, loader, device, tta=True):
    probs = [predict_proba_tta(m, loader, device) if tta else predict_logits(m, loader, device).___(1) for m in models]
    return torch.stack(probs).___(0)
""", r"""
def ensemble_proba(models, loader, device, tta=True):
    '''모델별 확률 평균 (N,C)'''
    # 작성하세요
""", r"""
def ensemble_proba(models, loader, device, tta=True):
    # 확률을 평균(soft voting). logits를 평균하면 크기가 큰 모델 하나가 결과를 좌우할 수 있다
    probs = [predict_proba_tta(m, loader, device) if tta else predict_logits(m, loader, device).softmax(1) for m in models]
    return torch.stack(probs).mean(0)
""")

run(r"""
Q4_EXTRA_SEEDS = [7, 2024]
def _run_4_3():
    with ps.Timer("추가 모델 학습"):
        extra = [run_q3(q3_cfg, seed=s, verbose=False)[0] for s in Q4_EXTRA_SEEDS]
    models = [q3_model] + extra
    rows = []
    for name, fn in [("단일", lambda ld: predict_logits(q3_model, ld, DEVICE).softmax(1)),
                     ("단일+TTA", lambda ld: predict_proba_tta(q3_model, ld, DEVICE)),
                     (f"{len(models)}개 앙상블", lambda ld: ensemble_proba(models, ld, DEVICE, tta=False)),
                     (f"{len(models)}개 앙상블+TTA", lambda ld: ensemble_proba(models, ld, DEVICE, tta=True))]:
        rows.append({"방법": name, **{f"{n}_f1": round(f1_score(y_va, fn(ld).argmax(1).numpy(), average="macro"), 3)
                                    for n, ld in [("val", val_loader), ("val_small", val_small_loader)]}})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return models, table
q4_out = attempt("4-3", _run_4_3)
""")

md(r"""
### 4-4 최종 제출 (성능 기준)
`Q4_FINAL`에 `"single"`, `"tta"`, `"ensemble"`, `"ensemble_tta"` 중 하나를 **위 표(validation)의 근거로** 고르세요. 아래 셀이 test 확률을 만들고(`q4_test_pred`), 여러분이 3-7의 `make_submission`으로 `제출물/q4_submission.csv`를 저장합니다.
""")

write(r"""
Q4_FINAL = ___   # "single" | "tta" | "ensemble" | "ensemble_tta"
""", r"""
Q4_FINAL = ""    # 작성하세요
""", r"""
Q4_FINAL = "ensemble_tta"   # val·val_small 모두에서 가장 높았고, 다양한 모델·변형의 평균이 분산을 줄인다
""")

run(r"""
def _final_4_4():
    models = q4_out[0] if q4_out else [q3_model]
    fn = {"single": lambda: predict_logits(q3_model, test_loader, DEVICE).softmax(1),
          "tta": lambda: predict_proba_tta(q3_model, test_loader, DEVICE),
          "ensemble": lambda: ensemble_proba(models, test_loader, DEVICE, tta=False),
          "ensemble_tta": lambda: ensemble_proba(models, test_loader, DEVICE, tta=True)}[Q4_FINAL]
    return np.array(CLASSES)[fn().argmax(1).numpy()]
q4_test_pred = attempt("4-4 예측", _final_4_4)
""")

write(r"""
q4_sub = make_submission(sample_df, test_df["id"].to_numpy(), ___)
q4_sub.to_csv(ps.SUBMIT / ___, index=False)
""", r"""
# 작성하세요: q4_sub 만들고 제출물/q4_submission.csv 로 저장
""", r"""
q4_sub = make_submission(sample_df, test_df["id"].to_numpy(), q4_test_pred)
q4_sub.to_csv(ps.SUBMIT / "q4_submission.csv", index=False)
""")

run(r"""
print("q4 형식 점검:", ps.check_submission("q4") or "통과")
""")

md(r"""
### 4-5 분석 (서술)
`q4_analysis`: (1) 주로 혼동한 클래스 쌍과 그 이유(사진의 어떤 특징이 비슷한가), (2) TTA·앙상블이 준 변화(수치 인용)와 최종 선택 이유, (3) **시간이 더 있다면 시도할 개선안 2가지**
""")

write(r"""
q4_analysis = ""  # 작성하세요
""", r"""
q4_analysis = ""  # 작성하세요
""", r"""
q4_analysis = (
    "__Q4_ANALYSIS__"
)
""")

# ======================================================================  Q5
md(r"""
---
## Q5 상품 설명 → JSON (프롬프트 엔지니어링 · 구현 + 기준)

판매자가 자유롭게 쓴 **상품 설명 문장**(`data/texts/products.csv`: `id, split, text`)에서 정보를 뽑아, 쇼핑몰 소프트웨어가 바로 쓸 수 있는 JSON으로 만듭니다. `dev` 40개는 정답(`dev_labels.json`)이 있고 `test` 30개는 없습니다.
- **FAQ: 지정 모델은 이미지 모델이라 채팅이 안 됩니다.** 이 문항은 언어모델을 실행하지 않고, `ps.mock_llm(id)`가 **미리 만든 응답 문자열**을 돌려줍니다(프롬프트와 무관). 실제 응답에서 흔한 형식 오류(코드 블록, 설명문, 한국어 값, `"39,000원"` 같은 문자열 숫자, 범위 표기 `"S~XL"`, 파이썬 dict 표기, 끝 쉼표, 끊긴 응답, 여분의 키)를 섞었습니다.
- **기준(가정): dev 필드 정확도 ≥ 0.95(test ≥ 0.93 — 끊긴 응답은 복구하지 않으므로 상한이 있음), 자료형 위반 레코드 0건.** 제출: `제출물/q5_submission.jsonl` (`{"id": ..., "record": {...}}` 한 줄씩)

**출력 스키마 (키 8개)**

| 필드 | 자료형 | 규칙 |
|---|---|---|
| category | 20개 클래스 이름(영문) 또는 null | 이미지 과제와 같은 `CLASSES` |
| brand | 문자열 또는 null | `[브랜드]` 표기. 없으면 null |
| price_krw | 정수(원) 또는 null | `39,000원`·`3만 9천원`·`₩45,000` → 정수. 할인가가 있으면 할인가. **외화(`$`)·"가격 문의"는 null** |
| colors | 배열 (`black, white, gray, red, blue, green, beige, brown, navy, pink`) | 없으면 `[]` |
| sizes | 문자열 배열 | `S~XL` → `["S","M","L","XL"]`, `FREE`·`프리사이즈` → `["FREE"]`, 신발 `240/250` → `["240","250"]`. 없으면 `[]` |
| material | `cotton, polyester, wool, leather, denim, nylon` 또는 null | **혼방은 null** |
| in_stock | true / false / null | "품절"·"일시품절"·"재입고 예정"은 false, "재고 있음"·"즉시 출고"는 true, 언급 없으면 null |
| free_shipping | true / false / null | "무료배송" true, "배송비 3,000원" false, **"5만원 이상 무료배송"처럼 조건부면 null** |
""")

run(r"""
texts = pd.read_csv(ps.DATA / "texts" / "products.csv", dtype={"id": str})
dev_gold = json.loads((ps.DATA / "texts" / "dev_labels.json").read_text(encoding="utf-8"))
dev = texts[texts.split == "dev"].reset_index(drop=True)
test_texts = texts[texts.split == "test"].reset_index(drop=True)

SCHEMA_KEYS = ["category", "brand", "price_krw", "colors", "sizes", "material", "in_stock", "free_shipping"]
COLORS = ["black", "white", "gray", "red", "blue", "green", "beige", "brown", "navy", "pink"]
MATERIALS = ["cotton", "polyester", "wool", "leather", "denim", "nylon"]
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]
# 정규화 사전(제공)
CATEGORY_MAP = {"볼캡": "cap", "야구모자": "cap", "비니": "beanie", "반팔 티셔츠": "tshirt", "반팔티": "tshirt", "셔츠": "shirt", "남방": "shirt",
                "후드티": "hoodie", "후드 티셔츠": "hoodie", "자켓": "jacket", "재킷": "jacket", "원피스": "dress", "스커트": "skirt", "치마": "skirt",
                "슬랙스": "pants", "바지": "pants", "반바지": "shorts", "숏팬츠": "shorts", "스니커즈": "sneakers", "운동화": "sneakers",
                "부츠": "boots", "워커": "boots", "샌들": "sandals", "슬리퍼": "sandals", "백팩": "backpack", "배낭": "backpack",
                "핸드백": "handbag", "토트백": "handbag", "반지갑": "wallet", "지갑": "wallet", "손목시계": "watch", "시계": "watch",
                "선글라스": "sunglasses", "양말": "socks", "머플러": "scarf", "스카프": "scarf", **{c: c for c in CLASSES}}
COLOR_MAP = {"블랙": "black", "검정": "black", "화이트": "white", "흰색": "white", "그레이": "gray", "회색": "gray", "레드": "red", "빨강": "red",
             "블루": "blue", "파랑": "blue", "그린": "green", "초록": "green", "베이지": "beige", "브라운": "brown", "갈색": "brown",
             "네이비": "navy", "남색": "navy", "핑크": "pink", "분홍": "pink", **{c: c for c in COLORS}}
MATERIAL_MAP = {"면 100%": "cotton", "면": "cotton", "코튼": "cotton", "폴리에스터": "polyester", "울": "wool", "양모": "wool",
                "천연가죽": "leather", "가죽": "leather", "데님": "denim", "나일론": "nylon", **{m: m for m in MATERIALS}}
BOOL_MAP = {"true": True, "yes": True, "y": True, "재고 있음": True, "즉시 출고": True,
            "false": False, "no": False, "n": False, "품절": False, "일시품절": False, "재입고 예정": False}
UNKNOWN = {"", "unknown", "n/a", "na", "none", "null", "없음", "미상", "모름"}

print(dev.text[1]); print("---- 모의 응답 ----"); print(ps.mock_llm(dev.id[1]))
""")

md(r"""
### 5-1 프롬프트 설계
`build_prompt(text)`: 언어모델에 보낼 **자연어 프롬프트**를 반환합니다. 다음을 모두 담으세요.
- 역할·작업, 출력은 **JSON 객체 하나만**(설명문·코드 블록 금지)
- 8개 필드 이름·자료형·허용값(category 20개 이름, 색상·소재 허용값)
- 정규화 규칙: 가격은 **원 단위 정수**(할인가 우선, 외화·문의는 null), 사이즈 범위 펼치기, 색상 영문화
- **미확정 정보 처리**: 없거나 불확실하면 추측하지 말고 null(배열은 `[]`). 혼방·조건부 무료배송은 null
- 예시 1개(few-shot), 상품 설명 원문을 구분 기호로 감싸 넣기
""")

write(r"""
FEW_SHOT_TEXT = "[노바핏] 여성 데일리 후드티 / 정가 59,000원 → 할인가 4만 1천원 / 색상: 네이비, 회색 / 사이즈 S~L / 면/폴리 혼방 / 5만원 이상 무료배송 / 품절"
FEW_SHOT_JSON = {"category": "hoodie", "brand": "노바핏", "price_krw": 41000, "colors": ["gray", "navy"], "sizes": ["S", "M", "L"],
                 "material": None, "in_stock": False, "free_shipping": None}

def build_prompt(text):
    prompt = f'''당신은 쇼핑몰 상품 설명에서 정보를 추출해 ___ 객체 하나만 출력하는 도구입니다. 설명 문장·코드 블록(```)·추가 키는 출력하지 마세요.

[출력 스키마] 키 8개를 모두 포함합니다.
- "category": 다음 중 하나 또는 null: {", ".join(CLASSES)}
- "brand": [대괄호] 안의 브랜드 문자열 또는 null
- "price_krw": ___ 단위 정수 또는 null. "39,000원"=39000, "3만 9천원"=39000. 할인가가 있으면 할인가. 외화($)나 "가격 문의"는 null
- "colors": 배열. 원소는 {", ".join(COLORS)} 중 하나(한국어 색 이름은 영문으로). 없으면 []
- "sizes": 문자열 배열. "S~XL"은 ["S","M","L","XL"]처럼 펼치고, 프리사이즈는 ["FREE"], 신발은 ["240","250"]. 없으면 []
- "material": {", ".join(MATERIALS)} 중 하나 또는 null. 혼방이면 null
- "in_stock": true/false/null ("품절"·"재입고 예정"은 false, "재고 있음"·"즉시 출고"는 true)
- "free_shipping": true/false/null ("무료배송" true, 배송비가 있으면 false)

[미확정 정보 규칙]
___

[예시]
설명: {FEW_SHOT_TEXT}
출력: {json.dumps(FEW_SHOT_JSON, ensure_ascii=False)}

[상품 설명]
<<<
{___}
>>>
출력:'''
    return prompt
""", r"""
def build_prompt(text):
    '''위 요구사항을 모두 담은 프롬프트 문자열'''
    prompt = ""  # 작성하세요
    return prompt
""", r"""
# 예시 1개: 할인가·범위 사이즈·혼방(null)·조건부 무료배송(null)·품절(false)이 한 번에 보이도록 골랐다
FEW_SHOT_TEXT = "[노바핏] 여성 데일리 후드티 / 정가 59,000원 → 할인가 4만 1천원 / 색상: 네이비, 회색 / 사이즈 S~L / 면/폴리 혼방 / 5만원 이상 무료배송 / 품절"
FEW_SHOT_JSON = {"category": "hoodie", "brand": "노바핏", "price_krw": 41000, "colors": ["gray", "navy"], "sizes": ["S", "M", "L"],
                 "material": None, "in_stock": False, "free_shipping": None}

def build_prompt(text):
    prompt = f'''당신은 쇼핑몰 상품 설명에서 정보를 추출해 JSON 객체 하나만 출력하는 도구입니다. 설명 문장·코드 블록(```)·추가 키는 출력하지 마세요.

[출력 스키마] 키 8개를 모두 포함합니다.
- "category": 다음 중 하나 또는 null: {", ".join(CLASSES)}
- "brand": [대괄호] 안의 브랜드 문자열 또는 null
- "price_krw": 원 단위 정수 또는 null. "39,000원"=39000, "3만 9천원"=39000. 할인가가 있으면 할인가. 외화($)나 "가격 문의"는 null
- "colors": 배열. 원소는 {", ".join(COLORS)} 중 하나(한국어 색 이름은 영문으로). 없으면 []
- "sizes": 문자열 배열. "S~XL"은 ["S","M","L","XL"]처럼 펼치고, 프리사이즈는 ["FREE"], 신발은 ["240","250"]. 없으면 []
- "material": {", ".join(MATERIALS)} 중 하나 또는 null. 혼방이면 null
- "in_stock": true/false/null ("품절"·"재입고 예정"은 false, "재고 있음"·"즉시 출고"는 true)
- "free_shipping": true/false/null ("무료배송" true, 배송비가 있으면 false)

[미확정 정보 규칙]
설명에 없거나 불확실한 정보는 추측하지 말고 null로 둡니다(배열 필드는 []).
"5만원 이상 무료배송"처럼 조건이 붙은 무료배송은 null, 소재가 혼방이면 null입니다.
숫자는 따옴표 없이 숫자 자료형으로, 참/거짓은 true/false로 씁니다.

[예시]
설명: {FEW_SHOT_TEXT}
출력: {json.dumps(FEW_SHOT_JSON, ensure_ascii=False)}

[상품 설명]
<<<
{text}
>>>
출력:'''
    return prompt
""")

run(r"""
def _check_5_1():
    p = build_prompt(dev.text[0])
    assert "___" not in p, "빈칸이 남아 있습니다"
    missing = ps.check_prompt(p, dev.text[0], SCHEMA_KEYS)
    print(p[:700] + ("..." if len(p) > 700 else ""))
    print("\n빠진 요소:", missing or "없음", "| 길이", len(p))
    assert not missing
attempt("5-1", _check_5_1)
""")

md(r"""
### 5-2 JSON 스키마
`RECORD_SCHEMA`(JSON Schema dict): 모든 키 필수, 여분 키 금지, 위 표의 자료형·허용값·null 허용, 배열 원소 스키마(`colors`는 `COLORS`의 enum, `sizes`는 문자열). 확인 셀은 dev 정답 40개가 통과하고 잘못된 예 7개가 걸러지는지 봅니다.
""")

write(r"""
RECORD_SCHEMA = {
    "type": "object",
    "required": SCHEMA_KEYS,
    "additionalProperties": ___,
    "properties": {
        "category": {"enum": CLASSES + [___]},
        "brand": {"type": ["string", "null"]},
        "price_krw": {"type": [___, "null"], "minimum": 0},
        "colors": {"type": "array", "items": {"enum": ___}},
        "sizes": {"type": "array", "items": {"type": ___}},
        "material": {"enum": MATERIALS + [None]},
        "in_stock": {"type": [___, "null"]},
        "free_shipping": {"type": ["boolean", "null"]},
    },
}
""", r"""
RECORD_SCHEMA = {}  # 작성하세요
""", r"""
RECORD_SCHEMA = {
    "type": "object",
    "required": SCHEMA_KEYS,                                  # 8개 키 모두
    "additionalProperties": False,                            # "comment" 같은 여분 키는 위반
    "properties": {
        "category": {"enum": CLASSES + [None]},               # 이미지 과제와 같은 20개 이름만
        "brand": {"type": ["string", "null"]},
        "price_krw": {"type": ["integer", "null"], "minimum": 0},   # "39,000원" 같은 문자열 차단
        "colors": {"type": "array", "items": {"enum": COLORS}},     # 배열 원소마다 검사
        "sizes": {"type": "array", "items": {"type": "string"}},    # "S~XL" 문자열 자체는 배열이 아니라 위반
        "material": {"enum": MATERIALS + [None]},
        "in_stock": {"type": ["boolean", "null"]},
        "free_shipping": {"type": ["boolean", "null"]},
    },
}
""")

run(r"""
def _check_5_2():
    try:
        import jsonschema
    except ImportError:
        print("jsonschema 없음: 구조만 확인"); assert RECORD_SCHEMA.get("type") == "object"; return
    v = jsonschema.Draft202012Validator(RECORD_SCHEMA)
    bad_gold = [k for k, r in dev_gold.items() if list(v.iter_errors(r))]
    assert not bad_gold, f"정답인데 위반: {bad_gold[:3]}"
    g = copy.deepcopy(next(iter(dev_gold.values())))
    for broken in [{**g, "price_krw": "39,000원"}, {**g, "colors": ["블랙"]}, {**g, "sizes": "S~XL"}, {**g, "in_stock": "품절"},
                   {**g, "comment": "x"}, {k: g[k] for k in SCHEMA_KEYS[:-1]}, {**g, "category": "볼캡"}]:
        assert list(v.iter_errors(broken)), f"걸러야 할 예가 통과: {broken}"
    print("dev 정답 40개 통과, 잘못된 예 7개 차단")
attempt("5-2", _check_5_2)
""")

md(r"""
### 5-3 응답에서 JSON 꺼내기
`extract_json(raw)`: 첫 `{`~마지막 `}` 잘라내기 → `json.loads` → 실패하면 끝 쉼표 제거 후 다시 → 그래도 실패하면 `ast.literal_eval` → dict가 아니거나 모두 실패하면 None
""")

write(r"""
def extract_json(raw):
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.___("}")
    if start == -1 or end <= start:
        return None
    body = raw[start:end + 1]
    for text in [body, re.sub(r",\s*([}\]])", ___, body)]:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    try:
        obj = ast.___(body)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None
""", r"""
def extract_json(raw):
    '''응답 문자열에서 dict를 꺼낸다. 실패하면 None'''
    # 작성하세요
""", r"""
def extract_json(raw):
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:                 # 끊긴 응답: 복구하지 않고 None (추측 금지)
        return None
    body = raw[start:end + 1]                       # ```json·설명문 제거
    for text in [body, re.sub(r",\s*([}\]])", r"\1", body)]:   # 원본, 끝 쉼표 제거본
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    try:                                            # {'price_krw': None, ...} 파이썬 표기
        obj = ast.literal_eval(body)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError):
        return None
""")

run(r"""
def _check_5_3():
    cases = ['```json\n{"a": 1}\n```', '결과입니다.\n{"a": 1}\n끝', '{"a": [1, 2,], "b": 3,\n}', "{'a': None, 'b': True}", '{"a": 1, "b": ', "없음"]
    want = [{"a": 1}, {"a": 1}, {"a": [1, 2], "b": 3}, {"a": None, "b": True}, None, None]
    for c, w in zip(cases, want):
        assert extract_json(c) == w, f"{c!r} → {extract_json(c)!r}"
    ok = sum(extract_json(ps.mock_llm(i)) is not None for i in dev.id)
    print(f"dev 응답 {len(dev)}개 중 {ok}개 파싱 성공")
attempt("5-3", _check_5_3)
""")

md(r"""
### 5-4 정규화와 미확정 정보 처리
`normalize_record(obj)` → 스키마를 지키는 레코드(키 순서 = `SCHEMA_KEYS`). `obj`가 None이면 모든 필드 null, 배열은 `[]`.
- 문자열 비교는 앞뒤 공백 제거·소문자. `UNKNOWN`에 속하면 null
- `category`, `material`, 색상 원소, `in_stock`·`free_shipping`: 제공 사전으로 변환, 사전에 없으면 null(색상은 그 원소 제외). **bool은 그대로**
- `parse_price_krw(v)`: 정수는 그대로(bool은 null). 문자열은 `,`·공백·`₩`·`원` 제거 후, `$`/`usd`가 있으면 null, `만`이 있으면 `만` 앞 숫자×10000 + `천` 앞 숫자×1000, 아니면 첫 숫자. 숫자가 없으면 null
- `parse_sizes(v)`: None·미확정 → `[]`. 문자열이면 대문자로 바꾸고 `FREE`/`프리`/`프리사이즈` → `["FREE"]`, `A~B`(둘 다 `SIZE_ORDER`) → 그 사이 전부, 그 밖에는 `,`·`/`로 나눔. 결과는 중복 없이 **SIZE_ORDER 순, 숫자는 크기 순**
- `colors`: 문자열이면 `,`·`/`·`·`로 나눈 뒤 변환, 중복 제거 후 정렬
""")

write(r"""
def _is_unknown(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ___)

def _map(v, table):
    if not isinstance(v, str) or _is_unknown(v):
        return None
    return table.get(v.strip().lower(), table.get(v.strip()))

def _to_bool(v):
    return v if isinstance(v, ___) else _map(v, BOOL_MAP)

def parse_price_krw(v):
    if isinstance(v, bool) or _is_unknown(v):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).replace(",", "").replace(" ", "").replace("₩", "").replace(___, "")
    if "$" in s or "usd" in s.lower():
        return ___
    if "만" in s:
        man, _, rest = s.partition("만")
        m = re.search(r"(\d+)천", rest)
        return int(man) * ___ + (int(m.group(1)) * 1000 if m else 0)
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def _size_key(z):
    if z in SIZE_ORDER:
        return (0, SIZE_ORDER.index(z))
    return (1, int(z)) if z.isdigit() else (2, 0)

def parse_sizes(v):
    if v is None or _is_unknown(v):
        return []
    if isinstance(v, str):
        s = v.strip().upper()
        if s in ("FREE", "프리", "프리사이즈"):
            return ["FREE"]
        if "~" in s:
            a, b = [x.strip() for x in s.split("~")]
            if a in SIZE_ORDER and b in SIZE_ORDER:
                return SIZE_ORDER[SIZE_ORDER.index(a):SIZE_ORDER.index(b) + ___]
        v = re.split(r"[,/]", s)
    out = [str(x).strip().upper() for x in v if str(x).strip()]
    return sorted(dict.fromkeys(out), key=___)

def parse_colors(v):
    if v is None or _is_unknown(v):
        return []
    items = re.split(r"[,/·]", v) if isinstance(v, str) else v
    return sorted({c for c in (_map(x, COLOR_MAP) for x in items) if c is not None})

def normalize_record(obj):
    obj = obj if isinstance(obj, dict) else {}
    brand = obj.get("brand")
    rec = {
        "category": obj.get("category") if obj.get("category") in CLASSES else _map(obj.get("category"), ___),
        "brand": None if _is_unknown(brand) or not isinstance(brand, str) else brand.strip(),
        "price_krw": parse_price_krw(obj.get("price_krw")),
        "colors": parse_colors(obj.get("colors")),
        "sizes": parse_sizes(obj.get("sizes")),
        "material": _map(obj.get("material"), MATERIAL_MAP),
        "in_stock": _to_bool(obj.get("in_stock")),
        "free_shipping": _to_bool(obj.get(___)),
    }
    return {k: rec[k] for k in SCHEMA_KEYS}
""", r"""
def normalize_record(obj):
    '''위 규칙으로 스키마를 지키는 레코드. parse_price_krw, parse_sizes 같은 도우미를 나눠 써도 됩니다'''
    rec = {}
    # 작성하세요
    return rec
""", r"""
def _is_unknown(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in UNKNOWN)

def _map(v, table):
    # 사전 변환. 문자열이 아니거나 사전에 없으면 null (모르는 값을 추측하지 않음)
    if not isinstance(v, str) or _is_unknown(v):
        return None
    return table.get(v.strip().lower(), table.get(v.strip()))

def _to_bool(v):
    return v if isinstance(v, bool) else _map(v, BOOL_MAP)       # "Y"/"품절" → bool, "5만원 이상 무료" → None

def parse_price_krw(v):
    if isinstance(v, bool) or _is_unknown(v):                    # True==1 함정
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).replace(",", "").replace(" ", "").replace("₩", "").replace("원", "")
    if "$" in s or "usd" in s.lower():                           # 외화는 환율을 모르므로 변환하지 않고 null
        return None
    if "만" in s:                                                # '3만9천' → 3×10000 + 9×1000
        man, _, rest = s.partition("만")
        m = re.search(r"(\d+)천", rest)
        return int(man) * 10000 + (int(m.group(1)) * 1000 if m else 0)
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None                         # '가격문의' → 숫자 없음 → null

def _size_key(z):
    if z in SIZE_ORDER:
        return (0, SIZE_ORDER.index(z))                          # S < M < L 순서
    return (1, int(z)) if z.isdigit() else (2, 0)                # 신발 치수는 숫자 순

def parse_sizes(v):
    if v is None or _is_unknown(v):
        return []                                                # 배열 필드의 '없음'은 null이 아니라 []
    if isinstance(v, str):
        s = v.strip().upper()
        if s in ("FREE", "프리", "프리사이즈"):
            return ["FREE"]
        if "~" in s:                                             # 'S~XL' → SIZE_ORDER 구간
            a, b = [x.strip() for x in s.split("~")]
            if a in SIZE_ORDER and b in SIZE_ORDER:
                return SIZE_ORDER[SIZE_ORDER.index(a):SIZE_ORDER.index(b) + 1]
        v = re.split(r"[,/]", s)                                 # '240/250', 'M, L'
    out = [str(x).strip().upper() for x in v if str(x).strip()]
    return sorted(dict.fromkeys(out), key=_size_key)             # 중복 제거(순서 유지) 후 정렬

def parse_colors(v):
    if v is None or _is_unknown(v):
        return []
    items = re.split(r"[,/·]", v) if isinstance(v, str) else v
    return sorted({c for c in (_map(x, COLOR_MAP) for x in items) if c is not None})

def normalize_record(obj):
    obj = obj if isinstance(obj, dict) else {}                   # 파싱 실패 → 전부 null / []
    brand = obj.get("brand")
    rec = {
        "category": obj.get("category") if obj.get("category") in CLASSES else _map(obj.get("category"), CATEGORY_MAP),
        "brand": None if _is_unknown(brand) or not isinstance(brand, str) else brand.strip(),
        "price_krw": parse_price_krw(obj.get("price_krw")),
        "colors": parse_colors(obj.get("colors")),
        "sizes": parse_sizes(obj.get("sizes")),
        "material": _map(obj.get("material"), MATERIAL_MAP),     # '면/폴리 혼방' → 사전에 없음 → null
        "in_stock": _to_bool(obj.get("in_stock")),
        "free_shipping": _to_bool(obj.get("free_shipping")),
    }
    return {k: rec[k] for k in SCHEMA_KEYS}                       # 스키마 키 순서, 여분 키 없음
""")

run(r"""
def _check_5_4():
    raw = {"category": "반팔티", "brand": "없음", "price_krw": "3만 9천원", "colors": "블랙/화이트", "sizes": "S~L",
           "material": "면 100%", "in_stock": "품절", "free_shipping": "N", "comment": "x"}
    r = normalize_record(raw)
    want = {"category": "tshirt", "brand": None, "price_krw": 39000, "colors": ["black", "white"], "sizes": ["S", "M", "L"],
            "material": "cotton", "in_stock": False, "free_shipping": False}
    assert r == want, r
    n = normalize_record({"price_krw": "$30", "sizes": "260/240", "material": "면/폴리 혼방", "free_shipping": "5만원 이상 무료배송", "in_stock": True})
    assert n["price_krw"] is None and n["sizes"] == ["240", "260"] and n["material"] is None and n["free_shipping"] is None and n["in_stock"] is True
    assert normalize_record({"price_krw": True})["price_krw"] is None and normalize_record({"price_krw": "₩45,000"})["price_krw"] == 45000
    e = normalize_record(None)
    assert list(e) == SCHEMA_KEYS and e["colors"] == [] and e["sizes"] == [] and e["price_krw"] is None
    print("정규화 규칙 확인 통과")
attempt("5-4", _check_5_4)
""")

md(r"""
### 5-5 자료형 검사와 평가·제출
`type_errors(rec)`: 라이브러리 없이 위반 목록(문자열 리스트). 키 집합, `category`·`material` 허용값(또는 null), `brand` 문자열/null, `price_krw` 정수(**bool 제외**)/null·0 이상, `colors` 리스트이고 원소가 `COLORS`, `sizes` 문자열 리스트, `in_stock`·`free_shipping` bool/null.

아래 셀이 dev 40개에 `응답 → extract_json → normalize_record → type_errors`를 돌려 필드 정확도를 계산하고, test 30개를 제출 파일로 저장합니다.
""")

write(r"""
def type_errors(rec):
    errors = []
    if set(rec) != set(SCHEMA_KEYS):
        return [f"키 불일치: {sorted(set(rec) ^ set(SCHEMA_KEYS))}"]
    if rec["category"] not in CLASSES + [None]:
        errors.append(f"category: 허용값 아님 ({rec['category']!r})")
    if rec["brand"] is not None and not isinstance(rec["brand"], ___):
        errors.append("brand: 문자열/null")
    p = rec["price_krw"]
    if p is not None and (not isinstance(p, int) or isinstance(p, ___) or p < 0):
        errors.append(f"price_krw: 0 이상 정수/null ({p!r})")
    if not isinstance(rec["colors"], list) or any(c not in ___ for c in rec["colors"]):
        errors.append(f"colors: 허용 색 배열 ({rec['colors']!r})")
    if not isinstance(rec["sizes"], list) or any(not isinstance(s, str) for s in rec["sizes"]):
        errors.append(f"sizes: 문자열 배열 ({rec['sizes']!r})")
    if rec["material"] not in MATERIALS + [None]:
        errors.append(f"material: 허용값 아님 ({rec['material']!r})")
    for k in ["in_stock", "free_shipping"]:
        if rec[k] is not None and not isinstance(rec[k], ___):
            errors.append(f"{k}: bool/null ({rec[k]!r})")
    return errors
""", r"""
def type_errors(rec):
    '''자료형 위반 설명 리스트 (없으면 [])'''
    errors = []
    # 작성하세요
    return errors
""", r"""
def type_errors(rec):
    errors = []
    if set(rec) != set(SCHEMA_KEYS):
        return [f"키 불일치: {sorted(set(rec) ^ set(SCHEMA_KEYS))}"]
    if rec["category"] not in CLASSES + [None]:
        errors.append(f"category: 허용값 아님 ({rec['category']!r})")
    if rec["brand"] is not None and not isinstance(rec["brand"], str):
        errors.append("brand: 문자열/null")
    p = rec["price_krw"]
    if p is not None and (not isinstance(p, int) or isinstance(p, bool) or p < 0):   # bool은 int의 하위 클래스!
        errors.append(f"price_krw: 0 이상 정수/null ({p!r})")
    if not isinstance(rec["colors"], list) or any(c not in COLORS for c in rec["colors"]):
        errors.append(f"colors: 허용 색 배열 ({rec['colors']!r})")
    if not isinstance(rec["sizes"], list) or any(not isinstance(s, str) for s in rec["sizes"]):
        errors.append(f"sizes: 문자열 배열 ({rec['sizes']!r})")
    if rec["material"] not in MATERIALS + [None]:
        errors.append(f"material: 허용값 아님 ({rec['material']!r})")
    for k in ["in_stock", "free_shipping"]:
        if rec[k] is not None and not isinstance(rec[k], bool):
            errors.append(f"{k}: bool/null ({rec[k]!r})")
    return errors
""")

run(r"""
_extract = extract_json if attempt("5-3 재확인", lambda: extract_json('{"a": 1}')) == {"a": 1} else (lambda raw: None)

def _field_match(a, b, k):
    if k in ("colors", "sizes"):
        return isinstance(a[k], list) and sorted(map(str, a[k])) == sorted(map(str, b[k]))
    return a[k] == b[k] and type(a[k]) is type(b[k])

def _run_5_5():
    assert type_errors({**normalize_record(None), "price_krw": True}) != [], "bool을 정수로 통과시킴"
    assert type_errors(normalize_record(None)) == []
    rows, n_err, n_fail = [], 0, 0
    for tid in dev.id:
        obj = _extract(ps.mock_llm(tid))
        n_fail += obj is None
        rec = normalize_record(obj)
        n_err += bool(type_errors(rec))
        rows.append({k: _field_match(rec, dev_gold[tid], k) for k in SCHEMA_KEYS})
    acc = pd.DataFrame(rows).mean()
    print("필드별 정확도\n", acc.round(3).to_string())
    print(f"전체 필드 정확도 {acc.mean():.3f} | 파싱 실패 {n_fail}건 | 자료형 위반 레코드 {n_err}건")
    print("기준(필드 정확도 ≥ 0.95, 위반 0건):", bool(acc.mean() >= 0.95 and n_err == 0))
    raw = np.mean([[(_extract(ps.mock_llm(t)) or {}).get(k) == dev_gold[t][k] for k in SCHEMA_KEYS] for t in dev.id])
    print(f"(참고) 정규화 없이 응답 그대로 쓸 때 필드 정확도 {raw:.3f}")
    out = [{"id": tid, "record": normalize_record(_extract(ps.mock_llm(tid)))} for tid in test_texts.id]
    ps.save_jsonl(out, "q5_submission.jsonl")
    print("q5 형식 점검:", ps.check_submission("q5") or "통과")
_ = attempt("5-5", _run_5_5)
""")

md(r"""
---
## 제출 전 점검 (제출가이드 + 코드 리뷰 대비)
""")

run(r"""
for k in ["q3", "q4", "q5"]:
    print(k, "→", ps.check_submission(k) or "형식 통과")
for name in ["q1_note", "q2_aug_note", "q3_analysis", "q4_analysis"]:
    print(f"{name}: {'작성함' if str(globals().get(name, '')).strip() else '비어 있음'}")
""")

md(r"""
- 형식 오류가 하나라도 있으면 그 문항은 0점입니다(FAQ).
- 코드 리뷰 관점: 전처리 값은 `resolve_data_config`에서 읽기(하드코딩 X), 함수마다 한 가지 일, 핵심 결정에 주석, seed 고정, test 정답·test 점수 미사용, 원본 불변, 장치 일치.
- 커널 재시작 → 모두 실행 후 출력이 남은 상태로 저장. 그다음 아래 **로컬 채점** 셀로 확인합니다.
""")

md(r"""
---
## 로컬 채점 — 기존 best와 비교 (Kaggle 리더보드 대신)

`정답_채점용/score_local.py`로 `제출물/`을 채점하고 **지금까지의 best와 이번 점수**를 비교합니다(기록: `제출물/_score_history.json`, 지우면 초기화). Q3(첫 제출) → Q4(개선 후 제출)의 변화도 보여 줍니다.

**이 점수를 보고 설정을 고치지 마세요.** 시험에는 test 정답이 없고, 리더보드에 맞춰 고치면 리더보드에 과적합합니다. 설정은 validation으로만 정합니다.
""")

run(r"""
import subprocess

def local_score():
    cmd = [sys.executable, "-I", "-X", "utf8", str(ROOT / "정답_채점용" / "score_local.py"), str(ps.SUBMIT), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    lines = proc.stdout.strip().splitlines()
    print("\n".join(l for l in lines if not l.startswith("SCORES_JSON")))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-500:])
    return json.loads(next(l for l in lines if l.startswith("SCORES_JSON"))[len("SCORES_JSON "):])

def compare_with_best(scored, history_path=ps.SUBMIT / "_score_history.json"):
    hist = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"best": {}, "runs": []}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for kind, r in scored["results"].items():
        new = r["score"] if r["score"] is not None else 0.0
        old = hist["best"].get(kind, {}).get("score")
        status = "첫 기록" if old is None else "▲ best 갱신" if new > old + 1e-9 else "▼ 하락 (best 유지)" if new < old - 1e-9 else "= 같음"
        rows.append({"문항": kind, "기준": scored["thresholds"][kind], "기존 best": old, "이번 점수": round(new, 4),
                     "변화": None if old is None else round(new - old, 4), "PASS": r["pass"], "상태": status})
        if old is None or new > old:
            hist["best"][kind] = {"score": new, "pass": r["pass"], "time": now}
    hist["runs"].append({"time": now, "scores": {k: (v["score"] or 0.0) for k, v in scored["results"].items()}})
    history_path.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    res = scored["results"]
    if "q3" in res and "q4" in res:
        a, b = res["q3"]["score"] or 0.0, res["q4"]["score"] or 0.0
        print(f"\n이미지 분류 이번 실행: Q3 첫 제출 {a:.4f} → Q4 개선 제출 {b:.4f} ({b - a:+.4f})")
    print(f"채점 {len(hist['runs'])}회째 | 기록: {history_path.relative_to(ROOT)}")

scored = attempt("로컬 채점", local_score)
if scored:
    compare_with_best(scored)
""")


# ======================================================================
def build(kind):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for cell_kind, a, b, c in CELLS:
        if cell_kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(a))
        elif cell_kind == "run":
            nb.cells.append(nbf.v4.new_code_cell(a))
        else:
            src = {"A": "# [작성하세요] ___ 를 채우세요\n" + a, "B": "# [작성하세요]\n" + b, "answer": "# [정답 예시]\n" + c}[kind]
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.cells[0].source += {"A": "\n\n> **버전 A — 핵심 빈칸.** `___` 자리만 채우면 됩니다. 서술 답안은 직접 씁니다.",
                           "B": "\n\n> **버전 B — 직접 구현.** 함수 이름·인자·설명만 주어집니다. 실전에 가장 가까운 판입니다.",
                           "answer": "\n\n> **정답·해설용입니다.** `[정답 예시]` 셀은 줄마다 이유를 주석으로 단 풀이 하나이며, 같은 요구사항을 만족하는 다른 코드도 정답입니다."}[kind]
    return nb


def fill(nb, answers, thresholds):
    for cell in nb.cells:
        for key, text in {**answers, "__Q3_TH__": str(thresholds["q3"]), "__Q4_TH__": str(thresholds["q4"])}.items():
            if key in cell.source:
                cell.source = cell.source.replace(key, text)
    return nb


if __name__ == "__main__":
    here = Path(__file__).parent
    answers = json.loads((here / "analysis_answers.json").read_text(encoding="utf-8")) if (here / "analysis_answers.json").exists() else {}
    th = json.loads((here / "thresholds.json").read_text(encoding="utf-8"))
    nbf.write(fill(build("A"), {}, th), ROOT / "문제_A_빈칸.ipynb")
    nbf.write(fill(build("B"), {}, th), ROOT / "문제_B_직접구현.ipynb")
    nbf.write(fill(build("answer"), answers, th), ROOT / "정답_채점용" / "정답_해설.ipynb")
    print("cells:", len(CELLS), "| write cells:", sum(k == "write" for k, *_ in CELLS))
