"""제출 형식 점검 (torch 없이 동작). practice_support와 채점기가 함께 씁니다."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SUBMIT = ROOT / "제출물"
CLASSES = ["cap", "beanie", "tshirt", "shirt", "hoodie", "jacket", "dress", "skirt", "pants", "shorts",
           "sneakers", "boots", "sandals", "backpack", "handbag", "wallet", "watch", "sunglasses", "socks", "scarf"]

GUIDE = {
    "q3": dict(file="q3_submission.csv"),     # Q3 첫 제출
    "q4": dict(file="q4_submission.csv"),     # Q4 개선 후 최종 제출
}


def check_submission(kind, folder=None):
    """제출가이드.md의 규칙으로 검사합니다. 문제 목록(빈 리스트면 통과)을 반환."""
    folder = SUBMIT if folder is None else Path(folder)
    if kind == "q5":
        path = folder / "q5_submission.jsonl"
        if not path.exists():
            return [f"{path.name} 없음"]
        ids = pd.read_csv(DATA / "texts" / "products.csv").query("split == 'test'")["id"].tolist()
        seen, problems = [], []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{n}번째 줄이 JSON이 아님")
                continue
            if set(row) != {"id", "record"} or not isinstance(row.get("record"), dict):
                problems.append(f"{n}번째 줄 형식이 {{'id', 'record'}}가 아님")
                continue
            seen.append(row["id"])
        if seen != ids:
            problems.append(f"id 목록·순서가 products.csv의 test와 다름 (제출 {len(seen)}줄, 기대 {len(ids)}줄)")
        return problems

    path = folder / GUIDE[kind]["file"]
    if not path.exists():
        return [f"{path.name} 없음"]
    sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"id": str})
    try:
        sub = pd.read_csv(path, dtype={"id": str})
    except Exception as exc:
        return [f"CSV로 읽을 수 없음: {exc}"]
    problems = []
    if list(sub.columns) != list(sample.columns):
        return [f"열 이름·순서가 다름: {list(sub.columns)} (기대 {list(sample.columns)})"]
    if len(sub) != len(sample):
        problems.append(f"행 수가 다름: {len(sub)} (기대 {len(sample)})")
    if sub["id"].duplicated().any():
        problems.append("id 중복")
    if sub["id"].tolist() != sample["id"].tolist():
        problems.append("id 순서가 sample_submission과 다름")
    if sub["label"].isna().any():
        problems.append("빈 값 있음")
    bad = set(sub["label"].astype(str)) - set(CLASSES)
    if bad:
        problems.append(f"허용되지 않은 label 값: {sorted(bad)[:5]} (영문 소문자 클래스 이름이어야 함)")
    return problems
