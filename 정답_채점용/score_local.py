"""로컬 연습 채점기 (Kaggle 리더보드 대신). 실제 SSAFY 채점·코드 리뷰가 아닙니다.

사용: python -I -X utf8 정답_채점용/score_local.py [제출 폴더] [--json]
FAQ 반영: 제출 형식이 틀리면 그 문항은 0점(형식 검사는 submission_check.check_submission).
성능 기준(이 모의의 가정)은 _build/thresholds.json (Q3·Q4 test macro-F1), Q5 test 필드 정확도 ≥ 0.93 + 스키마 위반 0 (노트북의 dev 기준은 0.95)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import submission_check as sc  # noqa: E402  (torch 없이 동작)

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
AS_JSON = "--json" in sys.argv
SUBMIT = Path(ARGS[0]).resolve() if ARGS else ROOT / "제출물"
_th = json.loads((HERE / "_build" / "thresholds.json").read_text(encoding="utf-8"))
THRESH = {"q3": _th["q3"], "q4": _th["q4"], "q5": _th.get("q5", 0.93)}   # q5 test: 끊긴 응답 2건은 복구 불가 → 상한 약 0.946
KEYS = ["category", "brand", "price_krw", "colors", "sizes", "material", "in_stock", "free_shipping"]
COLORS = ["black", "white", "gray", "red", "blue", "green", "beige", "brown", "navy", "pink"]
MATERIALS = ["cotton", "polyester", "wool", "leather", "denim", "nylon"]


def macro_f1(y, p):
    out = []
    for c in sorted(set(y)):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def schema_violation(rec):
    if set(rec) != set(KEYS):
        return True
    try:
        import jsonschema
    except ImportError:
        return False
    schema = {"type": "object", "required": KEYS, "additionalProperties": False, "properties": {
        "category": {"enum": sc.CLASSES + [None]}, "brand": {"type": ["string", "null"]},
        "price_krw": {"type": ["integer", "null"], "minimum": 0},
        "colors": {"type": "array", "items": {"enum": COLORS}}, "sizes": {"type": "array", "items": {"type": "string"}},
        "material": {"enum": MATERIALS + [None]}, "in_stock": {"type": ["boolean", "null"]},
        "free_shipping": {"type": ["boolean", "null"]}}}
    return bool(list(jsonschema.Draft202012Validator(schema).iter_errors(rec)))


RESULTS = {}


def record(kind, score, ok, note=""):
    RESULTS[kind] = {"score": None if score is None else round(float(score), 4), "pass": bool(ok), "note": note}
    shown = "0점(형식 오류)" if score is None else f"{score:.4f}"
    print(f"{kind}: {shown} (기준 {THRESH[kind]}) → {'PASS' if ok else 'FAIL'}{(' | ' + note) if note else ''}")


truth = pd.read_csv(HERE / "test_truth.csv", dtype={"id": str})
for kind in ["q3", "q4"]:
    problems = sc.check_submission(kind, SUBMIT)
    if problems:
        record(kind, None, False, f"형식 오류 {problems}")
        continue
    sub = pd.read_csv(SUBMIT / sc.GUIDE[kind]["file"], dtype={"id": str})
    m = truth.merge(sub, on="id", suffixes=("_t", "_p"))
    y, p = m["label_t"].to_numpy(), m["label_p"].to_numpy()
    f1 = macro_f1(y, p)
    per = {c: np.mean(p[y == c] == c) for c in sc.CLASSES}
    worst = min(per, key=per.get)
    record(kind, f1, f1 >= THRESH[kind], f"정확도 {np.mean(y == p):.3f} | 재현율 최저 {worst} {per[worst]:.2f}")

problems = sc.check_submission("q5", SUBMIT)
if problems:
    record("q5", None, False, f"형식 오류 {problems}")
else:
    gold = json.loads((HERE / "q5_truth.json").read_text(encoding="utf-8"))
    sub = {}
    for line in (SUBMIT / "q5_submission.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); sub[row["id"]] = row["record"]
    hits, bad = [], 0
    for tid, g in gold.items():
        rec = sub.get(tid, {})
        bad += schema_violation(rec)
        for k in KEYS:
            v = rec.get(k)
            if k in ("colors", "sizes"):
                hits.append(isinstance(v, list) and sorted(map(str, v)) == sorted(map(str, g[k])))
            else:
                hits.append(v == g[k] and type(v) is type(g[k]))
    acc = float(np.mean(hits))
    record("q5", acc, acc >= THRESH["q5"] and bad == 0, f"스키마 위반 레코드 {bad} (기준 0)")

print("\n요약:", " | ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in RESULTS.items()))
print("※ 성능 기준은 이 모의의 가정입니다. 서술·코드 리뷰 항목은 채점기준.md로 스스로 점검하세요.")
if AS_JSON:
    print("SCORES_JSON " + json.dumps({"results": RESULTS, "thresholds": THRESH}, ensure_ascii=False))
