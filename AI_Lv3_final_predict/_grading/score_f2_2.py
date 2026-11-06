"""F2_2 로컬 채점: 루트의 test.csv(answer 채워진 것) vs _grading/f2_2_test_truth.csv -> accuracy"""
import csv
from pathlib import Path
root = Path(__file__).resolve().parents[1]
truth = {r['id']: r['answer'] for r in csv.DictReader(open(root / '_grading' / 'f2_2_test_truth.csv', encoding='utf-8'))}
sub = {r['id']: r['answer'] for r in csv.DictReader(open(root / 'test.csv', encoding='utf-8'))}
missing = [k for k in truth if not str(sub.get(k, '')).strip()]
if missing:
    print(f'answer가 비어 있는 행 {len(missing)}개 - 추론 셀을 먼저 실행하세요.')
else:
    acc = sum(str(int(float(sub[k]))) == v for k, v in truth.items()) / len(truth)
    print(f'[Local Kaggle] test accuracy: {acc:.4f}  ({len(truth)}장)')
