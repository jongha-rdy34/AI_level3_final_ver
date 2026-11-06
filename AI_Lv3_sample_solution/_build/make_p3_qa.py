"""P3 노트북의 예시 1·2 마크다운에서 P3_qa.csv(task, question)를 만듭니다."""
import csv, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
nb = json.load(open(root / 'problems_original' / 'P3_problem.ipynb', encoding='utf-8'))
rows = []
for i in (10, 11):
    src = ''.join(nb['cells'][i]['source']).strip().split('\n')
    title = src[0].split(':', 1)[1].strip()          # '전세계약 정보 추출'
    body = '\n'.join(src[1:]).strip()
    rows.append({'task': title, 'question': body.replace('\n', '\n')})
with open(root / 'P3_qa.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'question']); w.writeheader(); w.writerows(rows)
print('P3_qa.csv', len(rows), [len(r['question']) for r in rows])
