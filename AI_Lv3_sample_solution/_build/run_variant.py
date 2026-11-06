"""노트북을 복사해 일부 셀을 바꿔 nbconvert로 실행합니다 (작업 폴더 = 루트).
사용: run_variant.py <src.ipynb> <out.ipynb> [json: {"셀에 포함된 문자열": "바꿀 source"}]"""
import json, sys
from pathlib import Path
import nbformat
root = Path(__file__).resolve().parents[1]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
subs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
nb = nbformat.read(src, as_version=4)
for c in nb.cells:
    if c.cell_type == 'code':
        if c.source.startswith("# 본인의 수험번호"):
            c.source = c.source.replace("student_id = ''", "student_id = '000000'")
        for key, rep in subs.items():
            if key in c.source:
                c.source = rep
tmp = root / f'_tmp_{out.stem}.ipynb'
nbformat.write(nb, tmp)
print(tmp)
