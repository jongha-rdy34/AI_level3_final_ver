"""두 폴더의 *_answer.ipynb 코드 셀에 줄별 설명 주석을 붙입니다 (실행 출력은 그대로).
사용: (Desktop에서) AI_Lv3_sample_solution/.venv/Scripts/python AI_Lv3_sample_solution/_build/annotate_answers.py
정답안을 다시 만든(build) 뒤에도 이 스크립트를 다시 실행하면 됩니다. 이미 주석이 있는 줄은 건너뜁니다."""
import ast
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from annotate_lib import annotate  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
comments = json.load(open(HERE / 'line_comments.json', encoding='utf-8'))
desktop = HERE.parents[1]
files = sorted(glob.glob(str(desktop / 'AI_Lv3_sample_solution' / 'P*_answer.ipynb'))) + \
        sorted(glob.glob(str(desktop / 'AI_Lv3_final_predict' / 'F*_answer.ipynb')))
missing = set()
for f in files:
    nb = json.load(open(f, encoding='utf-8'))
    changed = 0
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        new = annotate(src, comments, missing)
        if new != src:
            ast.parse(new)                       # 주석을 붙인 뒤에도 문법이 맞는지 확인
            ast_old = ast.dump(ast.parse(src))
            assert ast_old == ast.dump(ast.parse(new)), f'코드 의미가 바뀜: {f}'
            parts = new.split('\n')
            c['source'] = [p + '\n' for p in parts[:-1]] + [parts[-1]]
            changed += 1
    json.dump(nb, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(Path(f).name, '주석 단 셀:', changed)
print('설명이 없는 줄:', len(missing))
for m in sorted(missing):
    print('  ', m)
