import json, sys, glob
sys.path.insert(0, 'AI_Lv3_sample_solution/_build')
from annotate_lib import commentable_lines
sys.stdout.reconfigure(encoding='utf-8')
files = sorted(glob.glob('AI_Lv3_sample_solution/P*_answer.ipynb')) + sorted(glob.glob('AI_Lv3_final_predict/F*_answer.ipynb'))
seen = {}
for f in files:
    nb = json.load(open(f, encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        ok = commentable_lines(src)
        for i, line in enumerate(src.split('\n'), 1):
            if i in ok:
                seen.setdefault(line.strip(), f.split('/')[-1])
print(len(seen))
json.dump(seen, open(sys.argv[1], 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
