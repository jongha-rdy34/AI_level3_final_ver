import json, re, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
rows = []
for f in sorted(glob.glob('_build/sweep_*.ipynb')):
    bs, lr = re.findall(r'sweep_(\d+)_([0-9.]+)\.ipynb', f.replace(chr(92), '/'))[0]
    nb = json.load(open(f, encoding='utf-8'))
    text = ''.join(''.join(o.get('text', '')) for c in nb['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', []))
    f1 = [float(x) for x in re.findall(r'검증 F1: ([0-9.]+)', text)]
    rows.append({'batch_size': int(bs), 'learning_rate': float(lr), 'epochs': 20, 'best_f1': max(f1), 'best_epoch': f1.index(max(f1)) + 1})
    print(rows[-1])
json.dump(rows, open('_build/p2_1_sweep.json', 'w', encoding='utf-8'), indent=1)
