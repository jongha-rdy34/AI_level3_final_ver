import json,sys
sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open(sys.argv[1],encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        t=o.get('text') or (o.get('data',{}).get('text/plain') if 'image/png' not in o.get('data',{}) else None)
        if o.get('output_type')=='error': print(f'[{i}] ERROR', o['ename'], o['evalue'][:300])
        elif t: print(f'[{i}]', ''.join(t)[-int(sys.argv[2]) if len(sys.argv)>2 else -800:])
