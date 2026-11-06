"""공식 시험의 오프라인 docs/ 대신, 같은 경로에 '온라인 원문으로 이동' 페이지를 만듭니다.
노트북의 docs/pages/... 링크를 그대로 눌러도 동작합니다 (인터넷 필요)."""
import html
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P = 'https://docs.pytorch.org/'
M = {
 'arxiv.org/abs_2207.10666': 'https://arxiv.org/abs/2207.10666',
 'developers.openai.com/api_docs_guides_prompt-engineering': 'https://developers.openai.com/api/docs/guides/prompt-engineering',
 'developers.openai.com/api_docs_guides_structured-outputs': 'https://developers.openai.com/api/docs/guides/structured-outputs',
 'developers.openai.com/api_reference_python': 'https://developers.openai.com/api/reference/python',
 'developers.openai.com/api_reference_resources_chat': 'https://developers.openai.com/api/reference/resources/chat',
 'docs.python.org/3_library_datetime': 'https://docs.python.org/3/library/datetime.html',
 'docs.python.org/3_library_json': 'https://docs.python.org/3/library/json.html',
 'docs.pytorch.org/docs_2.11_notes_autograd': P + 'docs/2.11/notes/autograd.html',
 'docs.pytorch.org/docs_stable_generated_torch.optim.Adam': P + 'docs/stable/generated/torch.optim.Adam.html',
 'docs.pytorch.org/tutorials_beginner_basics_data_tutorial': P + 'tutorials/beginner/basics/data_tutorial.html',
 'docs.pytorch.org/tutorials_beginner_basics_optimization_tutorial': P + 'tutorials/beginner/basics/optimization_tutorial.html',
 'docs.pytorch.org/tutorials_beginner_basics_saveloadrun_tutorial': P + 'tutorials/beginner/basics/saveloadrun_tutorial.html',
 'docs.pytorch.org/tutorials_beginner_transfer_learning_tutorial': P + 'tutorials/beginner/transfer_learning_tutorial.html',
 'docs.pytorch.org/vision_0.26_auto_examples_transforms_plot_transforms_illustrations': P + 'vision/0.26/auto_examples/transforms/plot_transforms_illustrations.html',
 'docs.pytorch.org/vision_0.26_transforms': P + 'vision/0.26/transforms.html',
 'docs.pytorch.org/vision_main_transforms': P + 'vision/main/transforms.html',
 'github.com/microsoft_Cream_blob_main_TinyViT_models_tiny_vit.py': 'https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py',
 'huggingface.co/timm_tiny_vit_5m_224.dist_in22k_ft_in1k': 'https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k',
 'json-schema.org/understanding-json-schema_reference_null': 'https://json-schema.org/understanding-json-schema/reference/null',
 'numpy.org/doc_stable_reference_generated_numpy.load': 'https://numpy.org/doc/stable/reference/generated/numpy.load.html',
 'www.ecva.net/`papers_eccv_2022_papers_ECCV_papers_136810068-supp.pdf': 'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810068-supp.pdf',
}
for fn in ['clamp', 'exp', 'gather', 'log', 'mean', 'nn.Module', 'nn.functional.log_softmax', 'nn.functional.one_hot',
           'no_grad', 'optim.Adam', 'sum']:
    M[f'docs.pytorch.org/docs_2.11_generated_torch.{fn}'] = P + f'docs/2.11/generated/torch.{fn}.html'
for fn in ['ColorJitter', 'Normalize', 'RandomHorizontalFlip']:
    M[f'docs.pytorch.org/vision_stable_generated_torchvision.transforms.{fn}'] = P + f'vision/stable/generated/torchvision.transforms.{fn}.html'

links = set()
for nb in list(ROOT.glob('*.ipynb')) + list((ROOT / 'problems_original').glob('*.ipynb')):
    import re
    links |= set(re.findall(r'docs/pages/([^)#"\s]+?)--[0-9a-f]+\.html', nb.read_text(encoding='utf-8')))
missing = links - set(M)
assert not missing, missing
items = []
for nb_text in [None]:
    pass
for key in sorted(links):
    url = M[key]
    import re as _r
    full = [m for m in (ROOT / 'problems_original').glob('*.ipynb')]
    names = set()
    for nb in full:
        names |= set(_r.findall(r'docs/pages/(' + _r.escape(key) + r'--[0-9a-f]+\.html)', nb.read_text(encoding='utf-8')))
    for name in names:
        path = ROOT / 'docs' / 'pages' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={html.escape(url)}">'
                        f'<p>온라인 원문으로 이동: <a href="{html.escape(url)}">{html.escape(url)}</a></p>', encoding='utf-8')
        items.append((key, url, 'pages/' + name))
rows = '\n'.join(f'<li><a href="{html.escape(u)}">{html.escape(k)}</a></li>' for k, u, _ in sorted(items))
(ROOT / 'docs' / 'index.html').write_text(
    '<!doctype html><meta charset="utf-8"><title>참고 문서 목록</title><h1>참고 문서 목록 (연습용: 온라인 원문 링크)</h1>'
    '<p>시험장에서는 같은 문서가 오프라인 HTML로 제공됩니다.</p><ul>' + rows + '</ul>', encoding='utf-8')
print('docs pages:', len(items))
