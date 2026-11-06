"""[예측 문제] 공식 샘플(../AI_Lv3_sample_solution/problems_original)을 틀로 써서
F1_1, F1_2, F2_1, F2_2, F3 문제 노트북과 정답 노트북을 만듭니다.

문제: F*_problem.ipynb    정답: F*_answer.ipynb (같은 폴더, 작업 폴더=.venv 위치)
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT.parent / 'AI_Lv3_sample_solution' / 'problems_original'
sys.path.insert(0, str(ROOT / '_build'))
from f3_data import F3_SCHEMA_CODE, F3_STORY, F3_EXAMPLES_MD, F3_FORMAT_MD  # noqa: E402

THRESH = json.load(open(ROOT / '_build' / 'thresholds.json', encoding='utf-8')) if (ROOT / '_build' / 'thresholds.json').exists() else {}


def lines(text):
    text = text.strip('\n')
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines(text)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': lines(text)}


def tpl(name):
    nb = json.load(open(TPL / f'{name}_problem.ipynb', encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    return nb


def src(cell):
    return ''.join(cell['source'])


def save(nb, cells, path):
    nb = copy.deepcopy(nb)
    nb['cells'] = cells
    nb.setdefault('metadata', {})['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (.venv)', 'language': 'python'}
    json.dump(nb, open(ROOT / path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', path, len(cells))


def pair(name, cells_problem, answer_map, nb):
    """answer_map: {problem cell index: [replacement cells...]}  (첫 번째가 대체, 나머지는 뒤에 추가)"""
    save(nb, cells_problem, f'{name}_problem.ipynb')
    out = []
    for i, c in enumerate(cells_problem):
        if i in answer_map:
            out.extend(answer_map[i])
        else:
            out.append(copy.deepcopy(c))
    save(nb, out, f'{name}_answer.ipynb')


STUDENT_ANS = """# 본인의 수험번호를 따옴표 안에 입력하세요. 예: '001234'
student_id = '000000'   # [정답안] 예시 번호
if not student_id or not student_id.isascii() or not student_id.isdigit():
    raise ValueError('수험번호를 숫자로만 입력하세요. 앞자리 0도 그대로 적습니다.')
print('수험번호:', student_id)"""

ENV_ANS = """# [정답안] 환경 확인 - CPU/GPU 겸용 (문제 노트북은 GPU가 없으면 여기서 멈춥니다)
import sys
from pathlib import Path

print('Python:', sys.version.split()[0])
print('현재 커널:', sys.executable)
print('작업 폴더:', Path.cwd())
expected_env = Path.cwd() / '.venv'
if Path(sys.prefix).resolve() != expected_env.resolve():
    raise RuntimeError(f"{expected_env / 'Scripts' / 'python.exe'}를 노트북 커널로 선택하세요.")
import torch, torchvision, timm
print('PyTorch:', torch.__version__, '/ torchvision:', torchvision.__version__, '/ timm:', timm.__version__)
if torch.cuda.is_available():
    DEVICE, QUICK = 'cuda', False
    print('GPU:', torch.cuda.get_device_name(0), '-> 원본 설정 그대로 실행합니다.')
else:
    DEVICE, QUICK = 'cpu', True
    print('GPU 없음 -> QUICK 모드: 학습 데이터와 epoch를 줄여 CPU에서 끝까지 실행합니다 (점수는 낮게 나옴).')"""

QUICK_MD = md('> **[정답안 전용 셀]** 시험 노트북에는 없는 셀입니다. GPU가 있으면 아무 일도 하지 않습니다.')


def quick_data(n_train=2500, n_val=800):
    return [QUICK_MD, code(f"""# [정답안 전용] CPU QUICK 모드에서만 데이터를 줄입니다.
if QUICK:
    _rng = np.random.default_rng(0)
    _keep = np.sort(_rng.choice(len(train_labels), size=min({n_train}, len(train_labels)), replace=False))
    train_images, train_labels = train_images[_keep], train_labels[_keep]
    _keep = np.sort(_rng.choice(len(val_labels), size=min({n_val}, len(val_labels)), replace=False))
    val_images, val_labels = val_images[_keep], val_labels[_keep]
    print('QUICK 모드 데이터:', len(train_labels), len(val_labels))""")]


def quick_epochs(var='epochs', n=3):
    return [QUICK_MD, code(f"""if QUICK:
    {var} = {n}
    print('QUICK 모드 epoch:', {var})""")]


def banner(text):
    return md('> ## [예측 문제 정답안]\n> ' + text.replace('\n', '\n> '))


PREDICT_NOTE = """> **[예측 문제]** 공식 샘플의 형식·제약을 그대로 따르고 데이터와 함정만 바꾼 연습 문제입니다. 실제 시험 문제가 아닙니다."""


def th(key, default):
    return THRESH.get(key, default)


# ===================================================================== F1_1
def build_f1_1():
    nb = tpl('P1_1')
    c = nb['cells']
    hi, mid = th('f1_1', (0.80, 0.70))
    story = md(f"""# 문제 1-1. (예측)

{PREDICT_NOTE}

> 나는 온라인 패션 쇼핑몰의 데이터 담당자다. 이번 시즌부터 **협력사 세 곳이 보내 준 상품 사진을 기존 학습 데이터에 합쳐** 상품 분류 모델을 다시 학습했다. \\
> 그런데 합친 뒤로 오히려 **샌들·조리·힐, 셔츠·티셔츠, 운동화·캐주얼화** 쪽에서 오분류가 늘었다는 보고가 들어왔다.

>> “협력사 데이터가 들어오면서 학습 데이터 자체에 문제가 생긴 것 같습니다. 업로드 과정에서 파일이 깨지거나 같은 파일이 여러 번 올라온 흔적도 있어요.”

> 나는 학습 데이터를 점검하고, **학습에 쓸 항목을 골라내고(제외 가능)** 필요하면 보강하여 성능을 회복하려 한다.

### 문제 안내
서비스 팀에서 수정을 요청한 셀에는 표시가 되어 있습니다. 그 외 셀은 수정하지 마세요.

- 환경 제약으로 **TinyViT 5M을 Linear Probing으로 10 epoch** 학습한 모델만 사용합니다. 모델·Loss·학습률·batch 크기는 고정입니다.
- 제공된 train만 사용하세요. **train 항목을 학습에서 제외하는 것은 허용**됩니다.
- 클래스 번호와 원본 이미지 크기(높이 80 × 너비 60)를 유지하세요.""")
    data_cell = code("""# 수정 금지

data_path = Path('data/f1_1_noisy/prepared.npz')
if not data_path.is_file() or not data_path.with_name('classes.json').is_file():
    raise FileNotFoundError('데이터가 없습니다. 먼저 setup.cmd를 실행하고, 작업 폴더를 이 노트북이 있는 폴더로 맞추세요.')""")
    task_md = md(src(c[11]).replace('상품 이미지 6종', '상품 이미지 10종'))
    answer_md = md(src(c[13]).replace('위에서 확인한 데이터 분석에 따르면, 위 문제 중 하나의 문제가 가장 의심된다.',
                                      '위 목록 중 **두 가지 이상**이 섞여 있는 것 같다.')
                   .replace('데이터 증강은 데이터에 변화를 주어 마치 새로운 데이터를 합성한 것과 같은 효과를 줄 수 있으니, 이를 활용하여 문제를 해결하자!',
                            '문제가 있는 항목은 학습에서 빼고, 그 때문에 부족해진 클래스는 증강으로 보강하자!')
                   + """

### 추가 안내 (예측 문제)
- `processing_training_data(labels)`는 **라벨만** 인자로 받습니다. 이미지 내용을 확인하려면 이 함수 안에서 전역 변수 `train_images`를 읽어도 됩니다(읽기만, 수정 금지).
- 반환하는 `indices`에서 원본 번호를 **빼면 그 항목은 학습에서 제외**됩니다.
- 이미지 비교에는 NumPy 연산, `hashlib` 등 표준 라이브러리를 써도 됩니다.
- 아래 셀에 무엇을 발견했고 왜 그렇게 처리했는지 **주석으로 남기세요** (채점 요소).""")
    perf_md = md(src(c[19]).replace('- High-level: 0.75 초과\n- Mid-level: 0.60 초과 0.75 이하\n- Low-level: 0.60 이하',
                                    f'- High-level: {hi:.2f} 초과\n- Mid-level: {mid:.2f} 초과 {hi:.2f} 이하\n- Low-level: {mid:.2f} 이하'))
    hp = code(src(c[20]).replace("P1_1_{student_id}", "F1_1_{student_id}"))
    submit = md(src(c[27]).replace('P1_1_', 'F1_1_'))
    cells = c[:3] + [story] + c[4:9] + [data_cell, c[10], task_md, c[12], answer_md] + c[14:19] + [perf_md, hp] + c[21:27] + [submit]
    # 인덱스: 0..2, 3 story, 4..8, 9 data, 10, 11 task, 12, 13 answer md, 14, 15 starter, 16..18, 19 perf, 20 hp, 21..26, 27 submit

    answer = code("""# 수정 가능
# train_transform과 processing_training_data는 자유롭게 수정해도 됩니다.
# 다만, train_transform과 processing_training_data의 이름과 매개변수는 변경하지 마세요.
import hashlib

# [정답안] 보강용 약한 증강 (상품 정체성 유지, 흰 배경 채움, 80x60 유지)
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=255),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean, std),
])


def image_is_broken(image):
    \"\"\"손상 이미지: (1) 거의 단색(빈 흰/검은 화면)  (2) 이웃 픽셀 차이가 비정상적으로 큰 잡음.\"\"\"
    pixels = image.astype(np.int16)
    if pixels.std() < 3:
        return True
    roughness = np.abs(np.diff(pixels, axis=1)).mean()   # 실제 사진은 이웃 픽셀이 비슷 (보통 < 15)
    return roughness > 40


def processing_training_data(labels):
    \"\"\"[정답안]
    발견한 문제
      1) 라벨 오류: 내용이 '완전히 같은' 이미지가 서로 다른 라벨로 들어 있음
         (협력사 업로드 오류: Sandals<->Flip Flops, Shirts->Tshirts, Sports->Casual Shoes, Heels->Sandals)
      2) 낮은 품질: 빈 흰 화면, 검은 화면, 잡음 이미지 (라벨이 무작위)
      3) 중복 편중: 같은 사진이 15번씩 반복 (Watches, Handbags)
    처리
      - 모든 이미지를 md5로 묶어, 한 묶음에 라벨이 2개 이상이면 '어느 쪽이 맞는지 알 수 없으므로' 묶음 전체 제외
      - 라벨이 하나인 묶음은 1장만 남김 (반복 제거)
      - 손상 이미지는 제외
      - 정리 후 클래스 수가 달라지므로, 가장 많은 클래스 수까지 오버샘플링하고 반복분에만 증강(flag=True)
    \"\"\"
    labels = np.asarray(labels)
    groups = {}
    for index in range(len(labels)):
        key = hashlib.md5(train_images[index].tobytes()).hexdigest()
        groups.setdefault(key, []).append(index)

    keep = []
    removed = {'conflict': 0, 'duplicate': 0, 'broken': 0}
    for members in groups.values():
        if len(set(labels[members].tolist())) > 1:
            removed['conflict'] += len(members)
            continue
        first = members[0]
        removed['duplicate'] += len(members) - 1
        if image_is_broken(train_images[first]):
            removed['broken'] += 1
            continue
        keep.append(first)
    keep = np.array(sorted(keep))
    print('제외한 항목:', removed, '/ 남은 항목:', len(keep))

    rng = np.random.default_rng(42)
    counts = np.bincount(labels[keep], minlength=num_classes)
    print('정리 후 클래스별 수:', dict(zip(class_names, counts.tolist())))
    target = counts.max()
    indices, augment_flags = [], []
    for label in range(num_classes):
        members = keep[labels[keep] == label]
        indices.append(members)
        augment_flags.append(np.zeros(len(members), dtype=bool))
        if 0 < len(members) < target:
            indices.append(rng.choice(members, size=target - len(members), replace=True))
            augment_flags.append(np.ones(target - len(members), dtype=bool))
    return np.concatenate(indices).astype(int), np.concatenate(augment_flags).astype(bool)""")
    explain = md("""### [정답안 해설]
| 점검 | 방법 | 발견 |
|---|---|---|
| 같은 내용, 다른 라벨 | `md5(image.tobytes())`로 묶고 라벨 종류 수 확인 | 협력사 사본 약 1,370장이 엉뚱한 라벨 |
| 손상 이미지 | 표준편차(단색) + 이웃 픽셀 차이(잡음) | 흰/검은/잡음 180장 |
| 반복 편중 | 같은 묶음 크기 | 같은 사진 15회 반복 450장 |

- **라벨이 엇갈린 묶음은 통째로 제외**: 원본과 사본 중 어느 쪽이 맞는지 이 함수 안에서는 알 수 없습니다(다수결은 사본이 2개인 경우 틀린 쪽을 고름).
- 정리 후 일부 클래스가 줄어드므로 **오버샘플링 + 반복분 증강**으로 균형을 맞춥니다(1-1 샘플과 같은 기법).
- 시험에서는 `print`로 발견 개수를 남기고, 무엇을 왜 뺐는지 주석으로 쓰는 것이 채점 요소입니다.""")
    answer_map = {1: [code(STUDENT_ANS)], 6: [code(ENV_ANS)], 3: [story, banner('라벨 충돌 사본·손상·반복 이미지를 해시와 통계로 찾아 제외 → 오버샘플링 + 증강으로 균형')],
                  15: [answer, explain], 20: [hp] + quick_epochs('epochs', 3)}
    pair('F1_1', cells, answer_map, nb)


# ===================================================================== F1_2 (분포 차이, 증강만 수정)
def build_f1_2():
    nb = tpl('P2_1')
    c = nb['cells']
    hi, mid = th('f1_2', (0.70, 0.55))
    story = md(f"""# 문제 1-2. (예측)

{PREDICT_NOTE}

> 우리 쇼핑몰은 상품 사진으로 카테고리를 자동 분류한다. 학습 데이터는 **스튜디오에서 찍은 정돈된 상품컷**(흰 배경, 가운데, 꽉 찬 크기)이다. \\
> 이번에 **판매자 앱 업로드 기능**이 열리면서, 판매자가 직접 찍은 사진이 들어오기 시작했다. 이 사진들은 상품이 작게 찍히거나, 한쪽에 치우치거나, 기울어지거나, 어둡거나, 흑백인 경우가 있다.

검증(`val`) 데이터는 **판매자 앱 업로드 사진**에서 뽑았습니다. 학습 데이터는 바꿀 수 없고, **학습용 변환(`train_transform`)만** 수정할 수 있습니다.

## 공통 조건
- 모델은 TinyViT 5M `dist_in22k_ft_in1k`, **Full Fine-Tuning**, 구조·epoch(12)·학습률·batch·Loss는 고정입니다.
- 입력은 **높이 80 × 너비 60 RGB**로 유지합니다(텐서 변환과 정규화도 유지).
- train / val 구분과 정답은 변경하지 마세요.

## 평가 지표
- 최하단 `[Final] Validation macro-F1` 기준: High {hi:.2f} 초과 / Mid {mid:.2f} 초과 {hi:.2f} 이하 / Low {mid:.2f} 이하""")
    data_cell = code("""# 수정 금지

data_path = Path('data/f1_2_shift/prepared.npz')
if not data_path.is_file() or not data_path.with_name('classes.json').is_file():
    raise FileNotFoundError('데이터가 없습니다. 먼저 setup.cmd를 실행하고, 작업 폴더를 이 노트북이 있는 폴더로 맞추세요.')""")
    view = code("""# 수정 가능
# 확인용 그림입니다. 자유롭게 고쳐서 데이터를 살펴보세요.
fig, axes = plt.subplots(2, 8, figsize=(14, 4.5))
for axis, index in zip(axes[0], range(0, len(train_labels), len(train_labels) // 8)):
    axis.imshow(train_images[index]); axis.set_title('train: ' + class_names[train_labels[index]], fontsize=7)
for axis, index in zip(axes[1], range(0, len(val_labels), len(val_labels) // 8)):
    axis.imshow(val_images[index]); axis.set_title('val: ' + class_names[val_labels[index]], fontsize=7)
for axis in axes.flat:
    axis.axis('off')
plt.tight_layout()
plt.show()""")
    tf_md = md("""# **답안 작성 셀 - 학습용 변환**

학습 데이터가 검증(판매자 앱) 데이터의 조건에 대응할 수 있도록 `train_transform`을 설계하세요.
- 변환의 추가·삭제·순서·범위·확률을 바꿀 수 있습니다. 이름은 바꾸지 마세요.
- 출력은 정규화된 `torch.float32` 텐서 `(3, 80, 60)`이어야 합니다. `eval_transform`은 수정하지 마세요.
- 제공된 기본 변환 중 **오히려 해가 되는 변환**이 있을 수 있습니다. 선택 이유를 주석으로 남기세요.

### 공식 문서
- [torchvision Transform 전처리 요소 설명](docs/pages/docs.pytorch.org/vision_main_transforms--7c151205d9a0.html)
- [Transform 적용 결과 비교 예제](docs/pages/docs.pytorch.org/vision_0.26_auto_examples_transforms_plot_transforms_illustrations--0d046f058aa3.html)""")
    tf = code("""# 수정 가능

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomVerticalFlip(p=0.5),
    T.RandomResizedCrop((80, 60), scale=(0.2, 0.6)),
    T.ColorJitter(hue=0.5),
    T.ToTensor(),
    T.Normalize(mean, std),
])""")
    model_md = md("""## 고정 모델 (Full Fine-Tuning)
아래 모델·Loss·학습 설정은 수정하지 마세요.""")
    model = code("""# 수정 금지

class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(dim=64)
        self.stages = nn.Sequential(
            ConvStage(dim=64, depth=2),
            TinyViTStage(64, 128, depth=2, num_heads=4, window_size=3),
            TinyViTStage(128, 160, depth=6, num_heads=5, window_size=5),
            TinyViTStage(160, 320, depth=2, num_heads=10, window_size=3),
        )
        self.head = ClassifierHead(320, num_classes=1000)
        load_pretrained_weights(self)
        self.head.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(x)""")
    hp = code("""# 수정 금지

batch_size = 64
learning_rate = 0.0001
epochs = 12
loss_fn = nn.CrossEntropyLoss()
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
checkpoint = output_dir / f'F1_2_{student_id}.pt'""")
    loaders = code("""# 수정 금지

train_dataset = FashionDataset(train_images, train_labels, train_transform)
val_dataset = FashionDataset(val_images, val_labels, eval_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)""")
    train = code(src(c[35]).replace('loss_fn = custom_loss\n', ''))
    final = code("""# 수정 금지

model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
val_loss, val_f1 = evaluate(model, val_loader, loss_fn)
print(f'[Final] Validation macro-F1: {val_f1:.4f}')""")
    submit = md("""## 제출
- 완성한 노트북 `.ipynb`
- Best 체크포인트 `outputs/F1_2_<수험번호>.pt`
- 실제로 실행한 validation 결과를 노트북에 남기세요.""")
    cells = (c[:3] + [story] + c[4:7] + [c[8], data_cell, c[10], view, c[11], c[12], tf_md, tf]
             + c[14:24] + [model_md, model, hp, c[33], loaders, train, c[36], final, submit])
    # idx: 0-2, 3 story, 4,5,6(env), 7 imports, 8 data, 9 load, 10 view, 11 eval_tf, 12 dataset, 13 tf_md, 14 tf, ...
    answer_tf = code("""# 수정 가능

# [정답안] 검증(앱 업로드) 사진에서 본 변화만, 비슷한 크기로 넣습니다.
#  - 작게 찍힘 + 치우침: RandomAffine scale 0.5~1.0, translate 0.15 (빈 곳은 흰 배경 255)
#  - 기울어짐: degrees 15
#  - 어둡거나 대비 낮음: ColorJitter brightness/contrast
#  - 흑백 사진: RandomGrayscale p=0.25
# 제거한 기본 변환
#  - RandomVerticalFlip: 신발·가방이 뒤집힌 사진은 실제로 없음 -> 잘못된 단서
#  - RandomResizedCrop(scale 0.2~0.6): '확대'만 해서 검증 사진(축소)과 반대 방향, 상품 일부만 남김
#  - ColorJitter(hue=0.5): 색상이 뒤바뀌어 색 단서(양말, 지갑 등)를 망가뜨림
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.5, 1.0), fill=255),
    T.ColorJitter(brightness=0.35, contrast=0.3),
    T.RandomGrayscale(p=0.25),
    T.ToTensor(),
    T.Normalize(mean, std),
])""")
    explain = md("""### [정답안 해설]
- 위 그림의 val 줄을 보면 상품이 **작고(축소), 치우치고, 기울고, 어둡고, 일부는 흑백**입니다. 학습 데이터에는 이런 사진이 없으므로 **증강으로 만들어 줍니다**.
- 핵심은 *방향*입니다. 제공된 `RandomResizedCrop`은 확대만 하므로 축소된 검증 사진과 반대입니다. 축소는 `RandomAffine(scale=(0.5, 1.0))` + 흰색 채움으로 만듭니다.
- 해가 되는 변환(상하 반전, 색상 회전)을 빼는 것도 점수 요소입니다.""")
    answer_map = {1: [code(STUDENT_ANS)], 6: [code(ENV_ANS)], 3: [story, banner('val(앱 사진)의 축소·이동·회전·조명·흑백을 증강으로 재현, 해로운 기본 변환 제거')],
                  9: [c[10]] + quick_data(2500, 800), 14: [answer_tf, explain], 27: [hp] + quick_epochs('epochs', 3)}
    pair('F1_2', cells, answer_map, nb)


# ===================================================================== F2_1 (긴 꼬리 + Focal)
def build_f2_1():
    nb = tpl('P2_1')
    c = nb['cells']
    story = md(f"""# 문제 2-1. (예측)

{PREDICT_NOTE}

신상품 카테고리가 늘면서 **카테고리별 학습 사진 수가 크게 차이 나는(긴 꼬리, 최대 800장 ~ 최소 30장)** 데이터로 모델을 학습해야 합니다. 검증 데이터는 클래스마다 100장씩 균형입니다.

Task A, Task B, Task C로 이루어져 있으며, 수정해야 하는 셀에는 표시가 되어 있습니다. 그 외 셀은 수정하지 마세요.

- Task A는 `FashionModel`에서 학습 범위를 설정합니다. **고정: `patch_embed`, `stages[0]`, `stages[1]`, `stages[2]` / 학습: `stages[3]`, `head` 전체**
- Task B는 `custom_loss(logits, targets)`에 **클래스 불균형을 고려한 Loss**를 직접 구현합니다. (예: Focal Loss, 클래스 가중 CE, Logit Adjustment, Label Smoothing과의 조합)
- Task C는 `batch_size`, `learning_rate`, `epochs`를 조정합니다.
- 모델은 TinyViT 5M `dist_in22k_ft_in1k`, 입력 높이 80 × 너비 60 유지. 데이터 분할·전처리·구성 블록·가중치 로딩·학습/평가 코드는 수정 금지.

### 주의! 노트북에 반드시 포함
- 고정/학습 구간과 수정한 코드
- 선택한 Loss와 구현 코드, 선택 이유(주석)
- 비교한 하이퍼파라미터와 validation 결과, 최종 설정
- epoch별 validation macro-F1과 best validation macro-F1""")
    data_cell = code("""# 수정 금지

data_path = Path('data/f2_1_longtail/prepared.npz')
if not data_path.is_file() or not data_path.with_name('classes.json').is_file():
    raise FileNotFoundError('데이터가 없습니다. 먼저 setup.cmd를 실행하고, 작업 폴더를 이 노트북이 있는 폴더로 맞추세요.')""")
    counts_cell = code("""# 수정 금지
# Task B에서 사용할 수 있는 클래스별 학습 이미지 수입니다.

class_counts = torch.bincount(torch.as_tensor(train_labels), minlength=num_classes).float()
plt.figure(figsize=(9, 3))
plt.bar(range(num_classes), class_counts.numpy())
plt.xticks(range(num_classes), class_names, rotation=60, ha='right', fontsize=7)
plt.ylabel('Count')
plt.tight_layout()
plt.show()""")
    task_a_md = md(src(c[13]).replace('- 고정할 구간: `patch_embed`, `stages[0]`, `stages[1]`\n- 학습할 구간: `stages[2]`, `stages[3]`, `head` 전체(`norm`, `fc`)',
                                     '- 고정할 구간: `patch_embed`, `stages[0]`, `stages[1]`, `stages[2]`\n- 학습할 구간: `stages[3]`, `head` 전체(`norm`, `fc`)'))
    task_b_md = md(src(c[26]).replace('선택한 분류 Loss를 PyTorch 텐서 연산으로 직접 구현해야 합니다. Cross Entropy에 한정하지 않습니다.',
                                      '**클래스 불균형을 고려한** 분류 Loss를 PyTorch 텐서 연산으로 직접 구현해야 합니다. 위 셀의 `class_counts`(클래스별 학습 이미지 수, `(클래스 수,)` 실수 텐서)를 사용해도 됩니다.'))
    ckpt = code(src(c[31]).replace('P2_1_', 'F2_1_'))
    cells = c[:3] + [story] + c[4:9] + [data_cell, c[10], counts_cell] + c[11:13] + [task_a_md] + c[14:26] + [task_b_md] + c[27:31] + [ckpt] + c[32:39] + [md(src(c[39]).replace('P2_1_', 'F2_1_'))]
    # idx: 9 data, 10 load, 11 counts, 12 eval_tf, 13 dataset, 14 taskA md, ... model cell at 26, task_b md 27, loss 28, ... hp 31, ckpt 32
    model_ans = code("""# 수정 가능
class FashionModel(nn.Module):
    \"\"\"[정답안] 고정: patch_embed, stages[0..2] / 학습: stages[3], head\"\"\"
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(dim=64)
        self.stages = nn.Sequential(
            ConvStage(dim=64, depth=2),
            TinyViTStage(64, 128, depth=2, num_heads=4, window_size=3),
            TinyViTStage(128, 160, depth=6, num_heads=5, window_size=5),
            TinyViTStage(160, 320, depth=2, num_heads=10, window_size=3),
        )
        self.head = ClassifierHead(320, num_classes=1000)
        load_pretrained_weights(self)
        self.head.fc = nn.Linear(320, num_classes)

        self.requires_grad_(False)
        self.stages[3].requires_grad_(True)
        self.head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.patch_embed, self.stages[0], self.stages[1], self.stages[2]):
            module.train(False)          # 고정 구간 BatchNorm 통계 유지
        return self

    def forward(self, x):
        with torch.no_grad():
            x = self.patch_embed(x)
            x = self.stages[0](x)
            x = self.stages[1](x)
            x = self.stages[2](x)
        x = self.stages[3](x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(x)""")
    loss_ans = code("""# 수정 가능

def custom_loss(logits, targets):
    \"\"\"[정답안] 클래스 가중 Focal Loss (gamma=2, 가중치 = (1/n_c)^0.5 정규화)

    선택 이유 (주석)
      - 긴 꼬리 데이터에서 보통 CE는 다수 클래스(800장) 쪽으로 결정 경계가 밀려 소수 클래스(30장) F1이 낮습니다.
      - 클래스 가중치: 적은 클래스의 오류에 더 큰 벌점. 1/n 그대로 쓰면 30장 클래스가 너무 커서 불안정하므로
        제곱근으로 완화하고 평균 1이 되도록 정규화합니다.
      - Focal 항 (1-p_t)^gamma: 이미 잘 맞히는 쉬운 샘플(대부분 다수 클래스)의 Loss를 줄여
        어려운 샘플에 학습을 집중합니다.
    수식: loss_i = -w[y_i] * (1 - p_i[y_i])^gamma * log p_i[y_i],  배치 가중 평균
    \"\"\"
    gamma = 2.0
    weights = class_counts.to(logits.device).clamp(min=1).pow(-0.5)
    weights = weights / weights.mean()
    log_probs = F.log_softmax(logits.float(), dim=1)
    log_pt = log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    w = weights[targets]
    loss = -w * (1.0 - pt).pow(gamma) * log_pt
    return loss.sum() / w.sum()""")
    hp_ans = code("""# 수정 가능

# [정답안] 비교 (Task A + 가중 Focal, 시드 42) -> README의 실행 결과 표 참고
batch_size = 32
learning_rate = 0.0003
epochs = 20""")
    explain = md("""### [정답안 해설]
- 학습 범위가 `stages[3]`+`head`로 좁으므로 학습률은 샘플(2-1)보다 높게(3e-4) 써도 안정적입니다.
- 불균형 대응은 **Loss에서만** 해야 합니다(데이터·샘플러 수정 금지). 가중 CE, Focal, Logit Adjustment(`logits + tau*log(prior)`) 중 하나를 고르고 **이유를 주석으로** 쓰는 것이 핵심입니다.
- 정확도(accuracy)가 아니라 **macro-F1**이므로 소수 클래스 성능이 그대로 점수에 반영됩니다.""")
    answer_map = {1: [code(STUDENT_ANS)], 6: [code(ENV_ANS)], 3: [story, banner('Task A stages[3]+head 학습 / Task B 클래스 가중 Focal Loss / Task C 학습률·batch 조정')],
                  10: [c[10]] + quick_data(3000, 800), 26: [model_ans], 28: [loss_ans],
                  31: [hp_ans, explain] + quick_epochs('epochs', 3)}
    pair('F2_1', cells, answer_map, nb)


# ===================================================================== F2_2 (Kaggle)
def build_f2_2():
    nb = tpl('P2_2')
    c = nb['cells']
    story = md(f"""# 문제 2-2. 이미지 분류 모델 개선 (Kaggle, 예측)

{PREDICT_NOTE}

상품 이미지로 TinyViT를 학습하고, 테스트 이미지의 클래스를 예측하세요. 가장 높은 accuracy가 목표입니다.\\
학습 데이터는 20개 클래스에 각각 400개씩, 총 8,000개입니다(적은 클래스는 같은 이미지 반복).\\
**검증과 테스트 데이터의 절반은 판매자 앱 업로드 사진**(작게·치우쳐·기울어·어둡게·흑백)입니다.\\
**# 수정 가능** 표시가 있는 증강, 모델 구조, 하이퍼파라미터, 학습 셀을 수정하세요.\\
데이터, 클래스 번호, 이미지 크기(높이 80 × 너비 60)는 유지하세요.\\
노트북과 같은 폴더의 `test.csv`(`id,answer`)를 추론 셀이 채웁니다. 완료된 `test.csv`를 이름 변경 없이 제출하세요.""")
    load = code("""# 수정 금지

with (data_dir / 'classes.json').open(encoding='utf-8') as file:
    class_names = json.load(file)
num_classes = len(class_names)

with np.load(data_path, allow_pickle=False) as data:
    train_images = data['train_images']
    train_labels = data['train_labels'].reshape(-1)
    val_images = data['val_images']
    val_labels = data['val_labels'].reshape(-1)

print('학습:', len(train_images), '검증:', len(val_images))
print('클래스:', num_classes, '/ 이미지 크기:', train_images[0].shape)""")
    path = code("""# 수정 금지

data_dir = Path('data/f2_2_kaggle')
data_path = data_dir / 'prepared.npz'""")
    data_md = md("""## 데이터 불러오기
`prepared.npz`의 학습 8,000장(클래스당 400장)과 검증 2,000장(절반은 앱 업로드 사진)을 읽습니다.\\
테스트 이미지는 `data/f2_2_kaggle/test/<id>`에 있습니다.""")
    ckpt = code(src(c[26]).replace('P2_2_', 'F2_2_'))
    cells = c[:3] + [story] + c[4:8] + [data_md, path, load] + c[11:26] + [ckpt] + c[27:34]
    # 같은 인덱스 유지 (8 md, 9 path, 10 load, 11.., 13 tf, 23 model, 25 hp, 26 ckpt, 29 fns, 30 train, 33 inference)
    import importlib.util
    spec = importlib.util.spec_from_file_location('ba', ROOT.parent / 'AI_Lv3_sample_solution' / '_build' / 'build_answers.py')
    # 샘플 정답안의 P2_2 셀 문자열 재사용 (빌드 부작용 없이 읽기만)
    text = (ROOT.parent / 'AI_Lv3_sample_solution' / '_build' / 'build_answers.py').read_text(encoding='utf-8')
    ns = {'__file__': str(ROOT.parent / 'AI_Lv3_sample_solution' / '_build' / 'build_answers.py')}
    start = text.index('P2_2_IMPORT = ')
    end = text.index("build('P2_2',")
    exec('from pathlib import Path\nimport json\n' + text[:text.index('# ============================================================ P1_1')] + text[start:end], ns)
    tf_ans = code("""# 수정 가능

# [정답안] 앱 업로드 사진(축소·치우침·기울기·조명·흑백)까지 대비하는 증강
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.5, 1.05), fill=255),
    T.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.15),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
])""")
    score = [md('> **[정답안 전용 셀]** 로컬 채점 (Kaggle 대신).'),
             code("""# [정답안 전용] 로컬 채점
import subprocess, sys
print(subprocess.run([sys.executable, '_grading/score_f2_2.py'], capture_output=True, text=True).stdout)""")]
    explain = md(ns['P2_2_EXPLAIN']['source'] and ''.join(ns['P2_2_EXPLAIN']['source']).replace(
        '| 증강 | 좌우 반전, 작은 이동·회전·확대(흰 배경 채움), 약한 색 변화, 작은 지우기 | 클래스당 600장(일부 반복) → 반복 이미지 암기 방지 |',
        '| 증강 | 좌우 반전, **축소(0.5배)·이동·회전**(흰 배경 채움), 밝기·대비, **흑백**, 작은 지우기 | 검증·테스트 절반이 앱 사진 → 분포 차이 대응 (1-2와 같은 원리) |')
        .replace('`python _grading/score_p2_2.py`', '`python _grading/score_f2_2.py`'))
    answer_map = {1: [code(STUDENT_ANS)], 6: [code(ENV_ANS)], 3: [story, banner('분포 이동 대비 증강 + AdamW/warmup-cosine + label smoothing + Dropout + eval 좌우반전 TTA')],
                  7: [code(ns['P2_2_IMPORT'])], 10: [load] + quick_data(3000, 800), 13: [tf_ans], 23: [code(ns['P2_2_MODEL'])],
                  25: [code(ns['P2_2_HP'])] + quick_epochs('epochs', 3), 29: [code(ns['P2_2_TRAIN_FN'])],
                  30: [code(ns['P2_2_TRAIN']), explain], 33: [cells[33]] + score}
    pair('F2_2', cells, answer_map, nb)


# ===================================================================== F3
def build_f3():
    nb = tpl('P3')
    c = nb['cells']
    def fix(cell):
        return md(src(cell).replace('26개 키', '모든 키').replace('P3_qa.csv', 'F3_qa.csv').replace('P3_', 'F3_'))

    load_md = md("""## 데이터 예시 및 응답 Schema

노트북과 같은 폴더의 `F3_qa.csv`에 프롬프트 테스트용 예제 2개가 제공됩니다. 한 행이 한 문제이며 `task`, `question` 두 열만 있습니다. 정답은 공개하지 않습니다.""")
    load = code(src(c[9]).replace("load_questions('P3_qa.csv')", "load_questions('F3_qa.csv')"))
    prompt_md = md(src(c[14]))
    starter = code(src(c[16]).replace("'주어진 녹취에서 계약 정보를 JSON으로 정리하세요.'", "'주어진 상담 녹취에서 반품·교환 접수 정보를 JSON으로 정리하세요.'"))
    run = code(src(c[22]).replace("f'P3_{student_id}.txt'", "f'F3_{student_id}.txt'"))
    submit = md(src(c[23]).replace('P3_', 'F3_'))
    example_cell = code("""# 수정 금지
# 형식 확인용 예시이며, 실제 문항의 정답이 아닙니다. API는 호출하지 않습니다.

example_answer = {name: None for name in response_schema['properties']}
example_answer['주문번호'] = 'A20300101-0001'
example_answer['결제금액_원'] = 39000
example_answer['요청유형'] = '교환'
example_answer['접수일'] = '2030-01-02'
example_response = json.dumps(example_answer, ensure_ascii=False)
parsed = parse_response(example_response)
print('요청유형:', parsed['요청유형'], '/ 환불예정금액_원:', parsed['환불예정금액_원'])""")
    parse = code(src(c[18]).replace(
        "        if rule.get('format') == 'date':",
        "        if 'enum' in rule and value not in rule['enum']:\n            raise ValueError(f'{name}: 허용된 값이 아닙니다. 가능한 값: {rule[\"enum\"]}')\n        if rule.get('format') == 'date':"))
    cells = (c[:4] + [md(F3_STORY)] + c[5:8] + [load_md, load] + [md(F3_EXAMPLES_MD), md(F3_FORMAT_MD), code(F3_SCHEMA_CODE)]
             + [prompt_md, c[15], starter, fix(c[17]), parse, example_cell, fix(c[20]), c[21], run, submit])
    # idx: 4 story, 8 load_md, 9 load, 10 examples, 11 format, 12 schema, 13 prompt_md, 14, 15 starter, 16, 17 parse, 18 example, 19, 20 call, 21 run, 22 submit
    from f3_data import F3_PROMPT_ANSWER
    explain = md("""### [정답안 해설]
- 샘플(전세계약)과 같은 구조: **형식 규칙 → 최종 합의/정정 원칙 → 항목별 정의 → 계산 규칙**.
- 이 문항의 함정: 고객이 말을 바꾸는 사이즈·색상, 상담원이 정정하는 주문번호, **판매자 귀책(불량·오배송)이면 배송비 0원**, 교환이면 환불예정금액 `null`, 회수 주소가 주문 배송지와 다른 경우, 회수희망일 상대 날짜 계산, enum 값(요청유형·사유유형·상품분류) 정확히 쓰기.""")
    local = [md('> **[정답안 전용 셀]** 예제 + 숨은 테스트 1건을 손 정답(`_grading/f3_truth.json`)과 항목별 비교합니다. 실제 키가 있어야 의미가 있습니다.'),
             code("""# [정답안 전용] 손 정답과 비교 (예제 2 + 숨은 테스트 1)
truth = json.load(open('_grading/f3_truth.json', encoding='utf-8'))
hidden = load_questions('_grading/f3_hidden.csv')
total = correct = 0
for index, problem in enumerate(examples + hidden):
    try:
        answer = parse_response(call_llm(problem, system_prompts()))
    except Exception as error:
        print(f'문항 {index + 1}: 호출/형식 실패 - {error}')
        continue
    expected = truth[index]
    wrong = {k: (answer[k], v) for k, v in expected.items() if answer[k] != v}
    total += len(expected); correct += len(expected) - len(wrong)
    print(f'문항 {index + 1}: {len(expected) - len(wrong)}/{len(expected)} 일치')
    for key, (got, want) in wrong.items():
        print(f'   {key}: 응답 {got!r} / 정답 {want!r}')
if total:
    print(f'필드 정확도: {correct / total:.3f}')""")]
    answer_map = {1: [code(STUDENT_ANS)], 4: [md(F3_STORY), banner('반품·교환 접수서 22개 항목을 뽑는 단일 시스템 프롬프트')],
                  15: [code(F3_PROMPT_ANSWER), explain], 21: [run] + local}
    pair('F3', cells, answer_map, nb)


if __name__ == '__main__':
    build_f1_1(); build_f1_2(); build_f2_1(); build_f2_2(); build_f3()
