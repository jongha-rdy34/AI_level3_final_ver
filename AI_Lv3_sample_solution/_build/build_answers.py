"""problems_original/*.ipynb -> answers/*_answer.ipynb (예시 정답안)

- '# 수정 가능' 답안 셀만 바꾸고, 바로 아래에 [정답안 해설] 마크다운을 넣습니다.
- 환경 확인 셀은 CPU/GPU 겸용으로 바꿉니다. GPU가 없으면 QUICK(연습) 모드:
  학습 데이터·epoch를 줄여서 끝까지 돌아가게만 합니다. (시험장은 GPU, 원본 설정 그대로)
- 정답안은 노트북 폴더 루트(.venv, data, 가중치가 있는 곳)에서 실행합니다.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'problems_original'
OUT = ROOT  # 정답안도 루트에 둡니다 (작업 폴더 = .venv 위치 조건 때문)


def lines(text):
    text = text.strip('\n')
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': lines(text)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': lines(text)}


STUDENT = """# 본인의 수험번호를 따옴표 안에 입력하세요. 예: '001234'
student_id = '000000'   # [정답안] 예시 번호
if not student_id or not student_id.isascii() or not student_id.isdigit():
    raise ValueError('수험번호를 숫자로만 입력하세요. 앞자리 0도 그대로 적습니다.')
print('수험번호:', student_id)"""

ENV = """# [정답안] 환경 확인 - CPU/GPU 겸용 (원본은 GPU가 없으면 여기서 멈춥니다)
import sys
from pathlib import Path

print('Python:', sys.version.split()[0])
print('현재 커널:', sys.executable)
print('작업 폴더:', Path.cwd())

expected_env = Path.cwd() / '.venv'
if Path(sys.prefix).resolve() != expected_env.resolve():
    raise RuntimeError(
        f"{expected_env / 'Scripts' / 'python.exe'}를 노트북 커널로 선택하세요. "
        '환경이 없다면 setup.cmd를 실행하고, 작업 폴더도 노트북이 있는 폴더로 맞추세요.'
    )

import torch
import torchvision
import timm
print('PyTorch:', torch.__version__, '/ torchvision:', torchvision.__version__, '/ timm:', timm.__version__)

if torch.cuda.is_available():
    DEVICE, QUICK = 'cuda', False
    print('GPU:', torch.cuda.get_device_name(0), '-> 원본 설정 그대로 실행합니다.')
else:
    DEVICE, QUICK = 'cpu', True
    print('GPU 없음 -> QUICK 모드: 학습 데이터와 epoch를 줄여 CPU에서 끝까지 실행합니다.')
    print('           점수는 시험 기준보다 낮게 나옵니다. 코드 흐름 확인용입니다.')"""

QUICK_NOTE = '> **[정답안 전용 셀]** 시험 노트북에는 없는 셀입니다. GPU가 있으면 아무 일도 하지 않습니다.'


def subsample_code(img_var, lab_var, n, extra=''):
    return f"""# [정답안 전용] CPU QUICK 모드에서만 학습 데이터를 클래스 비율대로 줄입니다.
if QUICK:
    _rng = np.random.default_rng(0)
    _keep = np.sort(_rng.choice(len({lab_var}), size=min({n}, len({lab_var})), replace=False))
    {img_var}, {lab_var} = {img_var}[_keep], {lab_var}[_keep]
{extra}    print('QUICK 모드 학습 데이터:', len({lab_var}))"""


def build(name, edits, inserts=None, out_name=None):
    """edits: {cell_index: new_source}; inserts: {cell_index: [cells to insert AFTER that index]}"""
    nb = json.load(open(SRC / f'{name}_problem.ipynb', encoding='utf-8'))
    inserts = inserts or {}
    cells = []
    for i, c in enumerate(nb['cells']):
        c = copy.deepcopy(c)
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
        if i in edits:
            c['source'] = lines(edits[i])
        cells.append(c)
        cells.extend(inserts.get(i, []))
    nb['cells'] = cells
    nb.setdefault('metadata', {})['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (.venv)', 'language': 'python'}
    path = OUT / (out_name or f'{name}_answer.ipynb')
    json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', path.name, len(cells), 'cells')


def title_note(text):
    return md('> ## [예시 정답안]\n> ' + text.replace('\n', '\n> '))


# ============================================================ P1_1
P1_1_TRANSFORM = """# 수정 가능
# train_transform과 processing_training_data는 자유롭게 수정해도 됩니다.
# 다만, train_transform과 processing_training_data의 이름과 매개변수는 변경하지 마세요.

# [정답안] 상품 사진(흰 배경, 정면)에 맞는 '약한' 증강만 씁니다.
#  - 좌우 반전: 상품 종류가 바뀌지 않음
#  - 작은 이동/회전/확대: 촬영 위치 차이. 빈 곳은 흰색(255)으로 채워 배경과 같게
#  - 밝기/대비 약간: 조명 차이. 색상(hue)은 건드리지 않음(색이 단서인 상품이 있음)
#  - 크기는 80x60 그대로 (Resize/Crop 없음)
train_transform = T.Compose([
    T.ToPILImage(),                       # 입력이 numpy(H, W, 3)라서 PIL로 바꿔야 Random* 변환이 동작
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=255),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean, std),
])


def processing_training_data(labels):
    \"\"\"학습에 사용할 인덱스와 항목별 transform 적용 여부를 반환합니다.

    [정답안] 진단: 클래스 불균형. Shirts, Sports Shoes가 각 40장뿐이고 나머지는 1,200장.
    해결: 모든 클래스를 가장 많은 클래스 수까지 오버샘플링합니다.
      - 원본 이미지는 모두 한 번씩 넣고 flag=False (원본 그대로 학습)
      - 부족한 만큼은 같은 클래스 이미지를 반복해서 넣되 flag=True -> train_transform으로
        매번 다른 모습이 되므로 '복사본 반복'으로 인한 과적합을 줄입니다.
    언더샘플링(다수 클래스를 40장으로 줄이기)은 정보를 너무 많이 버려서 쓰지 않습니다.
    \"\"\"
    labels = np.asarray(labels)
    rng = np.random.default_rng(42)
    counts = np.bincount(labels)
    target = counts.max()
    indices, augment_flags = [], []
    for label in range(len(counts)):
        members = np.flatnonzero(labels == label)
        if len(members) == 0:
            continue
        indices.append(members)
        augment_flags.append(np.zeros(len(members), dtype=bool))
        extra = target - len(members)
        if extra > 0:
            indices.append(rng.choice(members, size=extra, replace=True))
            augment_flags.append(np.ones(extra, dtype=bool))
    indices = np.concatenate(indices).astype(int)
    augment_flags = np.concatenate(augment_flags).astype(bool)
    return indices, augment_flags
"""

P1_1_EXPLAIN = md("""### [정답안 해설] 왜 '클래스 불균형'인가
- 위 막대그래프에서 **Shirts, Sports Shoes만 40장**이고 나머지 4종은 1,200장입니다. 문제의 *"일부 상품만 잘못 등록된다"*와 맞습니다.
- 소수 클래스가 하필 **Tshirts↔Shirts, Casual Shoes↔Sports Shoes**처럼 헷갈리는 짝이라, Linear Probing의 마지막 층이 다수 클래스 쪽으로 기울어 소수 클래스의 F1이 낮게 나옵니다.
- 모델·Loss·학습률·batch는 고정이므로 **데이터 구성(오버샘플링)과 증강**으로만 해결해야 합니다. (`class weight` 같은 Loss 변경은 금지 조건)

| 방법 | 설명 | 판단 |
|---|---|---|
| 오버샘플링 + 증강 (선택) | 소수 클래스를 1,200장까지 반복, 반복분만 증강 | 정보 손실 없음, 반복 과적합은 증강이 완화 |
| 언더샘플링 | 다수 클래스를 40장으로 축소 | 데이터 4,640장을 버림 → 전체 성능 하락 |
| 단순 복제(증강 없음) | 같은 40장을 30번씩 반복 | 같은 이미지 암기 → 효과 작음 |

- 증강은 **상품 정체성을 바꾸지 않는 것만**: 좌우 반전, 작은 이동/회전/확대(빈 곳은 흰 배경), 약한 밝기·대비. 크기 80×60, 정규화는 그대로입니다.""")

P1_1_QUICK = [md(QUICK_NOTE), code("""# [정답안 전용] CPU QUICK 모드에서는 epoch만 줄입니다 (데이터는 그대로).
if QUICK:
    epochs = 3
    print('QUICK 모드 epoch:', epochs)""")]

P1_1_FINAL = md("""### [정답안 해설] 결과 읽는 법
- 제공 코드만 실행(전처리 없음)했을 때와 비교해 **Shirts, Sports Shoes의 클래스별 F1**이 크게 오르는 것이 핵심입니다.
- 검증 데이터는 클래스마다 200장으로 균형이라, 소수 클래스를 못 맞추면 macro-F1이 바로 떨어집니다.
- 기준: High > 0.75. 실행 결과는 `README.md`의 '정답안 실행 결과' 표에 정리했습니다.""")

build('P1_1',
      edits={1: STUDENT, 6: ENV, 15: P1_1_TRANSFORM},
      inserts={3: [title_note('진단 = **클래스 불균형** → 소수 클래스 오버샘플링 + 반복분만 증강. 바뀐 셀: `train_transform`, `processing_training_data` 한 셀뿐입니다.')],
               15: [P1_1_EXPLAIN], 20: P1_1_QUICK, 26: [P1_1_FINAL]})

# ============================================================ P1_2
P1_2_TRANSFORM = """# 수정 가능
# 변환을 추가, 삭제, 조절할 수 있습니다.

# [정답안] train은 '곧바른 정면', val/test는 설치 카메라 사진(기울어짐·위치 이동·확대·밝기 차이).
# -> train을 val/test처럼 보이게 만드는 증강을 적당한 세기로 넣습니다.
#    제공 값(shift 0.80, shear 50도, 지우기 95%)은 얼굴이 화면 밖으로 나가거나 심하게 찌그러져
#    나이 단서(주름, 피부, 얼굴형)가 사라지므로 크게 줄입니다.
shift = 0.12            # 위치 이동: 이미지 크기의 ±12%
shear = (-8.0, 8.0)     # 비스듬한 각도: 약하게
tilt = 30.0             # 카메라 기울기: ±30도 (val에서 확인한 범위)
mask_probability = 0.25 # 일부 가림(안경, 손 등)은 가끔만
mask_area = (0.02, 0.08)

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(shift, shift), scale=(1.0, 1.25),
                   interpolation=InterpolationMode.BILINEAR, fill=fill),
    T.RandomAffine(degrees=0, shear=shear,
                   interpolation=InterpolationMode.BILINEAR, fill=fill),
    T.RandomRotation(degrees=tilt, interpolation=InterpolationMode.BILINEAR,
                     expand=False, fill=fill),
    T.ColorJitter(brightness=0.3, contrast=0.2),
    T.ToTensor(),
    T.RandomErasing(p=mask_probability, scale=mask_area,
                    ratio=(0.5, 2.0), value=mean),
    T.Normalize(mean, std),
])"""

P1_2_VIEW = """# 수정 가능
# 이미지 확인용 코드입니다. 일부를 수정하여 evaluation 데이터를 확인할 수도 있습니다.

# [정답안] train(위)과 val(아래)을 나란히 보고 '분포 차이'를 직접 확인합니다.
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for axis, row in zip(axes[0], train_rows[:5]):
    with Image.open(data_dir / row['filename']) as image:
        axis.imshow(image.convert('RGB'))
    axis.set_title(f"train Age {row['label']:g}")
for axis, row in zip(axes[1], val_rows[:5]):
    with Image.open(data_dir / row['filename']) as image:
        axis.imshow(image.convert('RGB'))
    axis.set_title(f"val Age {row['label']:g}")
with Image.open(data_dir / train_rows[0]['filename']) as image:
    sample_image = image.convert('RGB')
for axis in axes[2]:
    tensor = train_transform(sample_image)
    restored = tensor * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    axis.imshow(restored.permute(1, 2, 0).clamp(0, 1))
    axis.set_title('Train transform')
for axis in axes.flat:
    axis.axis('off')
plt.tight_layout()
plt.show()"""

P1_2_EXPLAIN = md("""### [정답안 해설]
- **문제의 핵심은 '데이터 분포 차이'** 입니다. 힌트에 train은 *정돈된 정면 사진*, val/test는 *현실 카메라 위치*라고 했습니다. 위 그림의 val 줄을 보면 기울어지고, 옮겨지고, 확대되고, 밝기가 다릅니다.
- 그래서 증강은 **val에서 본 변화만, 비슷한 크기로** 넣습니다: 회전 ±30°, 이동 ±12%, 확대 1.0~1.25배, 좌우 반전, 밝기·대비.
- 제공된 기본값은 **과한 증강**입니다(이동 80%면 얼굴이 화면 밖으로 나가고, shear 50°는 얼굴형을 망가뜨리고, 95% 확률로 가림). 나이 단서가 사라져 성능이 떨어집니다. → *부적절한 데이터 증강* 항목에 해당합니다.
- 빈 곳을 채우는 색 `fill=(124,116,104)`는 제공 값 그대로 씁니다(평균 색이라 정규화 후 0에 가까움).
- 고정: 입력 크기 112, ToTensor/Normalize, 모델, 25 epoch, `SmoothL1Loss(beta=5.0)`.""")

P1_2_QUICK_DATA = [md(QUICK_NOTE), code("""# [정답안 전용] CPU QUICK 모드에서만 학습/검증 데이터를 줄입니다.
if QUICK:
    train_rows = train_rows[:1500]
    val_rows = val_rows[:500]
    print('QUICK 모드 데이터:', len(train_rows), len(val_rows))""")]
P1_2_QUICK_EP = [md(QUICK_NOTE), code("""if QUICK:
    EPOCHS = 4
    print('QUICK 모드 epoch:', EPOCHS)""")]

build('P1_2',
      edits={1: STUDENT, 6: ENV, 16: P1_2_TRANSFORM, 17: P1_2_VIEW},
      inserts={3: [title_note('진단 = **train(정면) ↔ val/test(카메라 사진) 분포 차이** + 제공 증강이 너무 셈 → val에서 본 범위만큼만 증강. 바뀐 셀: 증강 셀, 확인용 그림 셀.')],
               10: P1_2_QUICK_DATA, 17: [P1_2_EXPLAIN], 33: P1_2_QUICK_EP})

# ============================================================ P2_1
P2_1_MODEL = """# 수정 가능
class FashionModel(nn.Module):
    \"\"\"사전학습 체크포인트와 호환되는 TinyViT 5M입니다.

    [정답안] 고정: patch_embed, stages[0], stages[1]
            학습: stages[2], stages[3], head(norm, fc)
    \"\"\"
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

        # (1) gradient: 전체를 끈 뒤, 학습할 구간만 다시 켭니다.
        self.requires_grad_(False)
        self.stages[2].requires_grad_(True)
        self.stages[3].requires_grad_(True)
        self.head.requires_grad_(True)          # head.norm + head.fc

    def frozen_modules(self):
        return [self.patch_embed, self.stages[0], self.stages[1]]

    def train(self, mode=True):
        # (2) 모드: model.train()이 불려도 고정 구간은 항상 eval.
        #     -> 고정 구간 BatchNorm의 running_mean/var가 바뀌지 않습니다.
        #     model.eval()은 mode=False로 이 함수를 부르므로 검증 때는 전체가 eval.
        super().train(mode)
        for module in self.frozen_modules():
            module.train(False)
        return self

    def forward(self, x):
        # (3) 고정 구간은 no_grad: 계산 그래프를 만들지 않아 메모리·시간 절약
        with torch.no_grad():
            x = self.patch_embed(x)
            x = self.stages[0](x)
            x = self.stages[1](x)
        # 학습 구간은 no_grad 밖: 역전파로 가중치가 갱신됩니다.
        x = self.stages[2](x)
        x = self.stages[3](x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(x)"""

P2_1_CHECK = [md('> **[정답안 전용 셀]** Task A 설정이 맞는지 스스로 확인하는 검사입니다. 제출 노트북에 남겨도 됩니다(제공 셀은 건드리지 않음).'),
              code("""# [정답안 전용] Task A 자가 점검
_m = FashionModel()
_trainable = {name.split('.')[0] + ('.' + name.split('.')[1] if name.startswith('stages') else '')
              for name, p in _m.named_parameters() if p.requires_grad}
print('학습하는 구간:', sorted(_trainable))
assert _trainable == {'stages.2', 'stages.3', 'head'}

_m.train()
print('train() 후 모드 - patch_embed:', _m.patch_embed.training, '/ stages[1]:', _m.stages[1].training,
      '/ stages[2]:', _m.stages[2].training)
assert not _m.patch_embed.training and not _m.stages[1].training and _m.stages[2].training

# 고정 구간 BatchNorm 통계가 학습 forward 후에도 그대로인지 확인
_bn = _m.patch_embed.conv1.bn
_before = _bn.running_mean.clone()
_m(torch.randn(4, 3, 80, 60))
assert torch.equal(_before, _bn.running_mean), '고정 구간 BN 통계가 바뀌었습니다.'
_m.eval()
assert not any(module.training for module in _m.modules())
print('Task A 점검 통과')
del _m""")]

P2_1_LOSS = """# 수정 가능

def custom_loss(logits, targets):
    \"\"\"배치 평균 분류 Loss를 반환합니다.

    [정답안] 선택: Label Smoothing Cross Entropy (epsilon = 0.1)
    선택 이유 (주석):
      - 20종 중 Shirts/Tshirts/Tops/Kurtas, Casual/Sports/Formal Shoes처럼 비슷한 클래스가 많아
        정답 하나에 확률 1을 몰아주는 보통 CE는 과신(overconfidence)과 과적합이 생기기 쉽습니다.
      - 정답에 1-eps, 나머지 클래스에 eps를 고르게 나눠 주면 결정 경계가 부드러워지고
        validation 성능이 보통 소폭 오릅니다. (eps=0이면 보통 CE와 같음)
      - 학습 데이터는 클래스별 수가 달라도(최대 1,000장) 극단적 불균형은 아니어서 Focal Loss보다
        Label Smoothing이 안정적이었습니다.
    수식: loss_i = -(1-eps) * log p_i[y_i] - eps * mean_k log p_i[k]
    사용 연산: F.log_softmax, gather, mean (완성형 Loss API 사용 안 함)
    \"\"\"
    epsilon = 0.1
    log_probs = F.log_softmax(logits.float(), dim=1)                    # (B, C) 수치적으로 안정
    nll = -log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)  # 정답 클래스의 -log p
    smooth = -log_probs.mean(dim=1)                                     # 모든 클래스 평균 -log p
    loss = (1.0 - epsilon) * nll + epsilon * smooth
    return loss.mean()                                                  # () 스칼라, logits와 같은 장치"""

P2_1_LOSS_CHECK = [md('> **[정답안 전용 셀]** 직접 구현한 Loss 검증: epsilon=0으로 바꾸면 `F.cross_entropy`와 같아야 합니다(검증용으로만 비교, 학습에는 쓰지 않음).'),
                   code("""# [정답안 전용] Loss 자가 점검
_logits = torch.randn(8, num_classes, requires_grad=True)
_targets = torch.randint(0, num_classes, (8,))
_loss = custom_loss(_logits, _targets)
assert _loss.shape == () and torch.isfinite(_loss)
_loss.backward()
assert _logits.grad is not None
_lp = F.log_softmax(_logits.detach(), 1)
_ce = -_lp.gather(1, _targets[:, None]).mean()
print('custom_loss:', float(_loss), '| 같은 입력의 보통 CE:', float(_ce), '(smoothing 때문에 약간 다름)')
print('Loss 점검 통과')""")]


def p2_1_hparams():
    sweep = ROOT / '_build' / 'p2_1_sweep.json'
    table = ''
    best = {'batch_size': 64, 'learning_rate': 0.0003, 'epochs': 20}
    if sweep.exists():
        rows = json.load(open(sweep, encoding='utf-8'))
        table = '\n'.join(f"#   batch {r['batch_size']:>3} | lr {r['learning_rate']:<7g} | epochs {r['epochs']:>2} "
                          f"| best val macro-F1 {r['best_f1']:.4f} (epoch {r['best_epoch']})" for r in rows)
        best = max(rows, key=lambda r: r['best_f1'])
    return f"""# 수정 가능

# [정답안] 비교한 설정과 validation 결과 (Task A 구간 + Task B Loss, 같은 시드, GPU)
{table or '#   (sweep 결과 없음)'}
# 최종 선택: 가장 높은 best validation macro-F1
batch_size = {best['batch_size']}
learning_rate = {best['learning_rate']}
epochs = {best['epochs']}""", best


P2_1_HP_SRC, P2_1_BEST = p2_1_hparams()

P2_1_HP_EXPLAIN = md("""### [정답안 해설] Task C 하이퍼파라미터
- 바꿀 수 있는 것은 `batch_size`, `learning_rate`, `epochs` 세 개뿐입니다. 스케줄러가 없으므로 **학습률이 가장 중요**합니다.
- 뒤쪽 stage까지 학습하므로 Linear Probing(3e-4)보다 **조금 낮은 학습률**이 안정적이고, batch를 줄이면 업데이트 횟수가 늘어 같은 epoch에서 더 많이 배웁니다.
- 체크포인트는 val macro-F1이 가장 높은 epoch에서 저장되므로 epochs를 넉넉히 줘도 손해가 적습니다(시간만 늘어남).
- 위 표는 `_build/p2_1_sweep.py`로 같은 조건(시드 42)에서 돌린 결과입니다. 시험에서는 시간 안에 2~4개 정도 비교하고 표로 남기면 됩니다.""")

P2_1_QUICK_DATA = [md(QUICK_NOTE), code(subsample_code('train_images', 'train_labels', 3000,
                                                       extra="    val_images, val_labels = val_images[:1000], val_labels[:1000]\n"))]
P2_1_QUICK_EP = [md(QUICK_NOTE), code("""if QUICK:
    epochs = 3
    print('QUICK 모드 epoch:', epochs)""")]

build('P2_1',
      edits={1: STUDENT, 6: ENV, 25: P2_1_MODEL, 27: P2_1_LOSS, 30: P2_1_HP_SRC},
      inserts={3: [title_note('Task A: 앞 구간 고정(`requires_grad` + 항상 eval + `no_grad`) / Task B: Label Smoothing CE 직접 구현 / Task C: 학습률·batch 비교 후 선택.')],
               10: P2_1_QUICK_DATA, 25: P2_1_CHECK, 27: P2_1_LOSS_CHECK, 30: [P2_1_HP_EXPLAIN] + P2_1_QUICK_EP})

# ============================================================ P2_2
P2_2_IMPORT = """# 수정 금지

import csv
import json
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
device = DEVICE   # [정답안] 원본은 'cuda' 고정. CPU에서도 돌도록 환경 셀의 값을 씁니다."""

P2_2_TRANSFORM = """# 수정 가능

# [정답안] 상품 사진용 증강 (흰 배경, 가운데 정렬, 80x60 유지)
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=255),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
])"""

P2_2_MODEL = """# 수정 가능

class FashionModel(nn.Module):
    \"\"\"원본 크기의 이미지에서 전체 파라미터를 미세조정하는 TinyViT 5M입니다.

    [정답안] 구조 변경
      1) head 앞 Dropout(0.2): 12,000장으로 5M 모델 전체를 학습하므로 과적합 완화
      2) 평가 모드에서 좌우 반전 TTA: 추론 셀은 수정 금지지만 model(images)를 부르므로,
         eval일 때 원본과 좌우 반전 logits를 평균해 돌려줍니다. 가중치 수는 그대로.
    \"\"\"
    def __init__(self, drop=0.2, tta=True):
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
        self.dropout = nn.Dropout(drop)
        self.tta = tta

    def forward_once(self, x):
        x = self.patch_embed(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        x = self.stages[3](x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(self.dropout(x))

    def forward(self, x):
        logits = self.forward_once(x)
        if self.tta and not self.training:
            logits = (logits + self.forward_once(torch.flip(x, dims=[3]))) / 2
        return logits"""

P2_2_HP = """# 수정 가능

# [정답안] 전체 미세조정 설정
batch_size = 64
learning_rate = 0.0005       # AdamW 최대 학습률 (warmup 후 cosine으로 감소)
epochs = 30
weight_decay = 0.05
warmup_epochs = 2
label_smoothing = 0.1"""

P2_2_TRAIN_FN = """# 수정 가능

def train_one_epoch(model, loader, optimizer, loss_fn, scheduler=None):
    model.train()
    total_loss = 0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # [정답안] 큰 gradient 방지
        optimizer.step()
        if scheduler is not None:
            scheduler.step()                                    # [정답안] 배치마다 스케줄 갱신
        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
    return total_loss / sample_count


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    accuracy = accuracy_score(targets, predictions)
    return total_loss / len(targets), accuracy"""

P2_2_TRAIN = """# 수정 가능
import math

torch.manual_seed(42)
model = FashionModel().to(device)

# [정답안] AdamW + weight decay는 가중치 행렬에만 (bias, norm, attention bias 제외)
decay, no_decay = [], []
for name, param in model.named_parameters():
    if param.ndim <= 1 or name.endswith('attention_biases'):
        no_decay.append(param)
    else:
        decay.append(param)
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': weight_decay},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=learning_rate)

# [정답안] warmup -> cosine 감소 (배치 단위)
total_steps = epochs * len(train_loader)
warmup_steps = warmup_epochs * len(train_loader)
def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)   # Cross Entropy (+ smoothing)
eval_loss_fn = nn.CrossEntropyLoss()
best_accuracy = -1
history = []
started = time.perf_counter()

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scheduler)
    val_loss, val_accuracy = evaluate(model, val_loader, eval_loss_fn)
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    })
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), checkpoint)
    print(f'{epoch + 1:2d}/{epochs}  학습 Loss: {train_loss:.4f}  '
          f'검증 Loss: {val_loss:.4f}  검증 Accuracy: {val_accuracy:.4f}')

with history_path.open('w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
    writer.writeheader()
    writer.writerows(history)
print(f'최고 검증 Accuracy: {best_accuracy:.4f} / 학습 시간: {time.perf_counter() - started:.1f}초')
print('체크포인트:', checkpoint)"""

P2_2_EXPLAIN = md("""### [정답안 해설] 성능을 올린 요소
| 요소 | 내용 | 이유 |
|---|---|---|
| 증강 | 좌우 반전, 작은 이동·회전·확대(흰 배경 채움), 약한 색 변화, 작은 지우기 | 클래스당 600장(일부 반복) → 반복 이미지 암기 방지 |
| 옵티마이저 | AdamW, weight decay 0.05 (bias·norm 제외) | ViT 계열 미세조정의 기본값 |
| 스케줄 | warmup 2 epoch → cosine | 사전학습 가중치가 초반 큰 gradient에 망가지는 것 방지, 후반 미세 수렴 |
| Loss | CE + label smoothing 0.1 | Shirts/Tshirts/Tops, 신발 3종처럼 비슷한 클래스 과신 방지 |
| 구조 | head 앞 Dropout 0.2, eval 때 좌우 반전 TTA | 과적합 완화 / 추론 셀 수정 없이 예측 안정화 |
| gradient clipping | norm 5.0 | 학습 안정 |

- 추론 셀(수정 금지)은 `model(images)`만 부르므로, TTA를 **모델의 eval forward 안에** 넣은 것이 요령입니다.
- 제출: 추론 셀이 채운 `test.csv`를 이름 그대로 Kaggle에 냅니다. 이 폴더에서는 `python _grading/score_p2_2.py`로 로컬 채점합니다.""")

P2_2_QUICK_DATA = [md(QUICK_NOTE), code(subsample_code('train_images', 'train_labels', 3000,
                                                       extra="    val_images, val_labels = val_images[:1000], val_labels[:1000]\n"))]
P2_2_QUICK_EP = [md(QUICK_NOTE), code("""if QUICK:
    epochs, warmup_epochs = 3, 1
    print('QUICK 모드 epoch:', epochs)""")]
P2_2_SCORE = [md('> **[정답안 전용 셀]** 로컬 채점 (Kaggle 대신). 정답 파일은 `_grading/`에 있습니다.'),
              code("""# [정답안 전용] 로컬 채점
import subprocess, sys
print(subprocess.run([sys.executable, '_grading/score_p2_2.py'], capture_output=True, text=True).stdout)""")]

build('P2_2',
      edits={1: STUDENT, 6: ENV, 7: P2_2_IMPORT, 13: P2_2_TRANSFORM, 23: P2_2_MODEL, 25: P2_2_HP,
             29: P2_2_TRAIN_FN, 30: P2_2_TRAIN},
      inserts={3: [title_note('증강 + AdamW/warmup-cosine + label smoothing + Dropout + eval 좌우반전 TTA. 수정 가능 셀만 바꿨습니다.')],
               10: P2_2_QUICK_DATA, 25: P2_2_QUICK_EP, 30: [P2_2_EXPLAIN], 33: P2_2_SCORE})

# ============================================================ P3
P3_PROMPT = r'''# 수정 가능
# 함수명은 변경하지 마세요.

def system_prompts():
    """모든 문항에 공통으로 적용할 시스템 프롬프트를 반환합니다.

    입력 인자는 없습니다. 녹취 원문은 호출 코드가 question에서 가져와
    별도의 user 메시지로 전달합니다.

    Returns:
        str: 역할, 추출 규칙, 출력 형식을 담은 시스템 지시문입니다.
    """
    schema_text = json.dumps(response_schema, ensure_ascii=False)
    keys_text = ', '.join(response_schema['properties'])
    system_prompt = f"""
당신은 주택 전세 계약서 작성을 돕는 정보 추출기입니다.
사용자 메시지는 전세 계약 협의 녹취 전문 하나입니다. 녹취를 끝까지 읽고, 계약서 폼에 들어갈 '최종 합의 내용'만 JSON 객체 하나로 출력합니다.

# 출력 형식 (가장 중요)
- 아래 JSON Schema를 만족하는 JSON 객체 하나만 출력합니다. Schema 자체가 아니라 값을 채운 객체입니다.
- 키는 정확히 다음 26개이며 모두 포함하고, 다른 키는 넣지 않습니다: {keys_text}
- 첫 글자는 {{ , 마지막 글자는 }} 입니다. 설명, 머리말, 코드 블록(```), 태그, 주석을 붙이지 않습니다.
- 키와 문자열은 큰따옴표를 씁니다. 값이 없으면 null (따옴표 없이). "None", "null", "" 문자열은 금지입니다.
- 금액은 원 단위 정수(쉼표·단위 없이), 전용면적은 숫자(소수 가능), 날짜는 "YYYY-MM-DD"(월·일 두 자리, 실제 존재하는 날짜)입니다.
- 마지막 항목 뒤에 쉼표를 붙이지 않습니다.
JSON Schema: {schema_text}

# 판단 원칙
1. 최종 합의만: 녹취 중 제안 → 반대 → 수정 → 정정이 이어집니다. 마지막으로 양측이 받아들인 값을 씁니다. 한쪽 제안이 거절되었거나 대답 없이 끝났으면 채택하지 않습니다.
2. 정정 우선: "잘못 말했다", "아니고", "바꾸죠"처럼 정정된 값이 이전 값을 대체합니다. 잘못 출력된 서류·예약표·앱 저장값·광고 숫자는 무시합니다.
3. 확정되지 않은 값(검토 중, 업체에 물어본 날짜, 아직 정해지지 않은 돈 등)과 녹취에 없는 값은 null입니다. 추측하지 않습니다. 단, 아래 계산 규칙으로 정해지는 값은 계산합니다.
4. 계약과 무관한 잡담(수리, 이사업체, 주차, 가구 크기, 다른 집, 연락처 주소)은 무시합니다.

# 항목별 규칙
- 임대인: 집 소유자(보증금을 받는 쪽)의 이름. 임차인: 계약서에 당사자로 서명하는 사람의 정정된 이름(배우자·가족·동생이 아님). 이름만 씁니다(호칭 '씨', '소장님' 제외).
- 중개인: 이번 계약서를 작성·서명하는 공인중개사 이름(문을 열어준 직원 등 다른 사람 아님). 중개사무소: 사무소의 전체 이름(예: '○○공인중개사사무소').
- 주소는 임대차 목적물(계약할 집)의 주소만 씁니다.
  - 시도: '서울특별시', '경기도'처럼 정식 명칭. 시군구: '송파구', 시와 구가 함께 있으면 '고양시 덕양구'처럼 공백으로 이어서.
  - 읍면: 주소가 읍·면 지역일 때만, 동 지역이면 null. 도로명: '올림픽로'처럼 도로명만.
  - 건물번호: 아라비아 숫자 문자열. '사백삼십오' → "435", '백십팔 다시 삼십' → "118-30".
  - 상세주소: "111동 902호" 형식(숫자, '동'과 '호' 사이 공백 하나). 법정동: '신천동'처럼 법정동 이름.
  - 건물명: 단지/건물 이름 그대로. 건물구조: '철근콘크리트조'처럼 말한 그대로.
- 전용면적_m2: 해당 호의 '전용' 면적 숫자(공급면적, 다른 평면·광고 숫자 아님). '팔십사 점 칠육' → 84.76.
- 금액: 한국어 수를 원 단위 정수로 바꿉니다. 대화의 금액은 보통 만 원 단위를 생략합니다:
  '사억 삼천' = 430000000, '오백' = 5000000, '삼천팔백' = 38000000, '백오십' = 1500000, '오백만원' = 5000000.
  - 전세보증금_원: 할인·조정이 반영된 최종 보증금. 깎아 달라는 요청이 거절되면 반영하지 않습니다.
  - 계약금_원: 최종 합의한 계약금 총액(먼저 보낸 가계약금 포함).
  - 중도금_원: 최종 중도금 총액(나중에 늘리거나 줄인 것 반영). 중도금이 없다고 명시되면 0.
  - 잔금_원: 말한 금액이 있으면 그 값, 없으면 전세보증금 - 계약금 - 중도금 으로 계산합니다.
- 날짜: 녹취 머리의 기록 날짜를 기준으로 연·월을 정합니다('이번 달 13일' → 기록 날짜의 달, 월 없이 말한 날짜는 앞뒤 문맥의 달). 실제 달력에 있는 날짜인지 확인합니다.
  - 계약체결일: 계약서에 서명하는 날.
  - 계약금지급일: 계약금 전액이 다 지급되는(마지막으로 채우는) 날. 이미 보낸 일부 금액의 날짜가 아닙니다.
  - 중도금지급일: 중도금을 주는 날. 중도금이 0이면 null.
  - 잔금지급일: 잔금을 주는 날.
  - 명도일: 임대인이 집과 열쇠를 넘겨주는 날. ('잔금 다음 날' 같은 표현은 계산합니다.)
  - 입주일: 임차인이 실제로 이사해 짐을 넣는 날. ('열쇠 받은 사흘 뒤' 등은 계산합니다.)
  - 임대차시작일, 임대차종료일: 합의한 계약 기간의 시작일과 끝일.

출력 전에 속으로 각 항목의 최종값과 날짜 계산을 다시 확인하고, JSON 객체만 출력하세요.
""".strip()
    return system_prompt'''

P3_KEY = """# 본인의 API 키를 따옴표 안에 입력하고 실행하세요. 이후 모델 호출에 사용됩니다.
# [정답안] 'dummy'면 gms_client의 모의 모드(모든 값 null 응답)로 형식 흐름만 확인합니다.
#          실제 프롬프트 품질은 SSAFY GMS 키를 넣어 확인하세요.
USER_API_KEY = 'dummy'
if not isinstance(USER_API_KEY, str) or not USER_API_KEY.strip():
    raise ValueError('USER_API_KEY에 본인의 API 키를 입력하세요.')
USER_API_KEY = USER_API_KEY.strip()"""

P3_ENV = """# [정답안] 환경 확인 (원본과 같음)
import sys
from pathlib import Path

print('Python:', sys.version.split()[0])
print('현재 커널:', sys.executable)
print('작업 폴더:', Path.cwd())
expected_env = Path.cwd() / '.venv'
if Path(sys.prefix).resolve() != expected_env.resolve():
    raise RuntimeError(f"{expected_env / 'Scripts' / 'python.exe'}를 노트북 커널로 선택하세요.")
import openai
print('OpenAI SDK:', openai.__version__)
print('커널 확인 완료. 다음 셀로 진행하세요.')"""

P3_EXPLAIN = md("""### [정답안 해설] 프롬프트 설계 포인트
1. **출력 형식을 가장 먼저, 가장 구체적으로**: 26개 키 목록, `{`로 시작 `}`로 끝, 코드 블록 금지, `null` 표기, 정수/날짜 형식. 제공 `parse_response()`는 첫 `{`~마지막 `}`만 자르므로 앞뒤 설명에 `{}`가 있으면 실패합니다.
2. **최종 합의 원칙**: 녹취에는 일부러 *제안 → 거절*, *정정*, *잘못 출력된 서류*, *광고 숫자*, *다른 집 주소*가 섞여 있습니다.
3. **애매한 항목 정의**: 계약금지급일(=완납일), 명도일(=집·열쇠 인도일) vs 입주일(=실제 이사일) vs 임대차시작일(=기간 시작)처럼 비슷한 날짜를 하나씩 정의합니다.
4. **한국어 금액 표기**: 만 원 단위 생략 관행을 예시로 고정합니다(`사억 삼천` → 430000000).
5. **계산 규칙**: 잔금 = 보증금 − 계약금 − 중도금, '다음 날', '사흘 뒤' 날짜 계산.
6. 스키마는 `json.dumps(response_schema, ensure_ascii=False)`로 넣고, 문항별 정답은 넣지 않습니다(단일 프롬프트 조건).""")

P3_LOCAL = [md("""> **[정답안 전용 셀]** 예제 2개의 **손으로 만든 정답**(`_grading/p3_example_truth.json`)과 비교합니다. 공식 정답은 공개되지 않았으므로 참고용입니다.
> 실제 키로 실행하면 응답을 정답과 항목별로 비교해 맞힌 수를 보여줍니다."""),
            code("""# [정답안 전용] 예제 응답을 손 정답과 비교
truth = json.load(open('_grading/p3_example_truth.json', encoding='utf-8'))
for index, problem in enumerate(examples):
    try:
        answer = parse_response(call_llm(problem, system_prompts()))
    except Exception as error:
        print(f'예제 {index + 1}: 호출/형식 실패 - {error}')
        continue
    expected = truth[index]
    wrong = {key: (answer[key], expected[key]) for key in expected if answer[key] != expected[key]}
    print(f'예제 {index + 1}: {len(expected) - len(wrong)}/{len(expected)} 일치')
    for key, (got, want) in wrong.items():
        print(f'   {key}: 응답 {got!r} / 정답 {want!r}')""")]

build('P3',
      edits={1: STUDENT, 2: P3_KEY, 6: P3_ENV, 16: P3_PROMPT},
      inserts={4: [title_note('계약서 26개 항목을 뽑는 단일 시스템 프롬프트. 형식 규칙 → 최종 합의 원칙 → 항목별 정의 → 금액·날짜 변환 순서로 썼습니다.')],
               16: [P3_EXPLAIN], 22: P3_LOCAL})
