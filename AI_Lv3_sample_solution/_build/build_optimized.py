"""PDF(AI_level3_대비.pdf)의 '외울 코드 최소화' 변경만 반영한 정답지 생성기.

실행 (AI_Lv3_sample_solution 폴더에서):  .venv\\Scripts\\python.exe _build\\build_optimized.py
기준: *_answer.ipynb(CPU QUICK 모드 지원)를 복사하고,
  - PDF가 바꾸는 '수정 가능' 셀 → PDF 코드(오타만 고침)
  - PDF가 다루지 않는 '수정 가능' 셀 → 문제 원본(*_problem.ipynb)의 기본 코드로 되돌림 (외울 것 최소화)
  - 기존 [정답안 해설]/[예시 정답안] 마크다운은 내용이 달라지므로 빼고, 맨 위에 변경 요약을 넣음
  - 실행 출력은 모두 지움 (예전 GPU 출력과 섞이지 않게)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------
P1_1 = {
"def processing_training_data": '''# 수정 가능
# train_transform과 processing_training_data는 자유롭게 수정해도 됩니다.
# 다만, train_transform과 processing_training_data의 이름과 매개변수는 변경하지 마세요.

train_transform = T.Compose([   # flag가 True인 반복분에만 적용되는 '약한' 증강
    T.ToPILImage(),                                                                  # 입력이 numpy라 Random* 앞에 먼저
    T.RandomHorizontalFlip(p=0.5),                                                   # 좌우 반전
    T.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=255),  # 작은 회전/이동/확대, 빈 곳 흰색
    T.ColorJitter(brightness=0.2, contrast=0.2),                                     # 밝기/대비 (색상 hue는 건드리지 않음)
    T.ToTensor(),
    T.Normalize(mean, std),
])


def processing_training_data(labels):
    labels = np.asarray(labels)                     # labels를 array로
    rng = np.random.default_rng(seed=42)            # 난수 생성기(재현 가능)
    counts = np.bincount(labels)                    # 클래스별 개수
    target = counts.max()                           # 가장 많은 클래스 개수에 맞춤
    indices, augment_flags = [], []                 # 결과를 모을 리스트
    for label in range(len(counts)):                # 클래스 번호마다
        members = np.flatnonzero(labels == label)   # 이 클래스의 원본 이미지 번호
        if len(members) == 0: continue              # 없으면 건너뜀
        indices.append(members)                     # 원본은 모두 한 번씩
        augment_flags.append(np.zeros(len(members), dtype=bool))
        extra = target - len(members)               # 부족한 개수
        if extra > 0:
            indices.append(rng.choice(members, size=extra, replace=True))   # 부족분만큼 중복 허용으로 뽑기
            augment_flags.append(np.ones(extra, dtype=bool))                # 반복분은 True → train_transform으로 증강
    indices = np.concatenate(indices).astype(int)               # 하나로 합치기
    augment_flags = np.concatenate(augment_flags).astype(bool)
    return indices, augment_flags
''',
}
P1_1_NOTE = """> ## [optimized 정답 — PDF 반영]
> 원래 정답안(`P1_1_answer.ipynb`)에서 **외울 코드만 줄인 버전**입니다. 바뀐 셀은 `train_transform` + `processing_training_data` 한 셀입니다.
> - 소수 클래스를 가장 많은 클래스 수까지 **중복 허용 오버샘플링**. 원본은 `augment_flags=False`, 반복분만 `True`
> - `train_transform`: `ToPILImage` → 좌우 반전 → 작은 `RandomAffine`(빈 곳 흰색 255) → `ColorJitter`(밝기·대비) → `ToTensor` → `Normalize`. P2-2 증강에서 `saturation`과 `RandomErasing`을 뺀 형태입니다.
> - flag만 True로 바꾸고 `train_transform`을 그대로 두면 증강이 일어나지 않습니다(문제 기본값이 `eval_transform`과 같음). 두 곳을 함께 바꿔야 합니다.
> - PDF의 오타는 고쳤습니다: `memebers`→`members`, `concantenate`→`concatenate`, 반복문 `for i in range(len(labels))` → `for label in range(len(counts))`
> - 원래 정답안과 같은 방식이며 GPU val macro-F1은 0.783(기준선 0.521)이었습니다."""

# ----------------------------------------------------------------------
P2_1 = {
"class FashionModel": '''# 수정 가능
class FashionModel(nn.Module):
    """사전학습 체크포인트와 호환되는 TinyViT 5M입니다."""
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

        self.requires_grad_(False)                   # (제공) 전체 고정
        self.stages[2:].requires_grad_(True)         # 학습할 구간: stage 2, 3  (슬라이스 한 줄로)
        self.head.requires_grad_(True)               # head(norm + fc)

    def train(self, mode=True):
        not_train = [self.patch_embed, self.stages[0], self.stages[1]]   # 고정 구간
        super().train(mode)                          # 원래 train 동작
        for module in not_train:                     # 고정 구간은 항상 eval (BN 통계 고정)
            module.train(False)
        return self

    def forward(self, x):
        with torch.no_grad():                        # 고정 구간은 기울기 계산 안 함
            x = self.patch_embed(x)
            x = self.stages[0](x)
            x = self.stages[1](x)
        x = self.stages[2](x)                        # 학습 구간
        x = self.stages[3](x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(x)
''',
"def custom_loss": '''# 수정 가능

def custom_loss(logits, targets):
    # Label Smoothing CE:  loss_i = -(1-eps) * log p_i[y_i] - eps * mean_k log p_i[k]
    epsilon = 0.1                                                     # smoothing factor
    log_probs = F.log_softmax(logits.float(), dim=1)                  # logit → log softmax
    nega_log = -log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)   # 정답 클래스의 -log p
    smooth = -log_probs.mean(dim=1)                                   # 모든 class 평균 -log p
    loss = (1.0 - epsilon) * nega_log + epsilon * smooth              # 두 값을 섞음
    return loss.mean()
''',
}
P2_1["batch_size ="] = '''# 수정 가능

# [기존 answer 값] 문제 기본값(batch 64, lr 0.0003, epochs 20)에서 batch·lr을 바꿈
#   기존 answer의 GPU 비교 결과 (Task A 구간 + Task B Loss, 같은 시드)
#   batch  32 | lr 0.0001  | epochs 20 | best val macro-F1 0.9437
#   batch  32 | lr 0.0002  | epochs 20 | best val macro-F1 0.9476  ← 선택
#   batch  64 | lr 0.0001  | epochs 20 | best val macro-F1 0.9410
#   batch  64 | lr 0.0003  | epochs 20 | best val macro-F1 0.9425
batch_size = 32         # 변경: 64 → 32
learning_rate = 0.0002  # 변경: 0.0003 → 0.0002
epochs = 20             # 문제 기본값 그대로
'''

P2_1_NOTE = """> ## [optimized 정답 — PDF 반영]
> 원래 정답안(`P2_1_answer.ipynb`)에서 **외울 코드만 줄인 버전**입니다. 바뀐 셀: `FashionModel`, `custom_loss`. 하이퍼파라미터 셀은 **기존 answer 값**(batch 32, lr 2e-4, epochs 20)이며, 바뀐 줄에 주석을 달았습니다.
> - 학습 범위 3종 세트: `requires_grad_` / `train()`을 덮어써 고정 구간 항상 `eval` / 고정 구간은 `no_grad`
> - PDF에 `stages[2]`만 적혀 있었지만, stage 3은 `no_grad` 밖에서 학습 구간으로 계산되므로 `self.stages[2:].requires_grad_(True)` 한 줄로 stage 2·3을 함께 켰습니다(원래 정답안과 같은 범위).
> - PDF의 `x = self.stages[3]`는 `x = self.stages[3](x)`로, `epsilone`은 `epsilon`으로 고쳤습니다.
> - 원래 정답안은 같은 하이퍼파라미터로 GPU best val macro-F1 0.948이었습니다. 이 버전(모델·Loss 코드만 PDF식으로 줄임)의 GPU 점수는 확인하지 않았습니다."""

# ----------------------------------------------------------------------
P2_2 = {
"train_transform =": '''# 수정 가능

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),                                                   # 좌우 반전
    T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=255),  # 회전/이동/확대, 흰 배경
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),                    # 밝기/대비/채도
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),                               # 일부 영역 지우기 (텐서 뒤)
])
''',
"class FashionModel": '''# 수정 가능

class FashionModel(nn.Module):
    """원본 크기의 이미지에서 전체 파라미터를 미세조정하는 TinyViT 5M입니다."""
    def __init__(self, p=0.2, tta=True):
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
        self.dropout = nn.Dropout(p)        # dropout 추가
        self.tta = tta                      # tta 사용 여부

    def forward_once(self, x):              # 원래 forward 내용
        x = self.patch_embed(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        x = self.stages[3](x)
        x = x.mean(dim=(2, 3))
        x = self.head.norm(x)
        return self.head.fc(self.dropout(x))    # fc 앞에 dropout

    def forward(self, x):
        logits = self.forward_once(x)
        if self.tta and not self.training:      # 평가 때만 TTA: 좌우 반전 결과와 평균
            logits = (logits + self.forward_once(torch.flip(x, dims=[3]))) / 2   # dims=[3] = (B, C, H, W)의 W
        return logits
''',
"def train_one_epoch": '''# 수정 가능

def train_one_epoch(model, loader, optimizer, loss_fn, scheduler=None):   # scheduler 인자 추가
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # gradient 크기 제한
        optimizer.step()
        if scheduler is not None:                                 # scheduler가 있으면 배치마다 갱신
            scheduler.step()
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
    return total_loss / len(targets), accuracy
''',
"torch.manual_seed(42)": '''# 수정 가능

torch.manual_seed(42)
model = FashionModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)   # Adam"W" + weight_decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(                                            # warmup → cosine 감소
    optimizer, max_lr=learning_rate, total_steps=epochs * len(train_loader),
    pct_start=2 / epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)    # 학습 Loss: label smoothing
eval_fn = nn.CrossEntropyLoss()                       # 검증 Loss: 보통 CE
best_accuracy = -1
history = []
started = time.perf_counter()

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scheduler)   # scheduler 전달
    val_loss, val_accuracy = evaluate(model, val_loader, eval_fn)                       # 검증은 eval_fn
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
print('체크포인트:', checkpoint)
''',
}
P2_2["batch_size ="] = '''# 수정 가능

# [기존 answer 값] 문제 기본값(batch 64, lr 0.0003, epochs 20)에서 lr·epochs를 바꿈
batch_size = 64          # 문제 기본값 그대로
learning_rate = 0.0005   # 변경: 0.0003 → 0.0005 (OneCycleLR의 max_lr로 쓰임)
epochs = 30              # 변경: 20 → 30
# 아래 값은 학습 셀에 직접 적었습니다(PDF 방식). 기존 answer와 비교:
#   weight_decay 0.05     : 기존 answer와 같음
#   warmup 2 epoch        : 기존 answer와 같음 (OneCycleLR pct_start = 2 / epochs)
#   label_smoothing 0.2   : PDF 값. 기존 answer는 0.1
'''

P2_2_NOTE = """> ## [optimized 정답 — PDF 반영]
> 원래 정답안(`P2_2_answer.ipynb`)에서 **외울 코드만 줄인 버전**입니다. 핵심 4가지: 데이터 증강 / Loss·Optimizer·스케줄러 / train step / Dropout + TTA.
> - 증강: 좌우 반전, `RandomAffine`(빈 곳 흰색), `ColorJitter`, `RandomErasing`
> - `CrossEntropyLoss(label_smoothing=0.2)`(검증은 보통 CE), `AdamW(weight_decay=0.05)`, `OneCycleLR(pct_start=2/epochs)` — 원래 정답안의 직접 만든 warmup+cosine 함수와 파라미터 그룹 분리를 한 줄씩으로 줄였습니다.
> - `train_one_epoch`: `scheduler` 인자, `clip_grad_norm_(…, 5.0)`, 배치마다 `scheduler.step()`
> - `FashionModel(p=0.2, tta=True)`: fc 앞 Dropout, 평가 모드에서 좌우 반전 TTA
> - 하이퍼파라미터 셀은 **기존 answer 값**(batch 64, lr 5e-4, epochs 30)이며, 바뀐 줄에 주석을 달았습니다. label smoothing만 PDF 값 0.2입니다(기존 answer 0.1).
> - 원래 정답안의 GPU test accuracy는 0.952였고, 이 버전(스케줄러를 OneCycleLR로, decay 그룹 분리 생략)의 GPU 점수는 확인하지 않았습니다.
> - PDF의 오타 `sceduler`, `evluate`, `filp`은 고쳤습니다."""

# ----------------------------------------------------------------------
P3 = {
"def system_prompts": '''# 수정 가능
# 함수명은 변경하지 마세요.

def system_prompts():
    schema_text = json.dumps(response_schema, ensure_ascii=False)     # 스키마를 한글 그대로 문자열로
    keys_text = ', '.join(response_schema['properties'])              # 키 이름 목록
    system_prompt = f"""
주어진 녹취에서 계약 정보를 JSON으로 정리하세요.
다음 json schema {schema_text} 를 만족하는 JSON 객체 하나만 출력하세요.
키는 {keys_text} 입니다. 모든 키를 넣고, 확정되지 않았거나 녹취에 없는 값은 null로 씁니다.
""".strip()
    return system_prompt
''',
}
P3_NOTE = """> ## [optimized 정답 — PDF 반영]
> 원래 정답안(`P3_answer.ipynb`)의 긴 규칙 프롬프트 대신, **스키마 문자열과 키 목록만 끼워 넣는 짧은 프롬프트**입니다.
> - 외울 두 줄: `json.dumps(response_schema, ensure_ascii=False)`, `', '.join(response_schema['properties'])`
> - 문장은 문제 기본 지시문 + 스키마 + 키 목록 + null 규칙 한 줄입니다(PDF의 한글 문구는 글자가 깨져 읽히지 않아, 같은 뜻으로 적었습니다).
> - 원래 정답안의 항목별 규칙(금액·날짜 변환 등)을 뺐으므로 실제 추출 품질은 낮아질 수 있습니다. 형식 점검은 모의 모드로만 확인했습니다."""

JOBS = [("P1_1", P1_1, P1_1_NOTE), ("P2_1", P2_1, P2_1_NOTE), ("P2_2", P2_2, P2_2_NOTE), ("P3", P3, P3_NOTE)]


def key_of(src, table):
    for k in table:
        if k in src:
            return k
    return None


def problem_editables(name):
    """문제 원본의 '수정 가능' 셀을 첫 정의 이름으로 찾을 수 있게 모은다."""
    nb = json.loads((HERE / f"{name}_problem.ipynb").read_text(encoding="utf-8"))
    return [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "수정 가능" in ''.join(c["source"])]


def signature(src):
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(("def ", "class ")) or ("=" in s and not s.startswith("#")):
            return s.split("(")[0].split("=")[0].strip()
    return None


def build(name, table, note):
    nb = json.loads((HERE / f"{name}_answer.ipynb").read_text(encoding="utf-8"))
    prob = problem_editables(name)
    cells, used, n_edit = [], set(), 0
    for c in nb["cells"]:
        src = ''.join(c["source"])
        if c["cell_type"] == "markdown" and ("[정답안 해설]" in src or "[예시 정답안]" in src):
            continue                                   # 내용이 달라지는 해설은 뺌
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            if "warmup_epochs = 3, 1" in src:              # optimized 버전에는 warmup 변수가 없음
                src = src.replace("epochs, warmup_epochs = 3, 1  # QUICK: 3 epoch, warmup 1", "epochs = 3  # QUICK: 3 epoch")
                c["source"] = src
            if "수정 가능" in src:
                k = key_of(src, table)
                pos = n_edit; n_edit += 1               # 문제와 정답의 '수정 가능' 셀은 개수·순서가 같음
                if k is not None:
                    c["source"] = table[k]; used.add(k)
                else:                                   # PDF가 다루지 않는 셀 → 문제 기본값
                    c["source"] = prob[pos]
        cells.append(c)
    missing = set(table) - used
    assert not missing, f"{name}: 바꿀 셀을 못 찾음 {missing}"
    cells.insert(1, {"cell_type": "markdown", "metadata": {}, "source": note})
    nb["cells"] = cells
    out = HERE / f"{name}_answer_optimized.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out.name)


PRACTICE_NOTE = """> ## [연습용] {name} — 자동완성으로 직접 채워 보기
> `수정 가능` 셀은 **문제 원본 상태**입니다. `{name}_answer_optimized.ipynb`(있으면)를 보지 않고 채운 뒤 실행해 보세요.
> - 이 PC처럼 GPU가 없으면 첫 환경 셀이 자동으로 **QUICK 모드**(epoch·데이터 축소)로 바꿔 끝까지 돌아갑니다. 점수는 낮게 나오며 코드가 도는지 확인하는 용도입니다.
> - 자동완성: VS Code에서 이 폴더를 열고 커널을 `.venv\\Scripts\\python.exe`로 고르면, 입력 중 제안이 뜨고 `Ctrl+Space`로 직접 부를 수 있습니다. 함수 인자는 `(` 뒤 `Ctrl+Shift+Space`."""


def build_practice(name):
    """정답안 구조(CPU QUICK 지원)는 그대로, '수정 가능' 셀만 문제 원본으로 되돌린 연습장."""
    nb = json.loads((HERE / f"{name}_answer.ipynb").read_text(encoding="utf-8"))
    prob = problem_editables(name)
    cells, n_edit = [], 0
    for c in nb["cells"]:
        src = ''.join(c["source"])
        if c["cell_type"] == "markdown" and ("[정답안 해설]" in src or "[예시 정답안]" in src):
            continue
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            if "수정 가능" in src:
                c["source"] = prob[n_edit]; n_edit += 1          # 같은 순서의 문제 원본 셀
            elif "warmup_epochs = 3, 1" in src:
                c["source"] = src.replace("epochs, warmup_epochs = 3, 1  # QUICK: 3 epoch, warmup 1", "epochs = 3  # QUICK: 3 epoch")
            elif "[정답안]" in src and "student_id" in src:
                c["source"] = src.replace("'000000'   # [정답안] 예시 번호", "'000000'   # 연습용 번호")
        cells.append(c)
    cells.insert(1, {"cell_type": "markdown", "metadata": {}, "source": PRACTICE_NOTE.format(name=name)})
    nb["cells"] = cells
    out = HERE / f"{name}_practice.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out.name)


if __name__ == "__main__":
    for job in JOBS:
        build(*job)
    for name in ["P1_1", "P1_2", "P2_1", "P2_2", "P3"]:
        build_practice(name)
