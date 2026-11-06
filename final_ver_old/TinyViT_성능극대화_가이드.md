# TinyViT-5M · 상품 이미지 20분류 — 성능 극대화 가이드

공식 정보(지정 모델 **TinyViT-5M**, 과제 **상품 이미지 20개 분류**)와 공지·FAQ(GPU 사용 가능, 인터넷 차단, 가중치 설치됨, 4시간, 성능 기준이 있는 문항 포함)를 전제로 한 전략입니다. 수치 범위는 일반적으로 쓰이는 출발점이며, **반드시 고정한 validation으로 확인하고 채택**하세요. 이 모의 세트(축소판 모델·합성 데이터)에서 실제로 측정한 효과는 마지막 절에 있습니다.

---

## 0. 가장 중요한 다섯 가지 (시간이 없으면 이것만)

1. **전처리를 사전학습과 맞추기**: 입력 224, ImageNet mean/std, bicubic — `resolve_data_config(model=model)`에서 읽기. 비율을 지킨 **흰색 정사각형 패딩** 후 크기 조정. RGBA는 **흰 배경 합성**.
2. **전체 미세조정**: head만 학습하면 성능 상한이 낮습니다. 전체를 풀고, 입력 쪽 lr을 줄이는 layer decay는 **1.0과 0.75 정도를 둘 다 시험**해 validation으로 고르세요(사전학습 데이터와 과제가 가까울수록 decay가 도움이 됨).
3. **AdamW + warmup + cosine + label smoothing 0.1**: ViT 미세조정의 기본 묶음. 기울기 clipping 1.0.
4. **데이터 정제와 test 분포 맞추기**: 깨진 파일·중복·레이블 오류를 지우고, test와 train의 차이(크기·배경·밝기)를 육안으로 찾아 증강·검증에 반영.
5. **마지막에 TTA(좌우 반전) + 앙상블(seed/fold 2~5개)**: 학습을 다시 하지 않고도 오르는 구간. 확률을 평균.

---

## 1. 4시간 운영 (모델링 문항 기준)

| 시간 | 할 일 | 산출물 |
|---|---|---|
| 0:00–0:20 | 문제·제출 가이드·지표 확인, 데이터 점검(표, 모드, 크기, 중복, 분포) | 정제 표, 메모 |
| 0:20–0:40 | **baseline을 끝까지**: 전처리 → 짧은 학습(3~5 epoch) → 예측 → **제출 형식 통과** | 첫 제출 파일 |
| 0:40–1:40 | 레시피 탐색(한 번에 하나씩): 학습 범위 → lr → epoch → 증강 강도 → smoothing/mixup | 실험 표 |
| 1:40–2:20 | 최종 설정으로 seed 2~3개(또는 5-fold) 학습 | 모델 여러 개 |
| 2:20–2:35 | TTA·앙상블 비교(validation), 최종 제출 | 최종 제출 |
| 남는 시간 | 다른 문항, 서술, 커널 재시작 → 전체 실행 → 저장 | |

- **먼저 제출 가능한 상태를 만든 뒤 개선합니다.** 성능 기준이 있으니 "기준 통과 → 여유분 확보" 순서.
- 실험은 표로 기록(설정·val 점수·시간). 서술 답안과 코드 리뷰의 근거가 됩니다.
- GPU에서 TinyViT-5M(224)은 이미지 수천 장 규모면 epoch당 수십 초~1분 수준일 가능성이 큽니다. 첫 epoch 시간을 재서 전체 예산을 정하세요.

---

## 2. 데이터 점검과 정제 (Q1·Q2 유형)

| 점검 | 방법 | 처리 |
|---|---|---|
| 없는·깨진 파일 | `Image.open(p)` 후 **`im.load()`** 를 try | 제거 |
| 색 모드 | `im.mode` 분포 (RGBA, L, P, CMYK) | **RGBA·투명 P → 흰 배경 합성**, 나머지 `convert("RGB")` |
| 레이블 표기 | `value_counts()`, 공백·대소문자·하이픈 | 정규화, 모르는 값 제거 |
| 완전 중복 | 파일 바이트 md5 | 같은 레이블이면 1장만, 엇갈리면 묶음 제거 |
| 거의 중복 | 작게 줄인 이미지의 해시/특징 거리 | train↔test 누수 확인용 |
| 불균형 | 클래스별 개수, 최다/최소 비율 | 층화 분할, 샘플러·가중치 비교 |
| 크기·비율 | `width`, `height` 분포 | 정사각형 패딩(흰색) 후 224 |
| **train↔test 차이** | 육안 격자 + 지표(물체 크기 비율, 밝기, 배경색) | 증강·**분포를 흉내 낸 검증**으로 연결 |

- 상품 사진은 **흰 배경**이 기본이라, 패딩·회전 빈 곳·투명 배경을 모두 **흰색**으로 채우는 것이 일관됩니다. 검은 테두리는 모델이 엉뚱한 단서를 배우는 원인이 됩니다.
- 비율이 다른 사진을 정사각형으로 **늘이면** 긴바지↔반바지, 부츠↔스니커즈처럼 길이·높이가 근거인 클래스가 헷갈립니다. 패딩을 쓰세요.

---

## 3. 학습 레시피 (Q3 유형)

### 3-1. 출발 설정 (GPU, 224 입력)

| 항목 | 출발값 | 조정 방향 |
|---|---|---|
| 학습 범위 | **전체(all)** | 데이터가 매우 적거나(클래스당 수십 장) 시간이 없으면 `stages.3 + head` |
| 옵티마이저 | AdamW, weight decay 0.05 (bias·norm은 0) | |
| 기본 lr(head 기준) | 5e-4 ~ 1e-3 (배치 64) | 손실이 출렁이면 ↓, 느리게 줄면 ↑. 배치를 2배로 하면 lr도 대략 2배 |
| layer decay | 0.75 | **1.0(없음)과 비교 필수.** ImageNet → 상품 사진처럼 가까우면 0.7~0.85가 흔히 도움, 멀면 1.0이 나을 수 있음(이 세트의 결과) |
| 스케줄 | warmup 5~10% → cosine | epoch가 아니라 **step 단위** |
| epoch | 10~30 | val 곡선이 아직 오르면 늘리고, 꺾이면 체크포인트로 최고 epoch 사용 |
| 배치 | 64~128 | GPU 메모리에 맞게. AMP를 쓰면 더 크게 가능 |
| label smoothing | 0.1 | |
| drop_path_rate | 0.1 (5M 모델은 0.05~0.1) | 과적합이면 ↑ |
| gradient clipping | max_norm 1.0 | |
| mixup / cutmix | 과적합이 보일 때만: mixup α 0.2~0.4 또는 cutmix α 1.0, 배치의 50% | 과소적합이면 끔 |
| EMA | decay 0.999 (`timm.utils.ModelEmaV2`) | 짧은 학습에선 decay를 낮게(0.99~0.995) |

### 3-2. 두 단계 미세조정 (LP → FT)
1. **head만** 1~2 epoch (lr 1e-3~3e-3): 무작위 head가 몸통에 큰 기울기를 보내 사전학습 특징을 망치는 것을 막음
2. **전체** 미세조정 (위 설정)

warmup이 같은 역할을 일부 하므로, 시간이 없으면 warmup만으로도 충분한 경우가 많습니다.

### 3-3. 불균형 (지표를 먼저 확인)
- 지표가 **정확도**면 불균형 보정은 오히려 손해일 수 있고, **macro-F1**이면 보정이 거의 필수입니다. Kaggle 대회 페이지의 Evaluation을 먼저 확인하세요.
- 선택지(하나씩 비교): 균형 샘플러(`WeightedRandomSampler`, 가중치 1/개수 또는 1/√개수) · 클래스 가중 CE · **logit adjustment**(학습 후 logits에서 `τ·log(사전확률)`을 빼기 — 재학습 없이 시험 가능)
- 샘플러와 가중 손실을 **같이 쓰면 보정이 두 번** 됩니다.

### 3-4. 증강 (상품 사진)
| 권장 | 이유 |
|---|---|
| 좌우 반전 | 상품은 좌우 대칭적으로 촬영됨 |
| `RandomAffine(degrees≈10, translate≈0.1, scale≈(0.6, 1.0), fill=흰색)` | 위치·크기 변화, **작게 찍힌 사진 대비** |
| `RandomResizedCrop(224, scale=(0.6, 1.0))` | 약한 확대. 강하게(0.08~) 자르면 물체가 잘려 레이블 근거가 사라짐 |
| ColorJitter(밝기·대비 0.2, 채도 0.2, 색조 ≤0.05) | 조명 차이. 이 과제는 **색이 레이블이 아니므로** 색 변화가 비교적 안전 |
| RandAugment / TrivialAugment | 데이터가 적고 과적합일 때 |
| RandomErasing(p 0.25) | 가림에 강하게 |

피할 것: 상하 반전(실제 상품 사진에 없음), 강한 크롭, 검은색 fill.

---

## 4. 검증 설계 (점수를 믿을 수 있게)

- **층화 분할을 한 번 고정**하고 모든 실험을 같은 validation으로 비교합니다.
- test 분포가 다르면 **그 차이를 흉내 낸 validation**을 하나 더 만듭니다(예: 축소·어둡게·배경 바꾸기). 이 모의 세트의 `val_small`이 그 예입니다.
- 클래스가 20개이고 소수 클래스는 validation에 몇 장뿐이라 macro-F1이 흔들립니다. **클래스별 재현율 최저값**과 혼동 상위 쌍을 함께 보세요.
- 시간이 되면 **5-fold**: 데이터 전체를 학습에 쓰면서 OOF로 공정하게 평가하고, fold 모델 5개가 그대로 앙상블이 됩니다.
- **리더보드(public) 점수로 설정을 고르지 않습니다.** 리더보드에 과적합하면 최종 점수가 떨어질 수 있습니다.

---

## 5. 추론·앙상블 (Q4 유형)

| 방법 | 비용 | 기대 효과 | 메모 |
|---|---|---|---|
| TTA: 원본 + 좌우 반전 | 추론 2배 | 작지만 거의 항상 + | **확률(softmax)을 평균** |
| TTA: + 약간 축소한 뷰 | 추론 3배 | test가 작게 찍혔다면 + | 학습에 쓴 변환 범위 안에서만 |
| 체크포인트/EMA | 0 | + | 마지막 epoch보다 최고·EMA 가중치 |
| seed 2~3개 | 학습 ×N | + | 예측 상관이 매우 높으면 이득이 작음 |
| 5-fold | 학습 ×5 | ++ | 공정한 OOF + 앙상블 |
| 다른 설정 섞기(증강·해상도·decay) | 학습 ×N | ++ | 다양성이 클수록 이득 |

- 분류 앙상블은 **확률 평균**(soft voting). logits 평균은 모델마다 크기가 달라 한 모델이 좌우할 수 있습니다.
- 가중 평균의 가중치는 validation(또는 OOF)으로 정합니다.

---

## 6. 오분류 분석 → 맞춤 개선

1. 혼동 행렬에서 **대각선을 뺀 상위 쌍**을 뽑습니다(예: 셔츠↔자켓, 볼캡↔비니, 긴바지↔반바지, 스니커즈↔부츠, 핸드백↔백팩).
2. 그 쌍의 틀린 사진을 **직접 봅니다**: 레이블 오류인가? 작게 찍혔나? 잘렸나? 배경이 다른가?
3. 원인별 처방
   - 레이블 오류 → 정제(근거를 서술에 남김)
   - 작게 찍힘·잘림 → 축소 증강, 패딩 방식 점검, TTA 뷰 추가
   - 미세한 차이(칼라·단추·길이) → 해상도 유지(224), 강한 크롭 줄이기, epoch 늘리기
   - 소수 클래스 → 샘플러·가중치·logit adjustment 비교

---

## 7. GPU에서 빠르게 (시간 = 실험 횟수)

```python
torch.backends.cudnn.benchmark = True
model = model.to(device, memory_format=torch.channels_last)
scaler = torch.amp.GradScaler("cuda")                       # fp16일 때. bf16이면 scaler 불필요
for xb, yb in loader:
    xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(xb), yb.to(device))
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                               # clipping 전에 unscale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update(); scheduler.step()
```
- `DataLoader(..., num_workers=4, pin_memory=True, persistent_workers=True)` (Windows Jupyter에서 num_workers가 문제를 일으키면 0으로)
- **디코딩·패딩·크기 조정을 미리 한 번** 해 두고(메모리에 uint8로) 증강만 매 epoch — 이 세트의 `ProductDataset` 방식
- 검증은 `torch.no_grad()` + 큰 배치

---

## 8. 실전 템플릿 (timm이 있을 때)

```python
import timm
from timm.data import resolve_data_config
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV2

model = timm.create_model("tiny_vit_5m_224", pretrained=True, num_classes=20, drop_path_rate=0.1,
                          pretrained_cfg_overlay=dict(file=WEIGHT_PATH))   # 로컬 가중치 경로는 문제 안내대로
cfg = resolve_data_config(model=model)                                     # 224, mean/std, bicubic
model.to(device)
optimizer = create_optimizer_v2(model, "adamw", lr=1e-3, weight_decay=0.05, layer_decay=0.75)
mixup = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, label_smoothing=0.1, num_classes=20)
loss_fn = SoftTargetCrossEntropy()
ema = ModelEmaV2(model, decay=0.999)
# 학습 루프: xb, yb = mixup(xb, yb) → loss_fn(model(xb), yb) → backward → clip → step → scheduler.step() → ema.update(model)
# 평가·제출은 ema.module 과 model 중 validation이 높은 쪽
```
- `pretrained_cfg_overlay`·`checkpoint_path` 같은 인자는 timm 버전마다 다를 수 있습니다. 안 되면 `pretrained=False`로 만들고 `load_state_dict`로 직접 불러온 뒤 `reset_classifier(20)`.
- `create_optimizer_v2(..., layer_decay=...)`가 TinyViT를 지원하지 않는 버전이면 이 세트의 `param_groups_llrd`처럼 직접 그룹을 만드세요.

---

## 9. 코드 리뷰에서 지적받기 쉬운 것
- 전처리 값을 하드코딩(224, mean/std) — `pretrained_cfg`에서 읽기
- validation에 무작위 증강, test 정답·리더보드로 설정 선택
- `requires_grad=False`로 고정했는데 BatchNorm 통계가 계속 바뀜(고정 단계의 BN은 `eval()`)
- `state_dict()`를 deepcopy 없이 보관 → 마지막 가중치가 저장됨
- 제출 행 순서를 test 파일 순서로 저장(sample_submission의 id 순서로 `reindex`)
- 커널 재시작 후 다시 실행하지 않아 출력과 제출 파일이 다른 실행에서 나옴

---

## 10. 이 모의 세트에서 측정한 효과 (축소판 TinyViT·합성 데이터·CPU)

실제 TinyViT-5M과 실제 데이터에서의 크기는 다르지만, **방향**을 확인하는 참고 자료입니다. 표는 `정답_채점용/채점기준.md`의 보정 실험과 같은 값입니다.

| 설정 (모두 lr 1e-3~2e-3, 4~12 epoch) | val | val_small | **test** |
|---|---:|---:|---:|
| baseline: head만, 4 epoch, 증강 없음 | 0.584 | 0.438 | 0.492 ✗ |
| head만, 12 epoch, 증강 | 0.692 | 0.565 | 0.565 ✗ |
| stages.3 + head, 증강 | 0.902 | 0.767 | 0.800 ✗ |
| 전체, layer decay 0.75, **증강 없음** | 0.952 | 0.715 | 0.760 ✗ |
| 전체, layer decay 0.75, 증강 | 0.976 | 0.867 | 0.897 ✓ |
| 전체, decay 0.75, 증강 + mixup + 균형 샘플러 | 0.961 | 0.884 | 0.903 ✓ |
| **전체, layer decay 없음(1.0), 증강** | 0.979 | 0.902 | **0.927** ✓ |

(✓/✗는 Q3 기준 0.85. Q4 기준 0.92는 단일 모델 중 위 마지막 줄만 넘고, 나머지는 TTA·앙상블이 필요합니다.)

- **학습 범위가 가장 큰 요인**: head만(≤ 0.57) → 마지막 단계(0.80) → 전체(0.90~0.93)
- **축소 증강이 두 번째**: 같은 설정에서 증강을 빼면 test 0.897 → 0.760. test에 작게 찍힌 사진이 많아서입니다(val_small이 이 차이를 미리 보여 줌).
- **layer decay 0.75는 이 세트에서는 손해**(0.897 vs 0.927): 연습용 모델이 상품과 거리가 먼 **도형**으로 사전학습되어, 입력 쪽 층도 많이 바뀌어야 했기 때문으로 보입니다. 실제 TinyViT-5M은 ImageNet으로 사전학습되어 상품 사진과 가깝기 때문에 0.7~0.85가 흔히 도움이 됩니다. 결론: **layer decay는 1.0과 0.75 정도를 둘 다 시험해 validation으로 고르는 하이퍼파라미터**입니다.
- `val`(일반 검증)은 모든 상위 설정을 0.95 이상으로 보여 차이를 숨겼고, `val_small`의 순위가 test 순위와 같았습니다.

- test에 작게 찍힌 사진이 많아 **축소 증강**과 그것을 흉내 낸 검증(`val_small`)이 선택을 좌우했습니다.
- TTA·앙상블은 val_small에서 꾸준히 올랐고, test에서도 Q3 첫 제출 대비 크게 올랐습니다.
