# timm · TinyViT (자주 쓰는 부분)

이 PC에는 timm이 없어 노트북은 같은 이름·API의 `mini_tinyvit`을 씁니다. 아래는 시험에서 timm을 쓸 때의 형태입니다(버전에 따라 세부가 다를 수 있음).

### `timm.create_model(model_name, pretrained=False, num_classes=None, drop_rate=0.0, drop_path_rate=None, checkpoint_path="", pretrained_cfg_overlay=None, **kwargs)`
- `model_name`: `"tiny_vit_5m_224"` (`timm.list_models("tiny_vit*")`로 확인)
- `pretrained=True`: 사전학습 가중치를 불러옴. 인터넷이 없으면 로컬 파일을 써야 합니다.
  - `pretrained_cfg_overlay=dict(file="경로/model.safetensors")` 또는
  - `pretrained=False`로 만든 뒤 `model.load_state_dict(torch.load(경로, map_location="cpu"))`, 또는 `checkpoint_path=경로`
- `num_classes=20`: head를 20출력으로 새로 만듦(사전학습 head는 버림)
- `drop_path_rate`: stochastic depth 최대 비율(깊은 블록일수록 커짐). 미세조정에서 0.05~0.1

### 모델 메서드·속성
- `model.pretrained_cfg` → dict: `input_size`, `mean`, `std`, `interpolation`, `crop_pct`, `num_classes` …
- `model.reset_classifier(num_classes)` → head의 fc만 교체
- `model.get_classifier()` → 출력층 모듈 (`model.head.fc`)
- `model.num_features` → head 입력 차원
- `model.forward_features(x)` → 마지막 단계의 특징 맵 / `model.forward_head(x, pre_logits=True)` → 풀링·정규화된 특징 벡터(분류 전)

### `timm.data.resolve_data_config(args=None, model=model)` → dict
- `{"input_size": (3, 224, 224), "interpolation": "bicubic", "mean": (...), "std": (...), "crop_pct": 0.95}`
- `timm.data.create_transform(**cfg, is_training=True, auto_augment="rand-m9-mstd0.5")`로 timm식 학습 변환(RandAugment 포함)을 만들 수도 있습니다.

### 학습에 유용한 timm 도구 (있으면 직접 구현 대신 사용)
| 도구 | 역할 | 직접 구현 (이 세트) |
|---|---|---|
| `timm.loss.LabelSmoothingCrossEntropy(smoothing=0.1)` | label smoothing | `smooth_targets` + `soft_target_ce` |
| `timm.data.Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=20)` | mixup·cutmix + smoothing, soft target 반환 | `mixup_batch` |
| `timm.loss.SoftTargetCrossEntropy()` | soft target CE | `soft_target_ce` |
| `timm.optim.create_optimizer_v2(model, "adamw", lr=…, weight_decay=0.05, layer_decay=0.75)` | **layer-wise lr decay** + no-decay 그룹 | `param_groups_llrd` |
| `timm.scheduler.CosineLRScheduler(opt, t_initial=…, warmup_t=…)` | warmup + cosine | `warmup_cosine` |
| `timm.utils.ModelEmaV2(model, decay=0.999)` | 가중치 지수이동평균(EMA) | (가이드 참고) |

### TinyViT 구조 이름 (학습 범위·LLRD에 씀)
```
patch_embed            # 합성곱 2개, 해상도 1/4
stages.0               # MBConv (합성곱) 블록
stages.1 ~ stages.3    # downsample(PatchMerging) + blocks(attention + local conv + MLP)
head.global_pool, head.norm, head.fc
```
- `model.named_parameters()` 이름 예: `stages.3.blocks.1.attn.qkv.weight`, `head.fc.bias`
- 깊이 번호(LLRD): head 0 → stages.3 1 → … → stages.0 4 → patch_embed 5. lr = base × decay^깊이

### ViT 계열 미세조정의 관례 (TinyViT 포함)
- 옵티마이저 AdamW, weight decay 0.05, **bias·정규화 가중치는 weight decay 0**
- warmup(전체의 5~10%) 후 cosine 감소, 배치마다 스케줄러 step
- 전체 미세조정이면 layer decay 0.65~0.85로 입력 쪽 lr을 줄임
- gradient clipping(norm 1.0~5.0), label smoothing 0.1
- 입력 크기는 사전학습 크기(224) 유지, 정규화는 `pretrained_cfg` 값
