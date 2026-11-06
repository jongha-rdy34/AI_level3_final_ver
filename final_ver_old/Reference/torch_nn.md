# torch.nn — 모듈 (자주 쓰는 부분)

```python
from torch import nn
import torch.nn.functional as F
```

### `class nn.Module`
- `forward(x)`를 정의하고, 호출은 `model(x)`로 합니다(`model.forward(x)` 직접 호출 X).
- 하위 모듈은 속성으로 접근: `model.fc`, `model.layer4.conv1`. 속성에 새 모듈을 대입하면 교체됩니다: `model.fc = nn.Linear(512, 4)`.

### 파라미터
- `model.parameters()` → 모든 파라미터(이터레이터)
- `model.named_parameters()` → `(이름, 파라미터)` 쌍. 이름 예: `"layer4.conv1.weight"`, `"fc.bias"`
- `p.requires_grad` (bool): False이면 기울기를 계산하지 않음(고정). `p.numel()`: 원소 수
- `model.named_modules()` → `(이름, 모듈)` 쌍. 이름 예: `"bn1"`, `"layer1.bn2"`, 맨 앞은 `""`(모델 자신)

### 모드
- `model.train()`: 학습 모드(Dropout 동작, BatchNorm은 배치 통계 사용 **및 running 통계 갱신**). 하위 모듈 전부에 적용.
- `model.eval()`: 평가 모드(Dropout 끔, BatchNorm은 저장된 running 통계 사용, 갱신 안 함).
- `m.training` (bool): 현재 모드.
- **주의:** `requires_grad=False`로 BN 가중치를 고정해도, `train()` 모드라면 `running_mean`, `running_var`는 계속 바뀝니다. 고정하려는 BN은 `train()` 호출 **뒤에** 그 모듈만 `eval()`로 둡니다.

### `nn.Linear(in_features, out_features, bias=True)`
- `y = x Wᵀ + b`. 속성: `in_features`, `out_features`, `weight` `(out, in)`, `bias` `(out,)`.

### `nn.Sequential(*modules)`
- 순서대로 적용하는 컨테이너. 인덱스로 접근: `seq[0]`, `seq[1]`.

### `nn.Dropout(p=0.5)`
- 학습 모드에서 확률 `p`로 원소를 0으로 만들고 나머지를 `1/(1-p)`배. 평가 모드에서는 그대로 통과.

### `nn.BatchNorm2d(num_features)`
- 채널별 정규화. 버퍼: `running_mean`, `running_var`(학습 모드에서 갱신).

### `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`
### `nn.AdaptiveAvgPool2d(output_size)`, `nn.ReLU(inplace=False)`, `nn.Flatten()`

### 상태 저장·복원
- `model.state_dict()` → 파라미터·버퍼 이름 → 텐서의 dict. **실제 텐서를 가리키므로**, 학습 중 최고 가중치를 보관하려면 `copy.deepcopy(model.state_dict())`.
- `model.load_state_dict(state_dict, strict=True)`: 이름과 모양이 맞아야 합니다. 구조를 바꾼 뒤 이전 가중치를 넣으면 모양 불일치 오류가 납니다.
- `torch.save(obj, path)`, `torch.load(path, map_location="cpu", weights_only=True)`
