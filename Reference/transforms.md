# torchvision.transforms.v2 (자주 쓰는 부분)

```python
from torchvision.transforms import v2 as T
```

이 PC처럼 torchvision이 없으면 노트북은 같은 이름의 연습용 대체 `tv_compat.transforms_v2`를 씁니다(아래 표시된 기능만 지원).

변환 객체는 **호출할 수 있는 객체**입니다: `tf = T.RandomHorizontalFlip(p=0.5)`, `y = tf(x)`. 무작위 변환은 호출할 때마다 새로 난수를 뽑습니다.

### `T.Compose(transforms)`
- `transforms`: 변환 객체의 리스트. 앞에서부터 차례로 적용합니다.

### `T.ToImage()`
- numpy 배열 `(H, W, C)`나 PIL 이미지를 `(C, H, W)` 이미지 텐서로 바꿉니다. **자료형은 그대로**(uint8이면 uint8).

### `T.ToDtype(dtype, scale=False)`
- 자료형을 바꿉니다. `scale=True`이면 값 범위도 맞춥니다: uint8 `[0, 255]` → float `[0, 1]`.
- 흔한 사용: `T.ToDtype(torch.float32, scale=True)`

### `T.RandomHorizontalFlip(p=0.5)` / `T.RandomVerticalFlip(p=0.5)`
- 확률 `p`로 좌우(마지막 축) / 상하(끝에서 두 번째 축)를 뒤집습니다.

### `T.ColorJitter(brightness=None, contrast=None, saturation=None, hue=None)`
- `brightness`: float `b`이면 배율을 `[max(0, 1-b), 1+b]`에서, 튜플 `(min, max)`이면 그 범위에서 고릅니다. 실수 이미지는 결과를 `[0, 1]`로 자릅니다.
- `hue`: 색조를 바꿉니다. **색이 레이블의 근거인 과제에서는 레이블을 바꿀 수 있어 위험**합니다. (tv_compat은 `saturation`, `hue` 미지원)
- 실수 `[0,1]` 이미지에 쓰는 것이 의도에 맞습니다. 정규화 **뒤**에 쓰면 의미가 달라집니다.

### `T.Normalize(mean, std)`
- 채널별 `(x - mean[c]) / std[c]`. **실수 텐서에만** 적용합니다(uint8이면 오류).
- `mean`, `std`는 **사전학습 때 쓴 값(모델 카드)**을 씁니다. torchvision의 ImageNet 가중치는 보통 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`이지만, 지정 모델이 다르면 그 값을 따릅니다.

### `T.RandomApply(transforms, p=0.5)`, `T.Lambda(fn)`
- 묶음을 확률 `p`로 적용 / 임의 함수를 변환으로.

### `T.RandomAffine(degrees, translate=None, scale=None, shear=None, fill=0)`
- `degrees`: 회전 범위(±값 또는 (min, max)). `translate=(a, b)`: 가로·세로 최대 이동 비율. `scale=(lo, hi)`: 확대·축소 배율. **`scale`이 1보다 작으면 물체가 작아짐(zoom-out).**
- `fill`: 변환으로 생긴 빈 곳의 값. **기본값 0은 검정**입니다. 흰 배경 상품 사진이면 `fill=255`(uint8 이미지) — 실수 이미지면 `fill=1.0`(정규화 전).

### `T.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True)` / `T.CenterCrop(size)`
- `size`가 정수면 짧은 변을 그 길이로(비율 유지), `(h, w)`면 그 크기로. (tv_compat은 정사각형으로만)
- 가로세로 비율이 다른 상품 사진은 먼저 **정사각형 패딩**을 하고 `Resize((S, S))`를 하면 물체 비율이 유지됩니다.

### `T.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3))`
- 원본 넓이의 `scale` 비율만큼 무작위 영역을 잘라 `size`로 키움(zoom-in). ImageNet 학습의 기본이지만 **물체 일부가 잘려 레이블 근거가 사라질 수 있어** 상품처럼 물체 전체 모양이 중요한 과제는 `scale=(0.6, 1.0)` 정도로 약하게.

### (torchvision) 자동 증강: `T.TrivialAugmentWide()`, `T.RandAugment(num_ops=2, magnitude=9)`, `T.AutoAugment()`
- 여러 변환 중 무작위로 골라 적용. uint8 이미지에 적용(ToDtype 전). tv_compat 미지원.

### (참고, tv_compat 미지원) `T.RandomResizedCrop(size, scale=(0.08, 1.0))`, `T.RandomRotation(degrees)`, `T.Resize(size)`
- 잘라 내기는 물체 일부를 없앨 수 있습니다. **개수·크기·위치가 레이블의 근거이면** 레이블과 맞지 않는 표본이 생깁니다.

## 권장 순서 (예)

```python
train_tf = T.Compose([
    T.ToImage(),                                  # (H,W,C) uint8 → (C,H,W) uint8
    T.RandomHorizontalFlip(p=0.5),                # 기하 변환(값 범위와 무관)
    T.ToDtype(torch.float32, scale=True),         # [0,1] 실수
    T.ColorJitter(brightness=0.3),                # 색·밝기 변환은 [0,1]에서
    T.Normalize(mean=MEAN, std=STD),              # 정규화는 마지막
])
valid_tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=MEAN, std=STD)])
```

검증·test에는 무작위 변환을 넣지 않습니다(TTA를 일부러 할 때만 예외).
