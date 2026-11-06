# torchvision.models — 사전학습 모델과 구조 변경

FAQ: 시험 환경에는 **사전학습 가중치가 미리 설치**되어 있고 인터넷은 차단됩니다. 그래서 `weights="DEFAULT"`로 내려받는 대신 **로컬 파일에서 불러오는** 형태일 가능성이 큽니다. 실제 경로·모델 이름은 문제의 안내를 따르세요.

### 구조 만들기
```python
from torchvision import models
model = models.resnet18(weights=None)            # 구조만 (가중치 무작위)
state = torch.load("가중치경로.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)                     # 로컬 사전학습 가중치
```
- 인터넷이 될 때의 형태(참고): `models.resnet18(weights=models.ResNet18_Weights.DEFAULT)`. 이때 `weights.transforms()`가 그 가중치에 맞는 전처리(크기·mean·std)를 알려 줍니다.

### 출력층 이름 (head 교체 위치)

| 모델 | 출력층 | 교체 예 |
|---|---|---|
| ResNet (`resnet18/34/50`) | `model.fc` (`Linear(512 또는 2048, 1000)`) | `model.fc = nn.Linear(model.fc.in_features, n)` |
| EfficientNet (`efficientnet_b0` 등) | `model.classifier` = `Sequential(Dropout, Linear)` | `model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)` |
| MobileNetV3 | `model.classifier[-1]` | `model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n)` |
| ConvNeXt | `model.classifier[2]` | 같은 방식 |
| ViT (`vit_b_16`) | `model.heads.head` | `model.heads.head = nn.Linear(model.heads.head.in_features, n)` |
| DenseNet | `model.classifier` | `model.classifier = nn.Linear(model.classifier.in_features, n)` |

- 모르는 모델은 `print(model)`로 마지막 층 이름을 확인합니다.
- **`in_features`를 읽어서** 쓰면 모델이 바뀌어도 코드가 그대로 동작합니다.
- 회귀(점수 하나): 출력 1 — `nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, 1))`.
- 교체한 층은 무작위 초기화 상태이므로 반드시 학습 범위에 넣습니다.

### ResNet의 하위 모듈 이름
`conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc`
- 학습 범위 예: 마지막 블록과 head만 → 이름이 `("layer4", "fc")`로 시작하는 파라미터만 `requires_grad=True`.

### 학습 범위를 고르는 감각
- 데이터가 적고 원래 과제와 비슷 → head만 (빠름, 과적합 적음)
- 원래 과제와 다른 특징이 필요(예: 무늬 분류 → 반점 개수) → 마지막 블록 1~2개까지
- 전체 미세조정은 학습률을 작게(예: 1e-4), 시간이 가장 많이 듭니다.
