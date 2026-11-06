# torch.utils.data (자주 쓰는 부분)

```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
```

### `class Dataset`
사용자 Dataset은 이 클래스를 상속하고 두 메서드를 구현합니다.
- `__len__(self)` → 표본 수(int)
- `__getitem__(self, idx)` → 표본 하나. 보통 `(입력 텐서, 정답 텐서)` 튜플
- 무작위 변환은 `__getitem__` 안에서 적용하면 **꺼낼 때마다 다른 증강**이 됩니다.

```python
class MyDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return self.transform(self.images[i]), torch.tensor(self.labels[i], dtype=torch.long)
```

### `TensorDataset(*tensors)`
- 첫 축 길이가 같은 텐서들을 묶은 Dataset. `ds[i]` → `(t1[i], t2[i], ...)`

### `DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, drop_last=False, generator=None)`
- 표본을 묶어 배치를 만듭니다. 튜플 표본은 원소별로 쌓여 `(xb, yb)`가 됩니다.
- `shuffle=True`: epoch마다 순서를 섞음. 학습용에만.
- `sampler`: 순서를 정하는 객체. **`sampler`를 주면 `shuffle`은 주지 않습니다**(함께 주면 오류).
- `generator`: 섞는 난수 생성기(재현성).
- `drop_last=True`: 마지막 작은 배치를 버림.
- `len(loader)`: 배치 수.

### `WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`
- 표본 i를 확률 `weights[i] / sum(weights)`로 `num_samples`번 뽑습니다.
- `weights`: **표본마다 하나**(클래스마다가 아님). 1차원 시퀀스(흔히 `torch.double`).
- `replacement=True`: 복원추출. 적은 클래스의 표본이 한 epoch에 여러 번 나올 수 있습니다.
- 클래스 균형: 표본 i의 가중치 = `1 / (표본 i의 클래스 개수)` → 클래스마다 가중치 합이 같아져 고르게 뽑힙니다.

```python
counts = pd.Series(labels).value_counts()
w = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=torch.Generator().manual_seed(42))
loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
```

**주의:** 균형 샘플러와 클래스 가중 손실을 함께 쓰면 보정이 두 번 됩니다. 비교할 때는 따로 시험하세요. 검증·test 로더에는 샘플러·셔플을 쓰지 않습니다.
