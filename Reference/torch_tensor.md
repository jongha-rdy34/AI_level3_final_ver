# torch — 텐서 (자주 쓰는 부분)

### `torch.tensor(data, dtype=None, device=None)` / `torch.as_tensor(data)` / `torch.from_numpy(ndarray)`
- `from_numpy`는 메모리를 공유합니다(배열을 바꾸면 텐서도 바뀜). 음수 stride 배열(`a[..., ::-1]`)은 먼저 `np.ascontiguousarray`.

### 자료형
- `torch.float32`(모델 입력·가중치), `torch.long`(=int64, 분류 정답), `torch.double`(=float64, `WeightedRandomSampler` 가중치에 흔히 씀)
- `x.float()`, `x.long()`, `x.to(dtype)`

### 모양
- `x.shape`, `x.view(*shape)`, `x.reshape(*shape)`
- `x.permute(*dims)`: 축 순서 바꾸기. `(N,H,W,C) → (N,C,H,W)`는 `permute(0, 3, 1, 2)`. **`reshape`로 축을 바꾸면 픽셀이 섞입니다.**
- `x.unsqueeze(dim)` / `x.squeeze(dim)`: 길이 1인 축 추가 / 제거
- `torch.cat(tensors, dim=0)`: 이어 붙이기 · `torch.stack(tensors, dim=0)`: 새 축으로 쌓기
- `torch.flatten(x, start_dim=1)`

### 연산
- `x.flip(dims)`: 지정 축 뒤집기. 이미지 `(C,H,W)`에서 좌우는 `flip(-1)`, 상하는 `flip(-2)`.
- `torch.rot90(x, k, dims)`: 90도 회전 `k`번. 배치 `(B,C,H,W)`는 `dims=(2, 3)`
- `x.mean(dim, keepdim=False)`: `keepdim=True`면 집계한 축을 길이 1로 남김(브로드캐스팅용)
- `x.clamp(min, max)`: 범위 자르기
- `x.clone()`: 사본(원본을 바꾸지 않으려면)
- `x.argmax(dim)`: 최댓값 위치 · `x.exp()`, `x.log()`, `x.mean()`, `x.sum()`
- `torch.bincount(x, minlength=0)`: 정수 텐서의 값별 개수

### `Tensor.gather(dim, index)`
- `index`와 같은 모양으로, `dim` 축에서 `index`가 가리키는 값을 모읍니다.
- 분류에서 정답 클래스의 값만 고르기: `log_p.gather(1, target.unsqueeze(1)).squeeze(1)` → `(B,)`

### 장치
- `torch.cuda.is_available()` → bool
- `torch.device("cuda")` / `torch.device("cpu")`
- `x.to(device)`, `model.to(device)`: **모델과 입력은 같은 장치에 있어야 합니다.** 다르면 `RuntimeError: Expected all tensors to be on the same device`.
- `x.cpu()`, `x.numpy()`: numpy로 바꾸려면 CPU 텐서여야 하고, 기울기 기록이 없어야 합니다(`no_grad` 안이거나 `x.detach()`).

### 기울기 기록
- `with torch.no_grad():` 블록 안에서는 연산 기록을 하지 않습니다(추론·평가용).
- `x.detach()`: 기록에서 떨어진 텐서.

### 난수
- `torch.manual_seed(seed)`: 전역 난수 고정
- `torch.Generator().manual_seed(seed)`: 별도 난수 생성기. `DataLoader(generator=g)`, `torch.randperm(n, generator=g)`, `torch.multinomial(w, n, replacement=True, generator=g)`에 넘김
- `torch.rand(n)`, `torch.randn(*shape)`, `torch.randint(lo, hi, shape)`, `torch.empty(*shape).uniform_(a, b)`
