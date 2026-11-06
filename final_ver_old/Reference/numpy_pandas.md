# NumPy · pandas (자주 쓰는 부분)

## NumPy

### `ndarray.shape`, `ndarray.dtype`, `ndarray.reshape(*shape)`
- `shape`: 각 축의 길이 튜플. 사진 묶음은 보통 `(N, H, W, C)`.
- `reshape(len(a), -1)`: 첫 축은 유지하고 나머지를 한 축으로 펼침. `-1`은 "남는 크기 자동".

### 집계 `a.mean(axis=None)`, `a.std(axis=None)`, `a.max()`, `a.min()`
- `axis`: 집계할 축(정수 또는 튜플). `images[..., 0].mean(axis=(1, 2))` → 사진별 R 채널 평균 `(N,)`.
- 반환: 집계한 축이 사라진 배열.

### 인덱싱
- `a[..., k]`: 마지막 축의 k번째 (채널 선택).
- `a[..., ::-1]`: 마지막 축 순서 뒤집기 (RGB ↔ BGR).
- 불리언 마스크 `a[mask]`, 위치 배열 `a[idx]`. 마스크 대입 `a[mask] = b`는 원본을 바꿉니다.

### `np.flatnonzero(mask)` → True인 위치의 1차원 정수 배열
### `np.isin(element, test_elements)` → `element`의 각 값이 `test_elements`에 있는지 bool 배열
### `np.digitize(x, bins)`
- `bins`가 오름차순일 때 각 값이 들어갈 구간 번호. `x < bins[0]` → 0, `bins[i-1] <= x < bins[i]` → i, `x >= bins[-1]` → `len(bins)`.
- 예: `np.digitize([0.2, 0.6, 3.9], [0.5, 1.5, 2.5, 3.5])` → `[0, 1, 4]`

### `np.arange(start, stop, step)`, `np.sort(a)`, `np.bincount(x, minlength=0)`
- `bincount`: 음이 아닌 정수 배열의 값별 개수.

### `np.random.default_rng(seed)` → `Generator`
- `rng.choice(a, size, replace=False)`, `rng.integers(lo, hi, size)`, `rng.normal(mu, sigma, size)`

### `ndarray.copy()`, `ndarray.astype(dtype)`, `ndarray.tobytes()`
- `copy()`: 원본과 메모리를 공유하지 않는 사본. `tobytes()`: 내용을 바이트열로 (해시 계산용).

## pandas

### `Series.str.strip()`, `Series.str.lower()` → 문자열 정리(원소별)
### `Series.unique()` → 고유값 배열 · `Series.nunique(dropna=True)` → 고유값 수
### `Series.value_counts(normalize=False)` → 값별 개수(내림차순). `normalize=True`면 비율
### `DataFrame.set_index(col)`, `Series.reindex(labels)`
- `reindex`: 주어진 라벨 순서로 다시 줄 세움. 없는 라벨은 `NaN`.
- **id로 맞추기**: `pd.Series(values, index=ids).reindex(sample["id"])` → `sample`의 id 순서로 정렬된 값.

### `DataFrame.assign(**cols)` → 열을 추가·교체한 새 DataFrame
### `DataFrame.sort_values(by, ascending=True)`, `DataFrame.reset_index(drop=True)`
### `DataFrame.idxmax()` / `Series.idxmax()` → 최댓값의 라벨
### `Series.astype(int)` → 자료형 변환

### `DataFrame.to_csv(path, index=True, encoding="utf-8")`
- **`index=False`를 주지 않으면 행 번호 열이 함께 저장**되어, 읽을 때 `Unnamed: 0` 열이 생깁니다(제출 형식 위반의 흔한 원인).

### `pd.read_csv(path, dtype=None)`
- `dtype={"id": str}`: id가 숫자처럼 보여도 문자열로 읽기.
