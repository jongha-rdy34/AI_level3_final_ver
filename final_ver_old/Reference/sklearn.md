# scikit-learn (자주 쓰는 부분)

### `sklearn.model_selection.train_test_split(*arrays, test_size=None, random_state=None, shuffle=True, stratify=None)`
- 배열들을 같은 위치로 나눠 `[a_train, a_test, b_train, b_test, ...]`를 반환합니다.
- `test_size`: 비율(0~1) 또는 개수. `random_state`: 재현성.
- `stratify=labels`: 나눈 두 쪽의 클래스 비율을 원래와 같게(층화). 불균형 데이터에서 소수 클래스가 한쪽에 몰리지 않게 합니다.
- 위치 배열을 나누는 형태: `tr, va = train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=y)`

### `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)`
- `skf.split(X, y)` → fold마다 `(train_idx, valid_idx)`. 각 fold의 클래스 비율을 전체와 같게 유지. `X`는 길이만 쓰므로 `np.zeros(len(y))`도 됩니다. 사용 예는 `증강_앙상블.md`.

### `sklearn.metrics.f1_score(y_true, y_pred, average="binary", labels=None)`
- `average="macro"`: 클래스별 F1의 단순 평균(클래스 크기와 무관하게 같은 무게). `"weighted"`: 클래스 크기로 가중 평균. `"micro"`: 전체 TP·FP·FN으로 계산(다중 클래스에서는 정확도와 같음).
- 불균형 데이터에서 소수 클래스를 놓치면 macro가 크게 떨어집니다.

### `sklearn.metrics.recall_score(y_true, y_pred, labels=None, average="binary")`
- `average=None`: 클래스별 재현율 배열. `labels=[k], average="macro"`: 클래스 k 하나의 재현율.

### `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None)`
- 혼동 상위 쌍 찾기: 대각선을 0으로 만든 뒤(`np.fill_diagonal(cm, 0)`) `np.argsort(cm, axis=None)[::-1]` → 위치 `i`는 행 `i // C`, 열 `i % C`

### `sklearn.metrics.classification_report(y_true, y_pred, target_names=None, digits=2, output_dict=False)`
- 클래스별 precision·recall·F1·support와 macro/weighted 평균을 표로. `output_dict=True`면 dict.

- `C[i, j]` = 실제 i인데 j로 예측한 개수. 행 = 실제, 열 = 예측. `labels=range(k)`로 모든 클래스를 고정하면 빠진 클래스가 있어도 모양이 일정합니다.

### `sklearn.metrics.cohen_kappa_score(y1, y2, labels=None, weights=None)`
- `weights="quadratic"`: **QWK**(quadratic weighted kappa). 순서가 있는 등급에서 멀리 틀릴수록 크게 벌점.
- 정의: 관측 행렬 O, 가중치 `W[i,j] = (i-j)² / (N-1)²`, 기대 행렬 `E = outer(실제 분포, 예측 분포) / n`, `κ = 1 - ΣW·O / ΣW·E`. 1이면 완전 일치, 0이면 우연 수준.
- 입력은 **정수 등급**이어야 합니다(연속 예측은 먼저 임계값으로 등급화).

### `sklearn.metrics.mean_squared_error(y_true, y_pred)`
- RMSE는 `np.sqrt(mean_squared_error(y, p))` (버전에 따라 `root_mean_squared_error`도 있음).
