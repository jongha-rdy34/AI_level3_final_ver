# AI 역량평가 Lv3 공식 샘플 — 실행 환경 + 예시 정답안

공식 샘플 zip(`AI역량평가_Lv3(샘플).zip`)에는 노트북 5개만 들어 있고, 노트북이 요구하는 `.venv`, `setup.cmd`, `data/`, 가중치, `docs/`, `gms_config.pkl`, `P3_qa.csv`는 없습니다. 이 폴더는 그것들을 **같은 이름·같은 형식으로 재현**하고, 각 문항의 **예시 정답안**을 붙인 것입니다.

> 정답안은 공식 정답이 아닙니다. 데이터 구성도 공개되지 않아 Kaggle 원본으로 **형식과 함정만** 재현했으므로 점수는 시험과 다를 수 있습니다.

## 1. 설치 (Windows, 한 번만)
1. Python **3.12** 설치 확인: `py -3.12 --version`
2. 이 폴더에서 `setup.cmd` 실행. 하는 일:
   - `.venv` 생성 → torch 2.11 (NVIDIA GPU가 있으면 CUDA 12.8, 없으면 CPU) → `requirements.txt` (timm 1.0.31 등) → Jupyter 커널 등록
   - 가중치 `tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors` (HuggingFace `timm/tiny_vit_5m_224.dist_in22k_ft_in1k`, 21MB)
   - Kaggle 공개 데이터 (로그인 불필요): `paramaggarwal/fashion-product-images-small` → `raw/fashion.zip`, `jangedoo/utkface-new` → `raw/utkface.zip`
   - `prepare_data.py` → `data/` + `test.csv`, `make_gms_config.py` → `gms_config.pkl`
3. VSCode에서 **이 폴더**를 열고, 노트북 커널을 `.venv\Scripts\python.exe`로 선택 (작업 폴더 = 노트북 폴더여야 첫 셀 검사를 통과).

## 2. 파일
| 파일 | 설명 |
|---|---|
| `P1_1_problem.ipynb` … `P3_problem.ipynb` | 공식 샘플 원본 (수정 없음, `problems_original/`에도 보관) |
| `P1_1_answer.ipynb` … `P3_answer.ipynb` | **예시 정답안**. `[정답안 해설]` 셀, 자가 점검 셀 포함. GPU 실행 출력이 저장돼 있음 |
| `prepare_data.py` | Kaggle 원본 → `data/fashion_imbalance`, `data/fashion_finetune`, `data/utkface`, `test.csv` |
| `gms_client.py`, `make_gms_config.py` | P3 연결 객체 재현 (`USER_API_KEY='dummy'`면 모의 응답) |
| `docs/` | 노트북의 `docs/pages/...` 링크를 온라인 원문으로 넘겨 주는 페이지 |
| `_grading/` | `score_p2_2.py` (Kaggle 대신 로컬 채점), `p3_example_truth.json` (예제 손 정답) |
| `_build/` | 정답안 생성기, 기준선(원본 그대로 실행) 노트북, P2_1 하이퍼파라미터 비교 |

## 3. CPU / GPU
- **문제 원본**은 GPU가 없으면 첫 환경 셀에서 멈춥니다(시험 환경 그대로).
- **정답안**은 한 파일로 둘 다 지원합니다. GPU가 없으면 `QUICK = True`가 되어 `[정답안 전용]` 셀이 학습 데이터·epoch를 줄입니다. GPU에서는 아무것도 바꾸지 않습니다.

## 4. 재현한 데이터와 함정
| 문항 | 데이터 | 함정(재현) |
|---|---|---|
| P1_1 | Fashion 6종, 80×60. train: 4종 1,200장 + **Shirts·Sports Shoes 40장**, val 200장씩 | 클래스 불균형 (헷갈리는 짝의 소수 클래스) |
| P1_2 | UTKFace 112×112, train 6,000 / val 1,500 | train=정면, val=**카메라 사진**(회전 ±30°, 이동, 확대, 밝기, 좌우) + 과한 기본 증강 |
| P2_1 | Fashion 20종 (상위 20 articleType), train 클래스당 최대 1,000 / val 125씩 | 학습 범위·Loss·HP |
| P2_2 | 같은 20종, `p2_2_train_indices.npy`로 클래스당 600장(부족하면 반복), test 2,500 | Kaggle 형식 (`test.csv` answer 채우기) |
| P3 | 노트북 예시 녹취 2개 → `P3_qa.csv` | 정정·거절된 제안·잘못된 서류·광고 숫자 |

## 5. 정답안 실행 결과 (RTX 5060 Ti)
| 문항 | 제공 코드 그대로 (기준선) | 예시 정답안 | 공식 기준 |
|---|---|---|---|
| P1_1 | val macro-F1 **0.5211** (Shirts·Sports Shoes F1 = 0) | **0.7827** (Sports Shoes 0.55, Shirts 0.75) | High > 0.75, Mid > 0.60 |
| P1_2 | val CS@10 최고 **51.3%** (과한 기본 증강) | **74.7%** | 공개 안 됨 |
| P2_1 | (Loss 미구현이라 실행 불가) | **0.9476** (batch 32, lr 2e-4) | 공개 안 됨 |
| P2_2 | - | val acc **0.9476**, 로컬 test acc **0.9516** | Kaggle 순위 |
| P3 | - | 모의 모드에서 형식 검사 2/2 통과 (내용은 실제 GMS 키 필요) | 형식 + 정확도 |

P2_1 하이퍼파라미터 비교 (Task A + label smoothing CE, 20 epoch, `_build/p2_1_sweep.json`)

| batch | lr | best val macro-F1 | best epoch |
|---|---|---|---|
| 64 | 3e-4 (초기값) | 0.9425 | 19 |
| 64 | 1e-4 | 0.9410 | 16 |
| 32 | 1e-4 | 0.9437 | 13 |
| **32** | **2e-4** | **0.9476** | 6 |

- 기준선 실행 노트북: `_build/baseline/`
- 정답안 코드 셀에는 **줄마다 동작 설명 주석**이 붙어 있습니다(`_build/annotate_answers.py`, 설명 사전 `_build/line_comments.json`). 정답안을 다시 만들면 이 스크립트를 다시 실행하세요.

## 6. 문항별 정답 요지
- **P1_1**: 진단=클래스 불균형. `processing_training_data`에서 모든 클래스를 최대 수까지 오버샘플링, 원본은 flag False, 반복분만 True(증강). 증강은 `ToPILImage` 먼저(입력이 numpy), 좌우 반전·작은 affine(흰색 채움)·밝기/대비.
- **P1_2**: 진단=train/val 분포 차이 + 과한 기본 증강. 이동 0.8→0.12, shear 50°→±8°, 회전 0→30°, 지우기 95%→25%, 확대 1.0~1.25, 밝기/대비 추가.
- **P2_1**: Task A `requires_grad_` + `train()` 오버라이드로 고정 구간 항상 eval + `no_grad`. Task B label smoothing CE를 `log_softmax`+`gather`로 직접 구현. Task C 학습률·batch 비교 표.
- **P2_2**: 증강 + AdamW(no-decay 그룹) + warmup→cosine + label smoothing + gradient clipping + Dropout + **eval forward 안의 좌우 반전 TTA**(추론 셀 수정 없이).
- **P3**: 형식 규칙 → 최종 합의 원칙 → 항목 정의(명도일/입주일/임대차시작일 구분, 계약금지급일=완납일) → 금액(만 원 생략 관행)·날짜 계산.
