# 인계장 — AI 역량평가 Lv3 공식 샘플 대비 (2026-11-06 작성, SSAFY PC → 개인 노트북)

> 이 문서는 **개인 노트북에서 작업을 이어받는 사람/에이전트**용입니다. 위에서부터 읽고 그대로 따라 하면 됩니다.
> 시험: **2026-11-07(토) 13:00~17:00**, 서울캠퍼스(역삼 멀티캠퍼스). 12:50 착석.

## 0. 한눈에 보기
| 폴더 | 내용 | 시작 파일 |
|---|---|---|
| `AI_Lv3_sample_solution/` | **공식 샘플 5문항 원본 + 실행 환경 재현 + 예시 정답안** | `README.md`, `setup.cmd` |
| `AI_Lv3_final_predict/` | 샘플 형식 그대로 데이터·함정을 바꾼 **최종 예측 문제 5문항 + 정답안** | `README.md`, `setup.cmd` |
| `final_ver_old/` | 이전에 만든 v3 모의 세트 (TinyViT 20분류, 기존 저장소 내용 그대로 옮김) | `final_ver_old/README.md` |

공식 샘플에서 확인된 시험 형식 (중요)
- 노트북 5개: **P1_1·P1_2(데이터/증강), P2_1·P2_2(모델링/성능), P3(프롬프트)**.
- 셀마다 `# 수정 금지` / `# 수정 가능` 표시. `답안 작성 셀` 마크다운 뒤의 셀만 고침. 함수 이름·인자 유지.
- 첫 셀에 **수험번호**(문자열, 앞자리 0 포함). 출력 파일명에 자동으로 들어감 → `outputs/P?_<수험번호>.pt/.pth/.csv/.txt`.
- 모델은 **TinyViT 5M `dist_in22k_ft_in1k`** 를 노트북 안에 직접 구현한 코드(+ `.safetensors` 가중치). P1_1만 `timm.create_model` 사용.
- 입력 크기가 작음: Fashion 80×60(H×W), UTKFace 112×112. window size (3,3,5,3)으로 줄이고 attention bias를 보간.
- 성능 기준 예: P1_1 val macro-F1 High > 0.75 / Mid > 0.60. P2_2는 Kaggle 제출(`test.csv`의 `answer` 채우기, accuracy).
- P3: OpenAI SDK(`gms_config.pkl`의 클라이언트) + **본인 API 키**(SSAFY GMS) 입력. `system_prompts()`만 작성, 결과는 `outputs/P3_<수험번호>.txt`.
- 첫 환경 셀이 `.venv`가 **노트북과 같은 폴더**에 있는지, **CUDA GPU**가 있는지 검사(없으면 에러).

## 0-1. GPU 실행 결과 요약 (RTX 5060 Ti, 2026-11-06)
| 샘플 | 기준선(제공 코드 그대로) | 정답안 |  | 예측 | 기준선 | 정답안 |
|---|---|---|---|---|---|---|
| P1_1 | 0.521 | **0.783** (High>0.75) | | F1_1 | 0.706 | **0.756** |
| P1_2 | CS@10 51.3% | **74.7%** | | F1_2 | 0.368 | **0.867** |
| P2_1 | - | **0.948** | | F2_1 | 보통 CE 0.872 | **0.880** |
| P2_2 | - | val 0.948 / test **0.952** | | F2_2 | val 0.716 | val 0.922 / test **0.914** |
| P3 | - | 형식 2/2 (모의) | | F3 | - | 형식 2/2 (모의) |

정답안 코드 셀은 **줄마다 동작 설명 주석**이 달려 있습니다. 정답안을 다시 만들면 `AI_Lv3_sample_solution\_build\annotate_answers.py`를 다시 실행하세요.

## 1. 노트북에서 할 일 (순서대로)
1. 저장소 받기
   ```
   git clone https://github.com/jongha-rdy34/AI_level3_final_ver.git
   cd AI_level3_final_ver
   ```
2. **Python 3.12** 설치 확인: `py -3.12 --version` (없으면 python.org에서 3.12 설치, "py launcher" 포함).
3. 샘플 폴더 환경 만들기 (약 2~3GB 다운로드, 20~40분):
   ```
   AI_Lv3_sample_solution\setup.cmd
   ```
   - GPU(NVIDIA)가 있으면 CUDA 12.8 torch, 없으면 CPU torch를 자동 선택.
   - Kaggle 데이터 2개를 **로그인 없이** 공개 URL로 받음 (`raw/fashion.zip` 593MB, `raw/utkface.zip` 347MB).
   - HuggingFace에서 가중치 `tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors`(21MB).
   - `prepare_data.py`가 노트북용 `data/`와 `test.csv`를 만듦.
4. 예측 문제 폴더: `AI_Lv3_final_predict\setup.cmd` (샘플 폴더의 `.venv`를 junction으로 같이 씀).
5. VSCode로 **각 폴더를 따로 열고** 노트북 커널을 그 폴더의 `.venv\Scripts\python.exe`로 선택.

## 2. 노트북(저사양, GPU 없음 가정)에서 주의
- 원본 `*_problem.ipynb`는 GPU가 없으면 첫 환경 셀에서 멈춥니다(시험 환경 재현이라 그대로 둠).
- `*_answer.ipynb`(정답안)는 **CPU면 자동으로 QUICK 모드**: 학습 데이터·epoch를 줄여 끝까지 돌아갑니다. 점수는 낮게 나오며 흐름 확인용입니다.
- 원본 문제를 CPU에서 풀어 보고 싶으면, 환경 셀의 GPU 검사 두 줄(`if not torch.cuda.is_available(): raise ...`, `torch.zeros(1, device='cuda')`)만 주석 처리하고 epoch를 줄이세요.
- GPU 실행 결과(점수·그래프)는 정답안 노트북에 **출력이 저장된 상태**로 올라가 있습니다. 실행하지 않고 읽기만 해도 됩니다.

## 3. P3(프롬프트) 실행 조건
- `gms_config.pkl`은 공식 파일의 재현본: `gms_client.GMSClient`(OpenAI 호환, base_url `https://gms.ssafy.io/gmsapi/api.openai.com/v1`, model `gpt-4.1-mini`).
  **주소·모델 이름은 추정값**입니다. 다르면 `make_gms_config.py`의 두 값을 고치고 다시 실행.
- `USER_API_KEY = 'dummy'`면 모의 모드(모든 값 null 응답)로 형식 흐름만 확인됩니다. 실제 품질은 본인 GMS 키로 확인.
- 정답 비교: 샘플은 `_grading/p3_example_truth.json`(예제 2개 손 정답), 예측은 `_grading/f3_truth.json`(예제 2 + 숨은 1).
  공식 정답이 아니라 녹취를 읽고 만든 정답입니다. 특히 샘플 예제2의 '백오십'을 150만 원으로 해석했습니다.

## 4. 시험 전날 공부 순서 (권장, 약 3시간)
1. 샘플 정답안 5개를 읽기 (해설 셀 `[정답안 해설]`) — 60분
2. 예측 문제 `F*_problem.ipynb` 중 **F1_1, F2_1, F3**를 직접 풀기 (GPU 없으면 코드 작성 + QUICK 실행) — 90분
3. 정답안과 비교, 특히 **Loss 직접 구현 규칙**(완성형 Loss API 금지, `log_softmax`/`gather` 허용)과 **학습 범위 고정 3종 세트**(`requires_grad_`, 고정 구간 항상 `eval`, `no_grad`) 암기 — 30분

## 5. 파일 구조
```
AI_Lv3_sample_solution/
  README.md                 결과 표와 실행 방법
  setup.cmd, requirements.txt, prepare_data.py, make_gms_config.py, gms_client.py
  P1_1_problem.ipynb ... P3_problem.ipynb    공식 샘플 원본(그대로)
  P1_1_answer.ipynb  ... P3_answer.ipynb     예시 정답안 (GPU 실행 출력 포함)
  problems_original/       zip에서 꺼낸 원본 보관
  P3_qa.csv                P3 예제 2개 (원본 노트북 마크다운에서 추출)
  docs/                    노트북의 docs 링크 → 온라인 원문으로 이동하는 페이지
  _grading/                로컬 채점 (score_p2_2.py, P3 손 정답)
  _build/                  정답안 빌더, 기준선(원본 그대로 실행) 노트북, 하이퍼파라미터 비교 결과
AI_Lv3_final_predict/
  README.md                예측 근거, 문항 설명, 결과 표
  setup.cmd, prepare_final.py, make_gms_config.py, gms_client.py, F3_qa.csv, docs/
  F1_1_problem.ipynb ... F3_problem.ipynb    예측 문제
  F1_1_answer.ipynb  ... F3_answer.ipynb     정답안 (GPU 실행 출력 포함)
  _grading/  score_f2_2.py, f3_truth.json, f3_hidden.csv
  _build/    build_final.py(문제·정답 생성), f3_data.py, 기준선 노트북
```
`.venv/`, `raw/`, `data/`, `outputs/`, 가중치, `gms_config.pkl`, `test.csv`는 git에 없고 `setup.cmd`가 만듭니다.

## 6. 알려진 제한
- 공식 데이터 구성(어떤 6종/20종, 몇 장, 어떻게 나눴는지)은 공개되지 않아 **형식과 함정만 재현**했습니다. 점수는 시험과 다를 수 있습니다.
- `prepare_data.py`의 UTKFace 검증 이미지는 "설치 카메라" 조건(회전 ±30°, 이동 ±12%, 확대, 밝기, 좌우)을 제가 만들어 넣은 것입니다.
- Windows에서 `jupyter nbconvert --execute`가 커널 시작 실패하면 VSCode에서 직접 실행하세요(샌드박스/보안 프로그램 영향).

## 7. SSAFY PC 정리 (2026-11-06 18:00 전)
SSAFY PC의 `Desktop\AI_Lv3_sample_solution`, `Desktop\AI_Lv3_final_predict`(데이터·.venv 포함), 임시 clone, pip 캐시는 사용자 확인 후 정리합니다. 노트북에서는 이 저장소만으로 전부 다시 만들 수 있습니다.
