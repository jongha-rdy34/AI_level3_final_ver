# [claude] Lv3 모의평가 v3 — TinyViT-5M · 상품 이미지 20분류

**창작 모의문제입니다. 실제 시험 문제가 아닙니다.** 공식 정보 **"지정 모델 TinyViT-5M, 상품 이미지 20개 분류(모자·의류·신발 등)"**를 반영해 v2(공지·FAQ 반영 절충판)를 다시 만들었습니다.

- 성능을 올리는 전략은 **`TinyViT_성능극대화_가이드.md`**에 따로 정리했습니다(시험 환경: GPU, 224 입력 기준).

## 1. v2에서 바뀐 점

| 항목 | v2 | v3 (공식 정보 반영) |
|---|---|---|
| 지정 모델 | ResNet 계열 TinyResNet | **TinyViT**(합성곱 단계 + attention 단계) — timm과 같은 이름·API의 축소판 `mini_tinyvit` |
| 과제 | 잎 질병 4분류 + 등급 회귀 | **상품 이미지 20분류** (Q3 첫 제출 → Q4 개선 제출) |
| 데이터 형태 | npz 배열 | **이미지 파일(PNG/JPG) + CSV** — 크기·비율·색 모드(RGB/RGBA/L/CMYK)가 제각각 |
| 품질 함정 | 깨짐·중복·BGR·밝기 | 없는·깨진 파일(`load()`로만 잡힘), 레이블 표기 흔들림, 중복(엇갈린 레이블), **RGBA 투명 배경 → 검게 변함**, 작게 찍힌 test 사진 |
| 전처리 | 모델 카드 값 | **`resolve_data_config(model=model)`** 에서 읽기, 흰색 정사각형 패딩, `RandomAffine(fill=255)` |
| 모델링 기법 | fc 교체, 학습 범위, 가중 CE·focal | `reset_classifier`, 학습 범위(head/last/all), **layer-wise lr decay**, **label smoothing·mixup 직접 구현**, **warmup+cosine**, gradient clipping, drop_path |
| Q4 | 등급 예측 | **오분류 분석(혼동 상위 쌍) → TTA → 앙상블 → 최종 제출** |
| Q5 | 진료 메모 → JSON | **상품 설명 → JSON** (가격 `3만 9천원`·사이즈 `S~XL`·색상 정규화, 혼방·조건부 무료배송은 null) |
| Reference | | `timm_tinyvit.md`, `PIL_이미지.md` 추가, transforms·optim 페이지 보강 |

그대로 가져온 것: 버전 A(빈칸)/B(직접구현), 진행용 대체물, 제출가이드 + 형식 점검, 성능 기준 PASS/FAIL 채점기, **로컬 채점 셀(기존 best ↔ 이번 점수)**.

## 2. 시작하기 (VSCode)
1. VSCode에서 **이 폴더**를 엽니다.
2. `문제_B_직접구현.ipynb`(실전용) 또는 `문제_A_빈칸.ipynb`(워밍업)를 엽니다.
3. 커널을 torch가 설치된 Python으로 고르고 위에서부터 실행합니다. 막히면 `Reference/00_목차.md`.
4. 끝나면 커널 재시작 → 모두 실행 → 저장. 마지막 **로컬 채점** 셀로 기준 통과와 best 변화를 확인합니다.
5. `정답_채점용/채점기준.md`, `정답_채점용/정답_해설.ipynb`와 비교합니다.

버전 A는 빈칸을 채우기 전에 실행하면 그 셀에서 `NameError`가 납니다(정상). 버전 B는 빈 상태로도 끝까지 실행됩니다.

## 3. 문항과 시간 (권장 4시간, CPU 전체 실행 약 6분)

| 문항 | 유형 | 소문항 | 시간 |
|---|---|---|---:|
| Q1 데이터 점검 | 구현 | 1-1 표 점검·레이블 정리 · 1-2 파일 점검(모드·크기·깨짐·md5) · 1-3 `load_rgb`·패딩·물체 비율(+서술) · 1-4 정제 | 35분 |
| Q2 전처리 | 구현 | 2-1 transforms(모델 설정) · 2-2 미리 읽는 Dataset · 2-3 층화 분할·균형 샘플러 · 2-4 증강 판단(서술) | 30분 |
| Q3 전이학습 | 구현 + 성능 | 3-1 head 교체 · 3-2 학습 범위·**LLRD** · 3-3 **smoothing·mixup** · 3-4 **warmup+cosine** 학습 루프 · 3-5 체크포인트 · 3-6 baseline 대비 실험 · 3-7 제출 · 3-8 분석 | 65분 |
| Q4 성능 극대화 | 성능 | 4-1 혼동 상위 쌍 · 4-2 TTA · 4-3 앙상블 · 4-4 최종 제출 · 4-5 분석 | 45분 |
| Q5 프롬프트 | 구현 + 기준 | 5-1 프롬프트 · 5-2 스키마 · 5-3 JSON 꺼내기 · 5-4 정규화(가격·사이즈·색상) · 5-5 자료형 검사·제출 | 50분 |
| 점검·로컬 채점 | | 형식 점검, 재실행, **기존 best ↔ 이번 점수** | 15분 |

성능 기준(가정): `정답_채점용/_build/thresholds.json` — 노트북 머리말에 표시됩니다.

## 4. 폴더 구조
```
문제_A_빈칸.ipynb · 문제_B_직접구현.ipynb      ← 여기서 풉니다
TinyViT_성능극대화_가이드.md                    ← 시험용 성능 전략
제출가이드.md · Reference/ · assets/(모델카드.md, tiny_vit_practice.pt)
data/  train.csv · test.csv · sample_submission.csv · images/{train,test}/ · texts/
mini_tinyvit.py      ← timm TinyViT와 같은 이름·API의 연습용 축소판
practice_support.py  ← 제공 코드 · submission_check.py ← 형식 점검(torch 없이)
tv_compat.py         ← torchvision 없을 때만 쓰는 같은 이름의 대체
정답_채점용/  정답_해설.ipynb · score_local.py · 채점기준.md · test_truth.csv · q5_truth.json · _build/
```

## 5. 실제와 대체물
| 이름 | 정체 |
|---|---|
| `mini_tinyvit` + `assets/tiny_vit_practice.pt` | 실제 TinyViT-5M이 아닙니다. 같은 구조 이름(`patch_embed, stages, head.fc`)·API(`create_model`, `reset_classifier`, `pretrained_cfg`)를 가진 약 0.2M 파라미터 축소판(입력 64)을 합성 도형으로 사전학습했습니다. 실제 시험은 입력 224, 약 5.4M, ImageNet 사전학습 |
| `tv_compat.py` | torchvision.transforms.v2의 자주 쓰는 부분과 이름·인자를 맞춘 대체(공식 구현 아님) |
| `ps.mock_llm` | 실제 언어모델이 아닙니다(FAQ: 지정 모델은 이미지 모델). 미리 만든 응답 문자열 |
| 데이터 | 상품 사진·설명 모두 합성 자료. 브랜드 이름은 가상 |

## 6. 아직 모르는 것
- 실제 지표(accuracy인지 macro-F1인지), 성능 기준선, 데이터 규모·클래스 이름
- 가중치 경로와 불러오는 방식(timm 버전), 제출 방법 — 11/6 Kaggle 페이지·현장 안내로 확인
