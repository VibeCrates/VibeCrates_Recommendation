# VibeCrates Recommendation

[English](README.md) · **한국어**

VibeCrates는 "비 오는 밤에 혼자 듣기 좋은 노래"나 "잔잔한데 쓸쓸한 영화"처럼 원하는 분위기를 문장으로 적으면, 그 느낌에 맞는 영화·음악·책을 함께 찾아 주는 추천 서비스입니다. 찾은 작품들은 표지를 모은 콜라주 "보드"로 보여 줍니다.

이 저장소는 그중 **AI 추천 서버**를 담당합니다. 하는 일은 크게 세 가지입니다. 
- 추론 : 사용자의 검색어를 768차원 벡터로 바꿔 주는 API 서버를 운영합니다. 
- 학습 : 약 19만 개 작품을 같은 공간의 벡터로 바꾸는 모델을 학습하고, 그 결과를 백엔드에 전달합니다. 
- 평가 : 모델이 실제로 잘 찾는지 평가하는 실험을 관리합니다. 

사용자 계정, 보드 저장, 실제 벡터 검색은 백엔드 저장소가, 화면은 프론트엔드 저장소가 맡습니다.

---

## 목차

1. [서비스 안에서 이 서버가 하는 일](#1-서비스-안에서-이-서버가-하는-일)
2. [데이터](#2-데이터)
3. [모델](#3-모델)
4. [한국어 검색어 처리](#4-한국어-검색어-처리)
5. [API](#5-api)
6. [설치와 실행](#6-설치와-실행)
7. [학습과 평가](#7-학습과-평가)
8. [알려진 한계](#8-알려진-한계)
9. [저장소 구조](#9-저장소-구조)
10. [문서 안내](#10-문서-안내)

---

## 1. 서비스 안에서 이 서버가 하는 일



검색은 백엔드의 벡터 DB(Qdrant)가 하고, 이 서버는 **검색어를 벡터로 바꾸는 일**을 합니다.

```
[사용자 검색어] → 프론트엔드 → 백엔드 ──POST /api/v1/search/vector──▶ AI 추천 서버
                                                                        │ ① 한국어면 영어로 번역
                                                                        │ ② 쿼리 인코더로 768차원 벡터 생성
                                 백엔드 벡터 DB  ◀──────── 쿼리 벡터 ────┘
                         (작품 벡터 약 18만 개, 도메인별 컬렉션)
                                      │ 코사인 유사도로 검색
                                      ▼
                               결과 작품 → 보드 콜라주
```

작품 벡터는 이 저장소에서 미리 계산해 **전달 번들**(CSV)로 백엔드에 넘깁니다. **쿼리 벡터와 작품 벡터는 같은 체크포인트에서 나와야 합니다.** 다르면 검색은 오류 없이 성공하고 결과만 틀어집니다. API 응답과 번들의 모든 행에 들어 있는 `model_version`으로 둘이 같은지 확인합니다.

---

## 2. 데이터

세 도메인의 작품을 하나의 형식으로 맞춘 **canonical CSV**(`data/canonical/`)가 출발점입니다.

| 도메인 | 작품 수 | 원본 | 표지 이미지 |
|---|---:|---|---|
| 영화 | 39,515 | MovieGenre 데이터셋 + TMDB 메타데이터 | TMDB 포스터 재수집 (보유율 99.3%) |
| 음악 | 39,682 | Spotify 오디오 피처 데이터셋 | 원본 URL + Deezer 재수집 (92.4%) |
| 책 | 110,594 | Kindle + Goodreads(Zenodo) + BX 블러브 3개 소스를 합침 | 97.0% |

모델이 읽는 작품 텍스트는 원본 줄거리가 아니라, 원본 정보(줄거리, 블러브, 오디오 피처 등)를 바탕으로 LLM이 합성한 **분위기 설명(`description_synth`)**입니다. 세 도메인 모두 같은 형식으로 쓰여 있습니다. 음악의 오디오 피처(energy, valence 등)는 "차분한", "강렬한" 같은 자연어 표현으로 바뀌어 설명에 들어갑니다.

학습에는 작품마다 **학습 쿼리 3개**도 필요합니다. 사람이 그 작품을 찾을 때 쓸 법한 짧은 검색어로, Qwen2.5-VL이 표지와 설명을 보고 만듭니다. 이 쿼리와 작품을 짝지어 `scripts/prepare_dataset.py`가 학습용 `data/train_dataset.csv`를 만듭니다.

원본 데이터, 이미지, 학습 데이터, 모델 파일은 용량 때문에 저장소에 올리지 않습니다(`.gitignore`). 저장소에는 canonical CSV와 그것을 만드는 스크립트가 들어 있습니다.

---

## 3. 모델




모델은 **듀얼 인코더**입니다. 작품과 검색어를 각각 다른 인코더로 같은 768차원 공간에 놓고, 두 벡터의 코사인 유사도로 가까운 작품을 찾습니다. 작품 벡터는 미리 한 번만 계산해 두면 되므로, 실제 검색에서는 검색어 하나만 인코딩하면 됩니다. CPU에서도 수십 밀리초면 충분합니다.

<img width="769" height="248" alt="스크린샷 2026-11-16 오후 2 27 52" src="https://github.com/user-attachments/assets/216883ea-b2ca-497f-9069-e9757c5cda3d" />

- 추론 모델 그림 : 사용자 검색어(쿼리)를 벡터로 전환합니다.


```
작품 쪽                                              검색어 쪽
분위기 설명 ─ all-mpnet-base-v2(LoRA) + MLP ─ z_text ─┐
                                                ├─ ContentBlock ─ z_content (768, L2 정규화)
표지 이미지 ─ CLIP ViT-L/14 비전 + MLP ─ z_image ┘                       ↕ 코사인 유사도
                                     검색어 ─ CLIP ViT-L/14 텍스트(고정) + MLP ─ z_query (768)
```

학습은 두 단계로 나뉩니다(`src/training/trainer.py`).

<img width="1200" height="664" alt="스크린샷 2026-11-16 오후 2 27 24" src="https://github.com/user-attachments/assets/a6f8f180-1b27-4555-9d75-53f204bf75ca" />

- **1단계(대조 학습):** 설명·이미지·검색어 세 쪽의 MLP 헤드와 mpnet의 LoRA 어댑터를 학습합니다. 같은 작품에서 나온 세 벡터는 서로 가깝게, 다른 작품의 벡터와는 멀게 만드는 InfoNCE 손실을 세 쌍(설명–검색어, 설명–이미지, 이미지–검색어)에 걸어 더합니다.
- **2단계:** 나머지를 고정하고 `ContentBlock`만 학습합니다. 설명과 이미지를 합친 최종 작품 벡터 `z_content`가 검색어 벡터 `z_query`와 맞도록, 역시 InfoNCE로 학습합니다. 검색에 실제로 쓰이는 벡터가 바로 이 `z_content`입니다.

사전학습 인코더의 본체는 학습하지 않습니다. CLIP은 이미지 쪽과 검색어 쪽 모두 고정하고, mpnet에는 LoRA 어댑터만 붙여 학습합니다.

서비스 중인 체크포인트는 `best_stage2_qlora_off`이고, `model_version`은 `trained_model.pt@1786697109`입니다.

---

## 4. 한국어 검색어 처리

모델은 영어로만 학습됐고 CLIP 토크나이저도 영어 전용이라, 서버는 한국어 검색어를 **영어로 번역한 뒤 인코딩**합니다(`src/api/translation.py`). 백엔드는 한국어를 그대로 보내면 됩니다. 번역은 아래 순서로 시도합니다.

1. **번역 사전:** "시집", "로파이 비 오는 날"처럼 장르·분위기·상황 어휘로 끝까지 쪼갤 수 있는 검색어는 검수한 사전(`src/api/ko_en_dict.json`, 520개 항목)의 번역을 이어 붙입니다(로파이 비 오는 날 → *lo-fi rainy day*). 사전에 없는 단어가 하나라도 섞이면 다음 단계로 넘어갑니다.
2. **Gemma 3 4B:** 사전으로 풀리지 않는 문장형 검색어는 vLLM으로 띄운 Gemma 3 4B가 번역합니다. `LLM_TRANSLATE_URL`이 설정된 서버에서만 쓰입니다.
3. **opus-mt-ko-en:** Gemma가 없거나 시간 안에 답하지 않으면 Helsinki-NLP의 opus-mt로 번역합니다.

번역이 모두 실패하면 요청을 실패시키지 않고 원문을 그대로 인코딩합니다.

번역과 함께 **검색어가 지목한 도메인**도 읽습니다(`src/api/domain_intent.py`). "영화", "노래", "소설"처럼 작품 종류를 밝힌 명사가 있으면 응답의 `domain`이 `movie`·`music`·`book` 중 하나가 되고, 백엔드는 그 도메인에서만 검색합니다. 장르(재즈, 공포)는 단서로 쓰지 않고, 종류 명사가 없거나 여러 도메인을 함께 원하면 `all`입니다. 판별 규칙은 [`docs/api_for_backend.md`](docs/api_for_backend.md)에 있습니다.

---

## 5. API

FastAPI 서버이며, 실행한 뒤 `http://<서버>:8000/docs`에서 모든 엔드포인트를 직접 시험해 볼 수 있습니다. 요청 형식, 응답 필드, 전달 번들 형식, Qdrant 적재 방법은 [`docs/api_for_backend.md`](docs/api_for_backend.md)에 자세히 정리되어 있습니다.

| 메서드 · 경로 | 용도 |
|---|---|
| `POST /api/v1/search/vector` | **정본.** 검색어 한 건을 768차원 벡터로 바꿉니다 |
| `GET /api/v1/search/vector?keyword=…` | 위와 같은 결과를 GET으로 받습니다 |
| `POST /api/v1/embeddings` | 여러 검색어를 한 번에 벡터로 바꿉니다 (정리 예정) |
| `POST /api/v1/recommend` | 서버 안에서 검색까지 해서 결과를 돌려줍니다. 백엔드 검색 결과와 **대조하는 용도**이며 서비스 계약이 아닙니다 |
| `GET /api/v1/item/{domain}/{item_id}` | 작품 메타데이터를 조회합니다 |
| `GET /api/v1/ping?n=42` | 모델과 무관하게 통신만 확인합니다 |
| `GET /api/v1/health` | 모델과 인덱스 적재 상태를 확인합니다 |

```json
// POST /api/v1/search/vector
{ "text": "잔잔한 영화" }

// 응답
{ "text": "잔잔한 영화",
  "translated_text": "calm movie",
  "domain": "movie",
  "dim": 768,
  "normalized": true,
  "model_version": "trained_model.pt@1786697109",
  "vector": [0.0023, 0.0258, ...] }
```

벡터는 L2 정규화되어 있어서 내적이 곧 코사인 유사도입니다. `translated_text`는 번역이 일어났을 때만 채워지고, `null`이면 보낸 문장을 그대로 인코딩했다는 뜻입니다.

- 서버는 **HTTP/1.1만** 받습니다. HTTP/2로 보내면 본문이 전달되지 않아 422가 납니다.
- CLIP이 77토큰까지만 읽으므로, 그보다 긴 검색어는 뒷부분이 잘립니다.

---

## 6. 설치와 실행

### 환경 만들기

```bash
git clone https://github.com/VibeCrates/VibeCrates_Recommendation.git
cd VibeCrates_Recommendation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt          # 학습·실험까지 전부
# pip install -r requirements-serve.txt  # API 서버만 띄울 때
```

### 모델과 인덱스 준비

서버를 띄우려면 저장소에 없는 파일 두 가지가 필요합니다. 학습된 체크포인트 `models/trained_model.pt`(약 2.2GB)와, 작품 벡터를 미리 계산해 둔 `indexes/{movie,music,book}_embeddings.pt` 및 `_meta.parquet`입니다. 두 파일은 반드시 같은 학습에서 나온 것이어야 합니다. 파일이 없어도 서버는 뜨고 `/ping`은 동작하지만, 벡터 관련 경로는 503을 돌려줍니다.

### 서버 실행

```bash
./scripts/serve.sh            # 0.0.0.0:8000
PORT=9000 ./scripts/serve.sh  # 포트 변경
```

`serve.sh`는 `venv`의 uvicorn으로 서버를 띄우고, 이 컴퓨터의 LAN·Tailscale 주소를 출력합니다. 기동에 약 13초가 걸리고, 그동안은 연결이 거부됩니다.

주요 환경변수는 다음과 같습니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `models/trained_model.pt` | 체크포인트 경로 |
| `INDEX_DIR` | `indexes` | 작품 벡터 인덱스 폴더 |
| `DEVICE` | GPU가 있으면 `cuda`, 없으면 `cpu` | 추론 장치 |
| `TRANSLATE_QUERIES` | `1` | `0`이면 한국어 번역을 끕니다 |
| `LLM_TRANSLATE_URL` | (없음) | Gemma 번역 서버 주소(OpenAI 호환, 예: `http://127.0.0.1:8001/v1`). 비어 있으면 사전 → opus-mt만 씁니다 |
| `LLM_TRANSLATE_TIMEOUT` | `1.0` | Gemma 응답을 기다리는 시간(초) |

### Docker

이미지에는 코드만 들어가고, 모델과 인덱스는 실행할 때 볼륨으로 마운트합니다(`MODEL_PATH`, `INDEX_DIR`). 추론용이라 CPU 빌드 torch를 씁니다.

### 테스트

```bash
pytest tests/        # API, 데이터, 도메인 판별, 번역 (69개)
```

### 지금의 운영 환경

서비스 서버는 학교 GPU 서버(AstraGo, A100)에서 API 서버(포트 8000)와 Gemma 번역 서버(vLLM, 포트 8001)를 함께 돌립니다. 학습은 별도의 RTX 5090 서버에서 합니다. 접속 주소는 팀 내부 문서에 있습니다.

---

## 7. 학습과 평가

### 학습 파이프라인

```
canonical CSV
  → generate_item_descriptions.py / generate_music_descriptions.py   분위기 설명 합성 (vLLM)
  → generate_queries.py                                             작품별 학습 쿼리 3개 생성 (Qwen2.5-VL)
  → prepare_dataset.py                                              data/train_dataset.csv
  → extract_image_embeddings.py                                     표지 이미지 임베딩 미리 계산
  → train.py                                                        2단계 학습 + 인덱스 생성
  → convert_bundle_csv.py --require-image                           백엔드 전달용 작품 벡터 번들
  → export_query_vectors.py                                         백엔드 전달용 학습 쿼리 벡터
```

```bash
python scripts/train.py \
    --data-path data/train_dataset.csv \
    --image-embeddings <extract_image_embeddings.py가 만든 .pt> \
    --sample-one-query \
    --save-path models/trained_model.pt \
    --index-dir indexes
```

`--sample-one-query`는 매 스텝 작품의 학습 쿼리 3개 중 하나를 무작위로 골라 씁니다(서비스 모델의 설정). 그 밖에 `--query-lora`, `--title-dropout`, `--domain-balanced-batches` 같은 실험용 옵션이 있으며, 설명은 `python scripts/train.py --help`에 있습니다.

### 평가

두 모델(체크포인트와 인덱스)을 같은 평가 검색어로 돌려 비교합니다.

- **쌍대 판정:** `scripts/judge_pairwise.py`가 두 모델의 결과를 LLM 판정기(기본 Qwen2.5-VL-7B)에 나란히 보여 주고 어느 쪽이 나은지 고르게 합니다. 순서를 바꿔 두 번 묻고, 승률과 p값을 냅니다.
- **사람 라벨:** 판정기가 가리기 어려운 비교는 `scripts/build_label_sheet.py`로 라벨 시트를 만들어 사람이 고르고, `scripts/score_label_sheet.py`로 집계합니다.

실험 산출물은 `experiments/`에, 실험 보고는 `docs/`에 있습니다.

---

## 8. 알려진 한계

- **부정 표현을 거의 반영하지 못합니다.** "무섭지 않은 판타지 소설"을 "무서운 판타지 소설"과 비슷한 벡터로 바꿉니다. 고정된 CLIP 텍스트 인코더의 한계입니다.
- **은유적인 문장은 번역 품질이 떨어집니다.** 시적인 검색어일수록 번역문이 원문의 뜻에서 멀어질 수 있습니다.

각 문제의 원인과 현재 상태는 [`docs/model_issues.md`](docs/model_issues.md)에 있습니다.

---

## 9. 저장소 구조

```
VibeCrates_Recommendation/
├── src/
│   ├── api/                 # FastAPI 서버: 라우트, 모델 적재, 번역, 도메인 판별
│   ├── models/              # 듀얼 인코더 (TextBlock, ImageBlock, QueryBlock, ContentBlock)
│   ├── training/            # 2단계 트레이너, InfoNCE 손실, 학습 기록
│   └── data/                # 데이터셋, 로더, 전처리
├── scripts/                 # 데이터 수집·합성, 학습, 인덱스·번들 생성, 평가, 서버 실행
│   └── vibe_crate/          # 메인·보드 화면 디자인 시안 (정적 HTML)
├── data/canonical/          # 도메인별 정본 CSV (원본·이미지·학습 데이터는 gitignore)
├── experiments/             # 실험 결과와 사람 라벨 시트
├── docs/                    # 설계 문서, API 안내, 작업 기록, 세션 보고
├── reports/                 # 초기(1~20회차) 세션 보고
├── tests/                   # pytest
├── Dockerfile, docker/      # 서빙 컨테이너
├── requirements.txt         # 학습·실험 포함 전체 의존성
└── requirements-serve.txt   # API 서버 전용 의존성
```

`models/`, `indexes/`, `dist/`, `for_result_report/`는 저장소에 올리지 않습니다.

---

## 10. 문서 안내

| 문서 | 내용 |
|---|---|
| [`docs/api_for_backend.md`](docs/api_for_backend.md) | 백엔드 연동 안내: 엔드포인트, 전달 번들 형식, Qdrant 적재, 도메인 상한 규칙 |
| [`docs/model_issues.md`](docs/model_issues.md) | 지금까지 발견한 모델 문제를 해결 / 서비스 무관 / 남김 / 미확인으로 정리 |
| [`docs/worklog_2026-07_to_09.md`](docs/worklog_2026-07_to_09.md) | 7~9월에 한 일과 결과 |
| [`docs/worklist_2026-09-03.md`](docs/worklist_2026-09-03.md) | 9월 기준 할 일 목록 |
| `docs/report_session_*.md` | 9~10월 세션별 보고 (가장 최근: [`2026-10-31`](docs/report_session_2026-10-31.md)) |
| [`docs/design_poet_style_alignment.md`](docs/design_poet_style_alignment.md), [`docs/design_audio_feature_integration.md`](docs/design_audio_feature_integration.md) | 설계 노트 |
| [`docs/report_judge_validation_20261009.md`](docs/report_judge_validation_20261009.md) | LLM 판정기 검증 결과 |

---

## 라이선스

MIT License

## 연락처

maxsky9018@gmail.com
