# VibeCrates Recommendation

[English](README.md) · **한국어**

VibeCrates는 분위기를 텍스트로 입력받아("비 오는 밤에 혼자", "잔잔한 시골 풍경" 등) 그에 맞는 영화·음악·책을 추천하는 서비스입니다. 결과는 표지를 모은 콜라주 "보드"로 보여 줍니다.

이 저장소는 AI 추천 서버입니다.
- 추론 : 검색어를 768차원 벡터로 변환하는 API 서버
- 학습 : 약 19만 개 작품을 같은 벡터 공간에 넣는 모델의 학습, 결과물의 백엔드 전달
- 평가 : 검색 품질 평가 실험

사용자 계정, 보드 저장, 벡터 검색은 백엔드 저장소에서, 화면은 프론트엔드 저장소에서 다룹니다.

---

## 목차

1. [서비스 안에서 이 서버가 하는 일](#1-서비스-안에서-이-서버가-하는-일)
2. [데이터](#2-데이터)
3. [모델](#3-모델)
4. [한국어 검색어 처리](#4-한국어-검색어-처리)
5. [API](#5-api)
6. [로컬 실행](#6-로컬-실행)
7. [학습과 평가](#7-학습과-평가)
8. [개발 과정](#8-개발-과정)
9. [알려진 한계](#9-알려진-한계)
10. [저장소 구조](#10-저장소-구조)
11. [문서 안내](#11-문서-안내)

---

## 1. 서비스 안에서 이 서버가 하는 일

이 서버는 검색어를 벡터로 변환합니다. 검색은 백엔드의 벡터 DB(Qdrant)에서 합니다.

```
사용자 검색어
  → 프론트엔드 → 백엔드
  → AI 추천 서버 (POST /api/v1/search/vector)
      ① 한국어면 영어로 번역
      ② 쿼리 인코더로 768차원 쿼리 벡터 생성
  → 백엔드 벡터 DB (작품 벡터 약 18만 개, 도메인별 컬렉션)
      쿼리 벡터와 코사인 유사도로 검색
  → 결과 작품 → 보드 콜라주
```

작품 벡터는 이 저장소에서 미리 계산해 전달 번들(CSV)로 백엔드에 넘깁니다. 쿼리 벡터와 작품 벡터는 같은 체크포인트에서 나와야 하며, API 응답과 번들의 `model_version`으로 일치 여부를 확인합니다.

서비스 서버는 학교 GPU 서버(A100)에서 API 서버(8000)와 Gemma 번역 서버(vLLM, 8001)를 함께 돌립니다.

---

## 2. 데이터

아래 공개 데이터셋과 API를 조합해 세 도메인의 작품 데이터를 만들었습니다. 조합한 결과를 같은 형식으로 정리한 canonical CSV(`data/canonical/`)를 가공해서 사용합니다.

| 도메인 | 작품 수 | 표지 이미지 보유율 |
|---|---:|---:|
| 영화 | 39,515 | 99.3% |
| 음악 | 39,682 | 92.4% |
| 책 | 110,594 | 97.0% |

### 데이터 출처

<table>
  <tr><th>도메인</th><th>출처</th><th>사용한 내용</th><th>라이선스</th></tr>
  <tr><td rowspan="2">영화</td><td><a href="https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster">Movie Genre from its Poster</a> (Kaggle, neha1703)</td><td>작품 목록, IMDb ID, 장르</td><td>Kaggle 페이지 참조</td></tr>
  <tr><td><a href="https://www.themoviedb.org/">TMDB API</a></td><td>줄거리 등 메타데이터, 포스터</td><td>TMDB 이용 약관</td></tr>
  <tr><td rowspan="3">음악</td><td>Spotify 트랙 오디오 피처·가사 데이터셋 (Kaggle)</td><td>곡 목록, 오디오 피처, 가사, 장르</td><td>—</td></tr>
  <tr><td><a href="https://www.last.fm/api">Last.fm API</a></td><td>곡 설명</td><td>Last.fm API 이용 약관</td></tr>
  <tr><td><a href="https://developers.deezer.com/">Deezer API</a>, <a href="https://musicbrainz.org/">MusicBrainz</a> + <a href="https://coverartarchive.org/">Cover Art Archive</a></td><td>앨범 표지</td><td>각 서비스 약관</td></tr>
  <tr><td rowspan="4">책</td><td><a href="https://www.kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books">Amazon Kindle Books Dataset 2023 (130K Books)</a> (Kaggle, asaniczka)</td><td>작품 목록, 카테고리, 표지 URL</td><td>ODC-By</td></tr>
  <tr><td><a href="https://zenodo.org/records/4265096">Best Books Ever Dataset</a> (Casanova Lozano & Costa Planells, Zenodo, DOI 10.5281/zenodo.4265096)</td><td>작품 목록, 블러브, 장르, 표지 URL</td><td>CC BY-NC 4.0</td></tr>
  <tr><td><a href="https://www.kaggle.com/datasets/jdobrow/57000-books-with-metadata-and-blurbs">57,000+ Books with Metadata and Blurbs</a> (Kaggle, jdobrow)</td><td>작품 목록, 블러브</td><td>명시 없음</td></tr>
  <tr><td><a href="https://openlibrary.org/developers/api">Open Library API</a></td><td>표지, 카테고리, 평점</td><td>Open Library 이용 약관</td></tr>
</table>

책 데이터는 세 데이터셋에서 블러브가 있는 행을 합치고 중복을 제거해 만들었습니다(우선순위 Kindle > Goodreads > BX, `scripts/merge_book_sources.py`).

TMDB 이용 약관에 따른 고지: This product uses the TMDB API but is not endorsed or certified by TMDB.

모델에 들어가는 작품 텍스트는 LLM이 원본 정보(줄거리, 블러브, 오디오 피처 등)로 합성한 분위기 설명(`description_synth`)입니다. 세 도메인이 같은 형식을 씁니다. 음악의 오디오 피처(energy, valence 등)는 "차분한", "강렬한" 같은 표현으로 바뀌어 설명에 들어갑니다.

작품마다 학습 쿼리가 3개 있습니다. Qwen2.5-VL이 표지와 설명을 보고 만든 짧은 검색어입니다. `scripts/prepare_dataset.py`가 쿼리와 작품을 짝지어 `data/train_dataset.csv`를 만듭니다.

원본 데이터, 이미지, 학습 데이터, 모델 파일은 저장소에 없습니다(`.gitignore`).

---

## 3. 모델

듀얼 인코더입니다. 작품과 검색어를 각각의 인코더로 같은 768차원 공간에 놓고, 코사인 유사도로 가까운 작품을 찾습니다. 작품 벡터는 미리 저장해 둡니다. 실제 검색 시에는 검색어 하나만 인코딩하며, CPU에서 수십 ms 안에 끝납니다.

### 모델 전체 구조
<img width="1200" height="664" alt="스크린샷 2026-11-16 오후 2 27 24" src="https://github.com/user-attachments/assets/a6f8f180-1b27-4555-9d75-53f204bf75ca" />


```
작품 쪽
  분위기 설명 → SBERT 인코더(all-mpnet-base-v2, LoRA) + MLP → z_text
  표지 이미지 → CLIP ViT-L/14 이미지 인코더(고정) + MLP → z_image
  z_text + z_image → ContentBlock → z_content (768, L2 정규화)

검색어 쪽
  검색어 → CLIP ViT-L/14 텍스트 인코더(고정) + MLP → z_query (768)

점수 = cos(z_content, z_query)
```

학습은 두 단계입니다(`src/training/trainer.py`).


- **1단계(대조 학습):** 설명·이미지·검색어의 MLP 헤드와 SBERT 인코더의 LoRA 어댑터를 학습합니다. 손실은 세 쌍(설명–검색어, 설명–이미지, 이미지–검색어)의 InfoNCE 합입니다.
- **2단계:** `ContentBlock`만 학습합니다. 작품 벡터 `z_content`와 검색어 벡터 `z_query` 사이의 InfoNCE를 씁니다. 검색에 쓰는 벡터는 `z_content`입니다.
- CLIP 인코더(이미지·텍스트)는 고정이고, SBERT 인코더는 LoRA 어댑터만 학습합니다.

서비스 체크포인트: `best_stage2_qlora_off` (`model_version`: `trained_model.pt@1786697109`)

### 추론(쿼리 인코딩) 모델

<img width="769" height="248" alt="스크린샷 2026-11-16 오후 2 27 52" src="https://github.com/user-attachments/assets/216883ea-b2ca-497f-9069-e9757c5cda3d" />

- 추론 모델 그림 : 사용자 검색어(쿼리)를 벡터로 전환합니다.

---

## 4. 한국어 검색어 처리

모델은 영어로 학습됐고 CLIP 토크나이저도 영어 전용입니다. 한국어 검색어는 서버에서 영어로 번역한 뒤 인코딩합니다(`src/api/translation.py`). 번역 순서는 다음과 같습니다.

1. **번역 사전:** 검색어 전체가 사전(`src/api/ko_en_dict.json`, 520개 항목)의 장르·분위기·상황 어휘로 나뉘면 사전 번역을 이어 붙입니다(로파이 비 오는 날 → *lo-fi rainy day*). 사전에 없는 단어가 있으면 다음 단계로 넘어갑니다.
2. **Gemma 3 4B:** 문장형 검색어를 vLLM으로 띄운 Gemma 3 4B가 번역합니다. `LLM_TRANSLATE_URL`이 설정된 경우에만 동작합니다.
3. **opus-mt-ko-en:** Gemma가 없거나 제한 시간 안에 응답하지 않으면 opus-mt로 번역합니다.

번역이 모두 실패하면 원문을 그대로 인코딩합니다.

검색어가 지목한 도메인도 판별합니다(`src/api/domain_intent.py`). "영화", "노래", "소설" 같은 작품 종류 명사가 있으면 `domain`이 `movie`·`music`·`book` 중 하나가 되고, 없거나 여러 도메인이면 `all`입니다. 장르(재즈, 공포)는 단서로 쓰지 않습니다. 세부 규칙은 [`docs/api_for_backend.md`](docs/api_for_backend.md)에 있습니다.

---

## 5. API

FastAPI 서버입니다. 전체 엔드포인트 목록: `http://<서버>:8000/docs`. 요청·응답 형식, 전달 번들 형식, Qdrant 적재 방법은 [`docs/api_for_backend.md`](docs/api_for_backend.md)에 있습니다.

| 메서드 · 경로 | 용도 |
|---|---|
| `POST /api/v1/search/vector` | 검색어 한 건 → 768차원 벡터 (백엔드 사용 경로) |
| `GET /api/v1/search/vector?keyword=…` | 위와 동일, GET |
| `POST /api/v1/embeddings` | 여러 검색어 → 벡터 (정리 예정) |
| `POST /api/v1/recommend` | 서버 내부 검색 결과. 백엔드 검색 결과 대조용 |
| `GET /api/v1/item/{domain}/{item_id}` | 작품 메타데이터 |
| `GET /api/v1/ping?n=42` | 통신 확인 (모델 무관) |
| `GET /api/v1/health` | 모델·인덱스 적재 상태 |

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

벡터는 L2 정규화되어 있어 내적이 코사인 유사도와 같습니다. `translated_text`는 번역이 일어난 경우에만 값이 있고, 그 외에는 `null`입니다.

- CLIP 입력 한도는 77토큰이며, 넘는 부분은 잘립니다.

---

## 6. 로컬 실행

### 환경

```bash
git clone https://github.com/VibeCrates/VibeCrates_Recommendation.git
cd VibeCrates_Recommendation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt          # 학습·실험 포함 전체
# pip install -r requirements-serve.txt  # API 서버만
```

### 모델과 인덱스

서버 실행에는 저장소 밖의 파일 두 가지가 쓰입니다.
- 체크포인트: `models/trained_model.pt` (약 2.2GB)
- 작품 벡터 인덱스: `indexes/{movie,music,book}_embeddings.pt`, `indexes/{movie,music,book}_meta.parquet`

두 파일은 같은 학습에서 나온 것이어야 합니다. 파일이 없어도 서버는 뜨고 `/ping`은 동작하며, 벡터 경로는 503을 반환합니다.

### 서버 실행

```bash
./scripts/serve.sh            # 0.0.0.0:8000
PORT=9000 ./scripts/serve.sh  # 포트 변경
```

`serve.sh`는 `venv`의 uvicorn으로 서버를 띄우고 LAN·Tailscale 주소를 출력합니다. 기동에 약 13초가 걸리며, 그동안 연결은 거부됩니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `models/trained_model.pt` | 체크포인트 경로 |
| `INDEX_DIR` | `indexes` | 작품 벡터 인덱스 폴더 |
| `DEVICE` | `cuda` 또는 `cpu` | 추론 장치 |
| `TRANSLATE_QUERIES` | `1` | `0`이면 한국어 번역 끔 |
| `LLM_TRANSLATE_URL` | (없음) | Gemma 번역 서버 주소(OpenAI 호환, 예: `http://127.0.0.1:8001/v1`). 없으면 사전 → opus-mt |
| `LLM_TRANSLATE_TIMEOUT` | `1.0` | Gemma 응답 대기 시간(초) |

### 테스트

```bash
pytest tests/        # API, 데이터, 도메인 판별, 번역 (69개)
```

---

## 7. 학습과 평가

### 학습 파이프라인

```
canonical CSV
  → generate_item_descriptions.py / generate_music_descriptions.py   분위기 설명 합성 (vLLM)
  → generate_queries.py                                             작품별 학습 쿼리 3개 생성 (Qwen2.5-VL)
  → prepare_dataset.py                                              data/train_dataset.csv
  → extract_image_embeddings.py                                     표지 이미지 임베딩 계산
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

`--sample-one-query`는 매 스텝 작품의 학습 쿼리 3개 중 하나를 무작위로 씁니다(서비스 모델 설정). 그 밖의 실험용 옵션(`--query-lora`, `--title-dropout`, `--domain-balanced-batches` 등)은 `python scripts/train.py --help`에 있습니다.

### 평가

두 모델(체크포인트 + 인덱스)을 같은 평가 검색어로 돌려 비교합니다.

- **쌍대 판정:** `scripts/judge_pairwise.py`가 두 모델의 결과를 LLM 판정기(기본 Qwen2.5-VL-7B)에 나란히 보여 주고 더 나은 쪽을 고르게 합니다. 순서를 바꿔 두 번 묻고, 승률과 p값을 냅니다.
- **사람 라벨:** `scripts/build_label_sheet.py`로 라벨 시트를 만들고, `scripts/score_label_sheet.py`로 집계합니다.

실험 산출물은 `experiments/`, 실험 보고는 `docs/`에 있습니다.

---

## 8. 개발 과정

| 기간 (2026) | 단계 | 내용 |
|---|---|---|
| 05-06 | 모델 설계 | 듀얼 인코더 구조(TextBlock, ImageBlock, QueryBlock, ContentBlock)와 2단계 학습 손실 설계 |
| 05-20 ~ 05-22 | 데이터 탐색 | 영화·음악·책 원본 CSV 조사, IMDb ID 정리, 이미지 URL 유효성 검사 |
| 05-20 ~ 05-31 | 데이터 수집 | 책 설명(Google Books), 음악 설명(Last.fm), 음악 표지(MusicBrainz) 수집, 이미지 다운로드 파이프라인, 책 데이터를 Kindle 2023으로 교체 |
| 05-29 | 구현 | 전처리, 평가, 추론, 검색어 기반 API 구현 |
| 05-31 ~ 06-15 | 학습 데이터 생성 | Qwen2.5-VL-7B로 작품별 학습 쿼리 생성, 쿼리 언어를 한국어에서 영어로 전환 |
| 06-16 ~ 06-18 | 학습 | 학습 데이터셋 구성, CLIP 이미지 임베딩 사전 계산, 첫 학습, 조기 종료·체크포인트·인덱스 생성 |
| 06-18 | 평가 | 한국어·영어 쿼리 추천 품질 평가 |
| 07-06 ~ 07-16 | 데이터 처리 | canonical CSV 정리, TMDB 영화 메타데이터 병합, 음악 텍스트 진단, 분위기 설명 합성 시범 |
| 08-03 ~ 08-07 | 데이터 처리 | 책 데이터를 세 소스로 재구축, GPU 서버에서 vLLM으로 세 도메인 분위기 설명 합성, 학습 쿼리 재생성, 학습 전 QA |
| 08-13 ~ 08-18 | 모델 개선·학습 | 2단계 손실을 InfoNCE로 교체, 쿼리 1개 샘플링, CLIP 동결 수정, 제목·필드명 dropout 실험 |
| 08-14 ~ 08-31 | 서빙 | 백엔드 연동, 쿼리 벡터 엔드포인트, 작품 벡터 전달 번들, 한국어 검색어 번역, 음악 표지 추가 수집(Deezer) |
| 09-01 ~ 09-09 | 데이터 처리·평가 | 중복 작품 제거, LLM 판정기 쌍대 비교 도입, 도메인 상한 규칙 |
| 09-11 ~ 09-23 | 평가·학습 | 사람 라벨 시트, 도메인 균형 배치·필드명 통일 실험, 인기작 추출, 표지 없는 작품을 번들에서 제외 |
| 10-08 ~ 10-15 | 평가·서빙 | 판정기 검증, energy 축 측정, 학습 쿼리 벡터 번들, 한국어 번역 사전, 도메인 판별 |
| 10-23 ~ 10-31 | 서빙·분석 | 학교 GPU 서버로 이전, Gemma 3 4B 문장 번역, 재학습(newbase), 모델 문제 목록 정리, 부정 표현 측정 |

---

## 9. 알려진 한계

- **부정 표현:** "무섭지 않은 판타지 소설"이 "무서운 판타지 소설"과 가까운 벡터가 됩니다. 고정된 CLIP 텍스트 인코더의 한계입니다.
- **은유적 문장:** 은유적인 검색어는 번역문의 의미가 원문의 의미와 어긋나는 경우가 있습니다.

원인과 현재 상태는 [`docs/model_issues.md`](docs/model_issues.md)에 있습니다.

---

## 10. 저장소 구조

```
VibeCrates_Recommendation/
├── src/
│   ├── api/                 # FastAPI 서버: 라우트, 모델 적재, 번역, 도메인 판별
│   ├── models/              # 듀얼 인코더 (TextBlock, ImageBlock, QueryBlock, ContentBlock)
│   ├── training/            # 2단계 트레이너, InfoNCE 손실, 학습 기록
│   └── data/                # 데이터셋, 로더, 전처리
├── scripts/                 # 데이터 수집·합성, 학습, 인덱스·번들 생성, 평가, 서버 실행
│   └── vibe_crate/          # 메인·보드 화면 디자인 시안 (정적 HTML)
├── data/canonical/          # 도메인별 canonical CSV (원본·이미지·학습 데이터는 gitignore)
├── experiments/             # 실험 결과와 사람 라벨 시트
├── docs/                    # 설계 문서, API 안내, 작업 기록, 세션 보고
├── reports/                 # 초기(1~20회차) 세션 보고
├── tests/                   # pytest
├── Dockerfile, docker/      # 서빙 컨테이너
├── requirements.txt         # 학습·실험 포함 전체 의존성
└── requirements-serve.txt   # API 서버 전용 의존성
```

---

## 11. 문서 안내

| 문서 | 내용 |
|---|---|
| [`docs/api_for_backend.md`](docs/api_for_backend.md) | 백엔드 연동: 엔드포인트, 전달 번들 형식, Qdrant 적재, 도메인 상한 규칙 |
| [`docs/model_issues.md`](docs/model_issues.md) | 모델 문제 목록 (해결 / 서비스 무관 / 남김 / 미확인) |
| [`docs/worklog_2026-07_to_09.md`](docs/worklog_2026-07_to_09.md) | 7~9월 작업 기록 |
| [`docs/worklist_2026-09-03.md`](docs/worklist_2026-09-03.md) | 9월 기준 할 일 목록 |
| `docs/report_session_*.md` | 9~10월 세션 보고 (최근: [`2026-10-31`](docs/report_session_2026-10-31.md)) |
| [`docs/design_poet_style_alignment.md`](docs/design_poet_style_alignment.md), [`docs/design_audio_feature_integration.md`](docs/design_audio_feature_integration.md) | 설계 노트 |
| [`docs/report_judge_validation_20261009.md`](docs/report_judge_validation_20261009.md) | LLM 판정기 검증 결과 |

---

## 라이선스

MIT License

## 연락처

maxsky9018@gmail.com
