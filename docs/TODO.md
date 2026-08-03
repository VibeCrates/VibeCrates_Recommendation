# TODO — 도서 데이터셋 재구축 및 3도메인 텍스트 타입 통일

최종 갱신: 2026-08-03 (세션 17) · 선행 문맥: `reports/report_session_16.txt`,
`docs/design_poet_style_alignment.md`, `docs/design_audio_feature_integration.md`

## 배경 — 왜 이 작업을 하는가

두 개의 진단이 하나의 작업으로 수렴했다.

1. **세션 16 진단 (A)** — content_text의 텍스트 "타입"이 도메인마다 다르다. movie/book은
   3인칭 설명인데 music은 약 60%가 1인칭 가사 원문. SBERT가 이를 서로 다른 의미공간
   영역에 임베딩해 크로스도메인 추천이 어긋난다.
2. **poet 스타일 전역 실패** — 6월 평가에서 poet 최하(avg 0.86, 실패율 52.5%). 원인 3겹 중
   하나가 "콘텐츠 텍스트에 mood 어휘가 없어 추상 쿼리가 매칭할 표적이 없음".

공통 해법이 **3도메인 vibe description 통일**이고, 그러려면 각 도메인에 mood를 뽑아낼
소스가 있어야 한다. 그런데 **book은 블러브 커버리지가 15.0%(19,971/133,102)뿐**이라
나머지 85%는 제목·저자·카테고리·표지밖에 없었다. 이것이 이번 세션의 출발점.

---

## 이번 세션(17) 완료 사항

- [x] `scripts/generate_item_descriptions.py` 작성 — movie/book vibe description 합성기.
      `generate_music_descriptions.py`와 동일 계약(2~3문장 3인칭 mood 설명, film-synopsis
      register). dry-run 검증 완료.
      - movie: `text` 컬럼의 `줄거리 | 태그라인 | 키워드` 구조 파싱, 키워드를 별도 mood 신호
        블록으로 분리 (348건 검증, 오분류 0)
      - book: 블러브가 1인칭/광고문구인 경우가 흔해 3인칭 재서술을 명시 요구
      - 이미지 유무에 따라 grounding 문구 분기 (없는 포스터를 근거로 삼으라는 환각 유도 제거)
      - `description_synth_basis` 컬럼으로 어떤 신호로 합성했는지 기록
- [x] 외부 도서 데이터셋 3종 확보·검증 (아래 "데이터 소스 실측" 참조)
- [x] Open Library 커버 API 실측 — BX ISBN 기준 **히트율 94.0%**
- [x] Kindle 아마존 imgUrl 생존 확인 — 표본 80/80 (100%)

## 데이터 소스 실측 (2026-08-03)

| 소스 | 권수 | 블러브 | 커버 | 장르 | ID | 라이선스 |
|---|---:|---|---|---|---|---|
| Kindle (기존 canonical) | 133,102 | **15.0%** | 100% (생존확인) | 100% | ASIN | 원 데이터셋 조건 |
| Goodreads (Zenodo 4265096) | 52,478 | 97.5% | 98.8% | 91.2% | bookId | **CC BY-NC 4.0** |
| BX blurbs (Kaggle jdobrow) | 57,510 | 100% | **없음** | **없음** | ISBN(온전) | Unknown |

**주의 — Kaggle의 Goodreads 재업로드본(`arnabchaki/goodreads-best-books-ever`)은 쓰지 말 것.**
업로더가 Excel로 저장하면서 ISBN이 지수표기(`9.78044E+12`)로 파괴됐고 `coverImg`/`bookId`
컬럼이 누락된 23컬럼 버전이다. 반드시 Zenodo 원본(25컬럼)을 쓴다.
→ `data/raw/books_zenodo.csv` (73,839,808 bytes)

**세 소스는 모집단이 거의 겹치지 않는다** (Kindle 롱테일 / Goodreads 유명작 큐레이션 /
BX 2004년 종이책 카탈로그). 그래서 "기존 행의 빈 블러브를 채우는" 전략은 실패한다 —
두 외부 소스를 합쳐도 결손 113,131건 중 9,023건(8.0%)밖에 못 채운다.
가치는 "채우기"가 아니라 **블러브 온전한 행들의 합집합**에 있다.

---

## 목표 데이터셋 — 110,839권, 블러브 100%

블러브 온전한 행만 취하고 title+author 정규화로 중복 제거
(우선순위: Kindle > Goodreads > BX — 쿼리·이미지·장르 보유 순).

| 소스 | 권수 | 이미지 | 장르 | 쿼리 |
|---|---:|---|---|---|
| Kindle | 19,771 | 있음 | 있음 | 18,358 재사용 |
| Goodreads | 48,395 | 99.2% | 있음 | 생성 필요 |
| BX | 42,673 | **크롤링 필요** | **없음** | 생성 필요 |

현재 133,102권/블러브 15% → **110,839권/블러브 100%**.
도메인 균형도 개선된다 (movie 40K / music 40K / book 133K → 111K).

---

## 작업 단계

### 0단계 — 병합 스크립트 `scripts/merge_book_sources.py` (로컬, GPU 불필요)
- [ ] 세 소스를 canonical 스키마로 정규화
- [ ] 블러브 온전 행만 필터 → 소스 내/소스 간 dedup (Goodreads 내부 중복 758건 존재)
- [ ] ID 프리픽스 부여 (`kdl_`/`gr_`/`bx_`)로 충돌 방지
- [ ] Goodreads `genres` 리스트 문자열 평탄화, `publishDate`(`09/14/08`) 연도 정규화
- [ ] BX는 ISBN으로 Open Library URL을 미리 조립해 `imgUrl`에 채움
- [ ] 출력: `data/canonical/book_canonical_v2.csv`

### 1단계 — 이미지 확보 (무료, 약 4시간, 3.2GB)
- [ ] `download_images.py`에 Open Library 경로 추가
- [ ] **`?default=false` 필수** — 없으면 커버 없는 책에 43바이트 빈 이미지가 저장되어
      해당 책들이 전부 동일한 z_image를 갖게 된다 (조용히 망가지는 버그)
- [ ] 크기 하한(1KB 미만 폐기) 검증 추가
- [ ] 재개 가능한 캐시 (2.5시간 작업이므로 필수)

실측 속도: Open Library 4.8 req/s (worker 6, 레이트리밋 에러 0) / Amazon·Goodreads CDN 30~50 req/s

### 2단계 — description_synth 합성 (GPU, 110,839건)
- [ ] `generate_item_descriptions.py --domain book` 실행 (스크립트는 작성 완료)
- [ ] BX 42,673권은 장르가 없어 프롬프트에서 category 줄이 빠짐 — 동작 확인 필요
- [ ] `--limit 200`으로 품질·속도 먼저 측정

### 3단계 — 쿼리 생성 (GPU, 92,481건)
- [ ] `generate_queries.py --domain book`
- [ ] **순서 중요**: description_synth를 먼저 만들고 그걸 기반으로 쿼리 생성해야
      세션 16 진단 (D) 라벨 오염이 해소된다

### 4단계 — 파이프라인 반영 및 학습
- [ ] `_build_content_text` 3도메인 모두 `description_synth` 우선으로 통일,
      music의 raw lyrics 직접 주입(`preprocessing.py:178-180`) 제거
- [ ] `prepare_dataset.py` 재실행
- [ ] 학습 → 6월 eval(`experiments/eval_lang_20260618_report.txt`) 대비 poet/atmosphere 비교

### 세션 16에서 이월된 별도 작업 (미착수)
- [ ] `generate_music_descriptions.py` GPU 실제 실행 → music `description_synth` 채우기
- [ ] QueryBlock LoRA 또는 SBERT 병렬 경로 (poet 개선안 1) — `recommender.py:109` 동결 해제
- [ ] 학습 시 아이템당 쿼리 1개 확률 샘플링, mean-pool은 추론 전용
      (poet 개선안 2) — `recommender.py:191`
- [ ] 오디오 피처 경로 B(AudioBlock) 착수 여부 결정

---

## GPU 비용 추정

| 작업 | 건수 |
|---|---:|
| 쿼리 생성 | 92,481 |
| description_synth (book) | 110,839 |
| description_synth (music, 이월) | 40,036 |
| description_synth (movie, 이월) | 40,109 |

책 관련만 **203,320회**, 3도메인 전부면 **283,465회**.
속도 기록이 로그에 없어 정확한 시간 산출 불가 — `--limit 200`으로 측정할 것.
참고로 1 it/s면 56시간, 2 it/s면 28시간, 3 it/s면 19시간 (책 관련 기준).

## 결정 대기 항목

- **BX 장르 결손 42,673권(38.5%)** — (a) 그대로 두기 (b) Open Library API로 subject를
  이미지 크롤링과 동시에 수집 (c) BX 제외. **(b) 권장** — 어차피 ISBN으로 API를 때리므로
  추가 시간이 거의 없다.
- **라이선스 혼재** — Goodreads CC BY-NC 4.0(비상업), BX Unknown. 연구·학습용은 문제없으나
  로드맵상 "런칭" 시점에 Goodreads 부분(43.7%)이 걸림돌. 상업화 계획 확정 시 재검토.
- **이미지 없는 책 약 2,936권(2.6%)** — BX 미확보 2,560 + Goodreads 결손 376. 유지할지 제외할지.
