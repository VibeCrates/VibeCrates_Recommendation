# 다음 사이클 백로그

작성: 2026-08-07 (세션 19) · 선행 문맥: `reports/report_session_19.txt`,
`docs/design_poet_style_alignment.md`, `docs/design_audio_feature_integration.md`

이번 사이클(개선안 3 = 3도메인 description 통일 + 3-only 학습)을 돌리는 과정에서
"다음에 하자"고 미룬 것들을 모은 문서다. 미룬 이유가 항목마다 다르므로 함께 적는다 —
대부분은 **이번 사이클의 측정을 흐리지 않기 위해** 미뤘고, 지금 하면 개선안 3의 효과
크기를 잴 수 없게 된다.

우선순위는 A(다음 사이클에 반드시) / B(하면 좋음) / C(조건부·별도 트랙)로 나눈다.

---

## A. 다음 사이클에 반드시

### A1. 개선안 2 — 페르소나 평균 제거
- **무엇**: 학습 시 아이템당 쿼리 3개를 평균 내지 말고 매 스텝 1개를 무작위로 뽑는다.
- **왜**: `encode_query`가 Poet/Space/Philosopher 3개를 mean-pooling 하므로(`src/models/
  recommender.py:191`) 콘텐츠는 세 방향의 타협 centroid에 정렬된다. 개별 poet 쿼리가
  한 번도 검색 앵커로 학습되지 않아, 추론 시 순수 poet 쿼리가 centroid에서 벗어난다.
  6월 평가에서 poet이 전역 최하(0.86)였던 원인 3겹 중 하나다.
- **어디**: `src/data/dataset.py:83`에서 `random.choice`로 1개 샘플. mean-pool은
  추론(멀티 쿼리 입력) 전용으로 남긴다. 설계 노트의 "방법 B" — false negative가 없고
  배치·비용이 그대로다.
- **비용**: 구현 최소. 재학습 필요.

### A2. 개선안 1 — 쿼리 인코더 대칭화
- **무엇**: `QueryBlock`의 CLIP 텍스트 인코더 동결을 풀고 LoRA를 붙이거나, SBERT 병렬
  경로를 추가한다.
- **왜**: 현재 `src/models/recommender.py:109`에서 완전 동결이라 은유·추상 언어에 적응할
  capacity가 없다. 콘텐츠 쪽 `TextBlock`은 SBERT+LoRA라 **인코더가 비대칭**이다.
  SBERT 경로를 넣으면 콘텐츠와 같은 임베딩 공간이 생겨 추상 쿼리 정렬이 크게 개선된다.
- **비용**: 셋 중 가장 큼. A1 결과를 본 뒤에 착수한다(설계 노트: 셋은 상보적이며
  하나만 하면 절반 효과).

### A3. `build_synopsis` / `_build_content_text` 통합
- **무엇**: 같은 로직의 복제본 두 개를 `src` 쪽 하나로 합치고 `scripts`가 import 한다.
  자르기 한도와 포함 필드를 **명시적 파라미터**로 남겨 다시 갈라지지 않게 한다.
- **왜**: 이미 한 번 갈라졌다. 커밋 4bcc809(7/15)가 `_build_content_text`에만
  Director/Cast/Release Date를 추가해, 라벨을 만든 Qwen이 감독·배우를 본 적 없는
  상태가 됐다. 폴백 함수(`synth_text` / `_synth_text`)까지 이름만 다르게 복제돼 있다.
- **주의**: 절단을 없애는 게 아니라 파라미터화한다. 프롬프트 토큰 예산이라는 실제
  이유가 있으므로, 사고가 아니라 결정으로 남아야 한다.

### A4. 절단선 상향 + 감독/배우 라벨 쪽 복구
- **무엇**: 600자 → 1,200자로 올리고, `build_synopsis`에 Director/Cast를 넣은 뒤
  쿼리를 재생성한다(= 세션 19에서 보류한 "선택지 B").
- **왜**: 결정 A로 입력을 라벨 기준에 맞추면서 movie 30%의 description 꼬리와
  감독·배우 정보를 버렸다. 토큰 실측상 여유가 충분하다 — content_text가 평균 128,
  최대 244토큰인데 SBERT 한도는 384다(`recommender.py:60`은 512로 자른다).
- **비용**: 쿼리 전량 재생성 GPU 2~3시간. A3와 같이 하면 한 번에 끝난다.
- **의존**: A3 이후.

### A5. 메모리 최적화 — 배치 키우기
- **무엇**: gradient checkpointing 또는 SDPA/flash attention을 켜서 배치를 256까지 올린다.
- **왜**: 배치 256·128 모두 OOM이라 64로 학습했다. 원인은 누수가 아니라 mpnet이
  어텐션 행렬(배치×헤드12×길이²)을 12개 층 모두 역전파용으로 보관하는 것이고, 길이
  편차 때문에 128은 29스텝째에 터졌다. InfoNCE는 배치가 클수록 negative가 많아 유리하다.
- **선택지**: (a) gradient checkpointing — 메모리 1/3 이하, 30% 느려짐.
  (b) SDPA — 어텐션 행렬을 만들지 않아 메모리·속도 둘 다 이득. mpnet이 현재
  transformers 버전에서 지원하는지 확인이 먼저다. **(b) 우선 검토.**
- **의존**: 어차피 A1으로 재학습하므로 그때 함께.

### A6. 중복 ID 제거
- **무엇**: movie 593행(1.48%) / music 354행(0.88%)의 중복 ID를 정리한다. book은 0건.
- **왜**: 같은 아이템이 두 행으로 들어가면 InfoNCE 배치 안에서 **false negative**가 된다
  — 동일한 텍스트 둘을 서로 밀어내라고 가르치는 셈이다. A1의 false negative 논의와 같은 축.

### A7. `description_synth` 커버리지 가드
- **무엇**: `generate_queries.py` 시작 시 `description_synth` 컬럼 존재와 채움률을 확인하고
  임계 미만이면 중단한다.
- **왜**: 8/5 사고의 재발 방지. 합성이 크래시해 CSV에 컬럼이 안 쓰인 상태에서 쿼리가
  돌았고, 폴백이 "아직 합성 안 됨"과 "합성했는데 CSV에 안 실림"을 구별하지 못해
  예외도 로그도 없이 23,147건이 원문 기반으로 오염됐다.

---

## B. 하면 좋음

### B1. music 이미지 커버리지 66.8% → 개선
- 26,746/40,036장뿐이라 **13,290행이 시각 신호 없이(0 벡터) 학습**된다. movie 포스터를
  TMDB로 복구했던 것과 같은 작업이 필요하다. 이번 사이클의 music 점수를 해석할 때
  반드시 이 사실과 함께 봐야 한다.

### B2. QA 지표 오탐 수정
- `scripts/qa_synth_outputs.py`의 두 정규식이 오탐이다:
  - "리뷰 인용"(music 0.404) → 실제로는 곡 제목 인용(`"Best Friend," from his album...`)
  - "독자 호명"(music 0.029) → 제목 속 your(`Put Your Records On`)
- 곡 제목/작품명과 겹치는 부분을 먼저 제거한 뒤 검사하도록 고친다.

### B3. `prepare_dataset.py` 경고 문구 정리
- `scripts/prepare_dataset.py:124`가 아직 "will use HTTP fallback during training"이라고
  찍는다. 사실이 아니다 — `dataset.py`가 이미 0 벡터로 처리한다(커밋 a1edee5).

### B4. description 길이 A/B
- 현재 계약은 2~3문장(평균 128토큰)이고 모델 한도(384)의 3분의 1만 쓴다. 4~6문장으로
  늘리면 고유 디테일이 늘지만 환각 여지도 늘고 mood 신호가 희석된다. **근거 없이 바꾸면
  변수만 늘어나므로**, 이번 baseline 결과를 본 뒤 A/B로 판단한다.

### B5. music 형식 계약 위반
- 실제 위반은 두 가지뿐이다: 1문장으로 쓴 것 7,403건(18.5%), 발매/디스코그래피 정보
  680건(1.7%). 내용은 계약대로이므로 이번 사이클에서는 재생성하지 않기로 했다.
  다음에 music을 재생성할 일이 생기면 프롬프트에 문장 수를 더 강하게 못박는다.
- 부수 관측: music description 중 중국어로 생성된 것이 최소 1건 있다. 언어 일관성
  점검을 재생성 시 같이 넣는다.

---

## C. 조건부 · 별도 트랙

### C1. 오디오 피처 모델 직접 투입 (세션 16 진단 C)
- `create_item_features`(danceability/energy/valence 등 9개)가 `preprocessing.py:117`에
  정의만 되어 있고 **호출 건수는 여전히 0**이다. 현재는 `verbalize_audio`가 피처를
  자연어로 바꿔 description을 경유해 텍스트로만 간접 반영된다.
- 모델이 수치를 직접 보는 경로(AudioBlock)는 `docs/design_audio_feature_integration.md`의
  별도 트랙이며, poet 개선 사이클과 독립적이다.

### C2. 라이선스 재검토
- Goodreads 부분(CC BY-NC 4.0, book의 43.7%)과 BX(Unknown)는 연구·학습용은 무방하나
  런칭 시점에 걸린다. 상업화 계획 확정 시 재검토. (세션 17에서 이월)

---

## 이번 사이클에 남은 것 (다음 사이클 아님)

- 7단계 — 학습 완료 후 `eval_lang` 비교. 기준은 `experiments/eval_lang_20260618_report.txt`
  (poet 평균 0.86 / 실패율 52.5%).
- 8단계 — 쿼리·콘텐츠 어휘 겹침의 **유형별 분해**. QA에서 신규 쿼리의 겹침이 오염본보다
  높게 나왔는데(0.243 vs 0.179), 겹치는 단어가 mood 형용사면 의도된 효과이고
  고유명사·줄거리 단어면 지름길 위험이다. 현재 지표로는 구별되지 않는다.

---

## D. 백엔드 연동 — 데이터 계약 (2026-08-18 추가)

7월에 정한 프론트 데이터 계약(`scripts/vibe_crate/SPEC.md:108`, `README.md:49`)과
현재 추천 API 응답(`src/api/schemas.py:16`)이 어긋나 있다. 오늘은 `/ping` 통신 확인만
하고 넘어가므로, 아래는 그다음 작업으로 미룬다.

계약의 아이템은 `{id, title, domain, year, image, url}` 평면 구조다. 여기서
**`image`는 표지 그림**(로컬 파일 `/Users/hyun/images/{domain}/{id}.jpg`)이고,
**`url`은 상세 페이지 링크**(hover 시 클릭)로 서로 다른 필드다.

### D1. 응답 스키마를 계약 모양으로 평탄화
- **무엇**: `RecommendationItem`의 `extra` 중첩을 없애고 `year`·`image`·`url`을 최상위
  필드로 낸다. `item_id` → `id`로 이름을 맞춘다. `score`는 계약에 없지만 백엔드
  정렬·디버깅용으로 남길지 정한다.
- **왜**: 현재는 도메인마다 `extra.poster` / `extra.img` / `extra.image`로 키가 갈리고,
  그 값이 원본 CSV의 외부 URL이다. 계약이 **쓰지 말라고 명시한** 바로 그 컬럼이다
  (Amazon CDN 플레이스홀더·IMDB 깨짐 이력, `README.md:78`).
- **어디**: `src/api/schemas.py:16`, `src/data/meta.py:4`.

### D2. `year` / `url` 채우기 — movie·music
- movie: `release_date`에서 연도 추출(결측 1.3%), `Imdb Link` 컬럼 그대로(결측 0%).
- music: `year` 컬럼(결측 10.2%), URL은 `https://open.spotify.com/track/{id}` 조립.
  주의 — music CSV의 `preview` 컬럼은 30초 미리듣기 오디오라 계약의 `url`이 아니다.

### D3. book 상세 링크 규칙 신설 — **결정 필요**
- **문제**: 학습·서빙이 쓰는 `book_canonical_v2.csv`에는 `productURL` 컬럼이 없다.
  goodreads·bx 출처 90,844권(82%)은 애초에 Amazon 상품이 아니라 원본에 상품 링크가
  존재하지 않는다. 옛 `book_canonical.csv`에서 끌어오려 해도 asin이 0% 겹치고(v2는
  `kdl_`/`gr_`/`bx_` 접두사 합성 id), 옛 CSV에 isbn 컬럼이 없어 ISBN 조인도 불가능하다.
- **선택지**: (a) isbn 있는 75.3%는 ISBN 기반 링크(Goodreads/Open Library 검색), 나머지는
  `url: null` (b) 전량 null (c) 링크 재수집.

### D4. book 표지 매칭 복구 — **결정 필요**
- **문제**: 로컬 표지 133,082장이 옛 asin 이름인데 v2 id는 접두사가 붙어 0% 매칭이다.
  접두사를 떼도 kindle 출처만 맞아 **17.9%**만 표지가 있다. 콜라주 화면이 책으로는
  거의 채워지지 않는다. (세션 17에서 Open Library 커버 API 히트율 94.0%를 실측해 뒀다 —
  `docs/TODO.md:38`.)
- **선택지**: (a) Open Library로 BX/Goodreads 표지 재수집 (b) 파일명 규칙 통일 후 재매핑
  (c) 표지 없는 책은 콜라주 후보에서 제외.
- **주의**: D3(링크)와 D4(표지)는 원인이 다른 별개 문제다. 82%와 17.9%가 같은 출처
  분포에서 나와 숫자가 비슷할 뿐이다.

### D5. 이미지 서빙 주체 결정 — **백엔드 협의**
- 계약의 `image`는 `images/movie/114709.jpg` 상대 경로인데, 추천 서버는 정적 파일을
  전혀 서빙하지 않고 `IMAGE_DIR` 기본값도 존재하지 않는 `data/images`를 가리킨다
  (`src/api/dependencies.py:23`, 실제 이미지는 `/Users/hyun/images/`).
- 추천 서버가 직접 서빙할지, 백엔드가 복사해 갈지, CDN에 올릴지 정해야 한다.
- 참고 커버리지: movie 73.0%, music 75.2%, book 17.9%.

### D6. 통합 검색 500 버그
- `RecommendationResponse.domain`이 필수 `str`이라(`src/api/schemas.py:26`) `domain`을
  생략한 통합 검색이 Pydantic ValidationError로 500이 된다. `Optional[str]`로 고친다.
  `docs/api_for_backend.md`가 이 동작을 이미 백엔드에 약속해 둔 상태다.

### D7. "보드는 백엔드가 조립한다"를 문서에 명시
- 계약의 상위 객체는 `{id, title, section, items}` 보드지만, 추천 API는 `items` 배열만
  준다. 보드 id 발급·`section` 분류(추천/내 보드/인기)·저장 상태는 백엔드 몫이다.
  경계를 `docs/api_for_backend.md`에 한 줄 적지 않으면 "보드 API는 어디 있냐"가 나온다.

### D8. 로컬 venv에 pyarrow 없음
- `pd.read_parquet`이 실패하는데 `initialize_dependencies`가 예외를 삼켜서
  (`src/api/dependencies.py:207`) 인덱스가 **조용히 안 올라가고** 추천이 계속 503이 된다.
  `requirements-serve.txt:29`에 8/7 서버에서 같은 원인으로 3일 멈춘 기록이 이미 있다.

### D9. 아키텍처 결정 — 벡터 API vs 추천 API — **결정 필요**
- **무엇**: 8/18 연동에서 백엔드가 `POST /api/v1/embeddings`를 두드렸다. 검색을 자기
  쪽에서 하겠다는 뜻이다. 그래서 쿼리 벡터 엔드포인트를 새로 열었다
  (`POST /search/vector`, `GET /search/vector?keyword=`, `POST /embeddings`).
- **문제**: 지금 두 구조가 **동시에 열려 있다**. 백엔드가 벡터를 받아 자기 벡터 DB에서
  검색하는 길과, 추천 서버가 검색까지 하고 결과 리스트를 주는 길(`/recommend`)이다.
  하나로 정하지 않으면 데이터 계약(D1~D5)도, 인덱스 전달 방식도 결정할 수 없다.
- **딸린 결정**: 백엔드 검색으로 가면 아이템 임베딩 655MB(movie 123 / music 123 /
  book 409, 768차원 float32, L2정규화)를 넘겨야 한다. 파일 전달 / 다운로드 엔드포인트 /
  벡터 DB 직접 적재 중 무엇으로 할지, 재학습 때마다 어떻게 갱신할지.

### D10. `serve.sh`가 venv를 쓰도록 고정
- **무엇**: PATH의 `uvicorn`을 쓰는 탓에 아나콘다 파이썬이 잡히면
  `No module named 'sentence_transformers'`로 모델 적재가 조용히 실패한다.
  `./venv/bin/uvicorn`이 있으면 그것을 쓰도록 고친다.
- **왜**: 8/18에 실제로 겪었다. 게다가 실패해도 앱은 뜨기 때문에(`initialize_dependencies`가
  예외를 삼킨다) "떴는데 추천만 503"이라는 헷갈리는 상태가 된다. D8과 같은 축의 사고다.

### D11. `QueryBlock`의 CLIPProcessor → CLIPTokenizer
- **무엇**: 현재 `transformers` 버전이 `openai/clip-vit-large-patch14`의 image processor를
  인식하지 못해(`Unrecognized image processor`) **모델 객체 생성 자체가 실패**한다.
- **왜 고칠 수 있나**: `QueryBlock`(`src/models/recommender.py:105`)은 텍스트만 쓰는데도
  이미지까지 포함하는 `CLIPProcessor`를 쓴다. `CLIPTokenizer`로 바꾸면 image processor를
  아예 거치지 않아 문제가 사라지고, 토큰화 결과는 동일하다.
- **대안**: 학습 때 쓰던 `transformers` 버전으로 고정. 다만 근본 원인은 그대로 남는다.
- **의존**: 이게 풀려야 진짜 쿼리 벡터를 내보낼 수 있다. 모델 파일(`models/trained_model.pt`)
  을 서빙 컴퓨터로 가져오는 것과 함께 필요하다.

### D12. 가짜 벡터 모드 정리
- **무엇**: `FAKE_VECTOR=1`이면 모델 없이도 검색어 시드 기반 난수 벡터를 내보낸다
  (`src/api/dependencies.py`의 `_fake_vector`). 백엔드가 응답 **모양**을 먼저 맞출 수
  있게 하려고 8/18에 넣었다.
- **정리할 것**: 진짜 모델이 올라간 뒤 이 모드를 남길지 정한다. 남긴다면 실수로 켜진 채
  운영되지 않도록 기동 로그에 경고를 크게 찍는다. `model_version`이 `FAKE-no-model`로
  나가는 것이 유일한 구별 수단이다.
- **곁다리**: pydantic이 `model_loaded` / `model_version` 필드에 대해 `protected_namespaces`
  경고를 찍는다. 동작에는 지장 없지만 기동 로그가 지저분하다. `model_config`로 끈다.
