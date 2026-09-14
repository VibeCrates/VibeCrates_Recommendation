# 추천 서버 API — 백엔드 연동 안내

최종 갱신: 2026-09-14 · 대상: 백엔드 담당자
8/18 버전은 쿼리 벡터를 열되 가짜를 내보내던 상태였다. 8/24에 모델과 인덱스를 GPU
서버에서 가져와 **진짜 벡터로 바뀌었다.**

## 지금 상태 한눈에

| | 상태 |
|---|---|
| 통신 (`/ping`) | ✅ 확인 완료 |
| 쿼리 벡터 (`/search/vector`) | ✅ **진짜 벡터** |
| 추천 (`/recommend`) | ✅ 동작 — 단 **대조용**, 계약 아님 |
| 아이템 벡터 번들 | ✅ 준비됨 (**786.4MB, CSV 형식 — 2026-09-14 변경**, 이메일로 직접 전달 예정) |
| 접속 주소 | `http://100.77.133.40:8000` (Tailscale) |

**8/18에 받아 둔 벡터는 버릴 것.** 그때 나간 것은 검색어를 시드로 한 난수였다. 응답의
`model_version`으로 구별한다 — `FAKE-no-model`이면 가짜, `trained_model.pt@...`이면
진짜다. 현재 값은 다음과 같다.

```
model_version: trained_model.pt@1786697109
```

이 값은 체크포인트 파일이 바뀔 때마다 달라진다. 지금 실려 있는 것은 8/14에 학습한
`best_stage2_qlora_off`이고, 8월 실험 중 판정 평균이 가장 높았던 설정이다.

---

## 1. 접속 주소

```
http://100.77.133.40:8000
```

Tailscale 사설망 주소다. 인터넷에는 아무것도 열려 있지 않고, 같은 tailnet에 들어온
기기끼리만 보인다. 서버가 추천팀 맥북에서 도는 동안에만 살아 있다.

브라우저로 `http://100.77.133.40:8000/docs`를 열면 모든 엔드포인트를 눌러서 시험해볼
수 있는 화면이 나온다(FastAPI 자동 생성). 요청 형태가 헷갈릴 때 여기서 확인하는 것이
가장 빠르다.

### ⚠ HTTP/1.1로 보낼 것

**이 서버는 HTTP/2를 지원하지 않는다.** uvicorn이 HTTP/1.1만 처리한다.

8/18에 실제로 이것 때문에 막혔다. 백엔드가 HTTP/2로 요청을 보내는 동안 본문이 제대로
실리지 않아 **422가 반복**됐고, HTTP/1.1로 바꾸자 바로 정상화됐다. 서버 로그에는
"형식이 안 맞는 요청"으로만 보여서 원인을 찾는 데 시간이 걸렸다. 클라이언트 설정을
확인할 것.

---

## 2. 쿼리 벡터 — 백엔드가 자기 쪽에서 검색하는 경로

자연어 검색어를 받아 **768차원 벡터**로 바꿔 돌려준다. 검색은 하지 않는다.

> **이 경로를 쓰세요.** 아래 세 가지는 같은 일을 하는 세 가지 모양이고, 8/18 연동에서
> 요청 형태가 몇 번 바뀌며 하나씩 늘어난 것이다. 검색창이 하나이므로 검색어도 한 건이고,
> **`POST /api/v1/search/vector`가 정본**이다. 나머지 둘(`GET /search/vector`,
> `POST /embeddings`)은 지금 쓰고 있는 곳이 없으면 정리한다 —
> **어느 것을 쓰고 있는지 알려 주세요.**

### `POST /api/v1/search/vector` — 한 건 (정본)

```json
// 요청
{ "text": "하늘" }

// 응답
{ "text": "하늘",
  "dim": 768,
  "normalized": true,
  "model_version": "FAKE-no-model",
  "vector": [0.0023, 0.0258, 0.0003, ...] }
```

- 필드 이름은 `text` / `query` 둘 다 받는다
- 같은 경로에 `GET /api/v1/search/vector?keyword=하늘` 도 있다. 결과는 동일하다
- `normalize`(기본 `true`)가 참이면 L2 정규화된 벡터라 **내적이 곧 코사인 유사도**다

### `POST /api/v1/embeddings` — 여러 건 (정리 대상)

```json
// 요청
{ "query": ["하늘", "바다"] }

// 응답
{ "dim": 768, "normalized": true, "model_version": "...",
  "vectors": [[...], [...]] }
```

키가 `vectors`(복수)라는 점만 다르다. `text` / `query` 둘 다, 문자열과 배열 둘 다 받는다.

한 가지 주의 — 배치는 개별 항목을 `null`로 표시할 수 없어서, 하나라도 번역되면
`translated_queries`에 **번역되지 않은 것까지 포함한 전체 목록**이 나온다. 단건 경로는
`null`이냐 문자열이냐로 깔끔하게 갈린다. 이것도 단건을 정본으로 두는 이유다.

### 아이템 벡터도 받아야 한다

쿼리 벡터만으로는 검색할 수 없다. 검색 대상인 **아이템 벡터 190,739건**을 백엔드 벡터
DB에 미리 적재해야 한다. 벡터와 메타데이터와 대조용 manifest를 묶은 전달 번들을 준비해
뒀다 — **5절**을 볼 것.

그리고 **쿼리 벡터와 아이템 벡터는 반드시 같은 체크포인트에서 나와야 한다.** 어긋나면
검색은 오류 없이 성공하고 결과만 엉뚱해진다 — 조용히 틀리는 종류의 사고다. 응답의
`model_version`이 그 대조용이고, 재학습할 때마다 값이 바뀐다.

### 한국어 검색어 — 서버가 번역해서 인코딩한다 (2026-08-31)

**검색어를 그냥 한국어로 보내면 된다.** 백엔드에서 할 일은 없다.

내부적으로는 한글이 들어오면 영어로 번역한 뒤 인코딩한다. 쿼리 인코더가 CLIP 텍스트
인코더인데 영어 BPE라 한글을 바이트 조각으로 부수기 때문이다. 번역 없이 넣으면 검색이
사실상 동작하지 않는다 — 같은 뜻의 한/영 쿼리 40쌍에서 상위 10개 겹침이 **전부 0%**였고,
쿼리 벡터의 코사인도 0.058로 사실상 직교였다.

| | 상위10 겹침(영어 대비) | 최고 점수 |
|---|---:|---:|
| 번역 전 | 0.0% | 0.509 |
| 번역 후 | 29.0% | 0.645 |
| (영어 원문) | 100% | 0.659 |

응답에 `translated_query`(벡터 경로는 `translated_text`, 배치는 `translated_queries`)가
한 줄 더 실려 온다. **번역이 일어났을 때만 채워지고, 아니면 `null`이다** — `null`은
"보낸 문장 그대로 인코딩했다"는 뜻이지 값이 빠진 것이 아니다. 결과가 이상할 때 번역
탓인지 검색 탓인지 여기서 갈린다.

```json
{ "query": "비 오는 오후의 조용한 시간",
  "translated_query": "It's raining, quiet time in the afternoon.",
  "results": [ ... ] }
```

은유적인 문장일수록 번역이 흔들린다(스타일별 코사인: atmosphere 0.85 / direct 0.86 /
philosophical 0.78 / poet 0.63). 근본 해결은 다국어 인코더로 바꾸는 것이고 재학습이
필요해서, 다음 학습 사이클에서 판단한다.

### 검색어 길이 제한

쿼리 인코더(CLIP)가 77토큰까지만 본다. 그보다 긴 문장은 뒤가 잘린다. 한국어는 토큰이
더 많이 나와서 체감상 두세 문장이 한계다.

---

## 3. 통신 확인

### `GET /api/v1/ping`

모델과 무관하게 동작한다. 응답이 오면 네트워크·포트·경로가 정상이라는 뜻이다.

```
GET http://100.77.133.40:8000/api/v1/ping?n=42
→ {"value":42, "received":42, "server_time":"...", "model_loaded":false}
```

- `received`는 **쿼리 파라미터**로 보낸 `n`이 그대로 돌아온 것이다. 값이 맞으면 요청이
  중간에서 잘리지 않았다는 뜻이다
- `model_loaded: false`는 이 단계에서 정상이다
- `n`이 정수가 아니면 422

### `GET /api/v1/health`

```json
{ "status": "healthy", "model_loaded": false,
  "index_built": {"movie": false, "music": false, "book": false} }
```

서버 기동에 **13초쯤** 걸리고 그동안은 연결 자체가 안 된다. 첫 요청이 느린 것이 아니라
기동 중에 거부되는 것이므로, 타임아웃이 아니라 재시도로 다루는 편이 맞다.

---

## 4. 참고용 엔드포인트

### `POST /api/v1/recommend`

추천 서버가 검색까지 끝내고 결과 리스트를 돌려주는 경로다. 8/24부터 정상 동작한다.

```json
// 요청
{ "query": "비 오는 날 듣기 좋은", "domain": "music", "top_k": 10 }
```

`domain`은 `movie` | `music` | `book`이고, **생략하면 세 도메인 통합 검색**이다(그때
응답의 `domain`은 `null`). `top_k`는 1~100(기본 10).

### `GET /api/v1/item/{domain}/{item_id}`

아이템 메타데이터 조회. 없으면 404.

### `/recommend`의 위치 — 계약이 아니라 대조용이다

구조 결정(아래)에 따라 검색은 백엔드가 한다. `/recommend`는 **같은 검색어를 넣어
결과를 맞춰 보는 용도**로 남겨 둔 것이다. 백엔드 벡터 DB 검색이 이상할 때, 우리 쪽에
같은 질의를 넣어 결과가 같으면 벡터는 정상이고 백엔드의 거리 계산·정규화 설정 문제이며,
다르면 체크포인트가 어긋난 것이다. 응답 모양(`{item_id, ..., extra}`)은 프론트 계약과
다르며, **맞출 계획도 없다** — 계약을 따르는 것은 아래 번들의 items 쪽이다.

백엔드 검색이 정상 동작하는 것이 확인되면 이 엔드포인트는 닫는다.

---

## 5. 구조 결정 (2026-08-24) — 검색은 백엔드가 한다

두 길이 열려 있었고, **백엔드가 자기 벡터 DB에서 검색하는 쪽**으로 정했다.

```
[사용자 검색어] → 백엔드 → POST /api/v1/search/vector → 추천 서버
                                                          ↓ 768차원 벡터
                     백엔드 벡터 DB(아이템 벡터 190,739건) ← 검색 → 결과 아이템
```

추천 서버는 검색어를 벡터로 바꾸는 일만 한다. 따라서 아이템 벡터와 그 메타데이터를
백엔드가 들고 있어야 하고, 그것이 아래 **전달 번들**이다.

### 도메인 미지정 검색 — 결과 합칠 때 상한을 걸 것 (2026-09-09 측정)

`domain`을 지정하지 않은 통합 검색(세 컬렉션에서 각각 검색해 점수로 합치는 경우)에서
**한 도메인이 top-10을 통째로 먹는 경우가 40쿼리 중 12개(30%)였다.** 전역 분포는
고르지만 쿼리 단위로는 쏠린다. 검색 자체는 백엔드 쪽에서 하므로(위 구조 결정), 이 상한도
백엔드 검색 코드에 들어가야 한다. 규칙은 두 줄이다.

1. **도메인별 상위 5개만 남기고 합쳐서 top-10을 뽑는다.**
2. **단, 검색어가 도메인을 이름으로 지목하면 상한을 걸지 않는다.** 무조건 걸면 그런
   쿼리에서 유의하게 나빠진다(쌍대 승률 17.6%, p=0.0002) — "우주 배경 액션 **영화**"에
   상한을 걸면 음악·책이 억지로 끼어든다. 실측 40쿼리 중 direct 스타일 10개**전부**가
   도메인 단어를 담고 있었고 나머지 30개는 하나도 없었다.

```python
import re

DOMAIN_WORD = re.compile(
    r"\b(movie|movies|film|films|cinema|music|song|songs|track|tracks|album|albums|"
    r"book|books|novel|novels|fiction|nonfiction|memoir|drama|piano|guitar|rock|jazz|"
    r"soundtrack|documentary)\b", re.I)

def names_domain(query: str) -> bool:
    return bool(DOMAIN_WORD.search(query))
```

이 예외를 붙인 쪽(`cap5_guard`)이 쏠림을 절반으로 줄이면서(독식 12 → 6, 남은 6개는 전부
도메인을 지목한 direct라 의도된 결과) 판정 손해가 없었다 — 오히려 평균이 0.815 → 0.840로
올랐다(0.6 SE, 유의하지 않음). 무조건 상한(`cap5`)은 손해가 났다.

**한국어 검색어에 적용할 때 주의.** 서버가 번역한 뒤(위 "한국어 검색어" 절) 번역된 영어
문장에 이 정규식을 적용하면 대체로 맞는다. 다만 **번역이 도메인 단어를 떨어뜨리면 예외가
발동하지 않는다** — 이 경우는 아직 실측하지 않았다.

**한계.** 영어 40쿼리로 잰 값이고 정규식은 휴리스틱이다. 이 실험을 검증한 쌍대 판정기
자체의 위치 편향이 39.9%로 신뢰 구간(40~60%) 경계에 걸쳐 있어, 더 큰 표본으로
재확인할 여지가 있다.

### 전달 번들 (2026-09-14 형식 변경 — CSV)

`dist/bundle_20260914_csv/` 한 폴더로 넘긴다(786.4MB). 백엔드가 벡터도 CSV 열로 받는 편이
편하다고 요청해(2026-09-14) npz에서 바꿨다.

| 파일 | 내용 | 크기 |
|---|---|---:|
| `movie_vectors.csv` | 39,515행 — `id`, `vector_b64`, `model_version` | 163.3MB |
| `music_vectors.csv` | 39,682행 — 〃 | 164.6MB |
| `book_vectors.csv` | 110,594행 — 〃 | 458.5MB |
| `manifest.json` | model_version · 건수 · sha256 · L2 노름 범위 · 필드 채움률 | — |

```python
import base64
import numpy as np
import pandas as pd

df = pd.read_csv("book_vectors.csv", dtype={"id": str})
v = np.frombuffer(base64.b64decode(df.loc[0, "vector_b64"]), dtype="<f4")  # (768,) float32
```

`vector_b64`는 (768,) float32(L2 정규화 완료, 내적 = 코사인 유사도)를 바이트 그대로 base64로
감싼 것이다 — 무손실이고, 행당 정확히 4,096바이트로 고정된다. 10진수 텍스트로 풀면 행당
8.8KB(전체 1.68GB, 지금 파일의 2배)이고, JSON 배열 문자열은 더 크다(행당 17KB, 전체 3.2GB —
`json.dumps`가 float를 더 긴 자릿수로 풀어쓰기 때문). 정밀도와 크기를 함께 잡는 선택이
base64다.

**id와 vector가 같은 행에 있다.** npz 때는 "i번째 ids가 i번째 vectors의 주인"이라는 위치
기반 약속에 기대야 했지만(그래서 `assert len(items) == len(vectors)`가 필요했다), CSV는
그 약속 자체가 필요 없다 — 순서가 밀리는 사고가 구조적으로 불가능하다.

`model_version`은 매 행에 반복해서 들어 있다. **적재 시점에 `/search/vector` 응답의
`model_version`과 이 열이 같은지 확인할 것** — 다르면 쿼리 벡터와 아이템 벡터가 다른
체크포인트에서 나온 것이고, 그때 검색은 오류 없이 성공하고 결과만 엉뚱해진다.

#### 이전 형식(9/11, npz)에서 바뀐 것

- 파일 형식만 CSV로 바뀌었다. **무엇을 담는지, 무엇을 담지 않는지는 9/11과 같다** — `row`와
  `point_id`를 넘기지 않는 이유는 아래에 그대로 있다.
- 벡터가 이진 배열이 아니라 문자열 열(`vector_b64`)이 됐다. 적재 전에 base64 디코드가
  한 단계 더 필요하다.

#### 이전 형식(9/11 이전, parquet+npy)에서 바뀐 것

- **`{domain}_items.parquet`을 더 넘기지 않는다.** 백엔드가 저장소의 canonical CSV를
  목적에 맞게 가공해 쓰기로 했으므로(2026-09-11), 우리가 줘야 하는 것은 *어느 벡터가 어느
  아이템인지*뿐이다. 제목·연도 등은 CSV에서 `id`로 조인하면 된다 — 번들의 id 집합과
  canonical의 id 집합은 세 도메인 모두 1:1로 일치한다(중복 0, 누락 0).
- **`row`를 넘기지 않는다.** 예전 형식을 순서대로 읽으면 나오는 값이다.
- **`point_id`를 넘기지 않는다.** 아래 참조.
- `model_version`이 벡터 파일 안에 함께 들어갔다. manifest는 옮기다 떨어질 수 있는데,
  그러면 벡터가 어느 체크포인트에서 나왔는지 알 방법이 사라진다.

> **주의 — CSV의 이미지 URL 열을 쓰지 말 것.** `Poster`/`img`/`imgUrl`은 원본 데이터셋이
> 갖고 있던 URL이고 우리 보유 현황과 다르다. music은 **7,205곡이 파일은 있는데 URL이 비어
> 있고**(Deezer로 따로 받은 것이라 CSV에 반영되지 않았다), 반대로 book은 3,003건이 URL은
> 있는데 파일이 없다. **이미지 유무는 파일 존재로 판단한다** — 파일명이 곧 id다
> (`{domain}/{id}.jpg`, 매칭률 100%, 고아 파일 0개).

### Qdrant에 넣는 법 (2026-08-31, 백엔드가 Qdrant로 확정)

Qdrant의 point id는 **부호 없는 정수나 UUID만** 받는다. 우리 `id`는 임의의 문자열이라
(`kdl_B00TZE87S4`, `3tjFYV6RSFtuktYl3ZtYcq`) 그대로는 못 쓴다. 그러나 **UUID는 받으므로
id에서 결정론적으로 만들면 된다.**

```python
POINT_NS = uuid.UUID("6f9b4a2c-0000-4000-8000-000000000001")   # 아무 고정 UUID
point_id = str(uuid.uuid5(POINT_NS, f"{domain}:{item_id}"))
```

**point id는 백엔드가 정한다.** 어떤 방식이든 상관없지만 **id에서 파생시키기를 권한다** —
그러면 우리가 재학습해 벡터를 다시 넘겨도 같은 아이템이 같은 point를 유지해 upsert로
갱신된다.

> **2026-09-11 폐기** — 8/31 번들이 담고 있던 `point_id`(`10억 × 도메인 + row`)는 쓰지
> 말 것. `row`는 인덱스에서 몇 번째냐일 뿐이라 아이템이 추가·삭제되면 뒤가 전부 밀리고,
> **같은 아이템의 기본키가 재학습마다 바뀐다.** 우리 내보내기 순서가 그쪽 DB의 기본키에
> 박히는 구조였다. 다음 재학습에서 실제로 행 수가 바뀐다.
>
> 같은 이유로 `row`(벡터 배열의 행 번호)도 기본키로 쓰면 안 된다. 도메인 안에서만 0부터
> 매겨지므로 한 컬렉션에 세 도메인을 넣으면 movie의 0번과 music의 0번이 충돌한다.

```python
from qdrant_client import QdrantClient, models
import base64, numpy as np, pandas as pd, uuid, requests

POINT_NS = uuid.UUID("6f9b4a2c-0000-4000-8000-000000000001")

# 적재 전에 체크포인트를 대조한다. 쿼리 벡터와 아이템 벡터가 다른 체크포인트에서 나오면
# 검색은 오류 없이 성공하고 결과만 엉뚱해진다 — 넣기 전에 잡는 편이 싸다.
live = requests.post("http://<추천서버>/api/v1/search/vector",
                     json={"text": "ping"}).json()["model_version"]

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    "vibecrates",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.DOT),
)

for domain in ("movie", "music", "book"):
    df = pd.read_csv(f"{domain}_vectors.csv", dtype={"id": str})
    bundle_version = df["model_version"].iloc[0]
    assert bundle_version == live, f"{domain}: 번들 {bundle_version} != 서버 {live}"

    for start in range(0, len(df), 1000):
        chunk = df.iloc[start:start + 1000]
        client.upsert("vibecrates", points=[
            models.PointStruct(
                id=str(uuid.uuid5(POINT_NS, f"{domain}:{r.id}")),   # ← id에서 파생
                vector=np.frombuffer(base64.b64decode(r.vector_b64), dtype="<f4").tolist(),
                payload={"id": r.id, "domain": domain},              # 나머지는 canonical CSV에서
            ) for r in chunk.itertuples()
        ])

# 도메인 필터를 쓸 거라면 인덱스를 만들어 둔다
client.create_payload_index("vibecrates", "domain",
                            field_schema=models.PayloadSchemaType.KEYWORD)
```

`distance=DOT`인 이유는 양쪽 벡터가 이미 L2 정규화돼 있어서 내적이 그대로 코사인
유사도이기 때문이다(실측: 내적과 코사인의 차이 5.96e-08, 상위 100위 순위 동일).
`COSINE`을 골라도 순위는 같고 나눗셈만 한 번 더 한다.

검색은 우리 API에서 받은 벡터를 그대로 넣으면 된다.

```python
hits = client.query_points("vibecrates", query=vector, limit=10).points
for h in hits:
    h.payload["title"], h.payload["image"], h.payload["url"], h.score
```

(구버전 클라이언트는 `client.search(collection_name=..., query_vector=...)`다.)

> **payload의 `id`를 꼭 넣을 것.** 저장된 보드처럼 오래 남아야 하는 데이터는 반드시
> 이 문자열 `id`로 저장한다. `point_id`와 `row`는 인덱스를 다시 만들면 번호가 밀리므로,
> 그것으로 저장해 두면 재학습 후 **다른 작품을 가리키게 된다**.

### 아이템 메타데이터 필드 채움률

제목·연도·이미지·url 같은 메타데이터는 벡터 CSV가 아니라 canonical CSV에서 `id`로
조인해서 가져온다(위 "이전 형식에서 바뀐 것" 참조). 적재 직전에 벡터 CSV의 행 수와
`manifest.json`의 `count`가 같은지 확인할 것 — 다르면 행이 빠진 것이다.

| 도메인 | 건수 | year | image | url |
|---|---:|---:|---:|---:|
| movie | 39,515 | 99.2% | 99.3% | 100% |
| music | 39,682 | 89.8% | 92.4% | 100% |
| book | 110,594 | 91.7% | 97.0% | 93.1% |

채워지지 않는 것은 `null`이다. music 표지는 8/31에 Deezer로 9,902곡을 보강해 67.4% →
92.4%가 됐다. 남은 3,008곡은 Deezer에도 없다. book의
url 6.9%는 isbn도 asin도 없는 goodreads 책이다. `image`가 null이면 **그 파일은 존재하지
않는다** — 경로를 만들어 두고 404를 맞는 일이 없도록 파일 목록과 대조해 채웠다.

`url` 규칙: movie는 IMDB 링크, music은 Spotify 트랙 URL, book은 isbn이 있으면
Open Library, kindle 출처면 Amazon.

### 갱신 규약 — 이것을 안 지키면 조용히 틀린다

재학습하면 아이템 벡터가 전부 바뀐다. 그때 **번들을 통째로 새로 받아야** 한다.
쿼리 벡터만 새 모델에서 나오고 아이템 벡터가 옛것이면 검색은 오류 없이 성공하고
결과만 엉뚱해진다.

대조 방법은 하나다. `manifest.json`의 `model_version`과 `/search/vector` 응답의
`model_version`이 **같은지 매 배포마다 확인**한다. 현재 값은 `trained_model.pt@1786697109`.

#### 부분 갱신은 하지 마세요 — 통째로 다시 넣습니다

바뀐 아이템만 골라 덮어쓰면 안 된다. 이유가 둘이다.

첫째, 재학습하면 **아이템 19만 개의 벡터가 전부 바뀐다.** 인코더가 달라졌으므로 같은
작품도 다른 768개 숫자가 된다. 일부만 갱신하면 새 모델의 벡터와 옛 모델의 벡터가 한
컬렉션에 섞이고, 둘은 서로 비교할 수 없는 값이라 순위가 뒤죽박죽이 된다.

둘째, `point_id`가 밀릴 수 있다. `point_id`는 아이템이 인덱스에 실린 순서에서 나오므로,
아이템이 하나 추가되거나 빠지면 **그 뒤의 모든 아이템이 한 칸씩 밀린다.** 앞으로
예정된 작업 중에 그런 것이 있다(중복 데이터 정리, book 데이터셋 보완 등).
참고로 8/24 → 8/31 번들 사이에서는 아이템 구성이 같아 한 건도 밀리지 않았지만,
그것을 규칙으로 삼으면 안 된다.

둘 다 **오류 없이 조용히 틀리는** 종류다. 검색은 정상으로 성공하고 결과만 어긋난다.

#### 무중단으로 바꾸는 방법 — 별칭 전환

새 컬렉션에 먼저 다 넣고, 끝난 뒤 별칭만 옮긴다. 적재하는 동안에도 검색은 옛
컬렉션으로 계속 돌고, 문제가 생기면 별칭만 되돌리면 된다.

```
1. vibecrates_20260831 컬렉션을 만들고 새 번들을 적재한다
2. 건수·model_version을 확인한다
3. 별칭 vibecrates 를 새 컬렉션으로 전환한다   (Qdrant의 alias 기능)
4. 옛 컬렉션을 지운다
```

애플리케이션은 항상 별칭 `vibecrates`만 바라보게 해 두면, 교체 때 코드를 건드릴 일이
없다. 19만 point·570MB라 적재 자체는 몇 분 수준이고, 재학습이 잦은 일도 아니다.

### 표지 이미지 (9.9GB)

`images/{domain}/{id}.jpg`로 `items.parquet`의 `image` 값과 그대로 대응한다.
백엔드가 복사해 가서 자기 정적 경로나 CDN에서 제공하기로 했다. 전달 방법은 협의한다.

| 도메인 | 장수 | 용량 |
|---|---:|---:|
| movie | 39,237 | 1.3GB |
| music | 36,674 | 5.0GB |
| book | 107,254 | 3.7GB |
| 합계 | 183,165 | 9.9GB |

music이 장수에 비해 무거운 것은 8/31에 Deezer에서 받은 원본이 1000×1000(장당 약 140KB)
이기 때문이다. 전송량이 문제가 되면 줄여서 보낼 수 있으니 말해 주세요.

### 아직 정할 것

1. ~~번들 전달 방법~~ — **[결정] 2026-09-14: 이메일로 직접 전달.** 벡터 CSV 번들과 표지
   이미지를 사람이 이메일에 첨부해 보낸다. GitHub/LFS 경로(100MB 단일 파일 한도, 재학습마다
   쌓이는 히스토리, 저장소 공개·book 라이선스 문제)는 쓰지 않는다.
2. **포트·경로 규칙** — 현재 8000, `/api/v1/...`. 백엔드 규약이 있으면 맞춘다
3. **인증** — 사설망 안이라 없다. 정책상 필요하면 헤더 방식으로 넣는다
4. **에러 형식** — 현재 FastAPI 기본(`{"detail": "..."}`)

## 6. 문제가 생겼을 때

### 서버가 살아 있는데 추천만 503일 때

`GET /api/v1/health`의 `errors`를 먼저 볼 것. 8/24부터 기동 시 실패한 것과 그 이유가
여기 그대로 실려 나온다. 비어 있으면 모델·인덱스가 정상으로 올라온 것이다.

```json
{ "status": "healthy", "model_loaded": true,
  "index_built": { "movie": true, "music": true, "book": true },
  "errors": {} }
```

이 필드를 만든 이유는, 적재에 실패해도 앱은 뜨기 때문이다(`/ping`은 모델과 무관하게
응답해야 한다). 예전에는 실패가 서버 로그에만 남아 백엔드 쪽에서는 원인을 알 수 없었다.

### 8/24에 함께 고친 것

- 통합 검색(도메인 미지정)이 500으로 떨어지던 문제 — 응답의 `domain`이 `null`을 허용하지
  않아 생긴 것이다. 이제 `domain` 없이 `POST /recommend`를 부르면 세 도메인을 합쳐서
  돌려주고, 응답의 `domain`은 `null`이다.
- `serve.sh`가 venv의 `uvicorn`을 쓰도록 고쳤다. 그전에는 아나콘다 파이썬이 잡히면
  `No module named 'sentence_transformers'`로 모델 적재만 조용히 실패했다.

---

# 부록 — 추천팀 내부용

> 여기부터는 **백엔드가 보지 않아도 되는 내용**이다. 추천 서버를 직접 띄우고 점검하는
> 사람을 위한 메모다.

## A. 서버 실행

```bash
FAKE_VECTOR=1 ./scripts/serve.sh     # 가짜 벡터 모드 (모델 없이 연동 시험)
./scripts/serve.sh                   # 평상시 (모델 없으면 임베딩도 503)
PORT=9000 ./scripts/serve.sh         # 포트 변경
```

실행하면 LAN·Tailscale 주소를 함께 출력한다.

```bash
python scripts/check_api.py --host 100.77.133.40   # 밖에서 보이는지 자체 점검
```
