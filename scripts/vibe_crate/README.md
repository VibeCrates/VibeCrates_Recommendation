# VibeCrates 추천 결과 화면 — 디자인 시안

`example.png`(콜라주 "vibe" 무드보드 밈)를 참고해 만든 추천 결과 화면 시안입니다.
레이아웃·타이포·인터랙션은 고정 디자인이고, 실제 추천 데이터를 연결하는 건
백엔드 쪽 작업입니다.

## 로컬에서 띄우는 법

서버도 빌드도 필요 없습니다. `index.html`을 더블클릭해서 브라우저로 열면 끝입니다.

```
index.html   더블클릭 (또는 브라우저에 파일 드래그)
```

`data.js`가 `<script src="data.js">`로 로드되기 때문에 `file://`로 열어도
(fetch/XHR과 달리) CORS 문제 없이 바로 동작합니다. 혹시 브라우저에서 막히면
아래처럼 로컬 서버 하나 띄워서 열어도 됩니다:

```
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

## 폴더 구성

```
vibe_crate/
  index.html   ← 레이아웃 · CSS · 렌더링 로직 (고정 디자인, 여기가 "큰 틀")
  data.js      ← 추천 아이템 배열 (지금은 예시 데이터, 실제로는 API 응답으로 대체)
  images/      ← 표지 이미지 (도메인별 폴더)
  README.md    ← 이 문서
```

## 데이터 계약

`index.html`은 전역 변수 `RECOMMENDATIONS`(`data.js`에서 정의)를 읽어서
아이템 배열 순서대로 화면 슬롯에 채웁니다. 아이템 하나의 형태:

```js
{
  "id": "114709",
  "domain": "movie",              // "movie" | "music" | "book"
  "title": "Toy Story",
  "year": 1995,                   // 결측이면 null (화면엔 "xxxx"로 표시됨)
  "image": "images/movie/114709.jpg",
  "url": "http://www.imdb.com/title/tt0114709"
}
```

**실제 연결 시 필요한 작업 (백엔드 담당)**

- `domain` 필드는 지금 DB 스키마 어디에도 없음 — movie/music/book 세 소스를
  merge할 때 직접 채워 넣어야 함
- `image`는 원본 DB의 `Poster`/`img`/`imgUrl` 컬럼(외부 URL)을 그대로 쓰지 말 것.
  Amazon CDN 플레이스홀더, IMDB 이미지 깨짐 등 이전에 확인된 문제가 있어서,
  `/Users/hyun/images/{domain}/{id}.jpg` 로컬 이미지(또는 이를 서빙하는 CDN)를
  id로 매핑해서 내려줘야 함
- `year`는 movie(`release_date`)·music(`year`)·book(`publishedDate`)에서
  연도만 추출. book은 결측률이 36.8%로 꽤 높으니, null이면 프론트가 자동으로
  "xxxx" 표시함 — 별도 처리 불필요
- `url`은 movie=`Imdb Link`, book=`productURL` 컬럼 그대로, music은 컬럼이
  없어서 `https://open.spotify.com/track/{id}`로 조립 (music의 `id`가 그대로
  Spotify 트랙 ID라 별도 수집 불필요)
- 화면에 노출되는 아이템 개수: 지금 시안은 12개 슬롯 고정. API가 그보다 적거나
  많은 개수를 내려줄 경우 어떻게 할지는 아직 미정 — 슬롯 개수에 맞춰 자르거나,
  슬롯 패턴을 늘리는 방향 논의 필요

## 화면 동작

- 이미지는 항상 보임 (기본 상태)
- 카드에 마우스를 올리면 `domain → title → year → url` 순서로 오버레이가
  살짝 딜레이를 두고 나타남 (`index.html`의 `.hc-domain/.hc-title/.hc-year/.hc-url`
  `transition-delay` 참고)
- `url`은 실제 클릭 가능한 링크 (새 탭으로 이동)

## 시안 데이터 다시 만들기 (디자인 쪽 작업)

`scripts/build_vibe_mockup.py`가 이 폴더(`data.js` + `images/`)를 생성합니다.
노출할 (도메인, id) 목록은 스크립트 상단 `SELECTION`에 있고, 제목/연도/URL은
전부 실제 데이터셋(`data/MovieGenre_enriched.csv`, `data/music_features.csv`,
`data/kindle_data-v2.csv`)에서 그때그때 조회합니다 — 하드코딩된 값 없음.

```
python3 scripts/build_vibe_mockup.py
```

레이아웃 자체(슬롯 위치·회전각·hover 카드 스타일)를 바꾸려면
`scripts/vibe_crate/index.html`을 수정한 뒤 스크립트를 다시 실행하면
`mockups/vibe_crate/index.html`에 반영됩니다.
