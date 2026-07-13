# VibeCrates — 메인 페이지 · 보드 페이지 디자인 시안

`example.png`(콜라주 "vibe" 무드보드 밈)를 참고해 만든 화면 시안입니다.
레이아웃·타이포·인터랙션은 고정 디자인이고, 실제 추천 데이터를 연결하는 건
백엔드 쪽 작업입니다.

핵심 개념은 **"보드"** 입니다. 보드 하나 = 검색어(제목) + 그 검색으로 나온
추천 아이템들의 콜라주. 메인 페이지는 이런 보드들을 모아 보여주고, 검색으로
새 보드를 만드는 진입점입니다.

## 로컬에서 띄우는 법

서버도 빌드도 필요 없습니다. `main.html`을 더블클릭해서 브라우저로 열면 끝입니다.

```
main.html   더블클릭 (또는 브라우저에 파일 드래그)
```

`boards.js`가 `<script src="boards.js">`로 로드되기 때문에 `file://`로 열어도
(fetch/XHR과 달리) CORS 문제 없이 바로 동작합니다. 혹시 브라우저에서 막히면
아래처럼 로컬 서버 하나 띄워서 열어도 됩니다:

```
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000/main.html 접속
```

**주의**: `board.html`의 "캡쳐하기" 버튼만 [html2canvas](https://html2canvas.hertzen.com/)를
CDN에서 불러와 동작합니다. 이 기능 하나만 인터넷 연결이 필요하고, 나머지
(검색·저장·나가기·hover 카드)는 전부 오프라인에서도 동작합니다.

## 폴더 구성

```
vibe_crate/
  main.html    ← 메인 페이지 (Crates 타이틀 + 검색창 + 보드 그리드)
  board.html   ← 보드 페이지 (제목=검색어, 검색창, 콜라주, 저장/캡쳐/나가기)
  style.css    ← 공용 스타일
  common.js    ← 공용 렌더링 로직 + localStorage 기반 저장/커스텀 보드 로직
  boards.js    ← 보드 배열 (지금은 예시 데이터, 실제로는 API 응답으로 대체)
  images/      ← 표지 이미지 (도메인별 폴더)
  README.md    ← 이 문서
```

## 데이터 계약

### 아이템 (보드 하나를 채우는 콜라주 조각 하나)

```js
{
  "id": "114709",           // 비노출, 이미지 파일 매핑 + DOM key 용도
  "title": "Toy Story",
  "domain": "movie",         // "movie" | "music" | "book"
  "year": 1995,              // 결측이면 null (화면엔 "xxxx"로 표시됨)
  "image": "images/movie/114709.jpg",
  "url": "http://www.imdb.com/title/tt0114709"
}
```

### 보드 (아이템들을 묶고, 제목=검색어를 가짐)

```js
{
  "id": "b1",
  "title": "비 오는 날 필요한 조합",   // = 사용자가 입력한 검색어
  "section": "recommended" | "mine" | "popular",
  "items": [ {...}, {...}, ... ]
}
```

`main.html`, `board.html` 둘 다 `boards.js`가 정의하는 전역 `BOARDS` 배열을
읽습니다.

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
- 보드 하나당 아이템 개수: 지금 시안은 12개 슬롯 고정. API가 그보다 적거나
  많은 개수를 내려줄 경우 어떻게 할지는 아직 미정
- `section`("오늘의 추천"/"내 보드"/"인기 보드")을 어떤 기준으로 나눌지
  (추천 알고리즘 vs 유저 저장 여부 vs 저장 횟수 집계)는 백엔드·기획 협의 필요

## 화면 동작

**메인 페이지 (`main.html`)**
- 상단 중앙에 사이트 타이틀 "Crates", 그 아래 중앙 정렬 검색창
- 검색창 placeholder: 회색 이탤릭체로 "create your own collection via searching"
- 검색창 아래 3개 섹션, 각 4열 그리드: 오늘의 추천 보드 / 내 보드 / 인기 보드
- 각 보드는 축소된 콜라주 썸네일 + 제목으로 표시, 클릭하면 `board.html?id=...`로 이동
- 검색창에 검색어를 입력하고 제출하면 **새 보드를 즉석 생성**해서 그 보드로
  이동함 (지금은 데모라 기존 아이템 풀을 무작위로 섞어서 채움 — 실제로는
  이 시점에 추천 API를 호출해 새 아이템을 받아와야 함). 새로 만든 보드는
  `localStorage`에 저장되어 "내 보드" 섹션에도 나타남

**보드 페이지 (`board.html`)**
- 상단에 sticky 헤더: 왼쪽 "나가기", 가운데 검색창(항상 표시, 현재 보드 제목이
  기본값으로 채워져 있음), 오른쪽 "내 컬렉션에 저장" / "캡쳐하기" 버튼
- 제목(H1) = 보드의 검색어, 그 아래 콜라주 캔버스
- 이미지는 항상 보임(기본 상태). 카드에 마우스를 올리면
  `domain → title → year → url` 순서로 오버레이가 살짝 딜레이를 두고 나타남
- **내 컬렉션에 저장**: 클릭 시 `localStorage`에 저장 상태 토글, 버튼 라벨이
  "저장됨 ✓"으로 바뀜 (지금은 브라우저 저장소 기반 — 실제로는 로그인 유저 계정에
  귀속되는 백엔드 API 호출로 대체)
- **캡쳐하기**: html2canvas로 제목+콜라주 영역을 PNG로 캡처해 다운로드
- **나가기**: 보드가 저장되지 않은 상태면 "보드를 저장하지 않고 이동하시겠습니까?"
  확인 모달이 뜨고, 저장된 상태면 바로 `main.html`로 이동

## 시안 데이터 다시 만들기 (디자인 쪽 작업)

`scripts/build_vibe_mockup.py`가 이 폴더(`boards.js` + `images/`)를 생성합니다.
아이템 풀은 스크립트 상단 `ITEM_POOL`, 보드 12개의 제목/섹션은 `BOARD_TITLES`에
있고, 아이템의 제목/연도/URL은 전부 실제 데이터셋(`data/enriched/movie_enriched.csv`,
`data/canonical/music_canonical.csv`, `data/canonical/book_canonical.csv`)에서 그때그때 조회합니다 —
하드코딩된 값 없음. 다만 지금은 보드 12개가 같은 아이템 풀을 순환 이동만
해서 재사용하는 상태라, 보드마다 실제로 다른 추천 결과를 보여주는 건 아닙니다
(레이아웃/인터랙션 데모 목적).

```
python3 scripts/build_vibe_mockup.py
```

레이아웃 자체(슬롯 위치·회전각·hover 카드·버튼·모달 스타일)를 바꾸려면
`scripts/vibe_crate/main.html`, `board.html`, `style.css`, `common.js`를
수정한 뒤 스크립트를 다시 실행하면 `mockups/vibe_crate/`에 반영됩니다.
