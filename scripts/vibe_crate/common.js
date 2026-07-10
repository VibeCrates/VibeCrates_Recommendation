// VibeCrates 공용 렌더링/저장 로직 — main.html, board.html 둘 다 로드함.
// boards.js(BOARDS 배열)보다 나중에 로드되어야 함.

var SLOTS = [
  { left: 2,  top: 8,  width: 19, rot: -8, z: 1 },
  { left: 14, top: 4,  width: 19, rot: 7,  z: 2 },
  { left: 35, top: 6,  width: 30, rot: 0,  z: 3 },
  { left: 70, top: 8,  width: 27, rot: 2,  z: 2 },
  { left: 2,  top: 40, width: 15, rot: -6, z: 1 },
  { left: 12, top: 44, width: 15, rot: 5,  z: 2 },
  { left: 74, top: 42, width: 20, rot: -3, z: 2 },
  { left: 3,  top: 62, width: 17, rot: -5, z: 1 },
  { left: 14, top: 66, width: 17, rot: 6,  z: 2 },
  { left: 36, top: 62, width: 16, rot: -4, z: 1 },
  { left: 50, top: 60, width: 16, rot: 5,  z: 2 },
  { left: 74, top: 66, width: 22, rot: -3, z: 2 },
];

var DOMAIN_LABEL = { movie: "Movie", music: "Music", book: "Book" };

// items: [{id,title,domain,year,image,url}, ...]
// opts.interactive = false 면 hover 카드/링크 없이 순수 이미지만 (썸네일용)
function renderBoardCanvas(canvasEl, items, opts) {
  opts = opts || {};
  var interactive = opts.interactive !== false;
  canvasEl.classList.toggle("thumb", !interactive);
  canvasEl.innerHTML = "";

  items.slice(0, SLOTS.length).forEach(function (item, i) {
    var slot = SLOTS[i % SLOTS.length];
    var fig = document.createElement("figure");
    fig.className = "obj-slot";
    fig.style.left = slot.left + "%";
    fig.style.top = slot.top + "%";
    fig.style.width = slot.width + "%";
    fig.style.setProperty("--rot", slot.rot + "deg");
    fig.style.zIndex = slot.z;

    var hoverHtml = "";
    if (interactive) {
      var yearText = item.year || "xxxx";
      hoverHtml =
        '<div class="hover-card">' +
          '<span class="hc-domain ' + item.domain + '">' + (DOMAIN_LABEL[item.domain] || item.domain) + '</span>' +
          '<span class="hc-title">' + item.title + '</span>' +
          '<span class="hc-year">' + yearText + '</span>' +
          '<a class="hc-url" href="' + item.url + '" target="_blank" rel="noopener">바로가기 ↗</a>' +
        '</div>';
    }

    fig.innerHTML =
      '<div class="frame">' +
        '<img src="' + item.image + '" alt="' + item.title + '" loading="lazy" />' +
        hoverHtml +
      '</div>';
    canvasEl.appendChild(fig);
  });
}

// ---------- localStorage 기반 "내 컬렉션" / 커스텀 보드 ----------
// 실제 서비스에서는 로그인한 유저 계정에 귀속되는 값이라 백엔드 API로 대체될 부분.

var STORAGE_SAVED_KEY = "vibecrates:savedBoardIds";
var STORAGE_CUSTOM_PREFIX = "vibecrates:board:";

function getSavedBoardIds() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_SAVED_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

function isBoardSaved(id) {
  return getSavedBoardIds().indexOf(id) !== -1;
}

function toggleBoardSaved(id) {
  var ids = getSavedBoardIds();
  var idx = ids.indexOf(id);
  if (idx === -1) {
    ids.push(id);
  } else {
    ids.splice(idx, 1);
  }
  localStorage.setItem(STORAGE_SAVED_KEY, JSON.stringify(ids));
  return idx === -1; // true면 방금 저장됨, false면 방금 저장 해제됨
}

function saveCustomBoard(board) {
  localStorage.setItem(STORAGE_CUSTOM_PREFIX + board.id, JSON.stringify(board));
}

function getCustomBoard(id) {
  try {
    var raw = localStorage.getItem(STORAGE_CUSTOM_PREFIX + id);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function getAllCustomBoards() {
  var out = [];
  for (var i = 0; i < localStorage.length; i++) {
    var key = localStorage.key(i);
    if (key && key.indexOf(STORAGE_CUSTOM_PREFIX) === 0) {
      try {
        out.push(JSON.parse(localStorage.getItem(key)));
      } catch (e) { /* skip */ }
    }
  }
  return out;
}

// 검색어로 새 보드 생성 (데모용 — 실제로는 추천 API 응답으로 items를 채워야 함)
function createBoardFromQuery(query, itemPool) {
  var id = "custom-" + Date.now();
  var shuffled = itemPool.slice().sort(function () { return Math.random() - 0.5; });
  return { id: id, title: query, section: "mine", items: shuffled };
}

function findBoardById(id) {
  var fromStatic = (window.BOARDS || []).find(function (b) { return b.id === id; });
  if (fromStatic) return fromStatic;
  return getCustomBoard(id);
}
