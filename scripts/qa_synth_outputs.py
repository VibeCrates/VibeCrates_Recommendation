"""
합성 description과 생성 쿼리의 품질 검사 (학습 전 관문).

왜 필요한가:
  세션 18의 3도메인 텍스트 타입 통일은 콘텐츠 텍스트와 학습 라벨을 **양쪽 다** 새로
  만드는 작업이다. 잘못 만들어졌을 때 학습 후 eval로 되짚으면 원인을 분리할 수 없다
  (poet 점수가 안 오른 게 데이터 탓인지 아키텍처 탓인지 구분 불가). 그래서 학습 전에
  데이터만 보고 판정할 수 있는 지표를 여기서 뽑는다.

모든 지표에 **대조군**을 붙이는 게 이 스크립트의 설계 원칙이다. "중복률 3%"는 그 자체로
좋은지 나쁜지 알 수 없다. "원문은 1%인데 합성본은 3%"여야 판단이 된다.
  - 세분성(③): 합성본 vs 원문(description_clean/text) 같은 지표
  - 라벨 오염(⑥): 새 쿼리 vs 폐기한 오염 쿼리 캐시(query_cache_book.contaminated_*.bak)

검사 항목:
  ① 커버리지 — 캐시 건수와 CSV 컬럼 non-null을 따로 세서 비교한다. 8/5 사고(합성 크래시로
     write_output이 안 돌아 캐시엔 있고 CSV엔 없던 상태)를 잡는 지점이다.
  ② 층화 — ID 프리픽스(kdl_/gr_/bx_)와 description_synth_basis로 모든 지표를 쪼갠다.
     BX 42,673권은 장르가 없어 프롬프트에서 Category 줄이 통째로 빠지는데, 그 조건의
     출력은 아직 아무도 본 적이 없다.
  ③ 세분성 붕괴 — 설계 노트가 경고한 함정. coarse mood로 뭉치면 변별력이 무너진다.
  ④ 규칙 위반 — COMMON_RULES 금지 항목(발매일·수상·독자 호명)이 재유입됐는지.
  ⑤ DSV 유효율 — validate_dsv 통과율. 실패가 특정 층에 몰리는지.
  ⑥ 라벨 오염 — 쿼리가 입력 어휘를 그대로 베낀 비율(진단 D의 정량 지표).

사용:
  # 서버에서 CPU로 (GPU는 학습에 쓴다 — 임베딩 대신 어휘 기반 지표만 쓰는 이유)
  python scripts/qa_synth_outputs.py
  python scripts/qa_synth_outputs.py --domains book --sample 30000
"""

import argparse
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.preprocessing import DOMAIN_CONFIG

CACHE_DIR = "data/cache"
DESC_CACHE = {
    "movie": f"{CACHE_DIR}/movie_desc_synth_cache.json",
    "music": f"{CACHE_DIR}/music_desc_synth_cache.json",
    "book":  f"{CACHE_DIR}/book_desc_synth_cache.json",
}
QUERY_CACHE = {d: f"{CACHE_DIR}/query_cache_{d}.json" for d in DOMAIN_CONFIG}
# 8/5에 description_synth 없이 만들어져 폐기한 book 쿼리. ⑥의 대조군으로만 쓴다.
CONTAMINATED_BOOK_QUERIES = f"{CACHE_DIR}/query_cache_book.contaminated_20260805.bak"

# 도메인별 (제목 컬럼, 합성 전 원문 컬럼 후보) — 원문은 ③의 대조군이다.
SOURCE_COLS = {
    "movie": ("Title", ["text"]),
    "music": ("name", ["description", "lyrics"]),
    "book":  ("title", ["description_clean", "description"]),
}

# COMMON_RULES가 금지한 항목들. 프롬프트에 썼다고 지켜지는 게 아니라서 실제로 센다.
RULE_PATTERNS = {
    "발매/출간 정보": r"\b(released|release date|published in|publication|first aired)\b",
    "판매/수상 실적": r"\b(bestsell\w*|award[- ]winning|chart[- ]topping|prize[- ]winning|no\.? ?1\b)\b",
    "독자 호명":      r"\b(you (?:will|'ll|are|can|may)|your \w+|readers will)\b",
    "홍보 문구":      r"\b(must[- ]read|unputdownable|critically acclaimed|beloved classic|page[- ]turner)\b",
    "리뷰 인용":      r"[\"“][^\"”]{25,}[\"”]",
}

# THIN_SOURCE_RULE("소스가 빈약하면 줄거리·인물·사건을 지어내지 말 것") 위반 프록시.
# 소스 없는 층(meta_only/poster/커버only)에서 이 비율이 높으면 환각이다 — 실제로 movie
# 덤프의 poster-only 항목이 "protagonist who grapples with personal demons"처럼
# 있지도 않은 인물·갈등을 만들어냈다. 소스 있는 층에서는 정상 서술이므로 층별로만 본다.
NARRATIVE_CLAIM = (
    r"\b(protagonists?|follows? (?:a|an|the)|story of|journey of|"
    r"(?:struggles?|grapples?|wrestles?) with|discovers?|must (?:find|save|escape|confront))\b"
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this",
    "that", "these", "those", "his", "her", "their", "he", "she", "they", "who", "which",
    "into", "through", "while", "when", "where", "what", "s", "t",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")


# ── 텍스트 유틸 ───────────────────────────────────────────────────────────────

def tokens(text) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def content_tokens(text) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 2}


def shingles(text: str, k: int = 5) -> set[str]:
    """단어 k-gram 집합. 문장 단위보다 재서술 겹침에 민감하다."""
    toks = tokens(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


# ── MinHash + LSH ────────────────────────────────────────────────────────────
# 11만 건 전수 쌍비교(60억 쌍)는 불가능하고, SBERT 임베딩은 GPU를 학습에서 뺏는다.
# MinHash 시그니처 + 밴딩으로 후보 쌍만 추려 정확 Jaccard를 잰다.

MH_PRIME = (1 << 31) - 1


def minhash_signatures(shingle_sets: list[set[str]], perms: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MH_PRIME, perms).astype(np.uint64)
    b = rng.integers(0, MH_PRIME, perms).astype(np.uint64)

    sigs = np.full((len(shingle_sets), perms), np.uint64(MH_PRIME), dtype=np.uint64)
    for i, s in enumerate(shingle_sets):
        if not s:
            continue
        h = np.fromiter(
            (zlib.crc32(x.encode()) & 0x7FFFFFFF for x in s), dtype=np.uint64, count=len(s)
        )
        sigs[i] = ((h[:, None] * a[None, :] + b[None, :]) % MH_PRIME).min(axis=0)
    return sigs


def candidate_pairs(sigs: np.ndarray, bands: int = 8, max_bucket: int = 200) -> set[tuple[int, int]]:
    """밴딩으로 후보 쌍 추출. perms=32/bands=8이면 J≈0.6 이상을 주로 잡는다.
    버킷이 과대하면(동일 텍스트 수백 건) 쌍 수가 폭발하므로 상한을 둔다."""
    n, perms = sigs.shape
    rows = perms // bands
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        chunk = sigs[:, band * rows:(band + 1) * rows]
        for i in range(n):
            buckets[chunk[i].tobytes()].append(i)
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            idxs = idxs[:max_bucket]
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    pairs.add((idxs[x], idxs[y]))
    return pairs


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def granularity_stats(texts: list[str], sample: int, seed: int = 0) -> dict:
    """③ 세분성. 완전중복률 / 근접중복률 / 평균 최대유사도."""
    texts = [t for t in texts if str(t).strip() not in ("", "nan", "None")]
    if not texts:
        return {"n": 0}

    counts = Counter(texts)
    exact_dup = sum(c for c in counts.values() if c > 1) - sum(1 for c in counts.values() if c > 1)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    sub = [texts[i] for i in idx]
    sets = [shingles(t) for t in sub]

    sigs = minhash_signatures(sets)
    pairs = candidate_pairs(sigs)

    near, max_j = set(), np.zeros(len(sub))
    for i, j in pairs:
        s = jaccard(sets[i], sets[j])
        max_j[i] = max(max_j[i], s)
        max_j[j] = max(max_j[j], s)
        if s >= 0.8:
            near.add(i)
            near.add(j)

    return {
        "n": len(texts),
        "완전중복률": exact_dup / len(texts),
        "근접중복률": len(near) / len(sub),      # J≥0.8 쌍에 속한 비율 (샘플 기준)
        "평균최대유사도": float(max_j.mean()),   # 0에 가까울수록 서로 구별됨
        "샘플": len(sub),
    }


# ── 개별 검사 ────────────────────────────────────────────────────────────────

def check_coverage(domain: str, df: pd.DataFrame, id_col: str) -> dict:
    """① 커버리지. 캐시와 CSV를 **따로** 센다 — 8/5 사고가 정확히 둘의 불일치였다.

    단순 건수 차이는 두 가지를 섞는다: (a) 캐시에만 있고 CSV엔 안 실린 것(=사고),
    (b) 중복 ID 때문에 캐시 1건이 여러 행에 매핑된 것(=데이터 문제). 나눠서 센다.
    """
    cache = json.load(open(DESC_CACHE[domain])) if os.path.exists(DESC_CACHE[domain]) else {}
    qcache = json.load(open(QUERY_CACHE[domain])) if os.path.exists(QUERY_CACHE[domain]) else {}
    ids = df[id_col].astype(str)

    out = {
        "rows": len(df),
        "중복ID행": int(ids.duplicated().sum()),
        "캐시": len(cache),
        "쿼리캐시": len(qcache),
    }
    if "description_synth" in df.columns:
        filled = df["description_synth"].notna()
        out["CSV컬럼"] = int(filled.sum())
        # CSV엔 값이 있는데 캐시엔 없는 행 = 지난 실행의 잔존값(무효화된 합성 등)
        out["CSV_잔존"] = int((~ids[filled].isin(cache.keys())).sum())
        # 캐시엔 있는데 CSV의 어느 행에도 안 실린 항목 = write_output 미실행 (8/5 사고)
        out["CSV_미반영"] = len(set(cache) - set(ids))
    if "query" in df.columns:
        out["CSV쿼리"] = int(df["query"].notna().sum())
    return out


def check_rules(texts: list[str]) -> dict:
    """④ 규칙 위반율 + 길이/문장 수 (2~3문장 계약)."""
    out = {}
    n = max(len(texts), 1)
    for label, pat in RULE_PATTERNS.items():
        rx = re.compile(pat, re.I)
        out[label] = sum(1 for t in texts if rx.search(str(t))) / n
    lens = np.array([len(str(t)) for t in texts]) if texts else np.array([0])
    sents = np.array([len(re.findall(r"[.!?](?:\s|$)", str(t))) for t in texts]) if texts else np.array([0])
    out["평균길이"] = float(lens.mean())
    out["문장2~3개비율"] = float(((sents >= 2) & (sents <= 3)).mean())
    return out


def check_title_retention(df: pd.DataFrame, title_col: str) -> float:
    """제목의 고유 토큰이 합성문에 남았는지 — "이 작품 고유의 구체적 디테일" 규칙의 프록시."""
    hit = total = 0
    for title, synth in zip(df[title_col], df["description_synth"]):
        t_toks = {w for w in content_tokens(title) if len(w) >= 4}
        if not t_toks or str(synth).strip() in ("", "nan", "None"):
            continue
        total += 1
        hit += bool(t_toks & content_tokens(synth))
    return hit / total if total else 0.0


def check_queries(df: pd.DataFrame, source_col: str) -> dict:
    """⑤ DSV 유효율 + ⑥ 쿼리가 입력 어휘를 그대로 베낀 비율."""
    from scripts.generate_queries import validate_dsv

    valid = overlaps = counted = 0
    rows = df[df["query"].notna()]
    for q, src in zip(rows["query"], rows[source_col]):
        if validate_dsv(str(q)):
            valid += 1
        q_toks = content_tokens(str(q).replace("|", " "))
        s_toks = content_tokens(src)
        if q_toks and s_toks:
            overlaps += len(q_toks & s_toks) / len(q_toks)
            counted += 1
    return {
        "쿼리커버리지": len(rows) / max(len(df), 1),   # ← 실질 지표. 생성 실패분이 여기 반영된다
        "DSV형식_sanity": valid / len(rows) if len(rows) else 0.0,  # 캐시엔 유효분만 저장되므로 1.0이 정상
        "입력어휘겹침": overlaps / counted if counted else 0.0,
    }


def compare_contaminated_book(df: pd.DataFrame) -> dict | None:
    """⑥ 대조군. 폐기한 오염 쿼리는 '원문 블러브'에서, 새 쿼리는 'description_synth'에서
    나왔다. 각자 자기 입력과의 어휘 겹침을 재서 (D)가 실제로 줄었는지 본다."""
    if not os.path.exists(CONTAMINATED_BOOK_QUERIES):
        return None
    old = json.load(open(CONTAMINATED_BOOK_QUERIES))
    idx = df.set_index(df["asin"].astype(str))

    old_ov, new_ov, n = 0.0, 0.0, 0
    for item_id, old_q in old.items():
        if item_id not in idx.index:
            continue
        row = idx.loc[item_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        new_q, synth = row.get("query"), row.get("description_synth")
        blurb = row.get("description_clean") or row.get("description")
        if pd.isna(new_q) or pd.isna(synth):
            continue
        o = content_tokens(str(old_q).replace("|", " "))
        nq = content_tokens(str(new_q).replace("|", " "))
        if not o or not nq:
            continue
        old_ov += len(o & content_tokens(blurb)) / len(o)
        new_ov += len(nq & content_tokens(synth)) / len(nq)
        n += 1
    if not n:
        return None
    return {"공통아이템": n, "오염본_겹침": old_ov / n, "신규_겹침": new_ov / n}


# ── 층화 ─────────────────────────────────────────────────────────────────────

def stratum_key(domain: str, row) -> str:
    """② BX/Goodreads/Kindle과 합성 근거(basis)로 쪼갠다."""
    parts = []
    if domain == "book":
        parts.append(str(row.get("asin", ""))[:3].rstrip("_") or "?")
    basis = row.get("description_synth_basis")
    parts.append(str(basis) if basis and str(basis) != "nan" else "-")
    return "/".join(parts)


# ── main ─────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def run_domain(domain: str, args, report: list[str]) -> None:
    cfg = DOMAIN_CONFIG[domain]
    title_col, src_candidates = SOURCE_COLS[domain]
    say = lambda s: (print(s, flush=True), report.append(s))

    say(f"\n{'='*72}\n[{domain}] {cfg['csv']}\n{'='*72}")
    df = pd.read_csv(cfg["csv"], low_memory=False)

    # ① 커버리지
    cov = check_coverage(domain, df, cfg["id_col"])
    say("① 커버리지")
    for k, v in cov.items():
        say(f"    {k:10s} {fmt(v)}")
    if "description_synth" not in df.columns:
        say("    ⚠ description_synth 컬럼 없음 — 이하 검사 불가")
        return
    if cov["CSV_미반영"]:
        say(f"    ⚠ 캐시에만 있고 CSV엔 없는 항목 {cov['CSV_미반영']:,}건 — write_output 미실행 의심")
    if cov["CSV_잔존"]:
        say(f"    ⚠ 캐시에 없는데 CSV에 값이 있는 행 {cov['CSV_잔존']:,}건 — 지난 실행의 잔존값")
    if cov["중복ID행"]:
        say(f"    ⚠ 중복 ID {cov['중복ID행']:,}행 — 같은 아이템이 학습 배치에서 false negative가 된다")
    rate = cov["CSV컬럼"] / max(cov["rows"], 1)
    say(f"    채움률 {rate:.1%} {'✅' if rate >= 0.995 else '❌ (임계 99.5%)'}")

    synth = df["description_synth"].dropna().astype(str).tolist()
    src_col = next((c for c in src_candidates if c in df.columns), None)
    source = df[src_col].dropna().astype(str).tolist() if src_col else []

    # ③ 세분성 (대조군: 원문)
    say(f"\n③ 세분성 — 합성본 vs 원문({src_col})")
    g_new = granularity_stats(synth, args.sample)
    g_old = granularity_stats(source, args.sample) if source else {}
    keys = ["완전중복률", "근접중복률", "평균최대유사도"]
    say(f"    {'지표':16s} {'합성본':>10s} {'원문':>10s}")
    for k in keys:
        a = g_new.get(k, 0.0)
        b = g_old.get(k)
        flag = "" if b is None else ("  ⚠ 원문보다 뭉침" if a > b * 1.5 + 0.01 else "")
        say(f"    {k:16s} {a:>10.3f} {('%10.3f' % b) if b is not None else '         -'}{flag}")

    # ④ 규칙 위반 + 제목 토큰 보존
    say("\n④ 규칙 위반율 / 형식")
    for k, v in check_rules(synth).items():
        say(f"    {k:16s} {v:.3f}")
    say(f"    {'제목토큰보존':16s} {check_title_retention(df, title_col):.3f}")

    # ② 층화 — 위 지표를 층별로 다시
    say("\n② 층별 (ID 프리픽스 / 합성 근거)")
    df["_stratum"] = [stratum_key(domain, r) for _, r in df.iterrows()]
    say(f"    {'층':22s} {'건수':>8s} {'평균길이':>8s} {'제목보존':>8s} {'근접중복':>8s} {'서사주장':>8s}")
    narrative = re.compile(NARRATIVE_CLAIM, re.I)
    for key, grp in df.groupby("_stratum"):
        texts = grp["description_synth"].dropna().astype(str).tolist()
        if not texts:
            continue
        r = check_rules(texts)
        g = granularity_stats(texts, min(args.sample, len(texts)))
        claim = sum(1 for t in texts if narrative.search(t)) / len(texts)
        say(f"    {key:22s} {len(texts):>8,} {r['평균길이']:>8.0f} "
            f"{check_title_retention(grp, title_col):>8.3f} {g.get('근접중복률', 0):>8.3f} "
            f"{claim:>8.3f}")
    say("    ※ 서사주장 = 인물·사건 서술 비율. 소스 없는 층(meta_only/poster/cover)에서")
    say("      높으면 THIN_SOURCE_RULE 위반(환각)이다. 소스 있는 층에서는 정상이다.")

    # ⑤⑥ 쿼리
    if "query" in df.columns and df["query"].notna().any():
        say("\n⑤⑥ 쿼리")
        for k, v in check_queries(df, "description_synth").items():
            say(f"    {k:16s} {fmt(v)}")
        if domain == "book":
            cmp = compare_contaminated_book(df)
            if cmp:
                say("    ─ 대조: 폐기한 오염 쿼리(원문 기반) vs 신규(합성본 기반)")
                for k, v in cmp.items():
                    say(f"      {k:14s} {fmt(v)}")
                delta = cmp["오염본_겹침"] - cmp["신규_겹침"]
                say(f"      어휘 베끼기 {'감소' if delta > 0 else '증가'} {abs(delta):.3f}")
    else:
        say("\n⑤⑥ 쿼리 — 아직 없음 (쿼리 단계 미완료)")

    # 육안 검증용 덤프 — 층마다 골고루
    dump = f"{CACHE_DIR}/qa_samples_{domain}.txt"
    with open(dump, "w") as f:
        for key, grp in df.groupby("_stratum"):
            g = grp[grp["description_synth"].notna()].head(args.dump_per_stratum)
            f.write(f"\n{'='*72}\n### 층: {key} ({len(grp):,}건)\n{'='*72}\n")
            for _, r in g.iterrows():
                f.write(f"\n[{r[cfg['id_col']]}] {r[title_col]}\n")
                f.write(f"  원문 : {str(r.get(src_col, ''))[:200]}\n")
                f.write(f"  합성 : {r['description_synth']}\n")
                f.write(f"  쿼리 : {r.get('query', '')}\n")
    say(f"\n샘플 덤프 → {dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["movie", "music", "book"],
                    choices=list(DOMAIN_CONFIG))
    ap.add_argument("--sample", type=int, default=20000,
                    help="세분성 검사 표본 (MinHash+LSH 비용 관리)")
    ap.add_argument("--dump-per-stratum", type=int, default=20)
    ap.add_argument("--out", default=None, help="기본값: data/cache/qa_report_{날짜}.txt")
    args = ap.parse_args()

    report: list[str] = [f"QA 리포트 {datetime.now():%Y-%m-%d %H:%M}"]
    for domain in args.domains:
        run_domain(domain, args, report)

    out = args.out or f"{CACHE_DIR}/qa_report_{datetime.now():%Y%m%d_%H%M}.txt"
    with open(out, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n리포트 → {out}")


if __name__ == "__main__":
    main()
