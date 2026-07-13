"""
Movie 쿼리 캐시에서 불량 항목을 제거하여 재생성 대상으로 만듦.
불량 기준: 파이프 3분할 아님 | 번호 prefix | 페르소나 레이블 포함
영어 포함은 영화 제목 특성상 허용.

실행: /opt/conda/envs/ltv/bin/python scripts/clean_movie_cache.py
이후: /opt/conda/envs/ltv/bin/python scripts/generate_queries.py --domain movie --model-type qwen
"""
import re
import json
import pandas as pd

CACHE_PATH = "data/cache/query_cache_movie.json"
CSV_PATH   = "data/canonical/movie_canonical.csv"

LABEL_PATTERN = re.compile(r"시인:|공간:|철학자:|Poet:|Space:|Philosopher:")
NUM_PATTERN   = re.compile(r"^\d+\.")

def is_bad(query: str) -> bool:
    if not query or query == "nan":
        return True
    parts = [p.strip() for p in query.split("|")]
    if len(parts) != 3:
        return True
    if NUM_PATTERN.match(query):
        return True
    if LABEL_PATTERN.search(query):
        return True
    return False

with open(CACHE_PATH) as f:
    cache = json.load(f)

before = len(cache)
bad_ids = [k for k, v in cache.items() if is_bad(str(v))]
for k in bad_ids:
    del cache[k]
after = len(cache)

with open(CACHE_PATH, "w") as f:
    json.dump(cache, f, ensure_ascii=False)

# CSV query 컬럼도 갱신
df = pd.read_csv(CSV_PATH, engine="python")
df["imdbId"] = df["imdbId"].astype(str)
df["query"] = df["imdbId"].map(cache)
df.to_csv(CSV_PATH, index=False)

print(f"캐시 정리 완료")
print(f"  전체:    {before:,}건")
print(f"  제거:    {len(bad_ids):,}건")
print(f"  잔존:    {after:,}건")
print(f"  재생성 대상: {len(df) - after:,}건")
