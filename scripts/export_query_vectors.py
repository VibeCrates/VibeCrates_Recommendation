"""학습에 쓴 쿼리를 벡터로 만들어 백엔드에 넘길 CSV를 만든다 (2026-10-09 결정).

무엇을 넘기는가:
  `data/train_dataset.csv`의 `query` 열(아이템마다 Qwen이 만든 쿼리 3개, `|`로 이음)에서
  고유 쿼리 문장을 모아, 도메인별 CSV 세 개로 낸다. 열은 아이템 번들과 같은 규약이다.
      query, vector_b64, model_version
  - vector_b64: (768,) float32, L2 정규화, `.astype('<f4').tobytes()`를 base64로 감쌈
  - 같은 쿼리 문장이 여러 도메인의 학습 데이터에 나오면 각 도메인 파일에 같은 벡터로 들어간다.

왜 API 코드를 그대로 부르는가:
  벡터는 `/search/vector`가 내는 것과 같아야 백엔드가 두 출처를 섞어 써도 된다. 그래서
  `ModelManager.load_model`·`encode_queries`와 `prepare_queries`(한국어 번역)를 직접 호출한다.
  같은 로직을 여기 복제하면 언젠가 갈라진다(build_synopsis / _build_content_text 전례).

model_version을 직접 받는 이유:
  `checkpoint_version()`은 파일명과 수정 시각으로 만든다. 인코딩은 GPU가 있는 beast에서 하는데
  거기 파일 이름(best_stage2_qlora_off.pt)과 시각은 서빙 중인 Mac의 trained_model.pt와 다르다.
  그래서 서빙 체크포인트의 값을 받아 붙이되, `--expect-sha256`으로 두 파일이 바이트 단위로
  같은지 먼저 확인한다. 다르면 중단한다.

사용 (beast):
  python scripts/export_query_vectors.py --model models/best_stage2_qlora_off.pt \\
      --model-version trained_model.pt@1786697109 \\
      --expect-sha256 ec4cceaeec7926c442b134da465273debe807712765883423a828e4d4de246b4 \\
      --out dist/query_vectors_20261009
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DOMAINS = ("movie", "music", "book")
HANGUL = re.compile(r"[가-힣]")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/train_dataset.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-version", required=True, help="서빙 체크포인트의 checkpoint_version() 값")
    ap.add_argument("--expect-sha256", help="서빙 체크포인트의 SHA-256 — 다르면 중단")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    if args.expect_sha256:
        got = sha256(args.model)
        if got != args.expect_sha256:
            sys.exit(f"체크포인트 불일치: {args.model} = {got}, 기대값 {args.expect_sha256}")
        print(f"체크포인트 확인: {args.model} SHA-256 일치")

    # ── 쿼리 모으기 ────────────────────────────────────────────────────────
    d = pd.read_csv(args.dataset, usecols=["query", "domain"])
    q = d.assign(query=d["query"].astype(str).str.split("|")).explode("query")
    q["query"] = q["query"].str.strip()
    q = q[q["query"] != ""]
    per_domain = {dom: sorted(set(q.loc[q["domain"] == dom, "query"])) for dom in DOMAINS}
    uniq = sorted(set().union(*per_domain.values()))
    print(f"쿼리 {len(q):,}개 → 고유 {len(uniq):,}개 "
          f"({', '.join(f'{k} {len(v):,}' for k, v in per_domain.items())})")

    # ── 인코딩: API와 같은 경로 ────────────────────────────────────────────
    from src.api.dependencies import ModelManager
    from src.api.translation import prepare as prepare_queries

    hangul = [s for s in uniq if HANGUL.search(s)]
    print(f"한글이 섞인 쿼리 {len(hangul)}개 (API처럼 번역 후 인코딩)")

    mm = ModelManager()
    mm.load_model(args.model)
    vecs = np.empty((len(uniq), 768), dtype="<f4")
    t0 = time.time()
    for i in range(0, len(uniq), args.batch):
        chunk = uniq[i:i + args.batch]
        used, _ = prepare_queries(chunk)
        vecs[i:i + len(chunk)] = mm.encode_queries(used, normalize=True).numpy().astype("<f4")
        if (i // args.batch) % 50 == 0:
            print(f"  {i + len(chunk):,}/{len(uniq):,}  ({time.time() - t0:.0f}s)", flush=True)
    norms = np.linalg.norm(vecs, axis=1)
    print(f"인코딩 완료 {time.time() - t0:.0f}s, L2 노름 {norms.min():.6f} ~ {norms.max():.6f}")
    pos = {s: i for i, s in enumerate(uniq)}

    # ── 쓰기 ─────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "model_version": args.model_version,
        "model_sha256": args.expect_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": 768, "metric": "inner_product", "normalized": True, "format": "csv",
        "source": os.path.basename(args.dataset),
        "unique_queries_total": len(uniq),
        "note": ("각 {domain}_query_vectors.csv는 열 세 개 — query, vector_b64, model_version. "
                 "vector_b64는 아이템 번들과 같다: (768,) float32, L2 정규화, "
                 "`np.frombuffer(base64.b64decode(v), dtype='<f4')`. 같은 쿼리 문장이 여러 "
                 "도메인 파일에 있을 수 있고 그때 벡터는 같다. model_version이 아이템 번들의 "
                 "값과 같아야 같은 공간이다."),
        "domains": {},
    }
    for dom, qs in per_domain.items():
        path = os.path.join(args.out, f"{dom}_query_vectors.csv")
        idx = [pos[s] for s in qs]
        b64 = [base64.b64encode(vecs[i].tobytes()).decode("ascii") for i in idx]
        pd.DataFrame({"query": qs, "vector_b64": b64, "model_version": args.model_version}) \
            .to_csv(path, index=False, encoding="utf-8")
        manifest["domains"][dom] = {"count": len(qs), "file": os.path.basename(path),
                                    "sha256": sha256(path),
                                    "bytes": os.path.getsize(path)}
        print(f"  {dom}: {len(qs):,}행 → {path} ({os.path.getsize(path) / 1e9:.2f}GB)")
    sets = [set(v) for v in per_domain.values()]
    shared = sum(1 for s in uniq if sum(s in st for st in sets) > 1)
    manifest["queries_in_multiple_domains"] = shared
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"여러 도메인에 걸친 쿼리 {shared:,}개. manifest → {args.out}/manifest.json")


if __name__ == "__main__":
    main()
