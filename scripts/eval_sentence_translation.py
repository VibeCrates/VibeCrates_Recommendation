"""문장형 검색어 번역 후보를 서빙 모델의 쿼리 벡터로 채점한다 (Mac, 재학습 없음).

무엇을 재는가:
  같은 뜻의 영어 기준 문장을 인코딩한 벡터와, 번역문을 인코딩한 벡터의 코사인 유사도.
  1에 가까울수록 번역을 거친 검색이 영어로 검색한 것과 같아진다. 9월 측정(평가 쿼리 40쌍)은
  opus-mt 0.780, Qwen2.5-VL-7B 0.891이었다.

  0.70 미만은 "크게 빗나감"으로 따로 센다 — 평균이 비슷해도 몇 건이 뜻을 통째로 잃으면
  그 사용자에게는 검색이 실패한 것이다.

후보:
  ko_raw  — 번역 없이 한국어를 그대로 인코딩 (하한)
  served  — 지금 서빙 경로(사전 → opus-mt)
  <tag>   — experiments/sentence_translation/<tag>.json (scripts/translate_sentence_bench.py, beast)

기준 영어는 평가 쿼리 40쌍은 사람이 쓴 원문, 실사용형 100문장(data/translation/sentence_bench.tsv)은
2026-10-12에 직접 쓴 번역이다. 후보 번역기 어느 것의 출력도 보고 쓰지 않았다.

사용:
  python scripts/eval_sentence_translation.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F

IN_DIR = Path("experiments/sentence_translation")
MISS = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "models/trained_model.pt"))
    ap.add_argument("--out", default="experiments/sentence_translation/scores.csv")
    args = ap.parse_args()

    from scripts.eval_lang import load_model
    from src.api.translation import QueryTranslator

    runs = {p.stem: json.load(open(p)) for p in sorted(IN_DIR.glob("*.json"))}
    runs = {k: v for k, v in runs.items() if "items" in v}          # mac_latency.json 등은 뺀다
    if not runs:
        raise SystemExit(f"{IN_DIR}에 번역 결과가 없다 — beast에서 translate_sentence_bench.py 먼저")
    items = next(iter(runs.values()))["items"]
    df = pd.DataFrame([{"type": it["type"], "ko": it["ko"], "en": it["en"]} for it in items])
    for tag, run in runs.items():
        assert [it["ko"] for it in run["items"]] == df["ko"].tolist(), tag
        df[tag] = [it["translation"] for it in run["items"]]
    df["ko_raw"] = df["ko"]
    df["served"] = QueryTranslator().translate(df["ko"].tolist())

    model = load_model(args.model_path)
    model.eval()

    def enc(texts):
        with torch.no_grad():
            return F.normalize(model.encode_query(list(texts)), dim=1).cpu()

    ref = enc(df["en"])
    cands = ["ko_raw", "served"] + list(runs)
    for c in cands:
        df[f"cos_{c}"] = (enc(df[c]) * ref).sum(1).numpy()

    df["set"] = df["type"].where(~df["type"].str.startswith("eval_"), "eval40")
    df.loc[df["set"] != "eval40", "set"] = "real100"
    rows = []
    for c in cands:
        col = df[f"cos_{c}"]
        r = {"translator": c, "all": col.mean(), "miss<0.70": int((col < MISS).sum())}
        for s in ("eval40", "real100"):
            r[s] = col[df["set"] == s].mean()
        for t in sorted(df.loc[df["set"] == "real100", "type"].unique()):
            r[t] = col[df["type"] == t].mean()
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("all", ascending=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))

    df.to_csv(args.out, index=False)
    summary.round(4).to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
