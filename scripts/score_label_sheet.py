"""채워진 라벨 시트로 쌍대 판정기를 채점한다 — 특히 어휘가 오도하는 쌍에서.

무엇을 답하는가:
  쌍대 판정기는 어휘 단서가 없을 때 92.6%로 사람과 맞는다. 모르는 것은 **어휘가 반대
  방향을 가리킬 때**다. 이 시트는 (어휘 겹침 있음, 없음) 쌍을 사람에게 물어 그 경우를
  만들어 낸다 — 사람이 '없음' 쪽을 고른 쌍이 곧 역방향 쌍이다.

  그래서 두 값을 낸다.
    (1) 순방향(사람도 겹침 쪽) 정확도  — 구분이 안 되는 구간. 참고용.
    (2) **역방향(사람이 안 겹침 쪽) 정확도** — 이것이 저울을 믿을지 말지를 가른다.

사용:
  python scripts/score_label_sheet.py --sheet experiments/label_sheet_20260911.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.judge_eval_relevance import CACHE_DIR
from scripts.judge_pairwise import make_prompt, parse_choice, prompt_signature, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", help="기본값: <sheet>_key.csv")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="experiments/label_sheet_score.txt")
    args = ap.parse_args()

    sheet = pd.read_csv(args.sheet)
    key = pd.read_csv(args.key or args.sheet.replace(".csv", "_key.csv"))
    d = sheet.merge(key, on="pair_id", suffixes=("", "_k"))
    d["선택"] = d["선택"].astype(str).str.strip()
    done = d[d["선택"].isin(["1", "2"])]
    print(f"쌍 {len(d)}개 중 라벨된 것 {len(done)}개 (무응답·비슷함 {len(d)-len(done)})")
    if not len(done):
        print("라벨이 비어 있다. 시트의 '선택' 칸을 1 또는 2로 채워야 한다.")
        return

    # 사람이 어느 쪽을 골랐나 → 순방향/역방향 구분
    done = done.copy()
    done["human_picked_overlap"] = done["선택"].astype(int) == done["_ov_side"]
    n_rev = int((~done.human_picked_overlap).sum())
    print(f"  사람이 어휘 겹침 쪽 선택 {int(done.human_picked_overlap.sum())} / "
          f"안 겹침 쪽 선택 {n_rev}  ← 역방향 쌍")

    sig = prompt_signature()
    cache_path = os.path.join(CACHE_DIR, f"judge_cache_pair_{sig}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    # 판정기에게 같은 두 후보를 양방향으로 묻는다. a=후보1, b=후보2 (시트 그대로).
    units = []
    for r in done.itertuples():
        doms = {r._ov_item.split(":")[0], r._zero_item.split(":")[0]}
        units.append({"uid": f"labelsheet|{r.pair_id}", "query": r.query,
                      "noun": doms.pop() if len(doms) == 1 else "item",
                      "a": [str(r.후보1)], "b": [str(r.후보2)]})

    todo = [(u, f) for u in units for f in ("a", "b")
            if f"{sig}|후보1|후보2|{u['uid']}|{f}" not in cache]
    print(f"판정 대상 {len(todo)}건 (캐시 {len(cache):,}건)")
    if todo:
        from scripts.vllm_runner import VLLMRunner, chunks
        runner = VLLMRunner(args.model_id, max_new_tokens=4)
        os.makedirs(CACHE_DIR, exist_ok=True)
        for batch in chunks(todo, args.batch):
            items = [(make_prompt(u, f, True), None) for u, f in batch]
            for (u, f), raw in zip(batch, runner.generate(items)):
                c = parse_choice(raw)
                if c:
                    cache[f"{sig}|후보1|후보2|{u['uid']}|{f}"] = c
            json.dump(cache, open(cache_path, "w"))

    rows = []
    for u, (_, r) in zip(units, done.iterrows()):
        vs = {}
        for f in ("a", "b"):
            c = cache.get(f"{sig}|후보1|후보2|{u['uid']}|{f}")
            if c is None:
                break
            vs[f] = verdict(c, f)          # 'a' = 후보1, 'b' = 후보2
        if len(vs) < 2:
            continue
        agree_dir = vs["a"] == vs["b"]
        judge_pick = (1 if vs["a"] == "a" else 2) if agree_dir else None
        rows.append({"pair_id": r.pair_id, "style": r["style"],
                     "human": int(r["선택"]), "judge": judge_pick,
                     "reverse": not r.human_picked_overlap, "consistent": agree_dir})
    res = pd.DataFrame(rows)

    L = ["라벨 시트 채점", "=" * 60, "",
         f"라벨 {len(done)}개 / 양방향 일관 {int(res.consistent.sum())} "
         f"({res.consistent.mean():.1%})", ""]
    L += ["  %-28s %5s %6s %6s %9s" % ("구분", "쌍수", "일치", "불일치", "정확도")]
    for lab, g in (("순방향 (사람도 겹침 쪽)", res[~res.reverse]),
                   ("역방향 (사람이 안 겹침 쪽)", res[res.reverse]),
                   ("전체", res)):
        gg = g[g.judge.notna()]
        ok = int((gg.judge == gg.human).sum())
        L.append("  %-28s %5d %6d %6d %8.1f%%"
                 % (lab, len(g), ok, len(gg) - ok, 100 * ok / max(len(gg), 1)))
    L += ["", "※ 역방향 정확도가 높으면 어휘가 오도해도 쌍대 판정을 믿을 수 있다.",
          "   50% 근처면 판정기가 어휘를 따라간다는 뜻이고, 4-1은 이 저울로 평가할 수 없다."]
    rep = "\n".join(L)
    print("\n" + rep)
    open(args.out, "w").write(rep + "\n")
    res.to_csv(args.out.replace(".txt", "_rows.csv"), index=False)
    print(f"\n리포트 → {args.out}")


if __name__ == "__main__":
    main()
