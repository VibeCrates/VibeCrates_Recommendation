"""Mac(MLX)에서 LLM 번역 지연을 줄이는 방법들을 잰다 (2026-10-12).

기준: Qwen3-4B-Instruct 4비트, 140문장(sentence_bench.tsv 100 + 평가 쿼리 40)을 한 문장씩.
처음 측정은 중앙값 1.06초였다. 시간은 두 부분이다.
  - 프롬프트 읽기(prefill): 규칙 설명이 매번 같은데 매번 다시 읽는다
  - 생성(decode): 출력 토큰 수에 비례한다

변형:
  base   — 처음 측정과 같은 조건
  cache  — 규칙 부분의 KV 캐시를 한 번만 만들어 두고 매 요청은 한국어와 말미만 읽는다
  short  — cache + 출력을 짧은 검색 구절로 (출력 토큰 감소, 품질은 따로 채점)
  spec   — cache + Qwen3-0.6B 초안 모델로 추측 디코딩 (greedy라 출력은 base와 같아야 한다)
  batch  — 동시 요청 8개를 한 번에 생성할 때 요청당 시간

번역 결과는 experiments/sentence_translation/mac_<변형>.json으로 남기고
scripts/eval_sentence_translation.py로 채점한다.
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load, stream_generate, batch_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

from scripts.translate_sentence_bench import PROMPT, clean, load_items

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DRAFT = "mlx-community/Qwen3-0.6B-4bit"
OUT_DIR = _ROOT / "experiments/sentence_translation"

SHORT_PROMPT = (
    "Translate this Korean search query from a book/movie/music recommendation app into a "
    "short English search phrase.\n\n"
    "Rules:\n"
    "- Keep every mood, image, setting, genre, person and title. Do not turn a metaphor into a "
    "literal or religious term.\n"
    "- Drop request words (recommend, tell me, play, please) and question form.\n"
    "- Slang and internet abbreviations become plain English.\n"
    "- Titles and names use their usual English form.\n\n"
    "Korean: {ko}\n\n"
    "Output the English phrase only. No quotes, no explanation."
)


def prefill(model, ids):
    """ids를 읽은 KV 캐시. 생성은 하지 않는다."""
    cache = make_prompt_cache(model)
    model(mx.array(ids)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


class Translator:
    """규칙 부분(프롬프트 앞쪽) KV 캐시를 한 번 만들어 두고 복사해 쓴다."""

    def __init__(self, model, tok, template, use_cache=True):
        self.model, self.tok, self.use_cache = model, tok, use_cache
        self.template = template
        full = self._ids("@@KO@@")
        cut = self.tok.apply_chat_template(
            [{"role": "user", "content": template.format(ko="@@KO@@")}],
            add_generation_prompt=True, tokenize=False).index("@@KO@@") - 1   # "Korean: "의 공백 앞에서 자른다 — 공백은 뒤 토큰에 붙는다
        text = self._text("@@KO@@")
        self.prefix_ids = self.tok.encode(text[:cut], add_special_tokens=False)
        assert full[:len(self.prefix_ids)] == self.prefix_ids
        self.cache = None
        if use_cache:
            self.cache = prefill(model, self.prefix_ids)

    def _text(self, ko):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": self.template.format(ko=ko)}],
            add_generation_prompt=True, tokenize=False)

    def _ids(self, ko):
        return self.tok.encode(self._text(ko), add_special_tokens=False)

    def __call__(self, ko, draft=None):
        ids = self._ids(ko)
        kw = {}
        if self.use_cache:
            assert ids[:len(self.prefix_ids)] == self.prefix_ids
            ids = ids[len(self.prefix_ids):]
            kw["prompt_cache"] = copy.deepcopy(self.cache)
            if draft is not None:
                kw["prompt_cache"] = kw["prompt_cache"] + copy.deepcopy(self.draft_cache)
        text, last = "", None
        for r in stream_generate(self.model, self.tok, ids, max_tokens=96, draft_model=draft, **kw):
            text += r.text
            last = r
        return clean(text), last


def timed(fn, items):
    lat, outs = [], []
    fn(items[0]["ko"])                                   # 워밍업
    for it in items:
        t = time.perf_counter()
        out, r = fn(it["ko"])
        lat.append(time.perf_counter() - t)
        outs.append((out, r.generation_tokens if r else 0))
    return lat, outs


def report(name, lat, outs):
    s = sorted(lat)
    toks = statistics.mean(n for _, n in outs)
    print(f"{name:6s} p50 {s[len(s)//2]:.3f}s  p90 {s[int(len(s)*.9)]:.3f}s  max {s[-1]:.3f}s  "
          f"출력 {toks:.1f}토큰", flush=True)
    return {"p50": s[len(s) // 2], "p90": s[int(len(s) * .9)], "max": s[-1], "out_tokens": toks}


def save(tag, items, outs, template, stats):
    data = {"model": MODEL, "prompt": template, "latency": stats,
            "items": [{**it, "translation": o} for it, (o, _) in zip(items, outs)]}
    json.dump(data, open(OUT_DIR / f"{tag}.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base", "cache", "short", "spec", "batch"])
    args = ap.parse_args()

    items = load_items()
    model, tok = load(MODEL)
    stats = {}

    if "base" in args.variants:
        tr = Translator(model, tok, PROMPT, use_cache=False)
        lat, outs = timed(tr, items)
        stats["base"] = report("base", lat, outs)
        save("mac_qwen3_4b_base", items, outs, PROMPT, stats["base"])

    tr = Translator(model, tok, PROMPT)
    if "cache" in args.variants:
        lat, outs = timed(tr, items)
        stats["cache"] = report("cache", lat, outs)
        save("mac_qwen3_4b_cache", items, outs, PROMPT, stats["cache"])

    if "short" in args.variants:
        trs = Translator(model, tok, SHORT_PROMPT)
        lat, outs = timed(trs, items)
        stats["short"] = report("short", lat, outs)
        save("mac_qwen3_4b_short", items, outs, SHORT_PROMPT, stats["short"])

    if "spec" in args.variants:
        draft, _ = load(DRAFT)
        tr.draft_cache = prefill(draft, tr.prefix_ids)
        lat, outs = timed(lambda ko: tr(ko, draft=draft), items)
        stats["spec"] = report("spec", lat, outs)
        save("mac_qwen3_4b_spec", items, outs, PROMPT, stats["spec"])

    if "batch" in args.variants:
        prompts = [tok.encode(tr._text(it["ko"]), add_special_tokens=False) for it in items]
        per = []
        for i in range(0, len(prompts) - 7, 8):
            t = time.perf_counter()
            batch_generate(model, tok, prompts[i:i + 8], max_tokens=96)
            per.append(time.perf_counter() - t)
        print(f"batch  8개 묶음 중앙값 {statistics.median(per):.3f}s → 요청당 {statistics.median(per)/8:.3f}s")
        stats["batch8"] = {"batch_p50": statistics.median(per)}

    json.dump(stats, open(OUT_DIR / "mac_latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
