"""GPU(beast RTX 5090, vLLM)에서 LLM 번역 지연을 잰다 (2026-10-12).

Mac(MLX 4비트 Qwen3-4B)은 규칙 부분을 미리 계산해 둬도 문장당 0.5초였다. GPU로 옮기면
얼마나 빨라지는지 보려고 같은 140문장을 두 방식으로 보낸다.

  single — 한 문장씩 차례로 (사용자 한 명이 기다리는 시간)
  burst  — 20문장을 동시에 (백엔드 부하 테스트처럼 몰릴 때 마지막 요청이 기다리는 시간)

vLLM은 같은 앞부분(규칙 설명)의 KV를 자동으로 재사용한다(prefix caching). CUDA graph를
켜야 실제 서빙 속도가 나오므로 enforce_eager를 쓰지 않는다.

사용 (beast, source env.sh 후):
  ../venv_vllm/bin/python scripts/bench_gpu_translation.py --tag qwen3_4b
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from translate_sentence_bench import MODELS, PROMPT, clean, load_items

OUT = Path(__file__).resolve().parent.parent / "experiments/sentence_translation"


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(MODELS))
    args = ap.parse_args()
    model_id, need_gb = MODELS[args.tag]

    import torch
    from vllm import LLM, SamplingParams

    free, total = torch.cuda.mem_get_info()
    if free < (need_gb + 3) * 2**30:
        raise SystemExit(f"GPU 여유 {free / 2**30:.1f}GB 부족")
    mm = {"limit_mm_per_prompt": {"image": 0}} if args.tag in ("gemma3_27b", "gemma3_4b", "qwen25vl_7b") else {}
    llm = LLM(model=model_id, gpu_memory_utilization=(need_gb + 2) * 2**30 / total,
              max_model_len=1024, **mm)
    sp = SamplingParams(temperature=0, max_tokens=96)
    items = load_items()
    msg = lambda ko: [{"role": "user", "content": PROMPT.format(ko=ko)}]

    for it in items[:5]:                                   # 워밍업
        llm.chat([msg(it["ko"])], sp, use_tqdm=False)

    single = []
    for it in items:
        t = time.perf_counter()
        o = llm.chat([msg(it["ko"])], sp, use_tqdm=False)
        single.append(time.perf_counter() - t)
        it["translation"] = clean(o[0].outputs[0].text)

    burst = []
    for i in range(0, len(items) - 19, 20):
        t = time.perf_counter()
        llm.chat([msg(it["ko"]) for it in items[i:i + 20]], sp, use_tqdm=False)
        burst.append(time.perf_counter() - t)

    stats = {"model": model_id,
             "single_p50": pct(single, .5), "single_p90": pct(single, .9), "single_max": max(single),
             "burst20_p50": statistics.median(burst)}
    print(json.dumps(stats, indent=1))
    json.dump({"model": model_id, "prompt": PROMPT, "latency": stats, "items": items},
              open(OUT / f"gpu_{args.tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
