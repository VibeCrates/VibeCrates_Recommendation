"""부정 표현 검색어 100개를 사전을 거치지 않고 Gemma 3 4B(AstraGo vLLM)와 opus-mt(Mac)에 직접 넣어 번역한다.

서비스와 같은 프롬프트·후처리(src/api/translation.py)를 쓴다. 서비스 경로였다면 사전이 먼저 받았을지도 함께 적는다.
Gemma 제한 시간은 서비스(1초)보다 넉넉히 둔다. 번역 품질만 보려는 것이고, 걸린 시간은 따로 남긴다.
"""
import csv, json, sys, time, urllib.request
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer
from src.api.translation import LLM_PROMPT, LLMTranslator, DictTranslator, MT_MODEL, MT_BEAMS

OUT = Path(sys.argv[1])
GEMMA = "http://100.115.172.111:8001/v1/chat/completions"
rows = list(csv.DictReader(open(OUT / "queries.csv", encoding="utf-8")))
queries = [r["query"] for r in rows]

def gemma(text):
    body = json.dumps({"model": "gemma3-4b", "temperature": 0, "max_tokens": 96,
                       "messages": [{"role": "user", "content": LLM_PROMPT.format(ko=text)}]}).encode()
    req = urllib.request.Request(GEMMA, data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        out = LLMTranslator._clean(json.load(r)["choices"][0]["message"]["content"])
    return out, round((time.perf_counter() - t) * 1000)

tok = MarianTokenizer.from_pretrained(MT_MODEL)
mt = MarianMTModel.from_pretrained(MT_MODEL).eval()
opus = []
for i in range(0, len(queries), 20):
    batch = tok(queries[i:i + 20], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = mt.generate(**batch, max_new_tokens=64, num_beams=MT_BEAMS)
    opus += [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True)]

dictionary = DictTranslator()
res = []
for r, o in zip(rows, opus):
    g, ms = gemma(r["query"])
    rec = {**r, "dict": dictionary.translate(r["query"]), "gemma": g, "gemma_ms": ms, "opus": o}
    res.append(rec)
    print(f'{r["id"]:>3} {r["query"]} | G: {g} | O: {o}' + (f' | 사전: {rec["dict"]}' if rec["dict"] else ''), flush=True)
json.dump(res, open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
