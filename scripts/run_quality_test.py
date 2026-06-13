"""
실제 generate_queries.py의 프롬프트/파라미터 그대로 사용.
50건 랜덤 샘플, 도메인별 품질 측정.
"""
import os, sys, json, random
import torch
import pandas as pd
from io import BytesIO
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from generate_queries import (
    DOMAIN_CONFIGS, PROMPT_TEMPLATE, build_synopsis, load_image, validate_dsv,
    load_qwen, generate_query_qwen
)

SAMPLE_N = 50
SEED = 42

def run_test(domain: str, processor, model):
    cfg = DOMAIN_CONFIGS[domain]
    df = pd.read_csv(cfg["csv"], engine="python")
    random.seed(SEED)
    samples = random.sample(df.to_dict("records"), min(SAMPLE_N, len(df)))

    results = []
    for row in samples:
        row = pd.Series(row)
        item_id = str(row[cfg["id_col"]])
        synopsis = build_synopsis(domain, row)
        prompt = PROMPT_TEMPLATE.format(synopsis=synopsis, role=cfg["role"])

        try:
            image = load_image(domain, item_id, str(row[cfg["image_col"]])) if cfg["has_image"](row) else None
        except Exception:
            image = None

        raw = generate_query_qwen(processor, model, image, prompt)
        dsv = validate_dsv(raw)

        has_eng   = bool(__import__('re').search(r'[A-Za-z]', raw))
        has_num   = bool(__import__('re').match(r'^\d+\.', raw))
        has_label = bool(__import__('re').search(r'시인:|Space:|Poet:|Philosopher:|철학자:|공간:', raw))
        parts     = raw.split("|")

        results.append({
            "id": item_id,
            "raw": raw,
            "dsv_valid": dsv is not None,
            "has_eng": has_eng,
            "has_num_prefix": has_num,
            "has_label": has_label,
            "n_parts": len(parts),
        })
        sys.stdout.write(f"\r[{domain}] {len(results)}/{SAMPLE_N}")
        sys.stdout.flush()

    print(f"\n\n=== [{domain.upper()}] 품질 리포트 ({SAMPLE_N}건) ===")
    total = len(results)
    valid     = sum(1 for r in results if r["dsv_valid"])
    eng       = sum(1 for r in results if r["has_eng"])
    num_pre   = sum(1 for r in results if r["has_num_prefix"])
    label     = sum(1 for r in results if r["has_label"])
    bad_pipe  = sum(1 for r in results if r["n_parts"] != 3)

    print(f"  유효 DSV (파이프 3분할): {valid}/{total} ({valid/total*100:.0f}%)")
    print(f"  영어 포함:               {eng}/{total} ({eng/total*100:.0f}%)")
    print(f"  번호 prefix:             {num_pre}/{total} ({num_pre/total*100:.0f}%)")
    print(f"  페르소나 레이블 잔존:    {label}/{total} ({label/total*100:.0f}%)")
    print(f"  파이프 분할 이상:        {bad_pipe}/{total} ({bad_pipe/total*100:.0f}%)")
    print()
    print("  [샘플 출력 10건]")
    for r in results[:10]:
        mark = "✓" if r["dsv_valid"] else "✗"
        print(f"  {mark} {r['raw'][:80]}")

    return results

print("모델 로드 중...")
processor, model = load_qwen("Qwen/Qwen2.5-VL-7B-Instruct")
print("완료\n")

run_test("movie", processor, model)
run_test("music", processor, model)
run_test("book",  processor, model)
