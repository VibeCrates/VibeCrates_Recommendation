"""
Generates DSV queries (3 English evocative queries separated by |) for each
Movie/Music/Book item using Qwen2.5-VL or PaliGemma 2.
Checkpoints every 50 items to data/cache/query_cache_{domain}.json.

Usage:
  /opt/conda/envs/ltv/bin/python scripts/generate_queries.py --domain movie --model-type qwen
  /opt/conda/envs/ltv/bin/python scripts/generate_queries.py --domain music --model-type qwen --limit 100
  /opt/conda/envs/ltv/bin/python scripts/generate_queries.py --domain movie --model-type paligemma
"""

import os
import json
import argparse

import torch
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

CHECKPOINT_EVERY = 50
PLACEHOLDER_IMG = Image.new("RGB", (448, 448), (128, 128, 128))

DOMAIN_CONFIGS = {
    "movie": {
        "csv": "data/canonical/movie_canonical.csv",
        "id_col": "imdbId",
        "image_col": "Poster",
        "role": "a film marketing expert",
        "has_image": lambda row: pd.notna(row.get("Poster")) and str(row.get("Poster", "")).startswith("http"),
    },
    "music": {
        "csv": "data/canonical/music_canonical.csv",
        "id_col": "id",
        "image_col": "img",
        "role": "a music curator",
        "has_image": lambda row: pd.notna(row.get("img")) and str(row.get("img", "")) not in ("no", "nan", ""),
    },
    "book": {
        "csv": "data/canonical/book_canonical_v2.csv",
        "id_col": "asin",
        "image_col": "imgUrl",
        "role": "a book editor",
        "has_image": lambda row: pd.notna(row.get("imgUrl")) and str(row.get("imgUrl", "")).startswith("http"),
    },
}

PROMPT_TEMPLATE = (
    "[Context]\nSynopsis: {synopsis}\n\n"
    "[Instruction]\nYou are {role}. "
    "Combine the visual mood of the image (color tone, lighting, composition) with the text information "
    "to generate one English evocative search query per persona below.\n\n"
    "[Personas]\n"
    "1. Poet: metaphorical, lyrical imagery. e.g. 'fragments of a blue dawn', 'echo of forgotten seasons'\n"
    "2. Space: weather, place, sensory texture. e.g. 'rainy day vinyl bar', '3am convenience store glow'\n"
    "3. Philosopher: human essence, existential depth. e.g. 'at the edge of existence', 'solitude slowly ripening'\n\n"
    "[Examples]\n"
    "Synopsis: Title: Schindler's List / Genre: Drama, History / Overview: A businessman saves Jews during Holocaust.\n"
    "Output: ashes of memory|grey platform at dawn|humanity's last conscience\n\n"
    "Synopsis: Title: Toy Story / Genre: Animation, Comedy / Overview: Toys come to life when humans aren't watching.\n"
    "Output: abandoned doll's quiet dream|golden afternoon in a child's room|a journey to prove you exist\n\n"
    "Synopsis: Title: The Dark Knight / Genre: Action, Crime / Overview: Batman faces the Joker who wants to plunge Gotham into anarchy.\n"
    "Output: fracture of justice in darkness|rain-soaked Gotham alley|where good and evil lose their names\n\n"
    "[Output Rules]\n"
    "- Output in English only\n"
    "- Exactly 3 queries separated by | on a single line\n"
    "- Each query must be distinct\n"
    "- 5 words or fewer per query, noun-phrase style, no full sentences\n"
    "- No labels, numbers, or explanation — one DSV line only\n\n"
    "[Output]"
)


def synth_text(row: pd.Series) -> str:
    """합성된 vibe description. 세션 16 진단 (D) 해소의 핵심 —
    content_text와 쿼리(라벨)가 같은 오염된 원문에서 나오던 커플링을 끊는다.
    3도메인 모두 이걸 우선 쓰고, 아직 합성 전인 항목만 기존 원문으로 폴백한다.
    (합성이 100% 커버되면 폴백은 실행되지 않는다.)"""
    v = str(row.get("description_synth", "") or "").strip()
    return "" if v in ("", "nan", "None") else v


def build_synopsis(domain: str, row: pd.Series) -> str:
    synth = synth_text(row)

    if domain == "movie":
        text = f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"
        overview = synth or str(row.get("text", "")).strip()
        if overview and overview != "nan":
            text += f"\nOverview: {overview[:600]}"
        return text

    elif domain == "music":
        try:
            artists = json.loads(str(row.get("artists", "[]")))
            artist_str = ", ".join(artists)
        except Exception:
            artist_str = str(row.get("artists", ""))
        text = (
            f"Track: {row.get('name', '')}\nArtist: {artist_str}\n"
            f"Album: {row.get('album_name', '')}\nGenre: {row.get('genre', '')}"
        )
        desc = row.get("description")
        lyrics = row.get("lyrics")
        if synth:
            # 가사 원문(1인칭 콘텐츠 자체)이 라벨 생성 입력으로 들어가던 자리.
            # 합성 description은 movie/book과 같은 3인칭 설명 타입이다.
            text += f"\nDescription: {synth[:600]}"
        elif pd.notna(lyrics) and str(lyrics).strip() not in ("", "nan"):
            text += f"\nLyrics: {str(lyrics)[:500]}"
        elif pd.notna(desc) and str(desc).strip() not in ("", "nan"):
            text += f"\nDescription: {str(desc)[:500]}"
        return text

    else:  # book
        text = (
            f"Title: {row.get('title', '')}\nAuthor: {row.get('author', '')}\n"
            f"Category: {row.get('category_name', '')}"
        )
        desc = synth or str(row.get("description_clean", row.get("description", ""))).strip()
        if desc and desc != "nan":
            text += f"\nDescription: {desc[:600]}"
        return text


LOCAL_IMAGE_DIRS = {
    "movie": "data/images/movie",
    "music": "data/images/music",
    "book":  "data/images/book",
}


def load_image(domain: str, item_id: str, url: str) -> Image.Image:
    local_path = os.path.join(LOCAL_IMAGE_DIRS[domain], f"{item_id}.jpg")
    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")
    r = requests.get(
        url, timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"},
        stream=True,
    )
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


MIN_PARTS = 3


def validate_dsv(text: str) -> str | None:
    """레이블 줄을 건너뛰고 첫 유효 DSV 줄에서 앞 3개 쿼리를 취한다.

    이전에는 파트가 **정확히 3개**여야 통과시켰다. 그 결과 movie 12.9% / book 14.6%가
    폐기됐는데, 실패분 64건을 재생성해 원본 출력을 확인해 보니(2026-08-07) 89%가
    "쿼리가 3개가 아니라 7~16개"였다. 내용 자체는 정상이다:

      [Toy Story] 'vibrant toy adventure|childhood nostalgia|innocent playtime joy|
                   warmth from pixels|toy dreams awakening|friendship under stars|...'

    프롬프트가 3개를 요구해도 모델이 목록을 늘어놓는 것이고, 이건 품질 문제가 아니라
    형식 문제다. 앞 3개를 취해 2만 건 이상을 GPU 재생성 없이 회수한다.
    (3개 미만이거나 구분자가 없는 줄은 그대로 거절한다 — 실패의 나머지 4.7%.)
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= MIN_PARTS and all(parts[:MIN_PARTS]) and \
                all("\n" not in p for p in parts[:MIN_PARTS]):
            return "|".join(parts[:MIN_PARTS])
    return None


# ── Qwen ──────────────────────────────────────────────────────────────────────

def load_qwen(model_id: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    # 72B는 4-bit 양자화, 7B는 bfloat16 full
    use_4bit = any(x in model_id for x in ("72B", "72b", "32B", "32b"))
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return processor, model


def generate_query_qwen(processor, model, image: Image.Image | None, prompt: str) -> str:
    from qwen_vl_utils import process_vision_info

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


# ── PaliGemma ─────────────────────────────────────────────────────────────────

def load_paligemma(model_id: str):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return processor, model


def generate_query_paligemma(processor, model, image: Image.Image, prompt: str) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


# ── 배치 생성 (vLLM) ──────────────────────────────────────────────────────────

def run_vllm(args, cfg, to_process, cache, cache_path):
    """HF 경로와 프롬프트·디코딩 설정은 같고 배치만 다르다 (scripts/vllm_runner.py 참조).
    캐시 계약도 같아 엔진을 바꿔 이어 돌려도 기존 결과가 재사용된다."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.vllm_runner import VLLMRunner, chunks

    # 쿼리는 "3개 DSV 한 줄"이라 description보다 훨씬 짧다.
    runner = VLLMRunner(args.model_id, max_new_tokens=64)
    id_col = cfg["id_col"]
    done = valid = 0

    def fetch(item_id: str, row: dict):
        if not cfg["has_image"](row):
            return None
        try:
            return load_image(args.domain, item_id, str(row[cfg["image_col"]]))
        except Exception:
            return None

    for batch in chunks(to_process, args.batch):
        ids = [str(r[id_col]) for r in batch]
        with ThreadPoolExecutor(max_workers=16) as ex:
            images = list(ex.map(fetch, ids, batch))

        items = []
        for row, img in zip(batch, images):
            synopsis = build_synopsis(args.domain, pd.Series(row))
            items.append((PROMPT_TEMPLATE.format(synopsis=synopsis, role=cfg["role"]), img))

        for item_id, raw in zip(ids, runner.generate(items)):
            dsv = validate_dsv(raw)
            if dsv:
                cache[item_id] = dsv
                valid += 1

        done += len(batch)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  진행 {done:,}/{len(to_process):,} | 유효 DSV {valid:,}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["movie", "music", "book"])
    parser.add_argument("--model-type", default="qwen", choices=["qwen", "paligemma"])
    parser.add_argument("--model-id", default=None, help="HuggingFace 모델 ID (기본값 자동 선택)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                        help="hf=batch1 참조 경로 / vllm=배치 추론 (venv_vllm에서 실행)")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = (
            "Qwen/Qwen2.5-VL-7B-Instruct" if args.model_type == "qwen"
            else "google/paligemma2-28b-pt-448"
        )

    cfg = DOMAIN_CONFIGS[args.domain]
    cache_path = f"data/cache/query_cache_{args.domain}.json"

    df = pd.read_csv(cfg["csv"], engine="python")
    print(f"[{args.domain}] loaded {len(df):,} rows")

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"checkpoint loaded: {len(cache):,} entries")

    to_process = [
        row for row in df.to_dict("records")
        if str(row[cfg["id_col"]]) not in cache
    ]
    if args.limit:
        to_process = to_process[:args.limit]
    print(f"remaining: {len(to_process):,}\n")

    if args.engine == "vllm":
        run_vllm(args, cfg, to_process, cache, cache_path)
        write_output(cfg, df, cache)
        return

    print(f"model loading: {args.model_id} ({args.model_type})")
    if args.model_type == "qwen":
        processor, model = load_qwen(args.model_id)
        generate_fn = generate_query_qwen
    else:
        processor, model = load_paligemma(args.model_id)
        generate_fn = generate_query_paligemma

    valid_count = sum(1 for v in cache.values() if v and validate_dsv(str(v)))

    for i, row in enumerate(tqdm(to_process), 1):
        item_id = str(row[cfg["id_col"]])
        synopsis = build_synopsis(args.domain, pd.Series(row))
        prompt = PROMPT_TEMPLATE.format(synopsis=synopsis, role=cfg["role"])

        if args.model_type == "qwen":
            try:
                image = load_image(args.domain, item_id, str(row[cfg["image_col"]])) if cfg["has_image"](row) else None
            except Exception:
                image = None
            raw = generate_fn(processor, model, image, prompt)
        else:
            try:
                image = load_image(args.domain, item_id, str(row[cfg["image_col"]])) if cfg["has_image"](row) else PLACEHOLDER_IMG
            except Exception:
                image = PLACEHOLDER_IMG
            raw = generate_fn(processor, model, image, prompt)

        dsv = validate_dsv(raw)
        if dsv:
            cache[item_id] = dsv
            valid_count += 1

        if i % CHECKPOINT_EVERY == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"  {i:,}/{len(to_process):,} | 유효 DSV: {valid_count:,}")

    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    write_output(cfg, df, cache)


def write_output(cfg, df, cache):
    df[cfg["id_col"]] = df[cfg["id_col"]].astype(str)
    df["query"] = df[cfg["id_col"]].map(cache)
    df.to_csv(cfg["csv"], index=False)

    valid = len(cache)
    print(f"\n완료! 유효 쿼리: {valid:,} / {len(df):,} (실패: {len(df)-valid:,}건 — 재실행 시 자동 재처리)")
    print(f"저장: {cfg['csv']}")


if __name__ == "__main__":
    main()
