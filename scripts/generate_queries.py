"""
PaliGemma 2 28B (4-bit quantized)를 사용하여 Movie/Music/Book 각 인스턴스의
query 컬럼을 DSV(| 구분자, 4개 자연어 쿼리) 형태로 생성.
체크포인트: data/query_cache_{domain}.json (500건마다)

실행 예:
  python3 scripts/generate_queries.py --domain movie
  python3 scripts/generate_queries.py --domain music --model-id google/paligemma2-28b-pt-896
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
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig

CHECKPOINT_EVERY = 500
PLACEHOLDER_IMG = Image.new("RGB", (448, 448), (128, 128, 128))

DOMAIN_CONFIGS = {
    "movie": {
        "csv": "data/MovieGenre.csv",
        "id_col": "imdbId",
        "image_col": "Poster",
        "role": "영화 마케팅 전문가",
        "has_image": lambda row: pd.notna(row.get("Poster")) and str(row.get("Poster", "")).startswith("http"),
    },
    "music": {
        "csv": "data/music_features.csv",
        "id_col": "id",
        "image_col": "img",
        "role": "음악 큐레이터",
        "has_image": lambda row: pd.notna(row.get("img")) and str(row.get("img", "")) not in ("no", "nan", ""),
    },
    "book": {
        "csv": "data/Books_final.csv",
        "id_col": "ISBN",
        "image_col": "Image-URL-M",
        "role": "도서 편집자",
        "has_image": lambda row: pd.notna(row.get("Image-URL-M")) and str(row.get("Image-URL-M", "")).startswith("http"),
    },
}

PROMPT_TEMPLATE = (
    "[Context]\nSynopsis: {synopsis}\n\n"
    "[Instruction]\n당신은 {role}입니다. 입력된 이미지의 시각적 분위기(색감, 조명, 구도)와 "
    "제공된 텍스트 데이터를 융합하여 4가지 페르소나의 '감성 검색 쿼리'를 생성하세요.\n\n"
    "[Personas]\n"
    "Poet: 은유적이고 서정적인 표현 (예: '푸른 새벽의 조각')\n"
    "Trendy: MZ세대 스타일의 힙한 키워드 (예: '갓생 살기 챌린지')\n"
    "Space: 날씨, 공기, 장소의 질감 (예: '비 내리는 LP바')\n"
    "Philosopher: 인간 본질과 심연의 테마 (예: '고독의 끝에서')\n\n"
    "[Constraint]\n반드시 |를 구분자로 사용하는 DSV 형식으로 한 줄로 출력하세요.\n"
    "형식: Poet_Query|Trendy_Query|Space_Query|Philosopher_Query\n"
    "각 쿼리는 5단어 이내의 명사형으로 작성하고 설명조를 피하세요."
)


def build_synopsis(domain: str, row: pd.Series) -> str:
    if domain == "movie":
        return f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"

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
        if pd.notna(desc) and str(desc).strip() not in ("", "nan"):
            text += f"\nDescription: {str(desc)[:500]}"
        elif pd.notna(lyrics) and str(lyrics).strip() not in ("", "nan"):
            text += f"\nLyrics: {str(lyrics)[:500]}"
        return text

    else:  # book
        return (
            f"Title: {row.get('Book-Title', '')}\nAuthor: {row.get('Book-Author', '')}\n"
            f"Category: {row.get('main_category', '')}"
        )


LOCAL_IMAGE_DIRS = {
    "movie": "data/images/movie",
    "music": "data/images/music",
    "book":  "data/images/book",
}


def load_image(domain: str, item_id: str, url: str) -> Image.Image:
    local_path = os.path.join(LOCAL_IMAGE_DIRS[domain], f"{item_id}.jpg")
    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")
    r = requests.get(url, timeout=10, headers={"User-Agent": "VibeCrates/1.0"}, stream=True)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def validate_dsv(text: str) -> str | None:
    parts = [p.strip() for p in text.strip().split("|")]
    if len(parts) == 4 and all(parts):
        return "|".join(parts)
    return None


def load_model(model_id: str):
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


def generate_query(processor, model, image: Image.Image, prompt: str) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["movie", "music", "book"])
    parser.add_argument("--model-id", default="google/paligemma2-28b-pt-448")
    args = parser.parse_args()

    cfg = DOMAIN_CONFIGS[args.domain]
    cache_path = f"data/query_cache_{args.domain}.json"

    df = pd.read_csv(cfg["csv"], low_memory=False)
    print(f"[{args.domain}] {len(df):,}개 로드")

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"체크포인트 로드: {len(cache):,}개")

    processor, model = load_model(args.model_id)

    to_process = [
        row for row in df.to_dict("records")
        if str(row[cfg["id_col"]]) not in cache
    ]
    print(f"남은 처리 수: {len(to_process):,}개\n")

    valid_count = sum(1 for v in cache.values() if v and validate_dsv(str(v)))

    for i, row in enumerate(tqdm(to_process), 1):
        item_id = str(row[cfg["id_col"]])
        synopsis = build_synopsis(args.domain, pd.Series(row))
        prompt = PROMPT_TEMPLATE.format(synopsis=synopsis, role=cfg["role"])

        try:
            image = load_image(args.domain, item_id, str(row[cfg["image_col"]])) if cfg["has_image"](row) else PLACEHOLDER_IMG
        except Exception:
            image = PLACEHOLDER_IMG

        try:
            raw = generate_query(processor, model, image, prompt)
            dsv = validate_dsv(raw)
            cache[item_id] = dsv if dsv else raw
            if dsv:
                valid_count += 1
        except Exception:
            cache[item_id] = None

        if i % CHECKPOINT_EVERY == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"  {i:,}/{len(to_process):,} | 유효 DSV: {valid_count:,}")

    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    df[cfg["id_col"]] = df[cfg["id_col"]].astype(str)
    df["query"] = df[cfg["id_col"]].map(cache)
    df.to_csv(cfg["csv"], index=False)

    valid = df["query"].notna().sum()
    print(f"\n완료! 유효 쿼리: {valid:,} / {len(df):,}")
    print(f"저장: {cfg['csv']}")


if __name__ == "__main__":
    main()
