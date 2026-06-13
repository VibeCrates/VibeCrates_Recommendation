"""
Qwen2.5-VL / PaliGemma 2를 사용하여 Movie/Music/Book 각 인스턴스의
query 컬럼을 DSV(| 구분자, 4개 한국어 감성 쿼리) 형태로 생성.
체크포인트: data/query_cache_{domain}.json (500건마다)

실행 예:
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
        "csv": "data/kindle_data-v2.csv",
        "id_col": "asin",
        "image_col": "imgUrl",
        "role": "도서 편집자",
        "has_image": lambda row: pd.notna(row.get("imgUrl")) and str(row.get("imgUrl", "")).startswith("http"),
    },
}

PROMPT_TEMPLATE = (
    "[Context]\nSynopsis: {synopsis}\n\n"
    "[Instruction]\n당신은 {role}입니다. "
    "이미지의 시각적 분위기(색감, 조명, 구도)와 텍스트 정보를 융합하여 "
    "아래 3가지 페르소나 스타일의 한국어 감성 검색 쿼리를 각각 1개씩 생성하세요.\n\n"
    "[Personas]\n"
    "1. Poet(시인): 은유·서정적 표현. 예) '푸른 새벽의 파편', '잊혀진 계절의 잔향'\n"
    "2. Space(공간): 날씨·장소·감각적 질감. 예) '비 오는 날 LP바', '새벽 3시 편의점 불빛'\n"
    "3. Philosopher(철학자): 인간 본질·심연 테마. 예) '존재의 끝에서', '고독이 익어가는 시간'\n\n"
    "[Example]\n"
    "Synopsis: Title: Schindler's List / Genre: Drama, History / Overview: A businessman saves Jews during Holocaust.\n"
    "Output: 재의 기억|잿빛 기차역 플랫폼|인류의 마지막 양심\n\n"
    "Synopsis: Title: Toy Story / Genre: Animation, Comedy / Overview: Toys come to life when humans aren't watching.\n"
    "Output: 버려진 인형의 꿈|아이 방의 오후 햇살|존재 증명의 여정\n\n"
    "Synopsis: Title: The Dark Knight / Genre: Action, Crime / Overview: Batman faces the Joker who wants to plunge Gotham into anarchy.\n"
    "Output: 어둠 속 정의의 균열|고담시 빗속 골목|선과 악의 경계에서\n\n"
    "[Output Rules]\n"
    "- 반드시 한국어로만 출력\n"
    "- 정확히 3개의 쿼리를 | 로 구분하여 한 줄로만 출력\n"
    "- 3개의 쿼리는 서로 달라야 함\n"
    "- 각 쿼리는 5단어 이내 명사형, 설명문 금지\n"
    "- 추가 설명·번호·레이블 없이 DSV 한 줄만 출력\n\n"
    "[Output]"
)


def build_synopsis(domain: str, row: pd.Series) -> str:
    if domain == "movie":
        text = f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"
        overview = str(row.get("text", "")).strip()
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
        if pd.notna(desc) and str(desc).strip() not in ("", "nan"):
            text += f"\nDescription: {str(desc)[:500]}"
        elif pd.notna(lyrics) and str(lyrics).strip() not in ("", "nan"):
            text += f"\nLyrics: {str(lyrics)[:500]}"
        return text

    else:  # book
        text = (
            f"Title: {row.get('title', '')}\nAuthor: {row.get('author', '')}\n"
            f"Category: {row.get('category_name', '')}"
        )
        desc = str(row.get("description", "")).strip()
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


def validate_dsv(text: str) -> str | None:
    # "[Output]" 같은 레이블 라인 건너뛰고 첫 유효 DSV 줄 반환
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = [p.strip() for p in line.split("|")]
        # 정확히 4개이고, 각 파트가 비어있지 않으며, 각 파트 내 줄바꿈 없을 것
        if len(parts) == 3 and all(parts) and all("\n" not in p for p in parts):
            return "|".join(parts)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["movie", "music", "book"])
    parser.add_argument("--model-type", default="qwen", choices=["qwen", "paligemma"])
    parser.add_argument("--model-id", default=None, help="HuggingFace 모델 ID (기본값 자동 선택)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = (
            "Qwen/Qwen2.5-VL-7B-Instruct" if args.model_type == "qwen"
            else "google/paligemma2-28b-pt-448"
        )

    cfg = DOMAIN_CONFIGS[args.domain]
    cache_path = f"data/query_cache_{args.domain}.json"

    df = pd.read_csv(cfg["csv"], engine="python")
    print(f"[{args.domain}] {len(df):,}개 로드")

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"체크포인트 로드: {len(cache):,}개")

    print(f"모델 로드 중: {args.model_id} ({args.model_type})")
    if args.model_type == "qwen":
        processor, model = load_qwen(args.model_id)
        generate_fn = generate_query_qwen
    else:
        processor, model = load_paligemma(args.model_id)
        generate_fn = generate_query_paligemma

    to_process = [
        row for row in df.to_dict("records")
        if str(row[cfg["id_col"]]) not in cache
    ]
    if args.limit:
        to_process = to_process[:args.limit]
    print(f"남은 처리 수: {len(to_process):,}개\n")

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

    df[cfg["id_col"]] = df[cfg["id_col"]].astype(str)
    df["query"] = df[cfg["id_col"]].map(cache)
    df.to_csv(cfg["csv"], index=False)

    valid = len(cache)
    print(f"\n완료! 유효 쿼리: {valid:,} / {len(df):,} (실패: {len(df)-valid:,}건 — 재실행 시 자동 재처리)")
    print(f"저장: {cfg['csv']}")


if __name__ == "__main__":
    main()
