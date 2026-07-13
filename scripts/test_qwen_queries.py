"""
Qwen2.5-VL를 사용한 쿼리 생성 테스트 (movie 도메인, 5개 샘플)
실행: /opt/conda/envs/ltv/bin/python scripts/test_qwen_queries.py
"""

import json
import requests
from io import BytesIO

import torch
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PLACEHOLDER_IMG_PATH = None  # None이면 텍스트만 사용
TEST_N = 5

PROMPT_TEMPLATE = (
    "[Context]\nSynopsis: {synopsis}\n\n"
    "[Instruction]\n당신은 {role}입니다. "
    "입력된 이미지의 시각적 분위기(색감, 조명, 구도)와 제공된 텍스트 데이터를 융합하여 "
    "4가지 페르소나의 '감성 검색 쿼리'를 생성하세요.\n\n"
    "[Personas]\n"
    "Poet: 은유적이고 서정적인 표현 (예: '푸른 새벽의 조각')\n"
    "Trendy: MZ세대 스타일의 힙한 키워드 (예: '갓생 살기 챌린지')\n"
    "Space: 날씨, 공기, 장소의 질감 (예: '비 내리는 LP바')\n"
    "Philosopher: 인간 본질과 심연의 테마 (예: '고독의 끝에서')\n\n"
    "[Constraint]\n반드시 |를 구분자로 사용하는 DSV 형식으로 한 줄로 출력하세요.\n"
    "형식: Poet_Query|Trendy_Query|Space_Query|Philosopher_Query\n"
    "각 쿼리는 5단어 이내의 명사형으로 작성하고 설명조를 피하세요."
)


def build_synopsis(row: pd.Series) -> str:
    text = f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"
    overview = str(row.get("text", "")).strip()
    if overview and overview != "nan":
        text += f"\nOverview: {overview[:600]}"
    return text


def load_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            stream=True,
        )
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"  [이미지 로드 실패] {e}")
        return None


def validate_dsv(text: str) -> str | None:
    parts = [p.strip() for p in text.strip().split("|")]
    if len(parts) == 4 and all(parts):
        return "|".join(parts)
    return None


def build_messages(image: Image.Image | None, prompt: str) -> list:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main():
    print(f"모델 로드 중: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("모델 로드 완료\n")

    df = pd.read_csv("data/canonical/movie_canonical.csv", engine="python")
    samples = df.head(TEST_N)

    results = []
    for _, row in samples.iterrows():
        item_id = str(row["imdbId"])
        synopsis = build_synopsis(row)
        prompt = PROMPT_TEMPLATE.format(synopsis=synopsis, role="영화 마케팅 전문가")

        poster_url = str(row.get("Poster", ""))
        image = load_image(poster_url) if poster_url.startswith("http") else None

        messages = build_messages(image, prompt)
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
                max_new_tokens=128,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        dsv = validate_dsv(raw)

        result = {
            "id": item_id,
            "title": row.get("Title", ""),
            "has_image": image is not None,
            "raw": raw,
            "dsv_valid": dsv is not None,
            "dsv": dsv,
        }
        results.append(result)

        print(f"[{item_id}] {row.get('Title', '')} (이미지: {'O' if image else 'X'})")
        print(f"  raw  : {raw}")
        print(f"  valid: {dsv is not None} → {dsv}")
        print()

    valid_count = sum(1 for r in results if r["dsv_valid"])
    print(f"=== 결과 요약 ===")
    print(f"테스트: {len(results)}개 | 유효 DSV: {valid_count}개 ({valid_count/len(results)*100:.0f}%)")


if __name__ == "__main__":
    main()
