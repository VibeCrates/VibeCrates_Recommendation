"""
음악 트랙의 "통합 vibe description"을 LLM(Qwen2.5-VL)으로 합성한다.

동기 (세션 16 진단):
  현행 content_text는 음악에 대해 raw lyrics(약 60%) 또는 발매이력성 description을
  넣어, movie/book(3인칭 콘텐츠 설명)과 텍스트 "타입"이 어긋난다. 게다가 오디오 피처
  (danceability/energy/valence...)라는 음악 고유·객관 신호는 모델에서 통째로 버려진다.

목표:
  가진 신호(메타 + 오디오 피처 + 가사요약 + 기존 description + 앨범 커버)를 종합해
  movie overview와 동일한 register의 2~3문장 3인칭 mood/내용 설명을 전 트랙에 생성.
  → description_synth 컬럼으로 저장 (기존 description은 보존).

핵심 설계:
  - 오디오 피처를 자연어로 verbalize 해서 mood를 "환각"이 아니라 객관 신호에 grounding.
  - 가사는 인용 금지, 테마/정서 요약 소스로만 사용.
  - instrumental·무텍스트 트랙도 메타+오디오+커버로 커버 → 100% 커버리지.

사용:
  # 실제 합성 (GPU 서버)
  /opt/conda/envs/ltv/bin/python scripts/generate_music_descriptions.py --limit 200
  # 입력 프롬프트만 조립해서 파일로 덤프 (로컬, 모델 불필요 — 입력 품질/타입일관 검증용)
  python scripts/generate_music_descriptions.py --dry-run --sample 200 \
      --dump data/cache/desc_synth_dryrun.txt

체크포인트: data/cache/music_desc_synth_cache.json (50건마다)
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

CSV_PATH = "data/canonical/music_canonical.csv"
CACHE_PATH = "data/cache/music_desc_synth_cache.json"
CHECKPOINT_EVERY = 50
LOCAL_IMAGE_DIR = "data/images/music"


# ── 오디오 피처 → 자연어 verbalization ────────────────────────────────────────
# Spotify 오디오 피처를 mood 형용사구로 변환. 결측(-1 sentinel / NaN)은 건너뛴다.
# loudness는 dB 스케일(음수 정상)이라 별도 임계값 사용.

def _valid01(v) -> float | None:
    """0~1 범위 피처의 유효값만 반환 (결측 sentinel<0 / NaN 제외)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or f < 0:
        return None
    return f


def _bin(v: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    return labels[0] if v < low else (labels[2] if v >= high else labels[1])


def verbalize_audio(row: pd.Series) -> str:
    """오디오 피처를 'energetic and danceable, upbeat, mostly acoustic, ~120 BPM' 식 구로."""
    phrases: list[str] = []

    energy = _valid01(row.get("energy"))
    dance = _valid01(row.get("danceability"))
    valence = _valid01(row.get("valence"))
    acoustic = _valid01(row.get("acousticness"))
    instr = _valid01(row.get("instrumentalness"))
    speech = _valid01(row.get("speechiness"))
    live = _valid01(row.get("liveness"))

    if energy is not None:
        phrases.append(_bin(energy, 0.4, 0.7, ("calm and mellow", "moderately energetic", "high-energy and intense")))
    if dance is not None:
        phrases.append(_bin(dance, 0.4, 0.7, ("not very danceable", "moderately groovy", "very danceable")))
    if valence is not None:
        phrases.append(_bin(valence, 0.35, 0.65, ("melancholic and somber", "emotionally balanced", "upbeat and positive")))
    if acoustic is not None and acoustic >= 0.5:
        phrases.append("largely acoustic")
    elif acoustic is not None and acoustic < 0.2:
        phrases.append("electronic/produced")
    if instr is not None and instr >= 0.5:
        phrases.append("instrumental (little or no vocals)")
    if speech is not None and speech >= 0.33:
        phrases.append("spoken-word / rap-leaning delivery")
    if live is not None and live >= 0.5:
        phrases.append("live-performance feel")

    # tempo (BPM) — NaN 흔함(4092건)
    try:
        tempo = float(row.get("tempo"))
        if not np.isnan(tempo) and tempo > 0:
            band = "slow" if tempo < 90 else ("mid-tempo" if tempo < 130 else "fast")
            phrases.append(f"{band} tempo (~{round(tempo)} BPM)")
    except (TypeError, ValueError):
        pass

    # mode (1=major/bright, 0=minor/darker) — NaN 흔함
    try:
        mode = float(row.get("mode"))
        if mode == 1:
            phrases.append("major key (brighter)")
        elif mode == 0:
            phrases.append("minor key (darker)")
    except (TypeError, ValueError):
        pass

    return ", ".join(phrases) if phrases else "(audio features unavailable)"


# ── 가사 요약 소스 (인용 금지 신호) ───────────────────────────────────────────

def lyrics_hint(row: pd.Series, max_chars: int = 600) -> str | None:
    v = row.get("lyrics")
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    # 반복 후렴/애드립 노이즈를 줄이려 앞부분만 요약 소스로 제공
    return s[:max_chars]


def existing_desc(row: pd.Series, max_chars: int = 400) -> str | None:
    v = row.get("description")
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    return s[:max_chars]


# ── 프롬프트 조립 ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "[Task]\n"
    "You are a music curator. Write a 2-3 sentence, third-person description of the "
    "TRACK's mood, sonic texture, and thematic content — the same register as a film "
    "synopsis or a book blurb. Describe what the song FEELS and SOUNDS like and what it "
    "is ABOUT. Do NOT quote or paraphrase lyrics line-by-line. Do NOT list release dates, "
    "chart positions, or discography trivia. Ground the mood in the objective audio "
    "profile provided.\n\n"
    "[Track]\n{meta}\n\n"
    "[Objective audio profile]\n{audio}\n\n"
    "{extra}"
    "[Output]\nA single paragraph (2-3 sentences), no labels."
)


def build_meta(row: pd.Series) -> str:
    def clean(v) -> str | None:
        if pd.isna(v):
            return None
        s = str(v).strip()
        return None if s in ("", "nan", "None", "[]") else s

    try:
        artists = json.loads(str(row.get("artists", "[]")))
        artist_str = ", ".join(artists)
    except Exception:
        artist_str = str(row.get("artists", ""))

    fields = [
        ("Title", clean(row.get("name"))),
        ("Artist", clean(artist_str)),
        ("Album", clean(row.get("album_name"))),
        ("Genre", clean(row.get("genre"))),
    ]
    year = row.get("year")
    if pd.notna(year):
        try:
            fields.append(("Year", str(int(float(year)))))  # 2009.0 → 2009
        except (TypeError, ValueError):
            pass
    fields.append(("Subgenres", clean(row.get("niche_genres"))))
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def build_prompt(row: pd.Series) -> str:
    meta = build_meta(row)
    audio = verbalize_audio(row)
    extra = ""
    lh = lyrics_hint(row)
    ed = existing_desc(row)
    if lh:
        extra += (
            "[Lyrics excerpt — SUMMARIZE themes/emotion only, DO NOT quote]\n"
            f"{lh}\n\n"
        )
    if ed:
        extra += f"[Reference note (may contain trivia — extract only mood-relevant bits)]\n{ed}\n\n"
    return PROMPT_TEMPLATE.format(meta=meta, audio=audio, extra=extra)


def load_image(item_id: str, url: str):
    from PIL import Image
    local = os.path.join(LOCAL_IMAGE_DIR, f"{item_id}.jpg")
    if os.path.exists(local):
        return Image.open(local).convert("RGB")
    if isinstance(url, str) and url.startswith("http"):
        import requests
        from io import BytesIO
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return None


# ── 실제 합성 (Qwen2.5-VL) ────────────────────────────────────────────────────

def load_qwen(model_id: str):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return processor, model


def generate_qwen(processor, model, image, prompt: str) -> str:
    import torch
    from qwen_vl_utils import process_vision_info
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False, repetition_penalty=1.2)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="실제 합성 시 처리 건수 제한")
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 프롬프트만 조립")
    ap.add_argument("--sample", type=int, default=200, help="dry-run 샘플 수")
    ap.add_argument("--dump", default="data/cache/desc_synth_dryrun.txt")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"loaded {len(df):,} tracks")

    if args.dry_run:
        run_dry(df, args)
        return

    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    todo = [r for r in df.to_dict("records") if str(r["id"]) not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"remaining: {len(todo):,}")

    processor, model = load_qwen(args.model_id)
    from tqdm import tqdm
    for i, row in enumerate(tqdm(todo), 1):
        row = pd.Series(row)
        item_id = str(row["id"])
        prompt = build_prompt(row)
        try:
            img = load_image(item_id, str(row.get("img", "")))
        except Exception:
            img = None
        try:
            desc = generate_qwen(processor, model, img, prompt)
        except Exception as e:
            desc = None
            print(f"  [warn] {item_id}: {e}")
        if desc:
            cache[item_id] = desc
        if i % CHECKPOINT_EVERY == 0:
            json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)

    json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)
    df["id"] = df["id"].astype(str)
    df["description_synth"] = df["id"].map(cache)
    df.to_csv(CSV_PATH, index=False)
    print(f"done. description_synth 채워짐: {df['description_synth'].notna().sum():,} / {len(df):,}")


def run_dry(df: pd.DataFrame, args):
    """모델 없이 프롬프트만 조립해 덤프. 다양한 케이스가 섞이도록 층화 샘플."""
    def has(col):
        s = df[col].astype(str).str.strip()
        return df[col].notna() & ~s.isin(["", "nan", "None"])
    hd, hl = has("description"), has("lyrics")
    inst = pd.to_numeric(df["instrumentalness"], errors="coerce").fillna(0)

    buckets = {
        "lyrics_only": df[hl & ~hd],
        "has_desc": df[hd],
        "neither": df[~hl & ~hd],
        "instrumental": df[inst > 0.5],
    }
    per = max(1, args.sample // len(buckets))
    picks = []
    seen = set()
    for tag, sub in buckets.items():
        for _, r in sub.sample(min(per, len(sub)), random_state=42).iterrows():
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            picks.append((tag, r))

    lines = []
    for tag, row in picks:
        lines.append("=" * 88)
        lines.append(f"[bucket={tag}] id={row['id']}  name={row['name']}")
        lines.append("-" * 88)
        lines.append(build_prompt(row))
        lines.append("")
    os.makedirs(os.path.dirname(args.dump), exist_ok=True)
    with open(args.dump, "w") as f:
        f.write("\n".join(lines))
    print(f"dry-run: {len(picks)} prompts → {args.dump}")

    # 요약 통계: verbalize_audio가 실제로 신호를 뽑아내는지
    got_audio = sum(1 for _, r in picks if verbalize_audio(r) != "(audio features unavailable)")
    print(f"오디오 프로필 생성된 샘플: {got_audio}/{len(picks)}")


if __name__ == "__main__":
    main()
