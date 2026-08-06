"""
영화/책의 "통합 vibe description"을 LLM(Qwen2.5-VL)으로 합성한다.

동기 (세션 16 진단 A + docs/design_poet_style_alignment.md 개선안 3):
  content_text의 텍스트 "타입"을 3도메인이 동일 계약으로 맞춰야 SBERT 공간이 갈라지지
  않는다. 또한 6월 평가에서 poet/추상 쿼리가 전역 실패한 원인 중 하나가 콘텐츠 텍스트에
  mood 어휘가 없다는 것 — movie overview는 줄거리(사실) 위주, book 블러브는 주제/홍보
  위주라 "나직한", "야간의" 같은 poet 쿼리가 매칭할 표적이 없다.

목표:
  scripts/generate_music_descriptions.py 와 **동일 계약**(2~3문장 3인칭 mood/톤/내용 설명,
  movie overview register)의 설명을 movie/book 전 아이템에 생성 → description_synth 컬럼.
  소스만 도메인별로 교체한다.
    movie: Title/Genre/Overview/Director/Cast + 포스터(강한 시각 mood 신호)
    book : Title/Author/Category + 블러브 + 커버(약한 시각 신호)

사용:
  # 실제 합성 (GPU 서버)
  /opt/conda/envs/ltv/bin/python scripts/generate_item_descriptions.py --domain movie --limit 200
  # 입력 프롬프트만 조립해 덤프 (로컬, 모델 불필요 — 입력 품질/타입일관 검증용)
  python scripts/generate_item_descriptions.py --domain book --dry-run --sample 200 \
      --dump data/cache/book_desc_synth_dryrun.txt

체크포인트: data/cache/{domain}_desc_synth_cache.json (50건마다)

주의 — book 소스 희박:
  book_canonical.csv 133,102권 중 description 보유는 19,971권(15.0%)뿐이다. 나머지 85%는
  제목/저자/카테고리/표지만으로 합성해야 하므로, 프롬프트가 "줄거리를 지어내지 말고 톤과
  독서 경험 위주로 서술"하도록 제약한다. 어떤 신호로 만들어졌는지는 description_synth_basis
  컬럼에 기록되므로 후속 학습/평가에서 필터링·가중에 쓸 수 있다.
"""

import os
import re
import json
import argparse

import pandas as pd

CHECKPOINT_EVERY = 50
LOCAL_IMAGE_DIRS = {
    "movie": "data/images/movie",
    "book": "data/images/book",
}


# ── 공통 계약 ────────────────────────────────────────────────────────────────
# generate_music_descriptions.py 의 PROMPT_TEMPLATE 과 동일한 register/길이/인칭을
# 요구한다. 세 도메인이 같은 문체로 나와야 SBERT 공간에서 타입이 통일된다.

PROMPT_TEMPLATE = (
    "[Task]\n"
    "You are {role}. Write a 2-3 sentence, third-person description of the {noun}'s mood, "
    "atmosphere, and thematic content — the same register as a film synopsis. Describe what "
    "it FEELS like and what it is ABOUT. {grounding}\n"
    "{rules}\n\n"
    "[{noun_title}]\n{meta}\n\n"
    "{extra}"
    "[Output]\nA single paragraph (2-3 sentences), no labels."
)

# 모든 도메인 공통 규칙. 마지막 항목이 설계문서의 "세분성 함정" 방어 —
# 장르/카테고리 수준의 mood만 쓰면 비슷한 작품이 동일 텍스트로 뭉쳐 변별력이 붕괴한다.
COMMON_RULES = (
    "\n[Rules]\n"
    "- Do NOT list release dates, box office numbers, awards, ratings, sales rank, or "
    "publication trivia.\n"
    "- Do NOT use promotional copy (bestseller claims, review quotes, series marketing, "
    "author or cast biography).\n"
    "- Do NOT address the reader ('you will love...') and do NOT give an opinion or "
    "recommendation.\n"
    "- Lead with mood and tone words (e.g. tender, claustrophobic, sun-bleached, restless), "
    "not with a plot recap.\n"
    "- Include at least one concrete detail specific to THIS work (a setting, a conflict, a "
    "subject) so the description is not interchangeable with others in the same genre."
)

# 소스가 빈약할 때(특히 book 85%) 환각을 억제하는 추가 제약.
THIN_SOURCE_RULE = (
    "\n- The source material below is thin. Do NOT invent plot points, characters, or events. "
    "Infer mood and subject only from {signals}, and stay general about the story while still "
    "being concrete about tone."
)


def _clean(v) -> str | None:
    """nan/None/[] 같은 쓰레기 값을 프롬프트에 노출하지 않는다."""
    if pd.isna(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None", "[]", "no") else s


def _json_list(v, limit: int | None = None) -> str | None:
    """'["A", "B"]' 형태 문자열 컬럼을 'A, B'로."""
    try:
        items = json.loads(str(v))
    except Exception:
        return _clean(v)
    if not isinstance(items, list) or not items:
        return None
    if limit:
        items = items[:limit]
    return ", ".join(str(i) for i in items) or None


def _year(v) -> str | None:
    """release_date/publishedDate에서 연도만. 2009.0 / '1999-05-19' 모두 처리."""
    s = _clean(v)
    if not s:
        return None
    for token in (s[:4], s.split("-")[0]):
        try:
            y = int(float(token))
        except (TypeError, ValueError):
            continue
        if 1800 <= y <= 2100:
            return str(y)
    return None


# ── movie ────────────────────────────────────────────────────────────────────

def _looks_like_keywords(seg: str) -> bool:
    """'rescue, friendship, jealousy, ...' 같은 콤마 나열인지. 문장형 태그라인과 구분한다."""
    return seg.count(",") >= 2 and "." not in seg


def movie_meta(row: pd.Series) -> str:
    # Title은 98.7%가 "Foo (2011)" 형태 — Year 필드와 중복되므로 연도 접미사를 떼어낸다.
    title = _clean(row.get("Title"))
    if title:
        title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    fields = [
        ("Title", title),
        ("Genre", _clean(row.get("Genre"))),
        ("Year", _year(row.get("release_date"))),
        ("Director", _json_list(row.get("director"), limit=3)),
        # 캐스트는 앞 5명만 — 톤 추정용 신호이지 크레딧 나열이 아니다.
        ("Cast", _json_list(row.get("actor"), limit=5)),
    ]
    try:
        fields.append(("Runtime", f"{int(float(row.get('running_time')))} min"))
    except (TypeError, ValueError):
        pass
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def movie_extra(row: pd.Series) -> tuple[str, list[str]]:
    """(프롬프트 추가 블록, 사용된 신호 목록)"""
    overview = _clean(row.get("text"))
    if not overview:
        return "", []
    # text의 86%는 "줄거리 | (선택)태그라인/대체 시놉시스 | 키워드, 키워드, ..." 형태.
    # 키워드(labyrinth, first love, metaphor 등)는 mood 신호로 유용하지만 산문으로 오인되지
    # 않게 분리한다. 중간 세그먼트(태그라인)는 톤이 진한 문장이라 줄거리 쪽에 남긴다.
    parts = [p.strip() for p in overview.split("|") if p.strip()]
    keywords = ""
    if len(parts) > 1 and _looks_like_keywords(parts[-1]):
        keywords = parts.pop()
    plot = " ".join(parts)

    block = (
        "[Plot overview — RESTATE its tone and subject, do not copy it verbatim]\n"
        f"{plot[:900]}\n\n"
    )
    if keywords:
        block += f"[Theme keywords — mood signals, do not list them verbatim]\n{keywords[:300]}\n\n"
    return block, ["overview"]


# ── book ─────────────────────────────────────────────────────────────────────

def book_meta(row: pd.Series) -> str:
    fields = [
        ("Title", _clean(row.get("title"))),
        ("Author", _clean(row.get("author"))),
        ("Category", _clean(row.get("category_name"))),
        ("Year", _year(row.get("publishedDate"))),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def book_extra(row: pd.Series) -> tuple[str, list[str]]:
    # description_clean(홍보문구 제거본)을 우선, 없으면 원본 description.
    blurb = _clean(row.get("description_clean")) or _clean(row.get("description"))
    if not blurb:
        return "", []
    # 블러브는 1인칭 화자/광고 문구인 경우가 흔하다(예: "It's hate at first sight... for me").
    # 3인칭 재서술을 명시적으로 요구해야 도메인 간 텍스트 타입이 어긋나지 않는다.
    block = (
        "[Publisher blurb — RESTATE its tone and subject in THIRD PERSON, strip all marketing "
        "language. The blurb may be first-person or ad copy; your output must not be]\n"
        f"{blurb[:1200]}\n\n"
    )
    return block, ["blurb"]


DOMAIN_CONFIGS = {
    "movie": {
        "csv": "data/canonical/movie_canonical.csv",
        "id_col": "imdbId",
        "image_col": "Poster",
        "name_col": "Title",
        "role": "a film curator",
        "noun": "FILM",
        "noun_title": "Film",
        # 이미지가 실제로 첨부될 때만 포스터를 언급한다 — 없는 포스터를 근거로 삼으라고
        # 지시하면 곧장 환각이 된다.
        "grounding_img": (
            "Ground the mood in the poster's visual tone (color, lighting, composition) "
            "and in the genre and plot signals given below."
        ),
        "grounding_noimg": "Ground the mood in the genre and plot signals given below.",
        "thin_signals_img": "the title, genre, and poster",
        "thin_signals_noimg": "the title and genre",
        "meta_fn": movie_meta,
        "extra_fn": movie_extra,
        "has_image": lambda row: str(row.get("Poster", "")).startswith("http"),
    },
    "book": {
        "csv": "data/canonical/book_canonical_v2.csv",
        "id_col": "asin",
        "image_col": "imgUrl",
        "name_col": "title",
        "role": "a book editor",
        "noun": "BOOK",
        "noun_title": "Book",
        "grounding_img": (
            "Ground the mood in the cover's visual tone (color, typography, imagery) and in "
            "the category and blurb signals given below."
        ),
        "grounding_noimg": "Ground the mood in the category and blurb signals given below.",
        "thin_signals_img": "the title, category, and cover",
        "thin_signals_noimg": "the title and category",
        "meta_fn": book_meta,
        "extra_fn": book_extra,
        "has_image": lambda row: str(row.get("imgUrl", "")).startswith("http"),
    },
}


# ── 프롬프트 조립 ─────────────────────────────────────────────────────────────

def build_prompt(domain: str, row: pd.Series, has_img: bool | None = None) -> str:
    """has_img=None이면 URL 유무로 추정. 실제 합성 시에는 이미지 로드 성공 여부를 넘겨서
    (URL은 있지만 다운로드 실패한 경우) 없는 포스터를 근거로 삼지 않도록 한다."""
    cfg = DOMAIN_CONFIGS[domain]
    extra, sources = cfg["extra_fn"](row)
    if has_img is None:
        has_img = cfg["has_image"](row)

    rules = COMMON_RULES
    if not sources:
        signals = cfg["thin_signals_img"] if has_img else cfg["thin_signals_noimg"]
        rules += THIN_SOURCE_RULE.format(signals=signals)

    return PROMPT_TEMPLATE.format(
        role=cfg["role"],
        noun=cfg["noun"],
        noun_title=cfg["noun_title"],
        grounding=cfg["grounding_img"] if has_img else cfg["grounding_noimg"],
        rules=rules,
        meta=cfg["meta_fn"](row),
        extra=extra,
    )


def synth_basis(domain: str, row: pd.Series) -> str:
    """어떤 신호로 합성했는지 기록 (예: 'blurb+cover', 'meta_only').

    book의 85%처럼 소스가 빈약한 항목을 후속 단계에서 걸러내거나 가중하기 위한 컬럼.
    """
    cfg = DOMAIN_CONFIGS[domain]
    _, sources = cfg["extra_fn"](row)
    if cfg["has_image"](row):
        sources = sources + ["cover" if domain == "book" else "poster"]
    return "+".join(sources) if sources else "meta_only"


def load_image(domain: str, item_id: str, url: str):
    from PIL import Image
    local = os.path.join(LOCAL_IMAGE_DIRS[domain], f"{item_id}.jpg")
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


# ── 배치 합성 (vLLM) ──────────────────────────────────────────────────────────

def run_vllm(args, cfg, todo, cache, cache_path, id_col):
    """HF 경로와 프롬프트·디코딩 설정은 같고 배치만 다르다. 캐시 계약도 동일하므로
    두 경로를 섞어 돌려도(중단 후 엔진 교체) 이미 만든 건 그대로 재사용된다."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.vllm_runner import VLLMRunner, chunks

    runner = VLLMRunner(args.model_id)
    image_col = cfg["image_col"]
    done = 0

    def fetch(item_id: str, row: pd.Series):
        # 이미지 로드는 디스크·네트워크 대기다. GPU 구간과 겹치지 않게 배치 단위로
        # 미리 병렬로 채워 넣는다 (실패는 None = 텍스트 전용 요청).
        try:
            return load_image(args.domain, item_id, str(row.get(image_col, "")))
        except Exception:
            return None

    for batch in chunks(todo, args.batch):
        rows = [pd.Series(r) for r in batch]
        ids = [str(r[id_col]) for r in rows]
        with ThreadPoolExecutor(max_workers=16) as ex:
            images = list(ex.map(fetch, ids, rows))

        items = [
            (build_prompt(args.domain, row, has_img=img is not None), img)
            for row, img in zip(rows, images)
        ]
        for item_id, desc in zip(ids, runner.generate(items)):
            if desc:
                cache[item_id] = desc

        done += len(batch)
        json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
        print(f"  진행 {done:,}/{len(todo):,} (캐시 {len(cache):,})", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIGS))
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                    help="hf=batch1 참조 경로 / vllm=배치 추론 (venv_vllm에서 실행)")
    ap.add_argument("--batch", type=int, default=512,
                    help="vllm 엔진에서 한 번에 넘길 요청 수 (이미지 로드 단위이기도 함)")
    ap.add_argument("--limit", type=int, default=None, help="실제 합성 시 처리 건수 제한")
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 프롬프트만 조립")
    ap.add_argument("--sample", type=int, default=200, help="dry-run 샘플 수")
    ap.add_argument("--dump", default=None, help="기본값: data/cache/{domain}_desc_synth_dryrun.txt")
    args = ap.parse_args()

    cfg = DOMAIN_CONFIGS[args.domain]
    cache_path = f"data/cache/{args.domain}_desc_synth_cache.json"
    df = pd.read_csv(cfg["csv"], low_memory=False)
    print(f"loaded {len(df):,} {args.domain} items")

    if args.dry_run:
        run_dry(args.domain, df, args)
        return

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    id_col = cfg["id_col"]
    todo = [r for r in df.to_dict("records") if str(r[id_col]) not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"remaining: {len(todo):,}")

    if args.engine == "vllm":
        run_vllm(args, cfg, todo, cache, cache_path, id_col)
        write_output(args, cfg, df, cache, id_col)
        return

    processor, model = load_qwen(args.model_id)
    from tqdm import tqdm
    for i, row in enumerate(tqdm(todo), 1):
        row = pd.Series(row)
        item_id = str(row[id_col])
        try:
            img = load_image(args.domain, item_id, str(row.get(cfg["image_col"], "")))
        except Exception:
            img = None
        prompt = build_prompt(args.domain, row, has_img=img is not None)
        try:
            desc = generate_qwen(processor, model, img, prompt)
        except Exception as e:
            desc = None
            print(f"  [warn] {item_id}: {e}")
        if desc:
            cache[item_id] = desc
        if i % CHECKPOINT_EVERY == 0:
            json.dump(cache, open(cache_path, "w"), ensure_ascii=False)

    json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
    write_output(args, cfg, df, cache, id_col)


def write_output(args, cfg, df, cache, id_col):
    df[id_col] = df[id_col].astype(str)
    df["description_synth"] = df[id_col].map(cache)
    df["description_synth_basis"] = df.apply(lambda r: synth_basis(args.domain, r), axis=1)
    df.to_csv(cfg["csv"], index=False)
    filled = df["description_synth"].notna().sum()
    print(f"done. description_synth 채워짐: {filled:,} / {len(df):,}")


def run_dry(domain: str, df: pd.DataFrame, args):
    """모델 없이 프롬프트만 조립해 덤프. 소스 조합이 골고루 섞이도록 층화 샘플."""
    cfg = DOMAIN_CONFIGS[domain]
    dump = args.dump or f"data/cache/{domain}_desc_synth_dryrun.txt"

    def has(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        s = df[col].astype(str).str.strip()
        return df[col].notna() & ~s.isin(["", "nan", "None"])

    img_ok = df[cfg["image_col"]].astype(str).str.startswith("http")
    if domain == "movie":
        text_ok = has("text")
        buckets = {
            "overview+poster": df[text_ok & img_ok],
            "overview_only": df[text_ok & ~img_ok],
            "poster_only": df[~text_ok & img_ok],
            "meta_only": df[~text_ok & ~img_ok],
        }
    else:
        text_ok = has("description_clean") | has("description")
        buckets = {
            "blurb+cover": df[text_ok & img_ok],
            "blurb_only": df[text_ok & ~img_ok],
            "cover_only": df[~text_ok & img_ok],   # book의 85%가 여기
            "meta_only": df[~text_ok & ~img_ok],
        }

    per = max(1, args.sample // len(buckets))
    picks, seen = [], set()
    for tag, sub in buckets.items():
        if len(sub) == 0:
            print(f"  [note] bucket '{tag}' 비어 있음 (0건)")
            continue
        for _, r in sub.sample(min(per, len(sub)), random_state=42).iterrows():
            key = r[cfg["id_col"]]
            if key in seen:
                continue
            seen.add(key)
            picks.append((tag, r))

    lines = []
    for tag, row in picks:
        lines.append("=" * 88)
        lines.append(f"[bucket={tag}] id={row[cfg['id_col']]}  name={row[cfg['name_col']]}")
        lines.append(f"[basis={synth_basis(domain, row)}]")
        lines.append("-" * 88)
        lines.append(build_prompt(domain, row))
        lines.append("")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    with open(dump, "w") as f:
        f.write("\n".join(lines))
    print(f"dry-run: {len(picks)} prompts → {dump}")

    # 전체 커버리지 요약 — 어느 신호로 몇 건이 합성되는지
    basis = df.apply(lambda r: synth_basis(domain, r), axis=1)
    print(f"\n[{domain}] 전체 {len(df):,}건 신호 구성:")
    for k, v in basis.value_counts().items():
        print(f"  {k:20s} {v:>8,} ({v / len(df) * 100:5.1f}%)")
    thin = (basis == "meta_only").sum()
    print(f"  → 텍스트·이미지 모두 없는 항목: {thin:,} ({thin / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
