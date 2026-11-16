# VibeCrates Recommendation

**English** · [한국어](README_KR.md)

VibeCrates is a recommendation service where you describe a mood in plain words, such as "a song to listen to alone on a rainy night" or "a quiet, lonely film", and it finds movies, music and books that match that feeling. The results are shown as a "board", a collage of their covers.

This repository is the **AI recommendation server**. It does three things. It runs an API that turns a user's search text into a 768-dimensional vector. It trains the model that places about 190,000 works in the same vector space, and delivers those vectors to the backend. And it holds the experiments that check whether the model actually finds the right works. User accounts, saved boards and the vector search itself live in the backend repository, and the UI lives in the frontend repository.

---

## Contents

1. [What this server does in the service](#1-what-this-server-does-in-the-service)
2. [Data](#2-data)
3. [Model](#3-model)
4. [Korean queries](#4-korean-queries)
5. [API](#5-api)
6. [Setup and running](#6-setup-and-running)
7. [Training and evaluation](#7-training-and-evaluation)
8. [Current state and known limitations](#8-current-state-and-known-limitations)
9. [Repository layout](#9-repository-layout)
10. [Documentation](#10-documentation)

---

## 1. What this server does in the service

The backend runs the search; this server **only turns search text into a vector**. This was decided on 2026-08-24. The backend already operates a vector database (Qdrant), so loading the item vectors there and letting it search keeps the overall system simpler.

```
[search text] → frontend → backend ──POST /api/v1/search/vector──▶ AI recommendation server
                                                                     │ ① translate to English if Korean
                                                                     │ ② encode into a 768-d vector
                           backend vector DB  ◀────── query vector ──┘
                  (≈180k item vectors, one collection per domain)
                                │ cosine-similarity search
                                ▼
                         matching works → board collage
```

Item vectors are computed in this repository ahead of time and handed to the backend as a **delivery bundle** (CSV). One rule matters above all: **query vectors and item vectors must come from the same checkpoint.** If they come from different models, search still succeeds without any error and only the results are wrong, which makes the problem hard to notice. For that reason every API response and every bundle row carries a `model_version`, and the backend checks that the two match when it loads the vectors.

---

## 2. Data

The starting point is a set of **canonical CSVs** (`data/canonical/`) that bring the three domains into one format.

| Domain | Items | Sources | Cover images |
|---|---:|---|---|
| Movies | 39,515 | MovieGenre dataset + TMDB metadata | Re-fetched TMDB posters (99.3% coverage) |
| Music | 39,682 | Spotify audio-feature dataset | Original URLs + Deezer re-fetch (92.4%) |
| Books | 110,594 | Kindle + Goodreads (Zenodo) + BX blurbs, merged | 97.0% |

The raw text differed in kind from domain to domain. Movies and books had third-person plot descriptions, but about 60% of the music text field was first-person song lyrics. Trained on that, a model learns to tell domains apart by *the type of writing* rather than by mood. So for every item we **synthesized a mood description in a single shared format (`description_synth`) with an LLM**, based on the original information (plot, blurb, audio features, and so on). Music audio features such as energy and valence were turned into natural-language phrases like "calm" or "intense" and included in the description.

Training also needs **three training queries per item**: short search phrases a person might type to find that work. Qwen2.5-VL wrote them from each item's cover and description. `scripts/prepare_dataset.py` pairs these queries with the items to produce `data/train_dataset.csv`.

Raw data, images, training data and model files are not committed because of their size (see `.gitignore`). The repository contains the canonical CSVs and the scripts that build them.

---

## 3. Model

The model is a **dual encoder**. Items and search text are encoded by separate encoders into the same 768-dimensional space, and the nearest items are found by cosine similarity. Item vectors only need to be computed once, so a live search encodes just the query, which takes tens of milliseconds even on a CPU.

```
Item side                                                  Query side
mood description ─ all-mpnet-base-v2 (LoRA) + MLP ─ z_text ─┐
                                                            ├─ ContentBlock ─ z_content (768, L2-normalized)
cover image ─ CLIP ViT-L/14 vision + MLP ─ z_image ─────────┘                    ↕ cosine similarity
                                 search text ─ CLIP ViT-L/14 text (frozen) + MLP ─ z_query (768)
```

Training has two stages (`src/training/trainer.py`).

- **Stage 1 (contrastive):** trains the MLP heads on all three sides plus the mpnet LoRA adapter. An InfoNCE loss pulls the three vectors from the same item together and pushes apart those from different items. It is applied to three pairs (description–query, description–image, image–query) and summed.
- **Stage 2:** freezes everything else and trains only the `ContentBlock`, again with InfoNCE, so that the final item vector `z_content` (description and image combined) matches the query vector `z_query`. `z_content` is the vector actually used for search.

The pretrained encoder bodies stay frozen. CLIP is left untouched on both the image and the query side, and mpnet only gets a small LoRA adapter. In August a naming-rule bug made all of CLIP (262M parameters) trainable, and training on 190,000 items wiped out CLIP's pretrained knowledge. Fixing that was the single largest quality improvement so far.

**The serving checkpoint** is `best_stage2_qlora_off`, trained on 2026-08-14, and the API's `model_version` is `trained_model.pt@1786697109`. A retrain on the October-cleaned data (newbase) showed no difference in a pairwise comparison (win rate 49.2%, p=0.73), so it was not swapped in.

---

## 4. Korean queries

The model was trained only in English, and its query encoder, CLIP, uses an English tokenizer that breaks Korean text into byte fragments. When we compared 40 Korean–English query pairs with the same meaning, the cosine similarity of their query vectors was 0.058, which means essentially unrelated, and not a single pair shared any of its top-10 results.

So the server **translates Korean queries into English before encoding them** (`src/api/translation.py`). The backend simply sends Korean as is. Translation is attempted in three steps.

1. **Translation dictionary:** queries that can be split entirely into genre, mood and situation words, such as "시집" (poetry collection) or "로파이 비 오는 날" (lo-fi rainy day), are translated by joining entries from a reviewed dictionary (`src/api/ko_en_dict.json`, 520 entries). Most real queries look like this, and a general-purpose translator often gets them wrong (e.g. "시집" → *Indulgent*, "이별" → *DIVORCE*).
2. **Gemma 3 4B:** sentence-style queries that the dictionary cannot cover are translated by Gemma 3 4B served with vLLM. Human ratings clearly favored it over the older translator (4.42–4.52 vs 3.64 out of 5). It is used only on servers where `LLM_TRANSLATE_URL` is set.
3. **opus-mt-ko-en:** if Gemma is not configured or does not answer in time, Helsinki-NLP's opus-mt translates the query.

A failed translation never fails the request; the original text is encoded as is. Quality drops, but that is better than letting one translator stop search entirely.

Alongside translation, the server also reads **which domain the query names** (`src/api/domain_intent.py`). If the query contains a noun for a kind of work, such as "movie", "song" or "novel", the response's `domain` is `movie`, `music` or `book`, and the backend searches only that domain. Genres (jazz, horror) span several domains, so they are not used as cues. Restricting the search wrongly would hide results the user wanted, so when in doubt the value is `all`.

---

## 5. API

The server is built with FastAPI. Once it is running, every endpoint can be tried at `http://<server>:8000/docs`. Request formats, response fields, the delivery bundle format and how to load it into Qdrant are documented in [`docs/api_for_backend.md`](docs/api_for_backend.md) (in Korean).

| Method · path | Purpose |
|---|---|
| `POST /api/v1/search/vector` | **Canonical.** Turns one query into a 768-d vector |
| `GET /api/v1/search/vector?keyword=…` | Same result via GET |
| `POST /api/v1/embeddings` | Turns several queries into vectors at once (to be retired) |
| `POST /api/v1/recommend` | Searches inside this server and returns results. **For cross-checking** the backend's search, not part of the service contract |
| `GET /api/v1/item/{domain}/{item_id}` | Item metadata |
| `GET /api/v1/ping?n=42` | Connectivity check, independent of the model |
| `GET /api/v1/health` | Model and index loading status |

```json
// POST /api/v1/search/vector
{ "text": "잔잔한 영화" }

// response
{ "text": "잔잔한 영화",
  "translated_text": "calm movie",
  "domain": "movie",
  "dim": 768,
  "normalized": true,
  "model_version": "trained_model.pt@1786697109",
  "vector": [0.0023, 0.0258, ...] }
```

Vectors are L2-normalized, so the dot product equals cosine similarity. `translated_text` is filled only when a translation happened; `null` means the text was encoded as sent. When results look wrong, this field tells you whether the problem is the translation or the search.

Two things to watch out for. The server accepts **HTTP/1.1 only**; requests sent over HTTP/2 lose their body and keep failing with 422. And CLIP reads at most 77 tokens, so anything longer is cut off at the end.

---

## 6. Setup and running

### Environment

```bash
git clone https://github.com/VibeCrates/VibeCrates_Recommendation.git
cd VibeCrates_Recommendation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt          # everything, including training and experiments
# pip install -r requirements-serve.txt  # API server only
```

### Model and index files

Running the server needs two things that are not in the repository: the trained checkpoint `models/trained_model.pt` (about 2.2GB), and the precomputed item vectors `indexes/{movie,music,book}_embeddings.pt` with their `_meta.parquet` files. Both must come from the same training run. Without them the server still starts and `/ping` works, but the vector endpoints return 503.

### Starting the server

```bash
./scripts/serve.sh            # 0.0.0.0:8000
PORT=9000 ./scripts/serve.sh  # different port
```

`serve.sh` uses the uvicorn inside `venv` and prints this machine's LAN and Tailscale addresses. It does this because starting the app with another Python on the PATH (Anaconda, for example) can silently fail to load the model when packages are missing. Startup takes about 13 seconds, and connections are refused during that time.

Main environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/trained_model.pt` | Checkpoint path |
| `INDEX_DIR` | `indexes` | Item vector index folder |
| `DEVICE` | `cuda` if available, else `cpu` | Inference device |
| `TRANSLATE_QUERIES` | `1` | Set to `0` to turn off Korean translation |
| `LLM_TRANSLATE_URL` | (unset) | Gemma translation server (OpenAI-compatible, e.g. `http://127.0.0.1:8001/v1`). If empty, only dictionary → opus-mt is used |
| `LLM_TRANSLATE_TIMEOUT` | `1.0` | Seconds to wait for Gemma |

### Docker

The `Dockerfile` does not bake the model into the image; the model is mounted as a volume at run time. Model and indexes together are about 2.8GB, and baking them in would mean rebuilding the image after every retrain. Inference runs fine on a CPU, so the image installs the CPU build of torch.

### Tests

```bash
pytest tests/        # API, data, domain detection, translation (69 tests)
```

### Current deployment

The production server runs on a university GPU server (AstraGo, A100), with the API server on port 8000 and the Gemma translation server (vLLM) on port 8001. The backend has called this server since 2026-10-25. Training runs on a separate RTX 5090 server. Addresses and access procedures are kept in internal team documents, not in this public repository.

---

## 7. Training and evaluation

### Training pipeline

```
canonical CSV
  → generate_item_descriptions.py / generate_music_descriptions.py   synthesize mood descriptions (vLLM)
  → generate_queries.py                                             3 training queries per item (Qwen2.5-VL)
  → prepare_dataset.py                                              data/train_dataset.csv
  → extract_image_embeddings.py                                     precompute cover image embeddings
  → train.py                                                        two-stage training + index build
  → convert_bundle_csv.py --require-image                           item vector bundle for the backend
  → export_query_vectors.py                                         training query vectors for the backend
```

```bash
python scripts/train.py \
    --data-path data/train_dataset.csv \
    --image-embeddings <.pt produced by extract_image_embeddings.py> \
    --sample-one-query \
    --save-path models/trained_model.pt \
    --index-dir indexes
```

`--sample-one-query` picks one of each item's three training queries at random on every step. Earlier, the average of the three query vectors was used as the training target, which meant the model never learned the real case where a single query comes in. Other experimental flags such as `--query-lora`, `--title-dropout` and `--domain-balanced-batches` are described in `python scripts/train.py --help`.

### Evaluation

Recommendation has no single right answer, so the evaluation method itself has changed several times. These are the current principles.

- **Decide by pairwise comparison, not absolute scores.** At first an LLM judge graded each result as right, partial or wrong. Those grades turned out to depend heavily on whether the query and the item description shared words. For results with no shared words, humans marked 41% as right while the judge marked only 2.7%. Since September 2026 the judge is shown two models' results side by side and picks the better one (`scripts/judge_pairwise.py`), and decisions are made from the win rate and its p-value.
- **Humans evaluate experiments that are sensitive to wording.** The judge could not reliably reject results that share words with the query but are wrong. Such experiments are evaluated with human label sheets (`build_label_sheet.py`, `score_label_sheet.py`).
- **Recheck premises on the current model.** New experiments are not started from weaknesses that were measured on an older model.

Experiment outputs live in `experiments/`, and the reasoning behind decisions lives in `docs/`.

---

## 8. Current state and known limitations

The status and evidence for every known issue are in [`docs/model_issues.md`](docs/model_issues.md). The ones that affect the service are summarized here.

- **Negation is barely reflected.** "Fantasy novels that are not scary" is read much like "scary fantasy novels". The cause is the frozen CLIP text encoder, and retraining with negation data alone is unlikely to fix it. It is on hold because real search logs (about 40,000 unique queries) contain essentially no negated searches.
- **Metaphorical sentences translate unreliably.** The more poetic the query, the further its translation drifts from the original meaning. The real fix is a multilingual encoder, which needs retraining.
- **Recent retrains have not shown a measurable effect.** Four retrains in a row came out as "no difference" in pairwise comparison. We cannot yet tell whether the models are truly the same or whether the evaluation, with only 40 queries per language, is too coarse to see the difference.
- **There is no real user data yet.** Choosing the next improvement needs user feedback, so requirements for behavior logs (impressions, likes, board edits) have been handed to the backend.

---

## 9. Repository layout

```
VibeCrates_Recommendation/
├── src/
│   ├── api/                 # FastAPI server: routes, model loading, translation, domain detection
│   ├── models/              # Dual encoder (TextBlock, ImageBlock, QueryBlock, ContentBlock)
│   ├── training/            # Two-stage trainer, InfoNCE loss, training history
│   └── data/                # Dataset, loaders, preprocessing
├── scripts/                 # Data collection/synthesis, training, index/bundle export, evaluation, serving
│   └── vibe_crate/          # Main and board screen design mockups (static HTML)
├── data/canonical/          # Canonical per-domain CSVs (raw data, images, training data are gitignored)
├── experiments/             # Experiment outputs and human label sheets
├── docs/                    # Design notes, API guide, work logs, session reports
├── reports/                 # Early session reports (sessions 1–20)
├── tests/                   # pytest
├── Dockerfile, docker/      # Serving container
├── requirements.txt         # Full dependencies, including training and experiments
└── requirements-serve.txt   # API server dependencies only
```

`models/`, `indexes/`, `dist/` and `for_result_report/` are not committed because of size or because they are not meant to be public.

---

## 10. Documentation

Most documents are written in Korean.

| Document | Contents |
|---|---|
| [`docs/api_for_backend.md`](docs/api_for_backend.md) | Backend integration: endpoints, bundle format, loading into Qdrant, per-domain cap rule |
| [`docs/model_issues.md`](docs/model_issues.md) | Every model issue found so far, grouped as resolved / not affecting the service / kept / unconfirmed |
| [`docs/worklog_2026-07_to_09.md`](docs/worklog_2026-07_to_09.md) | What was done in July–September and what came of it |
| [`docs/worklist_2026-09-03.md`](docs/worklist_2026-09-03.md) | Task list as of September |
| `docs/report_session_*.md` | Session reports for September–October (latest: [`2026-10-31`](docs/report_session_2026-10-31.md)) |
| [`docs/design_poet_style_alignment.md`](docs/design_poet_style_alignment.md), [`docs/design_audio_feature_integration.md`](docs/design_audio_feature_integration.md) | Design notes |
| [`docs/report_judge_validation_20261009.md`](docs/report_judge_validation_20261009.md) | LLM judge validation results |

---

## License

MIT License

## Contact

maxsky9018@gmail.com
