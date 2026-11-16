# VibeCrates Recommendation

**English** · [한국어](README_KR.md)

VibeCrates is a recommendation service where you describe a mood in plain words, such as "a song to listen to alone on a rainy night" or "a quiet, lonely film", and it finds movies, music and books that match that feeling. The results are shown as a "board", a collage of their covers.

This repository is the **AI recommendation server**. It does three things.
- Inference: runs the API server that turns a user's search text into a 768-dimensional vector.
- Training: trains the model that places about 190,000 works in the same vector space, and delivers the results to the backend.
- Evaluation: manages the experiments that check whether the model actually finds the right works.

User accounts, saved boards and the vector search itself live in the backend repository, and the UI lives in the frontend repository.

---

## Contents

1. [What this server does in the service](#1-what-this-server-does-in-the-service)
2. [Data](#2-data)
3. [Model](#3-model)
4. [Korean queries](#4-korean-queries)
5. [API](#5-api)
6. [Setup and running](#6-setup-and-running)
7. [Training and evaluation](#7-training-and-evaluation)
8. [Known limitations](#8-known-limitations)
9. [Repository layout](#9-repository-layout)
10. [Documentation](#10-documentation)

---

## 1. What this server does in the service

The search runs in the backend's vector database (Qdrant); this server **turns search text into a vector**.

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

Item vectors are computed in this repository ahead of time and handed to the backend as a **delivery bundle** (CSV). **Query vectors and item vectors must come from the same checkpoint.** If they do not, search still succeeds without any error and only the results are wrong. The `model_version` carried by every API response and every bundle row is how you check that the two match.

---

## 2. Data

The starting point is a set of **canonical CSVs** (`data/canonical/`) that bring the three domains into one format.

| Domain | Items | Sources | Cover images |
|---|---:|---|---|
| Movies | 39,515 | MovieGenre dataset + TMDB metadata | Re-fetched TMDB posters (99.3% coverage) |
| Music | 39,682 | Spotify audio-feature dataset | Original URLs + Deezer re-fetch (92.4%) |
| Books | 110,594 | Kindle + Goodreads (Zenodo) + BX blurbs, merged | 97.0% |

The item text the model reads is not the original plot or blurb but a **mood description (`description_synth`)** that an LLM synthesized from the original information (plot, blurb, audio features, and so on). All three domains use the same format. Music audio features such as energy and valence are turned into natural-language phrases like "calm" or "intense" and included in the description.

Training also needs **three training queries per item**: short search phrases a person might type to find that work, written by Qwen2.5-VL from the item's cover and description. `scripts/prepare_dataset.py` pairs these queries with the items to produce `data/train_dataset.csv`.

Raw data, images, training data and model files are not committed because of their size (see `.gitignore`). The repository contains the canonical CSVs and the scripts that build them.

---

## 3. Model

The model is a **dual encoder**. Items and search text are encoded by separate encoders into the same 768-dimensional space, and the nearest items are found by cosine similarity. Item vectors only need to be computed once, so a live search encodes just the query, which takes tens of milliseconds even on a CPU.

<img width="769" height="248" alt="Inference model" src="https://github.com/user-attachments/assets/216883ea-b2ca-497f-9069-e9757c5cda3d" />

- Inference model: turns the user's search text (query) into a vector.

```
Item side                                                  Query side
mood description ─ all-mpnet-base-v2 (LoRA) + MLP ─ z_text ─┐
                                                            ├─ ContentBlock ─ z_content (768, L2-normalized)
cover image ─ CLIP ViT-L/14 vision + MLP ─ z_image ─────────┘                    ↕ cosine similarity
                                 search text ─ CLIP ViT-L/14 text (frozen) + MLP ─ z_query (768)
```

Training has two stages (`src/training/trainer.py`).

<img width="1200" height="664" alt="Training stages" src="https://github.com/user-attachments/assets/a6f8f180-1b27-4555-9d75-53f204bf75ca" />

- **Stage 1 (contrastive):** trains the MLP heads on all three sides plus the mpnet LoRA adapter. An InfoNCE loss pulls the three vectors from the same item together and pushes apart those from different items. It is applied to three pairs (description–query, description–image, image–query) and summed.
- **Stage 2:** freezes everything else and trains only the `ContentBlock`, again with InfoNCE, so that the final item vector `z_content` (description and image combined) matches the query vector `z_query`. `z_content` is the vector actually used for search.

The pretrained encoder bodies are not trained. CLIP is frozen on both the image and the query side, and mpnet only gets a LoRA adapter.

The serving checkpoint is `best_stage2_qlora_off`, with `model_version` `trained_model.pt@1786697109`.

---

## 4. Korean queries

The model was trained only in English and the CLIP tokenizer is English-only, so the server **translates Korean queries into English before encoding them** (`src/api/translation.py`). The backend simply sends Korean as is. Translation is attempted in this order.

1. **Translation dictionary:** queries that can be split entirely into genre, mood and situation words, such as "시집" (poetry collection) or "로파이 비 오는 날", are translated by joining entries from a reviewed dictionary (`src/api/ko_en_dict.json`, 520 entries), e.g. 로파이 비 오는 날 → *lo-fi rainy day*. If even one word is missing from the dictionary, the query moves to the next step.
2. **Gemma 3 4B:** sentence-style queries are translated by Gemma 3 4B served with vLLM. It is used only on servers where `LLM_TRANSLATE_URL` is set.
3. **opus-mt-ko-en:** if Gemma is not configured or does not answer in time, Helsinki-NLP's opus-mt translates the query.

If every translation step fails, the request does not fail; the original text is encoded as is.

Alongside translation, the server also reads **which domain the query names** (`src/api/domain_intent.py`). If the query contains a noun for a kind of work, such as "movie", "song" or "novel", the response's `domain` is `movie`, `music` or `book`, and the backend searches only that domain. Genres (jazz, horror) are not used as cues, and when no such noun appears or several domains are wanted together, the value is `all`. The full rules are in [`docs/api_for_backend.md`](docs/api_for_backend.md).

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

Vectors are L2-normalized, so the dot product equals cosine similarity. `translated_text` is filled only when a translation happened; `null` means the text was encoded as sent.

- The server accepts **HTTP/1.1 only**. Requests sent over HTTP/2 lose their body and fail with 422.
- CLIP reads at most 77 tokens, so anything longer is cut off at the end.

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

`serve.sh` starts the server with the uvicorn inside `venv` and prints this machine's LAN and Tailscale addresses. Startup takes about 13 seconds, and connections are refused during that time.

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

The image contains only the code; the model and indexes are mounted as volumes at run time (`MODEL_PATH`, `INDEX_DIR`). It is built for inference, so it uses the CPU build of torch.

### Tests

```bash
pytest tests/        # API, data, domain detection, translation (69 tests)
```

### Current deployment

The production server runs on a university GPU server (AstraGo, A100), with the API server on port 8000 and the Gemma translation server (vLLM) on port 8001. Training runs on a separate RTX 5090 server. Addresses are in internal team documents.

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

`--sample-one-query` picks one of each item's three training queries at random on every step (the setting used by the serving model). Other experimental flags such as `--query-lora`, `--title-dropout` and `--domain-balanced-batches` are described in `python scripts/train.py --help`.

### Evaluation

Two models (a checkpoint plus its index each) are run on the same evaluation queries and compared.

- **Pairwise judging:** `scripts/judge_pairwise.py` shows an LLM judge (Qwen2.5-VL-7B by default) the two models' results side by side and asks which is better. It asks twice with the order swapped and reports the win rate and p-value.
- **Human labels:** comparisons the judge cannot settle are turned into a label sheet with `scripts/build_label_sheet.py`, labeled by people, and tallied with `scripts/score_label_sheet.py`.

Experiment outputs are in `experiments/`, and experiment reports are in `docs/`.

---

## 8. Known limitations

- **Negation is barely reflected.** "Fantasy novels that are not scary" is turned into a vector close to "scary fantasy novels". This is a limitation of the frozen CLIP text encoder.
- **Metaphorical sentences translate less reliably.** The more poetic the query, the further its translation can drift from the original meaning.

The cause and current status of each issue are in [`docs/model_issues.md`](docs/model_issues.md).

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

`models/`, `indexes/`, `dist/` and `for_result_report/` are not committed.

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
