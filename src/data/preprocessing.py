"""
Preprocessing - Data preprocessing and feature engineering

Responsibilities:
  - Converts domain canonical CSVs (movie_canonical / music_canonical / book_canonical) into
    the standard schema expected by loader.py (content_text, image_path, query)
  - Missing value handling, categorical encoding, numerical feature normalization
  - Item feature matrix construction
  - Outlier removal, class imbalance handling
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

MUSIC_AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

DOMAIN_CONFIG = {
    "movie": {
        "csv": "data/canonical/movie_canonical.csv",
        "id_col": "imdbId",
        "image_col": "Poster",
    },
    "music": {
        "csv": "data/canonical/music_canonical.csv",
        "id_col": "id",
        "image_col": "img",
    },
    "book": {
        # v2 = Kindle/Goodreads/BX 3소스 병합본 (110,594권, 블러브 100%).
        # 기존 book_canonical.csv는 133,102권이지만 블러브 15.0%라 85%의 content_text가
        # 제목·저자·카테고리뿐이었다 — poet 등 추상 쿼리가 매칭할 mood 표적이 없었다.
        "csv": "data/canonical/book_canonical_v2.csv",
        "id_col": "asin",
        "image_col": "imgUrl",
    },
}


class DataPreprocessor:
    """Domain-agnostic preprocessing pipeline. Uses fit/transform pattern to apply consistent scalers across train/val/test."""

    def __init__(self):
        self.scalers: Dict[str, object] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Args:
            strategy: "mean" | "median" | "zero" | "drop" | "forward_fill"
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        str_cols = df.select_dtypes(include="object").columns.tolist()

        if strategy == "drop":
            return df.dropna()

        if strategy == "forward_fill":
            df[num_cols] = df[num_cols].ffill()
            df[str_cols] = df[str_cols].ffill().fillna("")
            return df

        fill_fns = {
            "mean":   lambda col: df[col].mean(),
            "median": lambda col: df[col].median(),
            "zero":   lambda col: 0,
        }
        fill_fn = fill_fns.get(strategy, fill_fns["mean"])
        for col in num_cols:
            df[col] = df[col].fillna(fill_fn(col))
        df[str_cols] = df[str_cols].fillna("")
        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """LabelEncoder를 fit(첫 호출) 또는 transform(이후 호출)."""
        df = df.copy()
        for col in categorical_cols:
            if col not in df.columns:
                continue
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        return df

    def normalize_numerical_features(
        self,
        X: np.ndarray,
        method: str = "standardization",
        key: str = "default",
    ) -> Tuple[np.ndarray, object]:
        """
        Args:
            method: "standardization" (StandardScaler) | "min-max" (MinMaxScaler)
            key: identifier for reusing the same scaler across calls on the same feature group
        Returns:
            (scaled_X, scaler)
        """
        if key not in self.scalers:
            scaler = StandardScaler() if method == "standardization" else MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[key] = scaler
        else:
            X_scaled = self.scalers[key].transform(X)
        return X_scaled, self.scalers[key]

    def create_item_features(self, item_df: pd.DataFrame) -> np.ndarray:
        """
        Normalizes Spotify audio features for the music domain.
        Returns a zero matrix for domains without audio features (movie/book).

        Returns:
            float32 array of shape (N, D), where D = number of audio features (9) or 1
        """
        audio_cols = [c for c in MUSIC_AUDIO_FEATURES if c in item_df.columns]
        if not audio_cols:
            return np.zeros((len(item_df), 1), dtype=np.float32)

        X = item_df[audio_cols].to_numpy(dtype=np.float32)
        col_means = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = col_means[nan_idx[1]]

        X_scaled, _ = self.normalize_numerical_features(X, method="standardization", key="item_audio")
        return X_scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Domain-specific standardization helpers
# ---------------------------------------------------------------------------

def _synth_text(row: pd.Series) -> str:
    """LLM이 합성한 vibe description.

    세션 16 진단 (A) "텍스트 타입 불일치"의 해소 지점이다. 이전에는 도메인마다 다른
    *타입*의 텍스트가 이 슬롯에 들어갔다 — movie/book은 3인칭 설명, music은 약 60%가
    1인칭 가사 원문. SBERT가 이를 서로 다른 의미공간 영역에 임베딩해 공유 텍스트 공간을
    쓰는 크로스도메인 추천이 어긋났다. description_synth는 3도메인 모두 동일 계약
    (2~3문장 3인칭 mood 설명, film-synopsis register)으로 생성된다.

    아직 합성되지 않은 항목만 기존 원문으로 폴백한다(합성 100% 커버 시 미실행).
    """
    v = str(row.get("description_synth", "") or "").strip()
    return "" if v in ("", "nan", "None") else v


def _build_content_text(domain: str, row: pd.Series) -> str:
    """Converts a domain CSV row into a content_text string.

    generate_queries.py::build_synopsis 와 **출력이 같아야 한다** (세션 18 결정 A).
    그쪽은 의사 라벨(query)을 만들 때 Qwen에게 주는 텍스트고, 이쪽은 학습 시 SBERT에
    넣는 텍스트다. 학습은 이 둘을 가깝게 당기므로, 한쪽에만 있는 정보는 대응하는 정답이
    없어 정렬에 기여하지 못한다. 실제로 두 군데가 갈라져 있었다:
      - 커밋 4bcc809(7/15)가 여기에만 Director/Cast/Release Date를 추가 — 라벨을 만든
        Qwen은 감독·배우를 본 적이 없다.
      - build_synopsis만 description을 600자로 자름 — description_synth 평균 548자,
        600자 초과가 movie 29.7% / music 17.6%로 그 꼬리가 입력에만 있었다.
    라벨 쪽을 넓히는 반대 방향은 쿼리 전량 재생성이 필요해 다음 사이클로 미뤘다.
    (근거 사실은 겹쳐야 하고 표현 형태는 갈라져야 한다 — 후자가 진단 D다.)
    """
    synth = _synth_text(row)

    if domain == "movie":
        text = f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"
        overview = synth or str(row.get("text", "")).strip()
        if overview and overview != "nan":
            text += f"\nOverview: {overview[:600]}"
        return text

    if domain == "music":
        try:
            artists = json.loads(str(row.get("artists", "[]")))
            artist_str = ", ".join(artists)
        except Exception:
            artist_str = str(row.get("artists", ""))
        text = (
            f"Track: {row.get('name', '')}\nArtist: {artist_str}\n"
            f"Album: {row.get('album_name', '')}\nGenre: {row.get('genre', '')}"
        )
        desc = row.get("description", "")
        lyrics = row.get("lyrics", "")
        if synth:
            # raw lyrics 직접 주입이 있던 자리. 가사는 "설명"이 아니라 콘텐츠 자체이므로
            # 이 슬롯의 오용이었다 (세션 16 진단 A).
            text += f"\nDescription: {synth[:600]}"
        elif pd.notna(lyrics) and str(lyrics).strip() not in ("", "nan"):
            text += f"\nLyrics: {str(lyrics)[:500]}"
        elif pd.notna(desc) and str(desc).strip() not in ("", "nan"):
            text += f"\nDescription: {str(desc)[:500]}"
        return text

    # book
    text = (
        f"Title: {row.get('title', '')}\nAuthor: {row.get('author', '')}\n"
        f"Category: {row.get('category_name', '')}"
    )
    desc = synth or str(row.get("description_clean", row.get("description", ""))).strip()
    if desc and desc != "nan":
        text += f"\nDescription: {desc[:600]}"
    return text


def _resolve_image_path(domain: str, item_id: str, url: str, image_base_dir: str) -> str:
    """Returns local file path if available, falls back to HTTP URL, empty string if neither exists."""
    local = os.path.join(image_base_dir, domain, f"{item_id}.jpg")
    if os.path.exists(local):
        return local
    if isinstance(url, str) and url.startswith("http"):
        return url
    return ""


def prepare_domain_df(
    domain: str,
    df: pd.DataFrame,
    image_base_dir: str = "data/images",
) -> pd.DataFrame:
    """
    Converts a raw domain DataFrame into the standard schema for loader.py.

    Output columns: item_id, content_text, image_path, query

    Args:
        domain: "movie" | "music" | "book"
        df: raw domain CSV as a DataFrame
        image_base_dir: local image root directory (as written by download_images.py)
    """
    cfg = DOMAIN_CONFIG[domain]
    id_col = cfg["id_col"]
    image_col = cfg["image_col"]

    records = []
    for _, row in df.iterrows():
        item_id = str(row[id_col])
        url = str(row.get(image_col, ""))
        records.append({
            "item_id": item_id,
            "content_text": _build_content_text(domain, row),
            "image_path": _resolve_image_path(domain, item_id, url, image_base_dir),
            "query": row.get("query", ""),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)


def remove_outliers(
    X: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        method: "iqr" | "z-score" | "isolation-forest"
        threshold:
          - iqr: IQR 배수 (기본 1.5)
          - z-score: 허용 표준편차 배수 (기본 3.0 권장)
          - isolation-forest: contamination 비율 (0~0.5, 기본 0.1 권장)
    Returns:
        (cleaned_X, bool_mask)  mask[i] = True 이면 정상 샘플
    """
    if method == "iqr":
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        mask = np.all((X >= q1 - threshold * iqr) & (X <= q3 + threshold * iqr), axis=1)

    elif method == "z-score":
        z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        mask = np.all(z < threshold, axis=1)

    elif method == "isolation-forest":
        from sklearn.ensemble import IsolationForest
        contamination = min(max(threshold, 0.0), 0.5)
        pred = IsolationForest(contamination=contamination, random_state=42).fit_predict(X)
        mask = pred == 1

    else:
        raise ValueError(f"Unknown method: {method!r}. 선택 가능: iqr | z-score | isolation-forest")

    return X[mask], mask


def resample_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    strategy: str = "oversample",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    클래스 불균형 해소.

    Args:
        strategy: "oversample" | "undersample" | "smote"
    """
    if y is None:
        return X, y

    from sklearn.utils import resample as sk_resample

    classes, counts = np.unique(y, return_counts=True)

    if strategy == "undersample":
        target_n = counts.min()
        replace = False
    elif strategy == "oversample":
        target_n = counts.max()
        replace = True
    elif strategy == "smote":
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
        return X_res, y_res
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. 선택 가능: oversample | undersample | smote")

    parts_X, parts_y = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        sampled = sk_resample(idx, replace=replace, n_samples=target_n, random_state=42)
        parts_X.append(X[sampled])
        parts_y.append(y[sampled])

    return np.vstack(parts_X), np.concatenate(parts_y)
