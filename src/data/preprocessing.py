"""
Preprocessing - Data preprocessing and feature engineering

Responsibilities:
  - Converts domain CSVs (MovieGenre / music_features / kindle_data-v2) into
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
        "csv": "data/MovieGenre.csv",
        "id_col": "imdbId",
        "image_col": "Poster",
    },
    "music": {
        "csv": "data/music_features.csv",
        "id_col": "id",
        "image_col": "img",
    },
    "book": {
        "csv": "data/kindle_data-v2.csv",
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

def _build_content_text(domain: str, row: pd.Series) -> str:
    """Converts a domain CSV row into a content_text string. Mirrors build_synopsis logic in generate_queries.py."""
    if domain == "movie":
        text = f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"
        overview = str(row.get("text", "")).strip()
        if overview and overview != "nan":
            text += f"\nOverview: {overview}"
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
        if pd.notna(desc) and str(desc).strip() not in ("", "nan"):
            text += f"\nDescription: {str(desc)[:500]}"
        elif pd.notna(lyrics) and str(lyrics).strip() not in ("", "nan"):
            text += f"\nLyrics: {str(lyrics)[:500]}"
        return text

    # book
    text = (
        f"Title: {row.get('title', '')}\nAuthor: {row.get('author', '')}\n"
        f"Category: {row.get('category_name', '')}"
    )
    desc = str(row.get("description", "")).strip()
    if desc and desc != "nan":
        text += f"\nDescription: {desc}"
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
