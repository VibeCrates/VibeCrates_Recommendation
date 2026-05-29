"""
Preprocessing - Data preprocessing and feature engineering

주요 역할:
  - 도메인별 CSV(MovieGenre / music_features / Books_final) →
    loader.py가 기대하는 표준 컬럼(content_text, image_path, query)으로 변환
  - 결측치 처리, 범주형 인코딩, 수치 피처 정규화
  - 사용자 프로필 집계, 아이템 피처 행렬 생성
  - 이상치 제거, 클래스 불균형 처리
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
        "csv": "data/Books_final.csv",
        "id_col": "ISBN",
        "image_col": "Image-URL-M",
    },
}


class DataPreprocessor:
    """도메인 독립적인 전처리 파이프라인. fit/transform 패턴으로 train/val/test에 동일 스케일러 적용."""

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
            key: 같은 피처 그룹에 같은 스케일러를 재사용하기 위한 식별자
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
        음악 도메인의 Spotify 오디오 피처를 정규화해 반환.
        해당 컬럼이 없는 도메인(movie/book)은 영벡터 행렬 반환.

        Returns:
            (N, D) float32 배열. D = 오디오 피처 수(9) 또는 1
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
# 도메인별 표준화 함수
# ---------------------------------------------------------------------------

def _build_content_text(domain: str, row: pd.Series) -> str:
    """도메인 CSV 행 → content_text 문자열 (generate_queries.py의 build_synopsis와 동일 로직)."""
    if domain == "movie":
        return f"Title: {row.get('Title', '')}\nGenre: {row.get('Genre', '')}"

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
    return (
        f"Title: {row.get('Book-Title', '')}\nAuthor: {row.get('Book-Author', '')}\n"
        f"Category: {row.get('main_category', '')}"
    )


def _resolve_image_path(domain: str, item_id: str, url: str, image_base_dir: str) -> str:
    """로컬 파일 우선, 없으면 HTTP URL, 둘 다 없으면 빈 문자열."""
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
    도메인별 원본 DataFrame → loader.py 호환 표준 DataFrame.

    출력 컬럼:
      item_id, content_text, image_path, query

    Args:
        domain: "movie" | "music" | "book"
        df: 원본 도메인 CSV를 읽은 DataFrame
        image_base_dir: 로컬 이미지 루트 디렉터리 (download_images.py의 저장 경로)
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
# 모듈 레벨 유틸리티
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
