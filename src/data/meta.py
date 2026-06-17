import pandas as pd


def build_meta_df(domain: str, raw_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    """item_id, title, domain, extra(dict) 컬럼을 가진 DataFrame."""
    records = []
    for i, (_, raw_row) in enumerate(raw_df.iterrows()):
        item_id = std_df.iloc[i]["item_id"]
        if domain == "movie":
            title = str(raw_row.get("Title", ""))
            extra = {"genre": str(raw_row.get("Genre", "")), "poster": str(raw_row.get("Poster", ""))}
        elif domain == "music":
            title = str(raw_row.get("name", ""))
            extra = {"artists": str(raw_row.get("artists", "")), "album": str(raw_row.get("album_name", "")),
                     "genre": str(raw_row.get("genre", "")), "img": str(raw_row.get("img", ""))}
        else:  # book
            title = str(raw_row.get("title", ""))
            extra = {"author": str(raw_row.get("author", "")),
                     "category": str(raw_row.get("category_name", "")),
                     "image": str(raw_row.get("imgUrl", ""))}
        records.append({"item_id": item_id, "domain": domain, "title": title, "extra": extra})
    return pd.DataFrame(records)
