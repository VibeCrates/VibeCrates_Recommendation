"""전달 번들을 npz 한 파일 형식으로 바꾼다 — id와 벡터를 같은 파일에 담는다.

왜 바꾸는가:
  기존 형식은 `{domain}_vectors.npy`(순수 숫자 배열)와 `{domain}_items.parquet`이
  **같은 순서**라는 약속으로 이어져 있었다. 약속은 사람이 지켜야 하고, 한쪽만 갱신하는
  사고가 가능하다. npz에 함께 담으면 그 실패가 구조적으로 불가능해진다.

  그리고 백엔드가 CSV를 자기 목적에 맞게 가공해 쓰기로 했으므로(2026-09-11) items
  parquet의 6개 필드는 더 이상 계약이 아니다. 우리가 줘야 하는 것은 **어느 벡터가 어느
  아이템인지**뿐이다.

담지 않는 것과 그 이유:
  - `row` — npz를 순서대로 읽으면 나오는 값이다. i번째 id가 i번째 벡터의 주인이다.
  - `point_id` — **백엔드 데이터베이스의 기본키이므로 그쪽이 정한다.** 우리가 주던
    `10억 + row`는 내보내기 순서에 묶여 있어 재학습하면 같은 아이템의 값이 바뀐다.
    Qdrant는 UUID도 받으므로 `uuid5(NAMESPACE, f"{domain}:{id}")`처럼 id에서
    결정론적으로 만들면 재학습과 무관하게 안정적이다.

id를 `dtype='S'`(바이트 문자열)로 저장하는 이유:
  pandas에서 그냥 꺼내면 dtype이 object가 되고, 그러면 numpy가 pickle로 저장해
  `allow_pickle=True` 없이는 로드가 거부된다. 유니코드 고정폭(`<U`)은 읽히지만 book id가
  최대 138자라 한 개가 552바이트를 먹어 파일이 오히려 커진다(349MB → 401MB).

사용:
  python scripts/convert_bundle_npz.py --src dist/bundle_20260831 --out dist/bundle_20260831_npz
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

DOMAINS = ("movie", "music", "book")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dist/bundle_20260831")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src_manifest = json.load(open(os.path.join(args.src, "manifest.json")))
    manifest = {k: v for k, v in src_manifest.items() if k != "domains"}
    manifest["format"] = "npz"
    manifest["note"] = (
        "각 {domain}_vectors.npz 안에 배열 세 개가 있다. ids = 아이템 id(바이트 문자열, "
        "np.char.decode로 문자열 변환), vectors = (N, 768) float32이며 L2 정규화돼 있다"
        "(내적 = 코사인 유사도), model_version = 이 벡터를 만든 체크포인트"
        "(d['model_version'].item().decode()로 읽는다). i번째 ids가 i번째 vectors의 주인이다. "
        "point id는 백엔드가 정한다 — Qdrant는 정수와 UUID를 받으므로 "
        "uuid5(NAMESPACE, f'{domain}:{id}')처럼 id에서 결정론적으로 만들면 재학습으로 "
        "벡터를 다시 받아도 같은 아이템이 같은 point를 유지해 upsert로 갱신된다. "
        "이미지는 파일명이 곧 id다({domain}/{id}.jpg). CSV의 이미지 URL 열은 우리 보유 "
        "현황과 다르므로 쓰지 말 것. 응답의 model_version이 이 값과 다르면 쿼리 벡터와 "
        "아이템 벡터가 다른 체크포인트에서 나온 것이고, 그때 검색은 오류 없이 성공하고 "
        "결과만 엉뚱해진다."
    )
    manifest["domains"] = {}

    for dom in DOMAINS:
        items = pd.read_parquet(os.path.join(args.src, f"{dom}_items.parquet"))
        vectors = np.load(os.path.join(args.src, f"{dom}_vectors.npy"))
        assert len(items) == len(vectors), f"{dom}: 행 수 불일치"
        ids = np.array(items["id"].astype(str).tolist(), dtype="S")

        # model_version을 파일 안에 함께 넣는다. manifest에도 있지만 manifest는 옮기다
        # 떨어질 수 있고, 그러면 벡터가 어느 체크포인트에서 나왔는지 알 방법이 사라진다.
        # 쿼리 벡터와 아이템 벡터의 체크포인트가 다르면 검색은 오류 없이 성공하고 결과만
        # 엉뚱해지므로, 적재 시점에 대조할 수 있어야 한다(응답의 model_version과 비교).
        path = os.path.join(args.out, f"{dom}_vectors.npz")
        np.savez_compressed(path, ids=ids, vectors=vectors,
                            model_version=np.array(src_manifest["model_version"], dtype="S"))

        norms = np.linalg.norm(vectors, axis=1)
        manifest["domains"][dom] = {
            "count": int(len(ids)),
            "file": f"{dom}_vectors.npz",
            "sha256": sha256(path),
            "id_max_len": int(max(len(i) for i in items["id"].astype(str))),
            "l2_norm_min": round(float(norms.min()), 6),
            "l2_norm_max": round(float(norms.max()), 6),
            "coverage": src_manifest["domains"][dom]["coverage"],
        }
        print(f"[{dom}] {len(ids):,}개 → {path} ({os.path.getsize(path)/1e6:.1f}MB)")

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\nmanifest → {args.out}/manifest.json")


if __name__ == "__main__":
    main()
