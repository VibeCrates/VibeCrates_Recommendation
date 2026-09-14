"""전달 번들을 CSV 한 파일 형식으로 바꾼다 — id와 벡터(base64)를 같은 행에 담는다.

왜 base64인가:
  10진수 텍스트로 풀면(예: "0.12345678,...") 행당 평균 8.8KB, 전체 약 1.68GB로 지금
  npz 번들(543MB)의 3배가 넘는다. JSON 배열 문자열은 더 크다(행당 17KB, 전체 3.2GB) —
  `json.dumps`가 float를 더 긴 자릿수로 풀어쓰기 때문이다. float32 바이트를 그대로
  base64로 감싸면 행당 정확히 4,096바이트(768×4바이트의 4/3배)로 고정되고, 원본과
  무손실이며 콤마·따옴표가 섞이지 않아 CSV 이스케이프 문제도 없다.

npz 대신 CSV로 바꾸는 이유:
  npz는 "i번째 ids가 i번째 vectors의 주인"이라는 위치 기반 약속에 의존한다
  (`convert_bundle_npz.py`의 `assert len(items) == len(vectors)`가 그 증거다).
  CSV는 id와 vector가 같은 행에 있어 그 약속 자체가 필요 없다 — 순서가 밀리는 사고가
  구조적으로 불가능해진다. 백엔드가 CSV 한 파일을 요청한 이유(2026-09-14)이기도 하다.

model_version을 매 행에 반복해서 넣는 이유:
  manifest.json은 옮기다 떨어질 수 있다. 쿼리 벡터와 아이템 벡터의 체크포인트가 다르면
  검색은 오류 없이 성공하고 결과만 엉뚱해지므로, 적재 시점에 파일 자체에서 대조할 수
  있어야 한다. 행마다 반복해도 문자열 하나라 크기에 미치는 영향은 무시할 만하다.

사용:
  python scripts/convert_bundle_csv.py --src dist/bundle_20260831 --out dist/bundle_20260914_csv
"""
import argparse
import base64
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
    manifest["format"] = "csv"
    manifest["note"] = (
        "각 {domain}_vectors.csv는 열 세 개 — id, vector_b64, model_version. "
        "vector_b64는 (768,) float32(L2 정규화 완료, 내적 = 코사인 유사도)를 "
        "`.astype('<f4').tobytes()` 한 뒤 base64로 감싼 것이다. 읽는 법: "
        "`v = np.frombuffer(base64.b64decode(row.vector_b64), dtype='<f4')` "
        "(결과 shape (768,)). model_version이 행마다 반복돼 있으므로 적재 시점에 "
        "`/search/vector` 응답의 model_version과 이 열이 같은지 확인할 것 — 다르면 "
        "쿼리 벡터와 아이템 벡터가 다른 체크포인트에서 나온 것이고, 그때 검색은 오류 "
        "없이 성공하고 결과만 엉뚱해진다."
    )
    manifest["domains"] = {}

    for dom in DOMAINS:
        items = pd.read_parquet(os.path.join(args.src, f"{dom}_items.parquet"))
        vectors = np.load(os.path.join(args.src, f"{dom}_vectors.npy")).astype("<f4")
        assert len(items) == len(vectors), f"{dom}: 행 수 불일치"
        ids = items["id"].astype(str).tolist()

        vector_b64 = [base64.b64encode(v.tobytes()).decode("ascii") for v in vectors]
        out_df = pd.DataFrame({
            "id": ids,
            "vector_b64": vector_b64,
            "model_version": src_manifest["model_version"],
        })
        path = os.path.join(args.out, f"{dom}_vectors.csv")
        out_df.to_csv(path, index=False)

        norms = np.linalg.norm(vectors, axis=1)
        manifest["domains"][dom] = {
            "count": int(len(ids)),
            "file": f"{dom}_vectors.csv",
            "sha256": sha256(path),
            "dim": int(vectors.shape[1]),
            "id_max_len": int(max(len(i) for i in ids)),
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
