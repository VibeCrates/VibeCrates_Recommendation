"""
canonical CSV의 중복 ID 행을 제거한다.

왜:
  같은 item_id가 두 행으로 들어가 있으면 InfoNCE 배치 안에서 false negative가 된다 —
  동일한 아이템을 서로 밀어내라고 가르치는 셈이다. 개선안 2가 다루는 false negative와
  같은 축의 문제다. 실측: movie 593행(1.48%) / music 354행(0.88%) / book 0행.

  검색 쪽에도 영향이 있다. 인덱스에 같은 아이템이 두 번 들어가면 추천 결과에 중복이
  노출될 수 있고, 상위 K개 중 한 자리를 낭비한다.

주의 — 실행 시점:
  이 스크립트는 canonical CSV를 고친다. 그런데 build_index.py가 그 CSV를 읽어 인덱스를
  만들고, train_dataset.csv는 이전 CSV로 만들어져 있다. 학습이나 인덱스 생성이 도는 중에
  실행하면 인덱스와 학습 데이터의 행이 어긋난다.
  **반드시 학습·인덱싱이 모두 끝난 뒤에 실행하고, 그다음 prepare_dataset.py부터 다시
  돌려야 한다.**

어느 행을 남기는가:
  description_synth와 query가 채워진 행을 우선한다. 둘 다 같으면 먼저 나온 행을 남긴다.
  중복이 서로 다른 내용을 담고 있을 수 있으므로(예: 한쪽만 합성 완료) 정보가 많은 쪽을
  남기는 것이 안전하다.

사용:
  python scripts/dedupe_canonical.py --dry-run     # 몇 건이 지워지는지만 확인
  python scripts/dedupe_canonical.py               # 실제 수정 (백업 후)
"""

import argparse
import shutil
from datetime import date

import pandas as pd

from src.data.preprocessing import DOMAIN_CONFIG

ID_COLS = {"movie": "imdbId", "music": "id", "book": "asin"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAIN_CONFIG), choices=list(DOMAIN_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for domain in args.domains:
        path = DOMAIN_CONFIG[domain]["csv"]
        id_col = ID_COLS[domain]
        df = pd.read_csv(path, low_memory=False)
        before = len(df)

        ids = df[id_col].astype(str)
        dup_mask = ids.duplicated(keep=False)
        n_dup_rows = int(dup_mask.sum())
        if n_dup_rows == 0:
            print(f"{domain:6s} {before:>7,}행 — 중복 없음")
            continue

        # 정보가 많은 행을 앞으로 보내 첫 번째를 남긴다.
        score = pd.Series(0, index=df.index)
        for col in ("description_synth", "query"):
            if col in df.columns:
                score += df[col].notna().astype(int)
        order = score.sort_values(ascending=False, kind="stable").index
        kept = df.loc[order].drop_duplicates(subset=[id_col], keep="first").sort_index()

        removed = before - len(kept)
        print(f"{domain:6s} {before:>7,}행 → {len(kept):>7,}행  (중복 ID를 가진 행 {n_dup_rows:,}개 중 "
              f"{removed:,}개 제거)")

        if not args.dry_run:
            backup = f"{path}.dup{date.today():%Y%m%d}.bak"
            shutil.copy(path, backup)
            kept.to_csv(path, index=False)
            print(f"       백업 {backup}")
            print(f"       저장 {path}")

    if args.dry_run:
        print("\n(dry-run — 파일은 수정하지 않았다)")
    else:
        print("\n다음 단계: prepare_dataset.py → extract_image_embeddings.py → 재학습")


if __name__ == "__main__":
    main()
