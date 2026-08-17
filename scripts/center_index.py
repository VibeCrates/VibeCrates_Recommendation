"""
도메인별 중심 벡터를 제거한 인덱스를 만든다 (도메인 편향 완화 실험).

왜:
  통합 검색에서 상위 10개가 한 도메인으로 쏠린다. 40개 쿼리 중 7개는 음악이 8개 이상,
  23개는 음악이 아예 없다. 원인은 아이템 벡터가 도메인끼리 뭉쳐 있기 때문이다 —
  같은 도메인 안 평균 코사인이 music +0.136 / movie +0.065 / book +0.028인데 도메인이
  다르면 음수다(movie↔music -0.024). 도메인 중심 벡터의 길이도 music 0.369로 커서,
  곡 벡터의 3분의 1 이상이 "이건 음악이다"를 표현하는 데 쓰이고 있다.

무엇을:
  각 아이템 벡터에서 자기 도메인의 평균 벡터를 빼고 다시 L2 정규화한다. 공통 성분이
  사라지므로 도메인이라는 이유만으로 가까워지는 효과가 없어진다(실측: centering 후
  같은 도메인·다른 도메인 평균 코사인이 모두 0.000).

한계 (해석 시 반드시 감안할 것):
  - 빼는 성분이 순수한 "도메인 표시"만은 아니다. 도메인이 실제로 공유하는 의미적 특성도
    함께 사라질 수 있다.
  - 쿼리 쪽에는 뺄 중심이 없다. 아이템만 옮기므로 두 벡터가 다른 좌표계에 놓이고,
    점수 자체가 달라진다. 좋은 방향인지는 검색 결과로 판정해야 한다.
  - 이것은 증상 완화다. 근본 원인은 content_text가 도메인마다 다른 머리말로 시작해
    SBERT가 그것을 도메인 신호로 학습한다는 데 있고, 그 수정은 재학습이 필요하다.

사용:
  python scripts/center_index.py --src indexes_qlora_off --dst indexes_qlora_off_centered
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F

DOMAINS = ("movie", "music", "book")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 인덱스 디렉토리")
    ap.add_argument("--dst", required=True, help="centering 결과를 쓸 디렉토리")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for domain in DOMAINS:
        emb_src = os.path.join(args.src, f"{domain}_embeddings.pt")
        meta_src = os.path.join(args.src, f"{domain}_meta.parquet")
        if not os.path.exists(emb_src):
            print(f"[{domain}] 원본 없음 — 건너뜀")
            continue

        z = torch.load(emb_src, map_location="cpu", weights_only=False)
        centroid = z.mean(0)
        z_centered = F.normalize(z - centroid, p=2, dim=1)

        torch.save(z_centered, os.path.join(args.dst, f"{domain}_embeddings.pt"))
        # 메타는 그대로 쓴다 — 행 순서가 임베딩과 1:1로 대응하므로 복사만 한다.
        shutil.copy(meta_src, os.path.join(args.dst, f"{domain}_meta.parquet"))
        # 중심 벡터도 남긴다. 나중에 쿼리 쪽에도 같은 보정을 시도하거나
        # 원본으로 되돌릴 때 필요하다.
        torch.save(centroid, os.path.join(args.dst, f"{domain}_centroid.pt"))

        print(f"[{domain}] {tuple(z.shape)}  중심 길이 {centroid.norm():.4f} → 제거 완료")

    print(f"\n저장 → {args.dst}/")


if __name__ == "__main__":
    main()
