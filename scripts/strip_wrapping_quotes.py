"""
description_synth 전체를 감싼 따옴표만 제거한다.

왜:
  music의 24.9%가 문장 전체를 "..."로 감싼 채 생성됐다(movie 0.3% / book 0.7%).
  임베딩 영향 자체는 미미하다 — 감싼 것과 벗긴 것의 SBERT 코사인이 평균 0.9744다.
  남는 문제는 도메인 간 표면 형태의 비대칭 하나다. 이번 사이클 전체가 "3도메인
  텍스트를 같은 타입으로 통일"하는 작업인데 music에만 있는 표면 특징을 남기면
  SBERT가 그것을 도메인 식별 신호로 쓸 여지가 생긴다(세션 16 진단 A와 반대 방향).

주의:
  곡 제목 인용은 건드리지 않는다. music의 67.8%가 '"Best Friend," from his album...'
  처럼 제목을 인용하며 시작하는 정상 문장이고, 이것까지 벗기면 문장이 깨진다.
  텍스트 **전체**가 한 쌍의 따옴표로 감싸인 경우만 대상이다.

사용:
  python scripts/strip_wrapping_quotes.py            # 전 도메인, 실제 수정
  python scripts/strip_wrapping_quotes.py --dry-run  # 건수만 확인
"""

import argparse

import pandas as pd

from src.data.preprocessing import DOMAIN_CONFIG

OPEN_QUOTES = '"“'
CLOSE_QUOTES = '"”'


def strip_wrapping(text: str) -> str:
    """전체를 감싼 따옴표 한 쌍만 제거. 내부 인용은 유지."""
    s = text.strip()
    if len(s) < 2 or s[0] not in OPEN_QUOTES:
        return text

    # 끝의 따옴표(뒤에 문장부호가 붙은 경우 포함)를 찾는다: ..." / ...". / ..."!
    end = len(s) - 1
    while end > 0 and s[end] in ".!?":
        end -= 1
    if s[end] not in CLOSE_QUOTES:
        return text

    inner = s[1:end].strip()
    if not inner:
        return text
    # 바깥 한 쌍임을 확신할 수 있을 때만 벗긴다. 두 가지 근거 중 하나면 충분하다:
    #   (a) 문서 전체에 따옴표가 정확히 두 개 = 그 둘이 바깥 쌍이다
    #   (b) 닫는 따옴표 바로 앞이 문장부호 = 완결된 문장 뒤에서 닫혔다("...resilience.")
    # 반례로 '"Best Friend," from his album, evokes "quiet warmth"'는 시작과 끝이 모두
    # 따옴표지만 제목 인용 + 문구 인용이다((a) 4개, (b) 앞이 'h') → 건드리지 않는다.
    # 벗기면 문장이 깨지므로 놓치는 쪽이 망가뜨리는 쪽보다 낫다.
    quote_count = sum(s.count(q) for q in set(OPEN_QUOTES + CLOSE_QUOTES))
    closes_after_sentence = end > 0 and s[end - 1] in ".!?"
    if quote_count != 2 and not closes_after_sentence:
        return text

    tail = s[end + 1:]          # 닫는 따옴표 뒤의 문장부호를 살린다
    if not inner[-1] in ".!?" and tail:
        inner += tail
    elif not inner[-1] in ".!?":
        inner += "."
    return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAIN_CONFIG), choices=list(DOMAIN_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for domain in args.domains:
        path = DOMAIN_CONFIG[domain]["csv"]
        df = pd.read_csv(path, low_memory=False)
        if "description_synth" not in df.columns:
            print(f"{domain:6s} description_synth 컬럼 없음 — 건너뜀")
            continue

        col = df["description_synth"]
        stripped = col.map(lambda v: strip_wrapping(v) if isinstance(v, str) else v)
        changed = int((stripped != col).sum())
        print(f"{domain:6s} {changed:>7,} / {int(col.notna().sum()):>7,} 건 수정 "
              f"({changed / max(int(col.notna().sum()), 1):.1%})")
        if changed and not args.dry_run:
            df["description_synth"] = stripped
            df.to_csv(path, index=False)
            print(f"       → {path} 저장")

    if args.dry_run:
        print("\n(dry-run — 파일은 수정하지 않았다)")


if __name__ == "__main__":
    main()
