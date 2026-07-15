"""
Book description 정제 스크립트.

Kindle 스크래핑 원본 description에는 실제 줄거리와 무관한 마케팅 문구가 섞여 있다.
  - 베스트셀러 배지: "NEW YORK TIMES BESTSELLER", "OVER 750,000 COPIES SOLD" 등
  - 서두의 리뷰 인용구: "..." — 평론가/유명인 이름 (예: "..."—Kirkus Reviews)

이 스크립트는 위 두 패턴을 문자열 맨 앞에서부터 반복적으로 제거해
`description_clean` 컬럼을 새로 추가한다. 원본 `description`은 보존한다
(보수적 규칙이라 일부 노이즈는 남을 수 있고, 중첩된 인용구는 첫 번째만 제거될 수 있음 —
 실제 줄거리 문장을 잘못 잘라내는 것을 피하기 위한 트레이드오프).

대상 파일: data/canonical/book_canonical.csv (학습 파이프라인과 db_export 공통 원본)

실행:
  python3 scripts/clean_book_description.py
"""
import re

import pandas as pd

BOOK_CANONICAL = "data/canonical/book_canonical.csv"

BADGE = re.compile(
    r"^\s*"
    r"(an?\s+)?(instant\s+)?(#\d+\s+)?"
    r"((new york times|usa today|wall street journal|amazon charts|indie|national|international|los angeles times)"
    r"\s*(,\s*|\s+and\s+|\s*/\s*)?)+"
    r"best.?sell(er|ing)\b[^.!\n]*[.!]?\s*"
    r"(•\s*)?",
    re.I,
)

COPIES_SOLD = re.compile(
    r"^\s*(over\s+)?[\d,\.]+\+?\s*(million|thousand)?\s*copies\s+sold!?\s*",
    re.I,
)

# 선두의 리뷰 인용구: "...." — Attribution  (attribution은 다음 인용부호나 문장 시작 전까지)
BLURB = re.compile(
    r'^\s*["“]([^"“”]{5,400})["”]\s*[-–—]{1,2}\s*'
    r"([A-Z][^\"“]{0,200}?)(?=[\"“]|\Z|(?<=[.\)])\s+[A-Z][a-z]+\s+[a-z]+\s+[a-z]+)",
)

# 다른 책 교차 홍보 인트로: "Don't miss X, ... available (to (pre-)order )now."
CROSS_PROMO = re.compile(
    r"^\s*don'?t miss\b.{0,200}?\bavailable\s+(to\s+(pre-?)?order\s+)?now\.{0,3}\s*",
    re.I,
)

# 아래 패턴들은 시험 결과 실제 줄거리를 잘못 잘라내는 사고가 확인되어 채택하지 않음:
#   - "#<n> bestseller ..." 순위 배지: 비탐욕 매칭이 실제 본문 중간의 유사 단어까지 삼켜버림
#   - 전체 대문자 문장 제거: 장르소설(스릴러/로맨스) 블러브는 대문자 훅 문장이 실제 콘텐츠인 경우가 많음


def clean_description(text) -> str:
    if pd.isna(text):
        return text
    s = str(text).strip()
    changed = True
    while changed:
        changed = False
        for pattern in (BADGE, COPIES_SOLD, CROSS_PROMO, BLURB):
            m = pattern.match(s)
            if m:
                s = s[m.end():].strip()
                changed = True
                break
    return s


def main():
    df = pd.read_csv(BOOK_CANONICAL, low_memory=False)

    before_notna = df["description"].notna().sum()
    df["description_clean"] = df["description"].apply(clean_description)

    diff_mask = df["description"].notna() & (df["description_clean"] != df["description"])
    print(f"description 보유 행: {before_notna}")
    print(f"정제로 변경된 행: {diff_mask.sum()}")

    df.to_csv(BOOK_CANONICAL, index=False, encoding="utf-8")
    print(f"저장 → {BOOK_CANONICAL} (description_clean 컬럼 추가)")


if __name__ == "__main__":
    main()
