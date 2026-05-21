"""
Books_filtered.csv의 main_category를 정규화.

3단계 처리:
  1) 명시적 동의어 매핑 (규칙 기반)
  2) 키워드 기반 매핑 (롱테일)
  3) 잔여 → Other

결과: main_category 컬럼을 덮어쓰고, 원본은 raw_category로 보존.
"""

import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV  = DATA_DIR / "Books_filtered.csv"
OUTPUT_CSV = DATA_DIR / "Books_filtered.csv"

# ── Step 1: 명시적 동의어 매핑 ────────────────────────────────────────────────
EXPLICIT_MAP: dict[str, str] = {
    # Fiction
    "Fiction in English":                           "Fiction",
    "Translations into English":                    "Fiction",
    "Classic Literature":                           "Fiction",
    "British and irish fiction (fictional works by one author)": "Fiction",
    "Nonfiction":                                   "Nonfiction",

    # Mystery / Thriller
    "Detective and mystery stories":                "Mystery/Thriller",
    "Mystery":                                      "Mystery/Thriller",
    "Women detectives":                             "Mystery/Thriller",
    "Private investigators":                        "Mystery/Thriller",
    "Murder":                                       "Mystery/Thriller",
    "Police":                                       "Mystery/Thriller",
    "American Detective and mystery stories":       "Mystery/Thriller",

    # Fantasy
    "Fantasy fiction":                              "Fantasy",
    "American Fantasy fiction":                     "Fantasy",
    "Magic":                                        "Fantasy",

    # Science Fiction
    "American Science fiction":                     "Science Fiction",
    "Science fiction":                              "Science Fiction",

    # Children's / YA
    "Juvenile fiction":                             "Children's/YA",
    "Children's fiction":                           "Children's/YA",
    "Children's stories":                           "Children's/YA",
    "Juvenile literature":                          "Children's/YA",

    # Romance
    "Romance fiction":                              "Romance",
    "Love stories":                                 "Romance",
    "Man-woman relationships":                      "Romance",

    # Horror
    "Horror tales":                                 "Horror",
    "Vampires":                                     "Horror",
    "Occult fiction":                               "Horror",

    # Biography / Memoir
    "Biographies":                                  "Biography/Memoir",
    "Biography":                                    "Biography/Memoir",
    "Autobiography":                                "Biography/Memoir",
    "Autobiographies":                              "Biography/Memoir",
    "Correspondence":                               "Biography/Memoir",
    "Authors":                                      "Biography/Memoir",
    "American Authors":                             "Biography/Memoir",

    # History
    "History and criticism":                        "History",
    "World War":                                    "History",
    "Civilization":                                 "History",

    # Travel
    "Guidebooks":                                   "Travel",
    "Description and travel":                       "Travel",

    # Poetry
    "Poetry (poetic works by one author)":          "Poetry",
    "American Poetry":                              "Poetry",

    # Short Stories
    "American Short stories":                       "Short Stories",
    "Short stories":                                "Short Stories",
    "Literary collections":                         "Short Stories",

    # Comics / Graphic Novels
    "Comic books":                                  "Comics/Graphic Novels",
    "Comics & graphic novels":                      "Comics/Graphic Novels",

    # Humor
    "Humorous stories":                             "Humor",
    "American wit and humor":                       "Humor",
    "Anecdotes":                                    "Humor",

    # Religion / Spirituality
    "Bible":                                        "Religion/Spirituality",
    "Christianity":                                 "Religion/Spirituality",
    "Christian life":                               "Religion/Spirituality",
    "Spiritual life":                               "Religion/Spirituality",
    "Buddhism":                                     "Religion/Spirituality",
    "Meditations":                                  "Religion/Spirituality",

    # Self-Help / Psychology
    "Conduct of life":                              "Self-Help/Psychology",
    "Psychology":                                   "Self-Help/Psychology",
    "Interpersonal relations":                      "Self-Help/Psychology",

    # Social / Political
    "Social life and customs":                      "Social/Political",
    "Politics and government":                      "Social/Political",
    "Social conditions":                            "Social/Political",
    "Race relations":                               "Social/Political",
    "African Americans":                            "Social/Political",
    "Women":                                        "Social/Political",

    # Science / Nature
    "Animals":                                      "Science/Nature",
    "Dinosaurs":                                    "Science/Nature",
    "Science":                                      "Science/Nature",

    # Cooking
    "Cookery":                                      "Cooking",

    # Drama
    "Drama":                                        "Drama",

    # Philosophy
    "Philosophy":                                   "Philosophy",
    "Literature":                                   "Philosophy",
    "Criticism and interpretation":                 "Philosophy",

    # Miscellanea / Reference
    "Miscellanea":                                  "Reference/Misc",
    "Dictionaries":                                 "Reference/Misc",
    "Handbooks":                                    "Reference/Misc",
    "Quotations":                                   "Reference/Misc",
    "Popular works":                                "Reference/Misc",
    "Case studies":                                 "Reference/Misc",

    # Fairy Tales / Folklore
    "Fairy tales":                                  "Fairy Tales/Folklore",
    "Folklore":                                     "Fairy Tales/Folklore",
    "Legends":                                      "Fairy Tales/Folklore",
}

# ── Step 2: 키워드 기반 매핑 (소문자 포함 여부 검사) ──────────────────────────
# 우선순위 순서 중요 — 앞쪽이 먼저 매칭
KEYWORD_MAP: list[tuple[str, str]] = [
    ("science fiction",         "Science Fiction"),
    ("detective",               "Mystery/Thriller"),
    ("mystery",                 "Mystery/Thriller"),
    ("thriller",                "Mystery/Thriller"),
    ("crime",                   "Mystery/Thriller"),
    ("fantasy",                 "Fantasy"),
    ("horror",                  "Horror"),
    ("vampire",                 "Horror"),
    ("ghost",                   "Horror"),
    ("romance",                 "Romance"),
    ("love stor",               "Romance"),
    ("juvenile",                "Children's/YA"),
    ("children",                "Children's/YA"),
    ("young adult",             "Children's/YA"),
    ("biography",               "Biography/Memoir"),
    ("autobio",                 "Biography/Memoir"),
    ("memoir",                  "Biography/Memoir"),
    ("history",                 "History"),
    ("historical",              "History"),
    ("travel",                  "Travel"),
    ("guidebook",               "Travel"),
    ("cook",                    "Cooking"),
    ("recipe",                  "Cooking"),
    ("poetry",                  "Poetry"),
    ("poem",                    "Poetry"),
    ("comic",                   "Comics/Graphic Novels"),
    ("graphic novel",           "Comics/Graphic Novels"),
    ("manga",                   "Comics/Graphic Novels"),
    ("humor",                   "Humor"),
    ("humour",                  "Humor"),
    ("comic",                   "Humor"),
    ("bible",                   "Religion/Spirituality"),
    ("christian",               "Religion/Spirituality"),
    ("religion",                "Religion/Spirituality"),
    ("spiritual",               "Religion/Spirituality"),
    ("buddhis",                 "Religion/Spirituality"),
    ("islam",                   "Religion/Spirituality"),
    ("psycholog",               "Self-Help/Psychology"),
    ("self-help",               "Self-Help/Psychology"),
    ("self help",               "Self-Help/Psychology"),
    ("conduct",                 "Self-Help/Psychology"),
    ("politic",                 "Social/Political"),
    ("social",                  "Social/Political"),
    ("fiction",                 "Fiction"),
    ("short stor",              "Short Stories"),
    ("philosoph",               "Philosophy"),
    ("science",                 "Science/Nature"),
    ("nature",                  "Science/Nature"),
    ("animal",                  "Science/Nature"),
    ("drama",                   "Drama"),
    ("play",                    "Drama"),
    ("fairy",                   "Fairy Tales/Folklore"),
    ("folk",                    "Fairy Tales/Folklore"),
    ("legend",                  "Fairy Tales/Folklore"),
    ("myth",                    "Fairy Tales/Folklore"),
]


def normalize(raw: str) -> str:
    stripped = raw.strip()

    # Step 1: 명시적 매핑
    if stripped in EXPLICIT_MAP:
        return EXPLICIT_MAP[stripped]

    # Step 2: 키워드 매핑
    lower = stripped.lower()
    for keyword, target in KEYWORD_MAP:
        if keyword in lower:
            return target

    # Step 3: 잔여
    return "Other"


def main():
    print(f"CSV 로딩: {INPUT_CSV}")
    with open(INPUT_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f"총 {len(rows):,}개 행\n")

    # raw_category 컬럼 추가 (원본 보존)
    if "raw_category" not in fieldnames:
        cat_idx = fieldnames.index("main_category")
        fieldnames.insert(cat_idx + 1, "raw_category")

    before_counter: Counter = Counter()
    after_counter: Counter = Counter()

    for row in rows:
        raw = row["main_category"]
        normalized = normalize(raw)
        row["raw_category"] = raw
        row["main_category"] = normalized
        before_counter[raw] += 1
        after_counter[normalized] += 1

    print(f"정규화 전 고유 카테고리: {len(before_counter):,}개")
    print(f"정규화 후 고유 카테고리: {len(after_counter):,}개\n")

    print("정규화 후 분포:")
    total = len(rows)
    for cat, cnt in after_counter.most_common():
        print(f"  {cnt:7,}  ({cnt/total*100:5.1f}%)  {cat}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
