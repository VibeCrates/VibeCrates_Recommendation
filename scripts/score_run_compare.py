"""두 실행 비교 시트(build_run_compare_sheet.py)의 사람 라벨을 정답지와 맞춰 집계한다.

미리 정한 판정(시트를 만들 때 정함): 답한 쌍(1·2)에서 비교 실행(b) 선호율, 양측 부호검정
p<0.05면 차이 있음. 애매(빈칸)는 빼고 따로 센다. 하위 집단(도메인필터·어휘 겹침·스타일)은
사후 분석이라 여러 번 검정한 것이다 — 가설로만 읽는다.

사용:
  python scripts/score_run_compare.py --prefix experiments/compare_titledrop_20261009
  → {prefix}_score.txt
"""
import argparse
import csv
from math import comb


def sign_p(k: int, n: int) -> float:
    """양측 정확 부호검정."""
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    lab = {r["pair_id"]: r for r in csv.DictReader(open(f"{args.prefix}_hyun.csv", encoding="utf-8-sig"))}
    key = {r["pair_id"]: r for r in csv.DictReader(open(f"{args.prefix}_key.csv", encoding="utf-8-sig"))}
    k0 = next(iter(key.values()))
    na, nb = k0["a_name"], k0["b_name"]
    lines = []

    def row(name, ids):
        a = b = skip = 0
        for i in ids:
            c = lab[i]["선택"]
            if c not in ("1", "2"):
                skip += 1
            elif c == key[i]["b_side"]:
                b += 1
            else:
                a += 1
        n = a + b
        rate = f"{b / n:.1%}" if n else "-"
        lines.append(f"  {name:22s} {nb} {b:3d} / {na} {a:3d} / 애매 {skip:2d}   {nb} 선호율 {rate:>6s}   p={sign_p(b, n):.3f}")

    ids = list(key)
    missing = [i for i in ids if i not in lab or lab[i].get("완료", "1") != "1"]
    lines.append(f"{args.prefix}: {len(ids)}쌍, 라벨 미완료 {len(missing)}")
    lines.append("")
    lines.append("[미리 정한 판정]")
    row("전체", ids)
    lines.append("")
    lines.append("[사후 분석 — 가설로만]")
    for d in ("all", "movie", "music", "book"):
        row(f"도메인필터={d}", [i for i in ids if key[i]["domain_filter"] == d])
    ov = lambda i: (int(key[i]["a_ov"]) > 0, int(key[i]["b_ov"]) > 0)
    row(f"{na}만 쿼리 단어 겹침", [i for i in ids if ov(i) == (True, False)])
    row(f"{nb}만 쿼리 단어 겹침", [i for i in ids if ov(i) == (False, True)])
    row("겹침 같음(둘 다/없음)", [i for i in ids if ov(i)[0] == ov(i)[1]])
    for st in sorted({key[i]["style"] for i in ids}):
        row(f"스타일={st}", [i for i in ids if key[i]["style"] == st])
    pos1 = sum(lab[i]["선택"] == "1" for i in ids)
    n12 = sum(lab[i]["선택"] in ("1", "2") for i in ids)
    lines.append("")
    lines.append(f"[위치 점검] 후보1 선택 {pos1}/{n12} ({pos1 / n12:.1%}), p={sign_p(pos1, n12):.3f}"
                 f" — {nb} 쪽이 1번인 쌍 {sum(key[i]['b_side'] == '1' for i in ids)}/{len(ids)}")

    text = "\n".join(lines)
    open(f"{args.prefix}_score.txt", "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
