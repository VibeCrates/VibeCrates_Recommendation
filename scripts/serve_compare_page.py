"""비교 라벨링 페이지를 띄우고, 누를 때마다 라벨을 CSV에 바로 기록한다.

브라우저 저장소(localStorage)는 브라우저를 바꾸거나 사이트 데이터를 지우면 사라지고,
"내보내기"를 잊으면 라벨이 파일로 남지 않는다. 그래서 페이지가 선택·메모가 바뀔 때마다
POST /labels로 전체 라벨을 보내고, 이 서버가 {prefix}_hyun.csv를 통째로 다시 쓴다
(임시 파일 → os.replace라 쓰다 끊겨도 반쪽 파일이 남지 않는다). 다시 열면 GET /labels로
CSV를 읽어 이어서 한다.

사용:
  python scripts/serve_compare_page.py --prefix experiments/compare_titledrop_20261009
  → http://localhost:8765 를 연다 (Ctrl+C로 종료)
"""
import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    page = Path(f"{args.prefix}_page.html").read_bytes()
    out = Path(f"{args.prefix}_hyun.csv")
    queries = {r["pair_id"]: r["query"]
               for r in csv.DictReader(open(f"{args.prefix}.csv", encoding="utf-8-sig"))}

    def read_labels() -> dict:
        if not out.exists():
            return {}
        return {r["pair_id"]: {"choice": r["선택"], "memo": r["메모"], "done": r.get("완료", "") == "1"}
                for r in csv.DictReader(open(out, encoding="utf-8-sig"))}

    def write_labels(labels: dict) -> None:
        tmp = out.with_suffix(".csv.part")
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "query", "선택", "메모", "완료"])
            for pid, q in queries.items():           # 시트 순서 그대로, 미응답 행도 남긴다
                l = labels.get(pid, {})
                w.writerow([pid, q, l.get("choice") or "", l.get("memo") or "", "1" if l.get("done") else ""])
        os.replace(tmp, out)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/labels":
                self._send(200, json.dumps(read_labels(), ensure_ascii=False).encode(), "application/json")
            else:
                self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/labels":
                return self._send(404, b"", "text/plain")
            labels = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            unknown = set(labels) - set(queries)
            if unknown:
                return self._send(400, f"모르는 pair_id: {sorted(unknown)[:5]}".encode(), "text/plain")
            write_labels(labels)
            self._send(200, b'{"ok":true}', "application/json")

        def log_message(self, *a):                   # 클릭마다 찍히는 접근 로그는 끈다
            pass

    print(f"http://localhost:{args.port}  →  라벨은 {out} 에 바로 기록된다 (Ctrl+C로 종료)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
