"""
추천 API가 밖에서 제대로 보이는지 점검한다 (백엔드 역할 흉내).

왜 필요한가:
  연동 중에 응답이 없으면 원인이 여러 개다 — 서버가 안 떴거나, 0.0.0.0이 아니라
  127.0.0.1에 묶였거나, 방화벽이 막았거나, 주소를 잘못 알려줬거나, 경로 오타이거나.
  이 스크립트는 그 후보를 순서대로 좁혀서 **어디까지 되고 어디서 막히는지**를 알려준다.
  화요일에 백엔드와 붙었을 때 "우리 쪽은 여기까지 정상"을 즉시 보일 수 있다.

사용:
  python scripts/check_api.py                          # 로컬 서버 점검
  python scripts/check_api.py --host 100.x.y.z         # Tailscale 주소 점검
  python scripts/check_api.py --host 100.x.y.z --port 9000
"""

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request

OK, FAIL, WARN = "✅", "❌", "⚠️ "


def get(url: str, timeout: float = 5.0):
    """(성공여부, 상태코드 or None, 본문 or 오류메시지, 소요 ms)"""
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode()
            return True, r.status, body, (time.time() - start) * 1000
    except urllib.error.HTTPError as e:
        return False, e.code, e.read().decode()[:200], (time.time() - start) * 1000
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}", (time.time() - start) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="점검할 주소 (기본 로컬)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"점검 대상: {base}\n")
    failed = 0

    # 1) TCP 연결 — 여기서 막히면 서버가 안 떴거나 방화벽/주소 문제이지 API 문제가 아니다.
    print("1. 포트가 열려 있는가")
    s = socket.socket()
    s.settimeout(args.timeout)
    try:
        s.connect((args.host, args.port))
        print(f"   {OK} {args.host}:{args.port} 연결됨")
    except Exception as e:
        print(f"   {FAIL} 연결 실패 — {type(e).__name__}: {e}")
        print("      → 서버가 실행 중인지, --host 0.0.0.0으로 떴는지, 주소가 맞는지 확인")
        sys.exit(1)
    finally:
        s.close()

    # 2) ping — 모델과 무관하게 200이어야 한다.
    print("\n2. /api/v1/ping (통신 확인)")
    ok, code, body, ms = get(f"{base}/api/v1/ping?n=42", args.timeout)
    if ok and code == 200:
        data = json.loads(body)
        print(f"   {OK} {code} · {ms:.0f}ms · {data}")
        if data.get("received") != 42:
            print(f"   {FAIL} received가 42가 아니다 — 쿼리 파라미터가 전달되지 않았다")
            failed += 1
        if not data.get("model_loaded"):
            print(f"   {WARN}model_loaded=false — 통신 확인 단계에서는 정상 "
                  "(추천 기능을 쓰려면 모델·인덱스가 필요하다)")
    else:
        print(f"   {FAIL} {code} · {body}")
        failed += 1

    # 3) 타입 검증 — 프록시가 중간에서 응답을 바꿔치기하지 않는지도 함께 드러난다.
    print("\n3. /api/v1/ping?n=abc (잘못된 입력 → 422여야 정상)")
    ok, code, body, ms = get(f"{base}/api/v1/ping?n=abc", args.timeout)
    print(f"   {OK if code == 422 else FAIL} {code} · {ms:.0f}ms")
    if code != 422:
        failed += 1

    # 4) health — 모델·인덱스 준비 상태
    print("\n4. /api/v1/health (준비 상태)")
    ok, code, body, ms = get(f"{base}/api/v1/health", args.timeout)
    if ok and code == 200:
        d = json.loads(body)
        print(f"   {OK} {code} · model_loaded={d.get('model_loaded')} · "
              f"index={d.get('index_built')}")
    else:
        print(f"   {FAIL} {code} · {body}")
        failed += 1

    # 5) 추천 — 아직 모델이 없으면 503이 정상이다. 500이면 진짜 버그다.
    print("\n5. /api/v1/recommend (아직 미동작이면 503이 정상)")
    req = urllib.request.Request(
        f"{base}/api/v1/recommend",
        data=json.dumps({"query": "비 오는 날", "top_k": 3}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            d = json.loads(r.read().decode())
            print(f"   {OK} 200 · 결과 {len(d.get('results', []))}건 — 추천 기능 동작 중")
    except urllib.error.HTTPError as e:
        if e.code == 503:
            print(f"   {OK} 503 · 모델 미적재 (현 단계에서는 정상)")
        else:
            print(f"   {FAIL} {e.code} · {e.read().decode()[:200]}")
            failed += 1
    except Exception as e:
        print(f"   {FAIL} {type(e).__name__}: {e}")
        failed += 1

    print("\n" + ("=" * 52))
    print(f"{OK} 모든 점검 통과 — 우리 쪽 준비 완료" if failed == 0
          else f"{FAIL} 실패 {failed}건")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
