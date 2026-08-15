#!/usr/bin/env bash
# 추천 API 서버 실행.
#
#   ./scripts/serve.sh              # 기본 8000 포트
#   PORT=9000 ./scripts/serve.sh    # 포트 변경
#
# --host 0.0.0.0 인 이유:
#   127.0.0.1(localhost)로 띄우면 이 컴퓨터 안에서만 접속된다. 백엔드가 다른 컴퓨터에서
#   부르려면 모든 네트워크 인터페이스에 열어야 하고 그게 0.0.0.0이다.
#   (Tailscale IP로 들어오는 요청도 이 설정이 있어야 받는다.)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "▶ 추천 API 시작 — http://${HOST}:${PORT}"
echo "  문서(브라우저):  http://127.0.0.1:${PORT}/docs"
echo "  통신 확인:       curl 'http://127.0.0.1:${PORT}/api/v1/ping?n=42'"

# 이 컴퓨터가 네트워크에서 어떤 주소로 보이는지 알려준다. 백엔드에게 이 주소를 준다.
if command -v ipconfig >/dev/null 2>&1; then           # macOS
  for iface in en0 en1; do
    ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
    [ -n "$ip" ] && echo "  이 컴퓨터의 LAN 주소: http://${ip}:${PORT}"
  done
fi
if command -v tailscale >/dev/null 2>&1; then
  ts=$(tailscale ip -4 2>/dev/null | head -1 || true)
  [ -n "$ts" ] && echo "  Tailscale 주소:       http://${ts}:${PORT}   ← 백엔드에게 줄 주소"
fi
echo

exec uvicorn src.api.main:app --host "$HOST" --port "$PORT"
