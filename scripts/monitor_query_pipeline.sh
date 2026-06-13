#!/bin/bash
# Monitors query generation chain and auto-restarts remaining domains on failure.
# Run in a separate tmux pane — does not interrupt the active query_gen session.

PYTHON=/opt/conda/envs/ltv/bin/python
WORKDIR=/home/data/Crates/VibeCrates_Recommendation
STATUS_FILE="$WORKDIR/logs/pipeline_monitor.log"
CHECK_INTERVAL=300  # seconds between checks

cd "$WORKDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS_FILE"
}

domain_completed() {
    local domain=$1
    local log="$WORKDIR/logs/query_${domain}.log"
    [ -f "$log" ] && grep -qE "완료!|^\[DONE\]" "$log"
}

domain_log_exists() {
    [ -f "$WORKDIR/logs/query_${domain}.log" ]
}

run_domain() {
    local domain=$1
    log "→ Starting: $domain"
    $PYTHON scripts/generate_queries.py --domain "$domain" --model-type qwen \
        2>&1 | tee -a "logs/query_${domain}.log"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "✓ $domain: 완료"
    else
        log "✗ $domain: 실패 (exit code: $exit_code)"
    fi
    return $exit_code
}

query_gen_running() {
    tmux list-panes -t query_gen -F "#{pane_current_command}" 2>/dev/null | grep -q "python"
}

# ── active_domain: 현재 어느 로그 파일이 가장 최근 수정됐는지로 판단
active_domain() {
    local newest=""
    local newest_time=0
    for d in movie music book; do
        local f="$WORKDIR/logs/query_${d}.log"
        if [ -f "$f" ]; then
            local t
            t=$(stat -c %Y "$f")
            if [ "$t" -gt "$newest_time" ]; then
                newest_time=$t
                newest=$d
            fi
        fi
    done
    echo "$newest"
}

log "=== 파이프라인 모니터 시작 (체크 주기: ${CHECK_INTERVAL}s) ==="

while true; do
    if query_gen_running; then
        active=$(active_domain)
        log "실행 중 — 활성 도메인: ${active:-unknown}"
    else
        log "query_gen 프로세스 종료 감지 — 도메인 상태 확인"

        all_done=true
        for domain in movie music book; do
            if domain_completed "$domain"; then
                log "✓ $domain: 이미 완료"
            else
                log "! $domain: 미완료 — 재시작"
                all_done=false
                run_domain "$domain"
                if [ $? -ne 0 ]; then
                    log "✗ $domain 실패로 파이프라인 중단"
                    exit 1
                fi
            fi
        done

        if $all_done; then
            log "=== 전체 완료: Movie + Music + Book ==="
            exit 0
        fi
    fi

    sleep $CHECK_INTERVAL
done
