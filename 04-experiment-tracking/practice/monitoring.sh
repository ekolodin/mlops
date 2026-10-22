#!/usr/bin/env bash
# Runs only the two native binaries owned by this practice. Ctrl+C stops both.
set -euo pipefail
course_practice=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$course_practice"

if [[ "${1:-}" == "install" ]]; then
  exec uv run --locked python install_monitoring.py
fi
if [[ $# -gt 0 ]]; then
  printf 'Usage: bash monitoring.sh [install]\n' >&2
  exit 2
fi

course_prometheus="$course_practice/.tools/prometheus-3.15.0"
course_grafana="$course_practice/.tools/grafana-13.2.3"
if [[ ! -x "$course_prometheus/prometheus" || ! -x "$course_grafana/bin/grafana" ]]; then
  printf 'Сначала выполните: bash monitoring.sh install\n' >&2
  exit 1
fi
# Do not start a second stack over somebody else's already-bound ports.
uv run --locked python -c 'import socket
for port in (9090, 3000):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Порт {port} занят; остановите предыдущий экземпляр.")'
"$course_prometheus/promtool" check config monitoring/prometheus.yml
mkdir -p state/prometheus state/grafana/logs state/grafana/plugins

course_prom_pid=""
course_graf_pid=""
cleanup() {
  trap - EXIT INT TERM
  # Only our own direct child PIDs: no killall or lookup by process name.
  if [[ -n "$course_prom_pid" ]]; then kill "$course_prom_pid" 2>/dev/null || true; fi
  if [[ -n "$course_graf_pid" ]]; then kill "$course_graf_pid" 2>/dev/null || true; fi
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$course_prometheus/prometheus" \
  --config.file="$course_practice/monitoring/prometheus.yml" \
  --storage.tsdb.path="$course_practice/state/prometheus" \
  --web.listen-address=127.0.0.1:9090 \
  > state/prometheus.log 2>&1 &
course_prom_pid=$!

# Absolute paths let Grafana locate configs and dashboards regardless of cwd.
export COURSE_DASHBOARDS_DIR="$course_practice/monitoring/dashboards"
export GF_PATHS_PROVISIONING="$course_practice/monitoring/provisioning"
export GF_PATHS_DATA="$course_practice/state/grafana"
export GF_PATHS_LOGS="$course_practice/state/grafana/logs"
export GF_PATHS_PLUGINS="$course_practice/state/grafana/plugins"
"$course_grafana/bin/grafana" server \
  --homepath="$course_grafana" \
  --config="$course_practice/monitoring/grafana.ini" \
  > state/grafana.log 2>&1 &
course_graf_pid=$!

printf 'Prometheus: http://127.0.0.1:9090\nGrafana: http://127.0.0.1:3000 (admin / seminar)\nЛоги: state/prometheus.log и state/grafana.log. Остановка: Ctrl+C.\n'
while kill -0 "$course_prom_pid" 2>/dev/null && kill -0 "$course_graf_pid" 2>/dev/null; do
  sleep 1
done
printf 'Один из процессов завершился. Проверьте его лог в state/.\n' >&2
exit 1
