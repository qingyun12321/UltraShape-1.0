#!/usr/bin/env bash
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULTRASHAPE_ROOT="$(cd "${API_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${API_DIR}/../.." && pwd)"
SYSTEMD_SRC="${API_DIR}/systemd"
SYSTEMD_DIR="/etc/systemd/system"
ENV_DIR="/etc/ultrashape"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (sudo)."
  exit 1
fi

install -d "${SYSTEMD_DIR}" "${ENV_DIR}"

sed "s|__WORKSPACE_ROOT__|${WORKSPACE_ROOT}|g" \
  "${SYSTEMD_SRC}/ultrashape.env" > "${ENV_DIR}/ultrashape.env"

install -m 0644 "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu1.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu1.service"
install -m 0644 "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu3.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu3.service"
install -m 0644 "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu0.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu0.service"
install -m 0644 "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu2.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu2.service"
install -m 0644 "${SYSTEMD_SRC}/ultrashape-refine-api.service" "${SYSTEMD_DIR}/ultrashape-refine-api.service"
install -m 0644 "${SYSTEMD_SRC}/ultrashape-stack.service" "${SYSTEMD_DIR}/ultrashape-stack.service"

systemctl daemon-reload

if [[ "${1:-}" == "--start" ]]; then
  systemctl enable --now ultrashape-stack.service
  systemctl status --no-pager ultrashape-stack.service
else
  echo "Installed systemd units. To start: sudo systemctl enable --now ultrashape-stack.service"
fi
