#!/usr/bin/env bash
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULTRASHAPE_ROOT="$(cd "${API_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${API_DIR}/../.." && pwd)"
HUNYUAN_ROOT="${WORKSPACE_ROOT}/Hunyuan3D-2.1"
SYSTEMD_SRC="${API_DIR}/systemd"
SYSTEMD_DIR="/etc/systemd/system"
ENV_DIR="/etc/ultrashape"

detect_conda_sh() {
  local conda_sh=""
  if command -v conda >/dev/null 2>&1; then
    local conda_base=""
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      conda_sh="${conda_base}/etc/profile.d/conda.sh"
    fi
  fi
  if [[ -z "${conda_sh}" ]]; then
    for candidate in \
      /opt/conda/etc/profile.d/conda.sh \
      /mnt/data/miniconda3/etc/profile.d/conda.sh \
      /root/miniconda3/etc/profile.d/conda.sh \
      /root/anaconda3/etc/profile.d/conda.sh; do
      if [[ -f "${candidate}" ]]; then
        conda_sh="${candidate}"
        break
      fi
    done
  fi
  echo "${conda_sh}"
}

CONDA_SH_PATH="$(detect_conda_sh)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root (sudo)."
  exit 1
fi

install -d "${SYSTEMD_DIR}" "${ENV_DIR}"

render_unit() {
  local src="$1"
  local dest="$2"
  sed -e "s|__WORKSPACE_ROOT__|${WORKSPACE_ROOT}|g" \
      -e "s|__ULTRASHAPE_ROOT__|${ULTRASHAPE_ROOT}|g" \
      -e "s|__HUNYUAN_ROOT__|${HUNYUAN_ROOT}|g" \
      -e "s|__CONDA_SH__|${CONDA_SH_PATH}|g" \
      "${src}" > "${dest}"
  chmod 0644 "${dest}"
}

render_unit "${SYSTEMD_SRC}/ultrashape.env" "${ENV_DIR}/ultrashape.env"
render_unit "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu0.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu0.service"
render_unit "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu1.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu1.service"
render_unit "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu2.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu2.service"
render_unit "${SYSTEMD_SRC}/ultrashape-hunyuan-gpu3.service" "${SYSTEMD_DIR}/ultrashape-hunyuan-gpu3.service"
render_unit "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu0.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu0.service"
render_unit "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu1.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu1.service"
render_unit "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu2.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu2.service"
render_unit "${SYSTEMD_SRC}/ultrashape-ultrashape-gpu3.service" "${SYSTEMD_DIR}/ultrashape-ultrashape-gpu3.service"
render_unit "${SYSTEMD_SRC}/ultrashape-refine-api.service" "${SYSTEMD_DIR}/ultrashape-refine-api.service"
render_unit "${SYSTEMD_SRC}/ultrashape-stack.service" "${SYSTEMD_DIR}/ultrashape-stack.service"

systemctl daemon-reload

if [[ "${1:-}" == "--start" ]]; then
  systemctl enable --now ultrashape-stack.service
  systemctl status --no-pager ultrashape-stack.service
else
  echo "Installed systemd units. To start: sudo systemctl enable --now ultrashape-stack.service"
fi
