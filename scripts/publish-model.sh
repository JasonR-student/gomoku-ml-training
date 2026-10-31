#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: publish-model.sh /path/to/run-directory}"
MODEL="${RUN_DIR}/checkpoints/$(find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)/model.gmk"
REPO="${GH_REPO:?set GH_REPO to owner/gomoku-ml-training}"
GH_TOKEN="${GH_TOKEN:?set GH_TOKEN before publishing}"

if [[ ! -f "${MODEL}" ]]; then
  echo "model.gmk not found in ${RUN_DIR}" >&2
  exit 1
fi

SHA256="$(sha256sum "${MODEL}" | awk '{print $1}')"
VERSION="model-$(date -u +%Y%m%d-%H%M%S)"
cp "${MODEL}" "${RUN_DIR}/gomoku-model.gmk"
cp "${RUN_DIR}/model-manifest.json" "${RUN_DIR}/model-manifest-${VERSION}.json"
cp "${RUN_DIR}/summary.json" "${RUN_DIR}/training-summary-${VERSION}.json"
cp "${RUN_DIR}/metrics.csv" "${RUN_DIR}/metrics-${VERSION}.csv"
echo "Model: ${MODEL}"
echo "SHA-256: ${SHA256}"
gh release create "${VERSION}" \
  --repo "${REPO}" \
  --title "${VERSION}" \
  --notes "Gomoku self-play model ${VERSION}" \
  "${RUN_DIR}/gomoku-model.gmk" \
  "${RUN_DIR}/model-manifest-${VERSION}.json" \
  "${RUN_DIR}/training-summary-${VERSION}.json" \
  "${RUN_DIR}/metrics-${VERSION}.csv"
