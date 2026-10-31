#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mvn -q -DskipTests package
exec java -jar target/gomoku-ml-training-1.0.0.jar train \
  --hours "${TRAIN_HOURS:-24}" \
  --output "${TRAIN_OUTPUT:-./runs}" \
  --serve-data \
  --data-host "${DATA_HOST:-0.0.0.0}" \
  --data-port "${DATA_PORT:-5000}" \
  --data-port-end "${DATA_PORT_END:-5005}" \
  --data-token "${DATA_TOKEN:-CHANGE_ME}"
