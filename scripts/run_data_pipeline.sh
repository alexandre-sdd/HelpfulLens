#!/usr/bin/env bash
# Orchestrate the end-to-end Yelp helpfulness data pipeline.

set -euo pipefail

CONFIG="${CONFIG:-src/config/config.yaml}"
INGEST_DATASETS="${INGEST_DATASETS:-business reviews users}"
CLEAN_DATASETS="${CLEAN_DATASETS:-reviews users business}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-}"
FORCE="${FORCE:-0}"
INGEST_LIMIT="${INGEST_LIMIT:-}"
CLEAN_LIMIT="${CLEAN_LIMIT:-}"

IFS=' ' read -r -a ingest_array <<< "${INGEST_DATASETS}"
IFS=' ' read -r -a clean_array <<< "${CLEAN_DATASETS}"

log() {
    printf "[run_data_pipeline] %s\n" "$*"
}

run_cmd() {
    log "Running: $*"
    "$@"
}

log "Starting ingest_raw step"
ingest_cmd=(python -m src.data.ingest_raw --config "${CONFIG}")
if [[ ${#ingest_array[@]} -gt 0 ]]; then
    ingest_cmd+=(--datasets "${ingest_array[@]}")
fi
if [[ -n "${CHUNK_SIZE}" ]]; then
    ingest_cmd+=(--chunk-size "${CHUNK_SIZE}")
fi
if [[ -n "${ROWS_PER_CHUNK}" ]]; then
    ingest_cmd+=(--rows-per-chunk "${ROWS_PER_CHUNK}")
fi
if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
    ingest_cmd+=(--force)
fi
if [[ -n "${INGEST_LIMIT}" ]]; then
    ingest_cmd+=(--limit "${INGEST_LIMIT}")
fi
run_cmd "${ingest_cmd[@]}"

log "Starting clean_yelp step"
clean_cmd=(python -m src.data.clean_yelp --config "${CONFIG}")
if [[ ${#clean_array[@]} -gt 0 ]]; then
    clean_cmd+=(--datasets "${clean_array[@]}")
fi
if [[ -n "${CLEAN_LIMIT}" ]]; then
    clean_cmd+=(--limit "${CLEAN_LIMIT}")
fi
run_cmd "${clean_cmd[@]}"

log "Starting make_dataset step"
run_cmd python -m src.data.make_dataset --config "${CONFIG}"

log "Pipeline finished successfully."
