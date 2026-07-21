#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-datasets/opv2v}"
mkdir -p "$DATA_DIR"

cat <<'EOF'
OPV2V is distributed by the official OpenCOOD project through external
cloud-hosted archives. Direct archive identifiers can change, so this helper
uses the maintained official pages instead of embedding a stale URL.

Official data guide:
  https://opencood.readthedocs.io/en/latest/md_files/data_intro.html

Official OpenCOOD repository:
  https://github.com/DerrickXuNu/OpenCOOD

Download the train/validate/test archives, then extract them under:
  datasets/opv2v/

Expected layout:
  datasets/opv2v/train
  datasets/opv2v/validate
  datasets/opv2v/test
  datasets/opv2v/test_culvercity

For split chunks:
  cat train.zip.part* > train.zip
  unzip train.zip -d datasets/opv2v
EOF

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "https://opencood.readthedocs.io/en/latest/md_files/data_intro.html" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "https://opencood.readthedocs.io/en/latest/md_files/data_intro.html" || true
fi
