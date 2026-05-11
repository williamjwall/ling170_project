#!/usr/bin/env bash
# Debian/Ubuntu/WSL: system Python is PEP 668–managed; `python` may be missing.
# Preferred: sudo apt install -y python3.12-venv && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m spacy download en_core_web_sm
# Fallback (user site-packages): run this script as your normal user.

set -euo pipefail
PY="${PY:-python3}"
$PY -m pip install --user --break-system-packages "spacy>=3.7,<4"
$PY -m pip install --user --break-system-packages \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
$PY -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy + en_core_web_sm OK')"
