#!/usr/bin/env bash
# End-to-end run: download -> train -> evaluate -> sample recommendations.
set -euo pipefail
cd "$(dirname "$0")"

python3 download_data.py
python3 train.py
python3 evaluate.py
python3 recommend.py --user_id 1 --top_n 10
