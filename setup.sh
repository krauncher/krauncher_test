#!/bin/bash
# Clone the current krauncher client from git and install it into a local venv.
# Pass a branch, tag or commit to pin the client: ./setup.sh <ref>
set -e
cd "$(dirname "$0")"
REPO=https://github.com/krauncher/krauncher.git
[ -d client ] || git clone -q "$REPO" client
git -C client fetch -q origin
git -C client checkout -q "${1:-origin/main}"
python3 -m venv .venv
.venv/bin/pip install -q -e client
echo "client $(git -C client rev-parse --short HEAD) installed into .venv"
