#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${1:-}"
if [[ -z "$REMOTE_URL" ]]; then
  echo "usage: scripts/publish_to_github.sh <git-remote-url>"
  exit 2
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git branch -M main
git push -u origin main
git tag -a v0.1.0 -m "Battery AI + COMSOL Lab v0.1.0" || true
git push origin v0.1.0
