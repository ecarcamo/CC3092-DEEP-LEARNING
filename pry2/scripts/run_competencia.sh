#!/usr/bin/env bash
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELO="$RAIZ/models/competencia/SpaceInvaders_Rainbow_FINAL.pt"

if [[ ! -f "$MODELO" ]]; then
    echo "ERROR: no existe el modelo final:"
    echo "$MODELO"
    echo
    echo "Descárgalo con:"
    echo "gh release download pry2-rainbow-v1 \\"
    echo "  --repo ecarcamo/CC3092-DEEP-LEARNING \\"
    echo "  --pattern SpaceInvaders_Rainbow_FINAL.pt \\"
    echo "  --dir \"$RAIZ/models/competencia\""
    exit 1
fi

if [[ -x "$RAIZ/.venv-qrdqn/bin/python" ]]; then
    PYTHON="$RAIZ/.venv-qrdqn/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" \
    "$RAIZ/scripts/competencia_final.py" \
    --episodes 5 \
    "$@"
