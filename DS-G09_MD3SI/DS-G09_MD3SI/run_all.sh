#!/usr/bin/env bash
# run_all.sh - Regenere TOUT (entrainement, resultats, figures, videos)
# en une seule commande :
#
#   bash run_all.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== 1/4 : Installation des dependances =="
pip install -r requirements.txt --quiet --break-system-packages

echo "== 2/4 : Tests de conformite de l'environnement =="
python3 -m tests.test_env

echo "== 3/4 : Entrainement (toutes les graines de configs/train.yaml) =="
rm -f results/results.csv
python3 -m src.train

echo "== 4/4 : Evaluation, figures et videos =="
python3 -m src.eval

echo ""
echo "Termine. Voir :"
echo "  - results/results.csv, results/eval_stats.csv"
echo "  - figures/learning_curve.pdf"
echo "  - videos/agent_before.mp4, videos/agent_after.mp4"
echo "  - logs/ (TensorBoard : tensorboard --logdir logs)"