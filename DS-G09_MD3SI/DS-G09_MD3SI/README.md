# DS-G07_MD3SI - Early-Containment Agent for a Ransomware Simulation

## Objectif
Ce projet construit un agent de Reinforcement Learning qui apprend à contenir une propagation de ransomware **simulée** dans un réseau de cinq PC. Aucun ransomware réel n'est exécuté : l'environnement applique uniquement des règles probabilistes de propagation et des actions défensives.

## Structure
```text
DS-G07_MD3SI/
├── README.md
├── requirements.txt
├── run_all.sh
├── configs/config.yaml
├── src/
│   ├── env.py
│   ├── dqn.py
│   ├── train.py
│   └── eval.py
├── results/
├── logs/
├── figures/
├── videos/
├── notebook.md
└── report.pdf
```

## Reproduction
Avec Python 3.11+ recommandé :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Sous Windows PowerShell, activer l'environnement avec `.venv\\Scripts\\Activate.ps1`, puis lancer le script dans Git Bash/WSL ou reproduire les commandes Python de `run_all.sh`.

## Sorties produites
- `results/training.csv` : statistiques d'entraînement (`seed`, `step`, `episode_return`, ...).
- `results/evaluation.csv` : statistiques d'évaluation.
- `results/dqn_ransomware.pth` : modèle DQN entraîné.
- `figures/*.pdf` : figures vectorielles, avec moyenne et écart-type pour la courbe d'apprentissage.
- `videos/before_training.mp4` et `videos/after_training.mp4` : comparaison avant/après entraînement.
- `report.pdf` : rapport final.

## Environnement
Chaque PC peut être `SAIN`, `INFECTE`, `ISOLE` ou `PATCHE`. L'observation est `[état_PC1, ..., état_PCN, étape]`. L'espace d'action est discret : `0` = ne rien faire, `1..N` = isoler, `N+1..2N` = patcher.

## Reproductibilité
Les seeds et hyperparamètres sont dans `configs/config.yaml`. Il n'y a pas de valeur d'entraînement principale à modifier dans le code source.

## Note importante
Le projet ne doit pas être utilisé avec un logiciel malveillant réel. La propagation est une simulation mathématique locale destinée à l'apprentissage du RL et à l'évaluation d'une stratégie de défense.
