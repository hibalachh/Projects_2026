# Weekly lab log

## Semaine 1
- Date: 2026-08-17
- Claim: Un environnement Gymnasium discret suffit pour représenter un petit problème d'Early-Containment.
- Evidence: `src/env.py` définit `observation_space`, `action_space`, `reset()` et `step()`.
- Decision: conserver 5 PC et 11 actions pour garder le projet simple.

## Semaine 2
- Date: 2026-08-17
- Claim: DQN est adapté aux observations vectorielles et aux actions discrètes de la simulation.
- Evidence: `src/dqn.py` utilise un réseau MLP et un replay buffer.
- Decision: utiliser un target network et epsilon-greedy.

## Semaine 3
- Date: 2026-08-17
- Claim: La reproductibilité demande plusieurs seeds et des paramètres externalisés.
- Evidence: `configs/config.yaml` contient les hyperparamètres et les cinq seeds; `results/training.csv` conserve `seed`, `step` et `episode_return`.
- Decision: comparer les runs par moyenne et écart-type.

## Semaine 4
- Date: 2026-08-17
- Claim: Le projet est vérifiable par une seule commande.
- Evidence: `run_all.sh` entraîne, évalue, génère les figures et vidéos puis vérifie les fichiers.
- Decision: considérer `run_all.sh` comme le point d'entrée officiel.
