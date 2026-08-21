import numpy as np
import matplotlib.pyplot as plt

from ransomware_env import RansomwareEnv


# ============================================================
# 1. Création de l'environnement
# ============================================================

env = RansomwareEnv(n_pc=5)

# 5 PC et 4 états possibles par PC
# => 4^5 = 1024 configurations possibles


# ============================================================
# 2. Paramètres du Q-Learning
# ============================================================

episodes = 5000

alpha = 0.1       # Learning rate
gamma = 0.95      # Importance des récompenses futures

epsilon = 1.0     # Exploration au début
epsilon_min = 0.05
epsilon_decay = 0.995


# ============================================================
# 3. Nombre d'états et d'actions
# ============================================================

n_network_states = 4 ** env.n_pc

n_states = n_network_states * (env.max_steps + 1)

n_actions = env.action_space.n

print("Nombre d'états :", n_states)
print("Nombre d'actions :", n_actions)


# ============================================================
# 4. Création de la Q-table
# ============================================================

# Chaque ligne = un état
# Chaque colonne = une action

Q = np.zeros((n_states, n_actions))


# ============================================================
# 5. Conversion observation -> état
# ============================================================

def get_state(observation):

    # États des PC
    pcs = observation[:env.n_pc].astype(int)

    # Étape actuelle
    step = int(observation[-1])

    # Transformer les états des PC en un seul nombre
    network_index = 0

    for pc in pcs:
        network_index = network_index * 4 + pc

    # Ajouter l'information sur l'étape
    state = network_index * (env.max_steps + 1) + step

    return state


# ============================================================
# 6. Choix de l'action : Epsilon-Greedy
# ============================================================

def choose_action(state, epsilon):

    # Exploration : action aléatoire
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation : meilleure action connue
    return np.argmax(Q[state])


# ============================================================
# 7. Entraînement
# ============================================================

rewards_history = []

for episode in range(episodes):

    # Réinitialisation de l'environnement
    observation, info = env.reset()

    state = get_state(observation)

    total_reward = 0

    terminated = False
    truncated = False


    while not terminated and not truncated:

        # Choisir une action
        action = choose_action(state, epsilon)

        # Exécuter l'action
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Obtenir le nouvel état
        next_state = get_state(next_observation)


        # ====================================================
        # Mise à jour de la Q-table
        # ====================================================

        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state, action]
        )


        # Passer au nouvel état
        state = next_state

        total_reward += reward


    # Sauvegarder la récompense de l'épisode
    rewards_history.append(total_reward)


    # ========================================================
    # Diminution de l'exploration
    # ========================================================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    # Affichage de la progression
    if (episode + 1) % 500 == 0:

        mean_reward = np.mean(rewards_history[-500:])

        print(
            f"Episode {episode + 1} | "
            f"Reward moyen : {mean_reward:.2f} | "
            f"Epsilon : {epsilon:.3f}"
        )


# ============================================================
# 8. Fin de l'entraînement
# ============================================================

print("\nEntraînement terminé.")

# Sauvegarder la Q-table
np.save("q_table.npy", Q)

print("Q-table sauvegardée.")


# ============================================================
# 9. Courbe d'apprentissage
# ============================================================

window = 100

moving_average = np.convolve(
    rewards_history,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(moving_average)

plt.xlabel("Episodes")
plt.ylabel("Reward moyen")

plt.title("Apprentissage du Q-Learning")

plt.grid()

plt.show()


# ============================================================
# 10. Test de l'agent
# ============================================================

print("\n===== TEST DE L'AGENT =====")

observation, info = env.reset()

state = get_state(observation)

total_reward = 0

terminated = False
truncated = False


while not terminated and not truncated:

    # Pendant le test, on exploite uniquement
    # ce que l'agent a appris
    action = choose_action(state, 0)

    observation, reward, terminated, truncated, info = env.step(action)

    state = get_state(observation)

    total_reward += reward

    print(
        f"Action : {action} | "
        f"Reward : {reward:.2f} | "
        f"PC infectés : {info['nb_infectes']}"
    )


print("\nReward total :", total_reward)

env.close()