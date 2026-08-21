# ============================================================
# train_dqn.py
# Entraînement de l'agent DQN
# ============================================================


# ------------------------------------------------------------
# IMPORTATIONS
# ------------------------------------------------------------

# NumPy permet de manipuler les tableaux numériques
import numpy as np

# PyTorch est utilisé pour sauvegarder le réseau de neurones
import torch

# Importation de notre environnement
from ransomware_env import RansomwareEnv

# Importation de notre agent DQN
from dqn import DQNAgent


# ============================================================
# 1. CRÉATION DE L'ENVIRONNEMENT
# ============================================================

# Création de l'environnement.
#
# Ici, on crée un environnement avec 5 PC.
#
# ATTENTION :
# Cette valeur doit correspondre à la configuration
# de ton ransomware_env.py.
env = RansomwareEnv(n_pc=5)


# ============================================================
# 2. DÉTERMINATION DE LA TAILLE DE L'ÉTAT ET DES ACTIONS
# ============================================================

# Nombre de valeurs qui représentent l'état.
#
# Exemple :
# Si nous avons 5 PC :
#
# PC1  PC2  PC3  PC4  PC5
#  ↓    ↓    ↓    ↓    ↓
# état = [0, 1, 0, 1, 0]
#
state_size = env.n_pc


# Nombre d'actions disponibles dans l'environnement.
#
# Par exemple, si l'environnement possède 11 actions :
#
# action = 0
# action = 1
# ...
# action = 10
#
action_size = env.action_space.n


# Affichage des informations
print("State size :", state_size)
print("Action size :", action_size)


# ============================================================
# 3. CRÉATION DE L'AGENT DQN
# ============================================================

# Création de l'agent DQN.
#
# state_size :
# nombre de valeurs dans l'état.
#
# action_size :
# nombre d'actions possibles.
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size
)


# Afficher si l'agent utilise le CPU ou le GPU
print("Device :", agent.device)


# ============================================================
# 4. PARAMÈTRES DE L'ENTRAÎNEMENT
# ============================================================

# Nombre total d'épisodes.
#
# Un épisode correspond à une partie complète
# entre reset() et la fin de l'environnement.
n_episodes = 5000


# Tous les combien d'épisodes
# nous mettons à jour le Target Network.
#
# Ici :
# tous les 10 épisodes.
target_update_frequency = 10


# ============================================================
# 5. BOUCLE PRINCIPALE D'ENTRAÎNEMENT
# ============================================================

# On répète l'entraînement pendant 5000 épisodes.
for episode in range(n_episodes):


    # --------------------------------------------------------
    # RÉINITIALISATION DE L'ENVIRONNEMENT
    # --------------------------------------------------------

    # reset() remet l'environnement à son état initial.
    #
    # observation :
    # nouvel état de l'environnement.
    #
    # info :
    # informations supplémentaires.
    observation, info = env.reset()


    # --------------------------------------------------------
    # EXTRACTION DE L'ÉTAT
    # --------------------------------------------------------

    # On récupère l'état utile pour le réseau DQN.
    #
    # Ici, on enlève la dernière valeur de observation.
    #
    # IMPORTANT :
    # Cette ligne dépend de la structure exacte
    # de ton ransomware_env.py.
    state = observation[:-1].astype(np.float32)


    # --------------------------------------------------------
    # VARIABLES DE L'ÉPISODE
    # --------------------------------------------------------

    # Indique si l'épisode est terminé
    # naturellement.
    terminated = False


    # Indique si l'épisode est arrêté
    # pour une autre raison, par exemple une limite.
    truncated = False


    # Stocke la récompense totale de l'épisode.
    total_reward = 0


    # Liste permettant de stocker les différentes losses.
    losses = []


    # ========================================================
    # 6. BOUCLE DE L'ÉPISODE
    # ========================================================

    # Tant que l'épisode n'est pas terminé,
    # l'agent continue à agir.
    while not (terminated or truncated):


        # ----------------------------------------------------
        # CHOIX D'UNE ACTION
        # ----------------------------------------------------

        # L'agent choisit une action.
        #
        # Le choix est fait avec epsilon-greedy :
        #
        # parfois → action aléatoire (exploration)
        #
        # parfois → meilleure action connue (exploitation)
        action = agent.choose_action(state)


        # ----------------------------------------------------
        # EXÉCUTION DE L'ACTION
        # ----------------------------------------------------

        # On envoie l'action à l'environnement.
        #
        # L'environnement retourne :
        #
        # next_observation :
        # nouvel état
        #
        # reward :
        # récompense obtenue
        #
        # terminated :
        # épisode terminé normalement
        #
        # truncated :
        # épisode arrêté pour une autre raison
        #
        # info :
        # informations supplémentaires
        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)


        # ----------------------------------------------------
        # EXTRACTION DU NOUVEL ÉTAT
        # ----------------------------------------------------

        # On récupère le nouvel état.
        #
        # Comme précédemment, on enlève la dernière valeur.
        next_state = next_observation[:-1].astype(
            np.float32
        )


        # ----------------------------------------------------
        # VÉRIFIER SI L'ÉPISODE EST TERMINÉ
        # ----------------------------------------------------

        # done vaut True si :
        #
        # terminated = True
        #
        # OU
        #
        # truncated = True
        done = terminated or truncated


        # ----------------------------------------------------
        # STOCKAGE DE L'EXPÉRIENCE
        # ----------------------------------------------------

        # On sauvegarde l'expérience dans le Replay Buffer.
        #
        # Une expérience contient :
        #
        # état actuel
        # action
        # récompense
        # nouvel état
        # épisode terminé ou non
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )


        # ----------------------------------------------------
        # ENTRAÎNEMENT DU RÉSEAU
        # ----------------------------------------------------

        # L'agent prend un batch d'expériences
        # dans le Replay Buffer et entraîne
        # le Policy Network.
        loss = agent.train_step()


        # Si une étape d'entraînement a réellement
        # été effectuée, on sauvegarde la loss.
        #
        # Au début, le Replay Buffer n'a peut-être
        # pas encore assez d'expériences.
        if loss is not None:
            losses.append(loss)


        # ----------------------------------------------------
        # PASSAGE AU NOUVEL ÉTAT
        # ----------------------------------------------------

        # Le nouvel état devient maintenant
        # l'état actuel.
        state = next_state


        # Ajouter la récompense à la récompense totale.
        total_reward += reward


    # ========================================================
    # 7. DIMINUTION D'EPSILON
    # ========================================================

    # Après chaque épisode, epsilon diminue.
    #
    # Au début :
    # epsilon est élevé → beaucoup d'exploration.
    #
    # Plus tard :
    # epsilon devient plus petit → plus d'exploitation.
    agent.update_epsilon()


    # ========================================================
    # 8. MISE À JOUR DU TARGET NETWORK
    # ========================================================

    # Tous les 10 épisodes, on copie
    # les poids du Policy Network
    # vers le Target Network.
    if (episode + 1) % target_update_frequency == 0:

        agent.update_target_network()


    # ========================================================
    # 9. AFFICHAGE DES RÉSULTATS
    # ========================================================

    # On affiche les résultats tous les 100 épisodes.
    if (episode + 1) % 100 == 0:


        # Si nous avons calculé des losses,
        # on calcule leur moyenne.
        if len(losses) > 0:

            average_loss = np.mean(losses)

        else:

            # Si aucune loss n'a encore été calculée,
            # on met la valeur à 0.
            average_loss = 0


        # Affichage des informations d'entraînement.
        print(
            f"Episode {episode + 1}/{n_episodes} | "
            f"Reward = {total_reward:.2f} | "
            f"Loss = {average_loss:.4f} | "
            f"Epsilon = {agent.epsilon:.3f}"
        )


# ============================================================
# 10. SAUVEGARDE DU MODÈLE
# ============================================================

# On sauvegarde les poids appris
# par le Policy Network.
#
# Le fichier sera créé dans le même dossier :
#
# dqn_ransomware.pth
torch.save(
    agent.policy_net.state_dict(),
    "dqn_ransomware.pth"
)


# ============================================================
# 11. FERMETURE DE L'ENVIRONNEMENT
# ============================================================

# On ferme proprement l'environnement.
env.close()


# ============================================================
# 12. MESSAGE FINAL
# ============================================================

print()
print("======================================")
print("Entraînement terminé !")
print("Modèle sauvegardé : dqn_ransomware.pth")
print("======================================")