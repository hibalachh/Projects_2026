# ============================================================
# dqn_agent.py
# Agent DQN pour l'environnement RansomwareEnv
# ============================================================


# ------------------------------------------------------------
# IMPORTATIONS
# ------------------------------------------------------------

# random permet de faire des choix aléatoires
# notamment pour l'exploration avec epsilon-greedy
import random


# deque est une structure de file très pratique
# pour stocker les expériences du Replay Buffer
from collections import deque


# NumPy permet de manipuler les tableaux numériques
import numpy as np


# PyTorch est utilisé pour créer et entraîner
# le réseau de neurones du DQN
import torch


# nn contient les composants nécessaires
# pour construire un réseau de neurones
import torch.nn as nn


# optim contient les algorithmes d'optimisation
# comme Adam
import torch.optim as optim


# ============================================================
# 1. DQN : RÉSEAU DE NEURONES
# ============================================================

class DQN(nn.Module):
    """
    Réseau de neurones utilisé par le DQN.

    Entrée :
        état de l'environnement

    Sortie :
        une Q-value pour chaque action possible
    """

    def __init__(self, state_size, action_size):

        # Initialise la classe nn.Module
        super().__init__()


        # ----------------------------------------------------
        # Création du réseau de neurones
        # ----------------------------------------------------

        self.network = nn.Sequential(

            # Première couche :
            # state_size valeurs en entrée
            # 64 neurones en sortie
            nn.Linear(state_size, 64),

            # Fonction d'activation ReLU
            nn.ReLU(),


            # Deuxième couche :
            # 64 neurones → 64 neurones
            nn.Linear(64, 64),

            # Deuxième ReLU
            nn.ReLU(),


            # Couche finale :
            # 64 neurones → nombre d'actions
            #
            # Dans notre environnement :
            # action_size = 11
            #
            # Donc :
            # 64 → 11
            #
            # Les 11 sorties représentent les
            # 11 Q-values des actions.
            nn.Linear(64, action_size)
        )


    def forward(self, state):
        """
        Cette fonction définit comment les données
        traversent le réseau.
        """

        # On donne l'état au réseau
        # et on récupère les Q-values
        return self.network(state)


# ============================================================
# 2. REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """
    Mémoire de l'agent.

    Elle permet de sauvegarder les expériences
    de l'agent pour les réutiliser pendant l'entraînement.
    """

    def __init__(self, capacity=10000):

        # deque permet de conserver les expériences
        #
        # maxlen=10000 signifie :
        # maximum 10 000 expériences.
        #
        # Lorsque le buffer est plein,
        # les expériences les plus anciennes
        # sont supprimées automatiquement.
        self.buffer = deque(maxlen=capacity)


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Ajoute une expérience dans la mémoire.

        Une expérience contient :

        state      → état actuel
        action     → action choisie
        reward     → récompense reçue
        next_state → nouvel état
        done       → épisode terminé ou non
        """

        # On ajoute l'expérience dans le buffer
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def sample(self, batch_size):
        """
        Prend un échantillon aléatoire d'expériences.

        Exemple :

        Buffer = 10 000 expériences

        batch_size = 64

        On prend 64 expériences aléatoires
        pour entraîner le réseau.
        """

        # Sélection aléatoire de plusieurs expériences
        batch = random.sample(
            self.buffer,
            batch_size
        )


        # On sépare les différents éléments
        # des expériences
        states, actions, rewards, next_states, dones = zip(*batch)


        # On transforme les données en tableaux NumPy
        # avec les types adaptés à PyTorch
        return (

            # États
            np.array(
                states,
                dtype=np.float32
            ),

            # Actions
            np.array(
                actions,
                dtype=np.int64
            ),

            # Rewards
            np.array(
                rewards,
                dtype=np.float32
            ),

            # Nouveaux états
            np.array(
                next_states,
                dtype=np.float32
            ),

            # États terminaux
            np.array(
                dones,
                dtype=np.float32
            )
        )


    def __len__(self):
        """
        Retourne le nombre d'expériences
        actuellement présentes dans le buffer.
        """

        return len(self.buffer)


# ============================================================
# 3. DQN AGENT
# ============================================================

class DQNAgent:
    """
    Agent DQN.

    Il contient :

    - Policy Network
    - Target Network
    - Replay Buffer
    - Optimizer
    - Epsilon-greedy
    - paramètres d'apprentissage
    """

    def __init__(
        self,
        state_size,
        action_size
    ):

        # Sauvegarde du nombre de valeurs
        # représentant l'état
        self.state_size = state_size


        # Sauvegarde du nombre d'actions possibles
        self.action_size = action_size


        # ----------------------------------------------------
        # HYPERPARAMÈTRES
        # ----------------------------------------------------

        # Gamma :
        # importance donnée aux récompenses futures.
        #
        # gamma proche de 1 :
        # l'agent regarde beaucoup le futur.
        self.gamma = 0.99


        # Learning rate :
        # vitesse d'apprentissage du réseau.
        self.learning_rate = 0.001


        # Nombre d'expériences utilisées
        # pour une étape d'entraînement.
        self.batch_size = 64


        # ----------------------------------------------------
        # EPSILON-GREEDY
        # ----------------------------------------------------

        # Au début :
        # epsilon = 1
        #
        # L'agent explore beaucoup.
        self.epsilon = 1.0


        # Valeur minimale d'epsilon.
        #
        # Même après beaucoup d'épisodes,
        # on garde un peu d'exploration.
        self.epsilon_min = 0.05


        # Permet de diminuer progressivement epsilon.
        self.epsilon_decay = 0.995


        # ----------------------------------------------------
        # CHOIX DU DEVICE
        # ----------------------------------------------------

        # Si un GPU CUDA est disponible,
        # on utilise le GPU.
        #
        # Sinon on utilise le CPU.
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        # ----------------------------------------------------
        # POLICY NETWORK
        # ----------------------------------------------------

        # Création du réseau principal.
        #
        # C'est ce réseau qui apprend.
        self.policy_net = DQN(
            state_size,
            action_size
        ).to(self.device)


        # ----------------------------------------------------
        # TARGET NETWORK
        # ----------------------------------------------------

        # Création du deuxième réseau.
        #
        # Il sert à calculer les valeurs cibles
        # pendant l'apprentissage.
        self.target_net = DQN(
            state_size,
            action_size
        ).to(self.device)


        # Au début, les deux réseaux sont identiques.
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )


        # Le Target Network n'est pas entraîné
        # directement à chaque étape.
        self.target_net.eval()


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        # Adam est utilisé pour modifier
        # les poids du réseau.
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )


        # ----------------------------------------------------
        # LOSS FUNCTION
        # ----------------------------------------------------

        # SmoothL1Loss mesure la différence
        # entre :
        #
        # Q-value prédite
        #
        # et
        #
        # Q-value cible
        self.loss_fn = nn.SmoothL1Loss()


        # ----------------------------------------------------
        # REPLAY BUFFER
        # ----------------------------------------------------

        # Création de la mémoire de l'agent.
        self.memory = ReplayBuffer(
            capacity=10000
        )


    # ========================================================
    # 4. CHOISIR UNE ACTION
    # ========================================================

    def choose_action(self, state):
        """
        Choisit une action avec epsilon-greedy.
        """

        # ----------------------------------------------------
        # EXPLORATION
        # ----------------------------------------------------

        # Génère un nombre aléatoire entre 0 et 1.
        #
        # Si ce nombre est inférieur à epsilon,
        # on choisit une action aléatoire.
        if random.random() < self.epsilon:

            # Choix aléatoire parmi toutes les actions
            return random.randrange(
                self.action_size
            )


        # ----------------------------------------------------
        # EXPLOITATION
        # ----------------------------------------------------

        # Transformation de l'état NumPy
        # en Tensor PyTorch.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )


        # Ajoute une dimension pour représenter
        # un seul élément dans un batch.
        #
        # Exemple :
        #
        # [0, 0, 1, 0, 0]
        #
        # devient :
        #
        # [[0, 0, 1, 0, 0]]
        state_tensor = state_tensor.unsqueeze(0)


        # On ne veut pas calculer les gradients
        # simplement pour choisir une action.
        with torch.no_grad():

            # Le réseau prédit les Q-values
            # de toutes les actions.
            q_values = self.policy_net(
                state_tensor
            )


        # On choisit l'action avec
        # la Q-value la plus élevée.
        #
        # Exemple :
        #
        # Q = [1.2, 0.5, -1, 7.4, 0.2]
        #
        # argmax = action 3
        return q_values.argmax(
            dim=1
        ).item()


    # ========================================================
    # 5. MÉMORISER UNE EXPÉRIENCE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Ajoute une expérience dans Replay Buffer.
        """

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )


    # ========================================================
    # 6. ENTRAÎNEMENT DU DQN
    # ========================================================

    def train_step(self):
        """
        Effectue une étape d'apprentissage.
        """

        # ----------------------------------------------------
        # Vérifier qu'on possède assez d'expériences
        # ----------------------------------------------------

        # Si on n'a pas encore 64 expériences,
        # on ne peut pas créer un batch de 64.
        if len(self.memory) < self.batch_size:

            return None


        # ----------------------------------------------------
        # PRENDRE UN BATCH ALÉATOIRE
        # ----------------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )


        # ----------------------------------------------------
        # TRANSFORMATION EN TENSORS
        # ----------------------------------------------------

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )


        actions = torch.tensor(
            actions,
            dtype=torch.int64,
            device=self.device
        )


        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )


        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )


        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )


        # ====================================================
        # CALCUL DE Q(s,a)
        # ====================================================

        # Le Policy Network calcule les Q-values
        # pour toutes les actions.
        #
        # Exemple :
        #
        # Q = [1.2, 0.5, 7.4, ...]
        current_q_values = self.policy_net(
            states
        )


        # On garde uniquement la Q-value
        # correspondant à l'action réellement choisie.
        #
        # C'est Q(s,a).
        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)


        # ====================================================
        # CALCUL DE LA MEILLEURE Q-VALUE FUTURE
        # ====================================================

        # On ne veut pas modifier le Target Network
        # pendant ce calcul.
        with torch.no_grad():

            # Le Target Network prédit les Q-values
            # du prochain état.
            next_q_values = self.target_net(
                next_states
            )


            # On garde la meilleure Q-value.
            #
            # max Q(s',a')
            next_q_values = next_q_values.max(
                dim=1
            )[0]


        # ====================================================
        # CALCUL DE LA TARGET
        # ====================================================

        # Formule DQN :
        #
        # Target =
        # reward + gamma * max Q(s',a')
        #
        # Si done = 1 :
        # l'épisode est terminé,
        # donc on ne regarde pas le futur.
        target_q_values = rewards + (
            self.gamma
            * next_q_values
            * (1 - dones)
        )


        # ====================================================
        # CALCUL DE LA LOSS
        # ====================================================

        # Compare :
        #
        # Q-value prédite
        #
        # avec
        #
        # Q-value cible.
        loss = self.loss_fn(
            current_q_values,
            target_q_values
        )


        # ====================================================
        # BACKPROPAGATION
        # ====================================================

        # Supprime les anciens gradients.
        self.optimizer.zero_grad()


        # Calcule les nouveaux gradients.
        loss.backward()


        # Met à jour les poids du Policy Network.
        self.optimizer.step()


        # Retourne la valeur de la loss
        # pour pouvoir l'afficher pendant l'entraînement.
        return loss.item()


    # ========================================================
    # 7. DIMINUER EPSILON
    # ========================================================

    def update_epsilon(self):
        """
        Diminue progressivement l'exploration.
        """

        # Exemple :
        #
        # 1.00 → 0.995 → 0.990 → ...
        #
        # Mais jamais en dessous de epsilon_min.
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


    # ========================================================
    # 8. METTRE À JOUR TARGET NETWORK
    # ========================================================

    def update_target_network(self):
        """
        Copie les poids du Policy Network
        vers le Target Network.
        """

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )