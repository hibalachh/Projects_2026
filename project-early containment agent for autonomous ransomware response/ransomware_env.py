"""
RansomwareEnv - Environnement Gymnasium personnalisé
======================================================

Ce module simule un petit réseau d'entreprise de 5 PC attaqué par un
ransomware. Un agent (le "défenseur") observe l'état du réseau à chaque
étape (step) et choisit une action pour limiter la propagation, tout en
minimisant les dégâts (PC infectés) et le coût de ses actions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------
# Étape 3 : Variables représentant le réseau
# ---------------------------------------------------------------------
#
# On représente chaque PC par un entier (son "état") :
#   0 = SAIN      -> le PC fonctionne normalement, pas infecté
#   1 = INFECTE   -> le ransomware est actif sur ce PC et peut se propager
#   2 = ISOLE     -> le PC a été coupé du réseau par l'agent (ne peut plus
#                    infecter ni être infecté, mais est indisponible)
#   3 = PATCHE    -> le PC a été immunisé par l'agent (ne peut plus être
#                    infecté, reste utilisable)
#
# Ces constantes évitent d'utiliser des "nombres magiques" dans le code.

SAIN = 0
INFECTE = 1
ISOLE = 2
PATCHE = 3

NB_ETATS = 4  # nombre de valeurs possibles pour l'état d'un PC


class RansomwareEnv(gym.Env):
    """
    Environnement Gymnasium simulant un réseau de N_PC PC face à un
    ransomware. L'agent doit choisir, à chaque étape, quel PC isoler
    ou patcher (ou ne rien faire) pour limiter la propagation.
    """

    # Métadonnées standard Gymnasium (utile si on veut du rendu graphique plus tard)
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_pc: int = 5, render_mode: str = None):
        super().__init__()

        # Nombre de PC dans le réseau (fixé à 5 pour ce projet)
        self.n_pc = n_pc
        self.render_mode = render_mode

        # État interne du réseau : un tableau numpy de taille n_pc,
        # chaque case contenant SAIN, INFECTE, ISOLE ou PATCHE.
        # Initialisé ici, mais rempli réellement dans reset().
        self.network_state = np.zeros(self.n_pc, dtype=np.int32)

        # Compteur du nombre d'étapes écoulées dans l'épisode courant
        self.current_step = 0

        # Nombre maximal d'étapes avant la fin forcée d'un épisode
        # (évite les épisodes infinis si le ransomware est totalement contenu)
        self.max_steps = 50

        # -----------------------------------------------------------------
        # Étape 4 : Observation Space
        # -----------------------------------------------------------------
        #
        # L'agent observe :
        #   - l'état de chacun des n_pc PC (valeur entre 0 et 3, voir constantes)
        #   - le nombre d'étapes écoulées dans l'épisode (normalisé plus tard
        #     n'est pas obligatoire ici, on le laisse en entier brut)
        #
        # On utilise un espace de type Box (vecteur de nombres) plutôt que
        # MultiDiscrete pour rester simple et compatible avec Stable-Baselines3.
        # Chaque case du réseau va de 0 (SAIN) à 3 (PATCHE) ; le dernier
        # élément du vecteur est le compteur d'étapes (0 à max_steps).
        #
        # Taille du vecteur d'observation = n_pc (états des PC) + 1 (compteur)
        low = np.zeros(self.n_pc + 1, dtype=np.float32)
        high = np.array(
            [NB_ETATS - 1] * self.n_pc + [self.max_steps], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # -----------------------------------------------------------------
        # Étape 5 : Action Space
        # -----------------------------------------------------------------
        #
        # L'agent doit choisir UNE action par étape, appliquée à UN PC cible.
        # On a 3 types d'action possibles : ISOLER, PATCHER, NE_RIEN_FAIRE.
        #
        # Astuce d'encodage : au lieu d'un espace à deux dimensions (type
        # action, PC cible), on "aplatit" tout en un seul entier discret.
        # C'est plus simple à gérer avec DQN (qui attend un Discrete unique).
        #
        # Encodage choisi :
        #   action = 0                         -> NE_RIEN_FAIRE (aucun PC ciblé)
        #   action = 1 .. n_pc                  -> ISOLER le PC (action-1)
        #   action = n_pc+1 .. 2*n_pc           -> PATCHER le PC (action-n_pc-1)
        #
        # Exemple avec n_pc = 5 :
        #   0       -> ne rien faire
        #   1 à 5   -> isoler le PC 0 à 4
        #   6 à 10  -> patcher le PC 0 à 4
        #
        # Nombre total d'actions possibles = 1 + n_pc (isoler) + n_pc (patcher)
        self.n_actions = 1 + 2 * self.n_pc
        self.action_space = spaces.Discrete(self.n_actions)

        # Constantes pratiques pour décoder l'action dans step() plus tard
        self.ACTION_NE_RIEN_FAIRE = 0

    # -----------------------------------------------------------------
    # Étape 6 : reset()
    # -----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement au début d'un nouvel épisode.

        Retourne :
            observation (np.ndarray) : l'état initial observé par l'agent
            info (dict)              : informations additionnelles (vide ici)
        """
        # Obligatoire par l'API Gymnasium : gère la graine aléatoire (seed)
        # pour que les expériences soient reproductibles.
        super().reset(seed=seed)

        # Tous les PC démarrent SAINS...
        self.network_state = np.full(self.n_pc, SAIN, dtype=np.int32)

        # ...sauf un PC choisi aléatoirement, qui est le point d'entrée
        # du ransomware (patient zéro). C'est réaliste : l'attaque démarre
        # souvent via un seul poste compromis (phishing, etc.).
        patient_zero = self.np_random.integers(0, self.n_pc)
        self.network_state[patient_zero] = INFECTE

        # Réinitialisation du compteur d'étapes
        self.current_step = 0

        # On construit l'observation à retourner à l'agent
        observation = self._get_observation()

        # Le dict info est requis par l'API Gymnasium, même vide ou
        # avec des infos de debug (ici, on indique le patient zéro)
        info = {"patient_zero": int(patient_zero)}

        return observation, info

    def _get_observation(self):
        """
        Construit le vecteur d'observation à partir de l'état interne.
        Méthode utilitaire réutilisée par reset() et step().
        """
        # On concatène l'état des PC (n_pc valeurs) avec le compteur
        # d'étapes courant, le tout converti en float32 comme attendu
        # par observation_space (Box).
        obs = np.append(
            self.network_state.astype(np.float32),
            np.float32(self.current_step),
        )
        return obs

    # -----------------------------------------------------------------
    # Étape 7 : step()
    # -----------------------------------------------------------------
    def step(self, action):
        """
        Applique une action de l'agent, fait avancer la simulation d'un pas,
        et retourne le résultat au format standard Gymnasium.

        Paramètres :
            action (int) : entier entre 0 et n_actions-1 (voir encodage Étape 5)

        Retourne :
            observation, reward, terminated, truncated, info
        """
        # --- 1. Décoder et appliquer l'action de l'agent ---
        self._appliquer_action(action)

        # --- 2. Faire progresser le ransomware d'un pas (Étape 8) ---
        nb_nouvelles_infections = self._propager_ransomware()

        # --- 3. Calculer la récompense de cette étape (Étape 9) ---
        reward = self._calculer_recompense(action, nb_nouvelles_infections)

        # --- 4. Avancer le compteur de temps ---
        self.current_step += 1

        # --- 5. Déterminer si l'épisode est terminé ---
        # "terminated" = fin naturelle (ex : ransomware totalement contenu
        # OU totalement propagé à tout le réseau)
        nb_infectes = np.sum(self.network_state == INFECTE)
        # Terminé si plus aucun PC infecté (victoire) ou si tout est compromis
        terminated = bool(nb_infectes == 0) or bool(nb_infectes == self.n_pc)

        # "truncated" = fin artificielle car on a atteint la limite de temps
        truncated = bool(self.current_step >= self.max_steps)

        observation = self._get_observation()

        info = {
            "nb_infectes": int(nb_infectes),
            "nb_nouvelles_infections": int(nb_nouvelles_infections),
        }

        return observation, reward, terminated, truncated, info

    # -----------------------------------------------------------------
    # Décodage et application de l'action (partie de l'Étape 7)
    # -----------------------------------------------------------------
    def _appliquer_action(self, action):
        """
        Traduit l'entier "action" (0 à n_actions-1) en effet concret sur
        le réseau, selon l'encodage défini à l'Étape 5.
        """
        action = int(action)

        # Cas 1 : ne rien faire
        if action == self.ACTION_NE_RIEN_FAIRE:
            return

        # Cas 2 : isoler un PC (action 1 à n_pc)
        elif 1 <= action <= self.n_pc:
            pc_cible = action - 1
            # On ne peut isoler que si le PC n'est pas déjà isolé/patché
            if self.network_state[pc_cible] in (SAIN, INFECTE):
                self.network_state[pc_cible] = ISOLE

        # Cas 3 : patcher un PC (action n_pc+1 à 2*n_pc)
        elif self.n_pc + 1 <= action <= 2 * self.n_pc:
            pc_cible = action - self.n_pc - 1
            # On ne peut patcher qu'un PC SAIN (on ne "guérit" pas un PC
            # déjà infecté avec un simple patch, il faudrait le nettoyer
            # d'abord : simplification volontaire et réaliste du modèle)
            if self.network_state[pc_cible] == SAIN:
                self.network_state[pc_cible] = PATCHE

    # -----------------------------------------------------------------
    # Étape 8 : Simulation de la propagation du ransomware
    # -----------------------------------------------------------------
    def _propager_ransomware(self):
        """
        Règle de propagation simple : chaque PC infecté a une certaine
        probabilité d'infecter chacun de ses PC voisins SAINS à chaque
        étape (modèle inspiré des modèles épidémiologiques SIR, très
        utilisé en recherche pour simuler la diffusion d'un ransomware
        dans un réseau, sans manipuler de code malveillant réel).

        Ici, pour rester simple et pédagogique, on considère un réseau
        "complètement connecté" : un PC infecté peut contaminer
        n'importe quel autre PC sain du réseau (pas de topologie précise).

        Retourne :
            nb_nouvelles_infections (int)
        """
        PROBA_INFECTION = 0.3  # probabilité qu'un PC infecté contamine un PC sain

        indices_infectes = np.where(self.network_state == INFECTE)[0]
        indices_sains = np.where(self.network_state == SAIN)[0]

        nb_nouvelles_infections = 0

        # Pour chaque PC sain, on regarde s'il se fait infecter par au
        # moins un des PC infectés (tirage aléatoire indépendant par paire)
        for pc_sain in indices_sains:
            for _ in indices_infectes:
                if self.np_random.random() < PROBA_INFECTION:
                    self.network_state[pc_sain] = INFECTE
                    nb_nouvelles_infections += 1
                    break  # inutile de retester une fois infecté

        return nb_nouvelles_infections

    # -----------------------------------------------------------------
    # Étape 9 : Calcul de la récompense
    # -----------------------------------------------------------------
    def _calculer_recompense(self, action, nb_nouvelles_infections):
        """
        Fonction de récompense : c'est ELLE qui définit ce qu'est un
        "bon" Early-Containment Agent. On combine plusieurs signaux :

          1. Pénalité forte pour chaque nouvelle infection (l'agent doit
             agir vite, avant que le ransomware ne se propage)
          2. Pénalité légère pour chaque action (isoler/patcher a un coût
             réel en entreprise : PC indisponible, temps IT, etc.)
          3. Petit bonus si le réseau est totalement sain (victoire)
          4. Grosse pénalité si tout le réseau est infecté (défaite totale)
        """
        reward = 0.0

        # 1. Pénalité pour chaque nouvelle infection : encourage un
        #    containment PRÉCOCE, cœur du concept "Early-Containment"
        reward -= 5.0 * nb_nouvelles_infections

        # 2. Coût des actions (pousse l'agent à ne pas isoler/patcher
        #    "au hasard" tout le réseau par précaution excessive)
        if action != self.ACTION_NE_RIEN_FAIRE:
            reward -= 1.0

        # 3. et 4. Bonus/malus de fin d'épisode
        nb_infectes = np.sum(self.network_state == INFECTE)
        if nb_infectes == 0:
            reward += 20.0  # ransomware totalement contenu : bonne nouvelle
        elif nb_infectes == self.n_pc:
            reward -= 20.0  # réseau entièrement compromis : échec total

        return float(reward)

    # -----------------------------------------------------------------
    # Étape 10 (partie environnement) : rendu texte pour le débogage
    # -----------------------------------------------------------------
    def render(self):
        """Affiche l'état du réseau dans la console (mode texte simple)."""
        symboles = {SAIN: "🟢", INFECTE: "🔴", ISOLE: "⚪", PATCHE: "🔵"}
        etat_str = " ".join(symboles[etat] for etat in self.network_state)
        print(f"Step {self.current_step:2d} | {etat_str}")

    def close(self):
        """Nettoyage éventuel de ressources (rien à faire ici)."""
        pass
