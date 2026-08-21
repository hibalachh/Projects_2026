"""
RansomwareEnv - Environnement Gymnasium personnalise


Simulation d'un petit reseau de 5 PC attaque par un ransomware.
Le but : entrainer un agent (RL) qui isole/patch les PC pour limiter
la propagation, avec le moins d'actions possible.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces 

#les etats possibles d'un pc 
SAIN = 0 #pc normal pas touche
INFECTE = 1 #pc infexte peut contaminer les autres
ISOLE = 2 #pc coupe  du reseau par agent 
PATCHE = 3 # pc protege par agent

NB_ETATS = 4  # juste pour dire qu'il y a 4 etats possible 


class RansomwareEnv(gym.Env):
    """
    Environnement Gym : reseau de n_pc + un ransomware qui se propage 
    l'agent choisit une action a chaque step pour limiterles degats 
    """

    
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_pc: int = 5, render_mode: str = None):
        super().__init__()

        # Nombre de PC dans mon reseau (fixe a 5 pour ce projet)
        self.n_pc = n_pc
        self.render_mode = render_mode

        # tableau qui stocke l'etat de chaque pc rempli pour de vrai dans reset() 
        self.network_state = np.zeros(self.n_pc, dtype=np.int32)

        # Compteur d'atapes dans l'episode en cours 
        self.current_step = 0

        # limite d'etapes pour pas avoir un episode infini
        self.max_steps = 50

        
        # Observation Space
        
        # On utilise un espace de type Box (vecteur de nombres) 
        # Chaque case du reseau va de 0 (SAIN) a 3 (PATCHE)  le dernier
        # element du vecteur est le compteur d'etapes (0 a max_steps)
        # Taille du vecteur d'observation = n_pc (etats des PC) + 1 (compteur)
        low = np.zeros(self.n_pc + 1, dtype=np.float32)
        high = np.array(
            [NB_ETATS - 1] * self.n_pc + [self.max_steps], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        
        # Action Space
        
        
        # L'agent doit choisir une  action par etape appliquee a un PC cible
        
        # Astuce d'encodage : au lieu d'un espace a  deux dimensions (type
        # action, PC cible), on "aplatit" tout en un seul entier discret
        # C'est plus simple a gerer avec DQN (qui attend un Discrete unique).
        #
        # Encodage choisi :
        #   action = 0                         -> NE_RIEN_FAIRE (aucun PC cible)
        #   action = 1 .. n_pc                  -> ISOLER le PC (action-1)
        #   action = n_pc+1 .. 2*n_pc           -> PATCHER le PC (action-n_pc-1)
        #
        
        #
        # Nombre total d'actions possibles = 1 + n_pc (isoler) + n_pc (patcher)
        self.n_actions = 1 + 2 * self.n_pc
        self.action_space = spaces.Discrete(self.n_actions)

        # petit reccourc pour lisibilite dans step()
        self.ACTION_NE_RIEN_FAIRE = 0

    
    # reset()
    
    def reset(self, seed=None, options=None):
        """
        remet le reseau a zero au debut d'un episode 
        """
        # Obligatoire par l'API Gymnasium pour gere la seed correctement
        
        super().reset(seed=seed)

        # Tous les PC demarrent SAINS 
        self.network_state = np.full(self.n_pc, SAIN, dtype=np.int32)

        # sauf un PC choisi aleatoirement qui est le point d'entree
        
        patient_zero = self.np_random.integers(0, self.n_pc)
        self.network_state[patient_zero] = INFECTE

        # on remetle compteur de steps  a 0 
        self.current_step = 0

        # On construit l'observation a retourner a l'agent
        observation = self._get_observation()

        # info sert surtout au debug ici 
        info = {"patient_zero": int(patient_zero)}

        return observation, info

    def _get_observation(self):
        """
        Construit le vecteur d'observation etats des pc + step
        """
        # je colle les etats des pc avec le step actuel 
        obs = np.append(
            self.network_state.astype(np.float32),
            np.float32(self.current_step),
        )
        return obs

    
    #  step()
    
    def step(self, action):
        """
        Applique une action de l'agent fait avancer la simulation d'un pas
        et retourne le resultat au format standard Gymnasium

        Parametres :
            action (int) : entier entre 0 et n_actions-1 (voir encodage etape 5)

        Retourne :
            observation, reward, terminated, truncated, info
        """
        # 1. j'applique l'action choisie par l'agent
        self._appliquer_action(action)

        # 2. le ransomware essaie de se prpager 
        nb_nouvelles_infections = self._propager_ransomware()

        # 3. je calcule la recompense en fanction de ce qui s'est passe 
        reward = self._calculer_recompense(action, nb_nouvelles_infections)

        #  4.Avancer le compteur de temps
        self.current_step += 1

        #  5. Determiner si l'episode est termine 
        
        nb_infectes = np.sum(self.network_state == INFECTE)
        # Termine si plus aucun PC infecte (victoire) ou si tout est compromis
        terminated = bool(nb_infectes == 0) or bool(nb_infectes == self.n_pc)

        # "truncated" = fin artificielle car on a atteint la limite de temps
        truncated = bool(self.current_step >= self.max_steps)

        observation = self._get_observation()

        info = {
            "nb_infectes": int(nb_infectes),
            "nb_nouvelles_infections": int(nb_nouvelles_infections),
        }

        return observation, reward, terminated, truncated, info

    
    # Decodage et application de l'action 
    
    def _appliquer_action(self, action):
        """
        transforme le nemuro d'action en effet reel sur le reseau 
        """
        action = int(action)

        # Cas 1 : ne rien faire
        if action == self.ACTION_NE_RIEN_FAIRE:
            return

        # Cas 2 : isoler un PC (action 1 a n_pc)
        elif 1 <= action <= self.n_pc:
            pc_cible = action - 1
            # On ne peut isoler que si le PC est sain ou infecte 
            if self.network_state[pc_cible] in (SAIN, INFECTE):
                self.network_state[pc_cible] = ISOLE

        # Cas 3 : patcher un PC (action n_pc+1 a 2*n_pc)
        elif self.n_pc + 1 <= action <= 2 * self.n_pc:
            pc_cible = action - self.n_pc - 1
            # On ne peut patcher qu'un PC SAIN (on ne "guerit" pas un PC
            # deja infecté avec un simple patch, il faudrait le nettoyer
            # d'abord : simplification volontaire et realiste du modele)
            if self.network_state[pc_cible] == SAIN:
                self.network_state[pc_cible] = PATCHE

    
    #  Simulation de la propagation du ransomware
    
    def _propager_ransomware(self):
        """ 
        chaque pc infecte a une chance de contaminer chaque pc sain 
        reseau considerer complet ici (pas de topologie prescise  on simplifie pour rester simple a coder )
        """
        PROBA_INFECTION = 0.3  # probabilite qu'un PC infecte contamine un PC sain

        indices_infectes = np.where(self.network_state == INFECTE)[0]
        indices_sains = np.where(self.network_state == SAIN)[0]

        nb_nouvelles_infections = 0

        # Pour chaque PC sain, on regarde s'il se fait infecter par au
        
        for pc_sain in indices_sains:
            for _ in indices_infectes:
                if self.np_random.random() < PROBA_INFECTION:
                    self.network_state[pc_sain] = INFECTE
                    nb_nouvelles_infections += 1
                    break  # pas besoin de retester il est deje infecte

        return nb_nouvelles_infections

    
    # Calcul de la recompense
    
    def _calculer_recompense(self, action, nb_nouvelles_infections):
        """
        Ici je definis ce qui est "bien" ou "mal" pour l'agent :
        - eviter les nouvelles infections (priorite n 1)
        - eviter de spammer des actions inutiles ca coute cher en vrai)
        - bonus si reseau sain, malus si tout est infecte
        """
        reward = 0.0

        # 1. penalite pour chaque nouvelle infection 
        reward -= 5.0 * nb_nouvelles_infections

        # 2. Coût des actions pour eviter qu'il isole/patch tous le reseau
        if action != self.ACTION_NE_RIEN_FAIRE:
            reward -= 1.0

        # 3. et 4. Bonus/malus de fin d'episode
        nb_infectes = np.sum(self.network_state == INFECTE)
        if nb_infectes == 0:
            reward += 20.0  # ransomware totalement contenu : bonne nouvelle
        elif nb_infectes == self.n_pc:
            reward -= 20.0  # réseau entièrement compromis : échec total

        return float(reward)

    
    # Affichage simple pour debug
    
    def render(self):
        """Affiche l'etat du reseau dans la console """
        symboles = {SAIN: "🟢", INFECTE: "🔴", ISOLE: "⚪", PATCHE: "🔵"}
        etat_str = " ".join(symboles[etat] for etat in self.network_state)
        print(f"Step {self.current_step:2d} | {etat_str}")

    def close(self):
        """Rien a nettoyer ici mais Gymnasium demande cette methode"""
        pass
