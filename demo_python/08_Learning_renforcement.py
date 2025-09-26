"""
APPRENTISSAGE PAR RENFORCEMENT - Q-LEARNING
==========================================

Exemple complet avec un environnement GridWorld simple où un agent
doit apprendre à naviguer vers un trésor en évitant des obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# =========================================
# PARTIE 1 : ENVIRONNEMENT GRIDWORLD
# =========================================

class GridWorld:
    def __init__(self, size=5):
        """
        Crée un environnement grille 5x5 avec :
        - Agent (A) : position de départ
        - Trésor (T) : objectif (+10 points)
        - Obstacles (X) : pénalité (-5 points)
        - Cases vides (.) : pénalité légère (-0.1 points)
        """
        self.size = size
        self.reset()
        
        # Définir les récompenses
        self.rewards = {
            'treasure': 10,      # Trésor trouvé
            'obstacle': -5,      # Obstacle touché
            'empty': -0.1,       # Case vide (pour encourager la rapidité)
            'wall': -1           # Essayer de sortir de la grille
        }
        
        # Actions possibles : 0=haut, 1=droite, 2=bas, 3=gauche
        self.actions = ['↑', '→', '↓', '←']
        self.action_effects = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def reset(self):
        """Remet l'environnement à zéro"""
        # Créer la grille
        self.grid = np.full((self.size, self.size), '.', dtype=str)
        
        # Position de départ de l'agent (coin haut-gauche)
        self.agent_pos = [0, 0]
        self.start_pos = [0, 0]
        
        # Position du trésor (coin bas-droite)
        self.treasure_pos = [self.size-1, self.size-1]
        self.grid[self.treasure_pos[0], self.treasure_pos[1]] = 'T'
        
        # Placer quelques obstacles aléatoirement
        num_obstacles = 3
        for _ in range(num_obstacles):
            while True:
                row, col = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if [row, col] not in [self.start_pos, self.treasure_pos]:
                    self.grid[row, col] = 'X'
                    break
        
        return self.get_state()
    
    def get_state(self):
        """Retourne l'état actuel (position de l'agent)"""
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        Exécute une action et retourne (nouvel_état, récompense, terminé)
        """
        # Calculer la nouvelle position
        dr, dc = self.action_effects[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        # Vérifier les limites de la grille
        if new_row < 0 or new_row >= self.size or new_col < 0 or new_col >= self.size:
            # Sortie de grille : pénalité mais pas de déplacement
            return self.get_state(), self.rewards['wall'], False
        
        # Déplacer l'agent
        self.agent_pos = [new_row, new_col]
        current_cell = self.grid[new_row, new_col]
        
        # Calculer la récompense et vérifier si terminé
        if current_cell == 'T':
            # Trésor trouvé !
            reward = self.rewards['treasure']
            done = True
        elif current_cell == 'X':
            # Obstacle touché
            reward = self.rewards['obstacle']
            done = True  # Episode terminé (échec)
        else:
            # Case vide
            reward = self.rewards['empty']
            done = False
        
        return self.get_state(), reward, done
    
    def render(self):
        """Affiche l'état actuel de la grille"""
        display_grid = self.grid.copy()
        agent_row, agent_col = self.agent_pos
        display_grid[agent_row, agent_col] = 'A'
        
        print("\n" + "="*20)
        for row in display_grid:
            print(" ".join(row))
        print("="*20)
        print("A=Agent, T=Trésor, X=Obstacle, .=Vide")

# =========================================
# PARTIE 2 : AGENT Q-LEARNING
# =========================================

class QLearningAgent:
    def __init__(self, num_actions=4, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Agent Q-Learning
        
        Args:
            num_actions: Nombre d'actions possibles
            learning_rate (α): Taux d'apprentissage [0,1]
            discount_factor (γ): Facteur d'actualisation [0,1] 
            epsilon (ε): Probabilité d'exploration [0,1]
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Table Q : Q(état, action) -> valeur attendue
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Statistiques pour le suivi
        self.episode_rewards = []
        self.episode_steps = []
    
    def choose_action(self, state):
        """
        Choisit une action selon la politique ε-greedy
        
        ε-greedy : avec probabilité ε, explorer (action aléatoire)
                   sinon, exploiter (meilleure action connue)
        """
        if random.random() < self.epsilon:
            # Exploration : action aléatoire
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation : meilleure action selon Q-table
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Met à jour la valeur Q selon l'équation de Bellman :
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        où :
        - α = learning_rate
        - γ = discount_factor  
        - r = reward
        - s = state, a = action
        - s' = next_state
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Équation de Bellman
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Réduit progressivement l'exploration"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

# =========================================
# PARTIE 3 : ENTRAÎNEMENT
# =========================================

def train_agent(env, agent, num_episodes=1000, verbose=True):
    """
    Entraîne l'agent dans l'environnement
    """
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT ({num_episodes} épisodes)")
    print("="*50)
    
    all_rewards = []
    all_steps = []
    
    for episode in range(num_episodes):
        # Réinitialiser l'environnement
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:  # Limite pour éviter les boucles infinies
            # Choisir une action
            action = agent.choose_action(state)
            
            # Exécuter l'action
            next_state, reward, done = env.step(action)
            
            # Mettre à jour la Q-table
            agent.update_q_value(state, action, reward, next_state)
            
            # Mise à jour pour la prochaine itération
            state = next_state
            total_reward += reward
            steps += 1
        
        # Enregistrer les statistiques
        all_rewards.append(total_reward)
        all_steps.append(steps)
        
        # Réduire l'exploration progressivement
        agent.decay_epsilon()
        
        # Affichage périodique
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_steps = np.mean(all_steps[-100:])
            print(f"Épisode {episode+1:4d}: Récompense moy.={avg_reward:6.2f}, "
                  f"Étapes moy.={avg_steps:5.1f}, ε={agent.epsilon:.3f}")
    
    return all_rewards, all_steps

# =========================================
# PARTIE 4 : TEST DE L'AGENT ENTRAÎNÉ
# =========================================

def test_agent(env, agent, num_tests=5):
    """
    Teste l'agent entraîné (pas d'apprentissage, pas d'exploration)
    """
    print(f"\n🎯 TEST DE L'AGENT ENTRAÎNÉ ({num_tests} tests)")
    print("="*50)
    
    # Sauvegarder l'epsilon original
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Pas d'exploration pendant le test
    
    successes = 0
    
    for test in range(num_tests):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        path = [state]
        
        print(f"\n--- Test {test+1} ---")
        env.render()
        
        while not done and steps < 20:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            path.append(next_state)
            total_reward += reward
            steps += 1
            state = next_state
            
            print(f"Étape {steps}: Action={env.actions[action]} → "
                  f"Position={next_state}, Récompense={reward:.1f}")
        
        # Résultat du test
        if env.agent_pos == env.treasure_pos:
            print(f"🎉 SUCCÈS ! Trésor trouvé en {steps} étapes (Récompense: {total_reward:.1f})")
            successes += 1
        else:
            print(f"❌ Échec. Position finale: {env.agent_pos} (Récompense: {total_reward:.1f})")
        
        env.render()
    
    # Restaurer epsilon
    agent.epsilon = original_epsilon
    
    print(f"\n📊 RÉSULTATS: {successes}/{num_tests} succès ({100*successes/num_tests:.1f}%)")

# =========================================
# PARTIE 5 : VISUALISATION DES RÉSULTATS
# =========================================

def plot_training_results(rewards, steps):
    """
    Graphiques des résultats d'entraînement
    """
    # Calculer les moyennes mobiles
    window = 50
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        steps_smooth = np.convolve(steps, np.ones(window)/window, mode='valid')
        x_smooth = range(window-1, len(rewards))
    else:
        rewards_smooth = rewards
        steps_smooth = steps
        x_smooth = range(len(rewards))
    
    # Créer les graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique des récompenses
    ax1.plot(rewards, alpha=0.3, color='blue', label='Récompenses brutes')
    if len(rewards_smooth) > 1:
        ax1.plot(x_smooth, rewards_smooth, color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    ax1.set_xlabel('Épisodes')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Évolution des récompenses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique du nombre d'étapes
    ax2.plot(steps, alpha=0.3, color='green', label='Étapes brutes')
    if len(steps_smooth) > 1:
        ax2.plot(x_smooth, steps_smooth, color='orange', linewidth=2, label=f'Moyenne mobile ({window})')
    ax2.set_xlabel('Épisodes')
    ax2.set_ylabel('Nombre d\'étapes')
    ax2.set_title('Évolution du nombre d\'étapes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_q_table(agent, env):
    """
    Visualise la politique apprise (meilleure action par état)
    """
    print(f"\n🧠 POLITIQUE APPRISE (meilleure action par état)")
    print("="*50)
    
    policy_grid = np.full((env.size, env.size), '?', dtype=str)
    
    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)
            if state in agent.q_table:
                best_action = np.argmax(agent.q_table[state])
                policy_grid[row, col] = env.actions[best_action]
    
    # Marquer les positions spéciales
    policy_grid[env.start_pos[0], env.start_pos[1]] = 'S'  # Start
    policy_grid[env.treasure_pos[0], env.treasure_pos[1]] = 'T'  # Treasure
    
    # Marquer les obstacles
    for row in range(env.size):
        for col in range(env.size):
            if env.grid[row, col] == 'X':
                policy_grid[row, col] = 'X'
    
    print("Politique (meilleure action par case):")
    for row in policy_grid:
        print(" ".join(f"{cell:>2}" for cell in row))
    print("\nS=Start, T=Treasure, X=Obstacle, ↑→↓←=Directions recommandées")

# =========================================
# PARTIE 6 : DÉMONSTRATION COMPLÈTE
# =========================================

def main_demonstration():
    """
    Démonstration complète de l'apprentissage par renforcement
    """
    print("🤖 APPRENTISSAGE PAR RENFORCEMENT - Q-LEARNING")
    print("=" * 60)
    
    # 1. Créer l'environnement
    print("\n1️⃣ CRÉATION DE L'ENVIRONNEMENT")
    env = GridWorld(size=5)
    env.render()
    
    # 2. Créer l'agent
    print("\n2️⃣ CRÉATION DE L'AGENT Q-LEARNING")
    agent = QLearningAgent(
        num_actions=4,
        learning_rate=0.1,    # α : vitesse d'apprentissage
        discount_factor=0.9,  # γ : importance du futur
        epsilon=0.3           # ε : taux d'exploration initial
    )
    
    print(f"Agent créé avec :")
    print(f"  - Taux d'apprentissage (α): {agent.learning_rate}")
    print(f"  - Facteur d'actualisation (γ): {agent.discount_factor}")
    print(f"  - Taux d'exploration initial (ε): {agent.epsilon}")
    
    # 3. Entraînement
    print("\n3️⃣ ENTRAÎNEMENT")
    rewards, steps = train_agent(env, agent, num_episodes=500, verbose=True)
    
    # 4. Visualisation des résultats
    print("\n4️⃣ VISUALISATION DES RÉSULTATS")
    try:
        plot_training_results(rewards, steps)
    except:
        print("Matplotlib non disponible pour les graphiques")
    
    # 5. Test de l'agent entraîné
    print("\n5️⃣ TEST DE L'AGENT ENTRAÎNÉ")
    test_agent(env, agent, num_tests=3)
    
    # 6. Visualisation de la politique
    print("\n6️⃣ POLITIQUE APPRISE")
    visualize_q_table(agent, env)
    
    # 7. Statistiques finales
    print(f"\n📈 STATISTIQUES FINALES")
    print("="*30)
    print(f"États explorés: {len(agent.q_table)}")
    print(f"Récompense moyenne (100 derniers épisodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Étapes moyennes (100 derniers épisodes): {np.mean(steps[-100:]):.1f}")
    print(f"Epsilon final: {agent.epsilon:.3f}")

# =========================================
# EXERCICES PRATIQUES
# =========================================

def exercices_supplementaires():
    """
    Exercices pour approfondir la compréhension
    """
    print(f"\n🎓 EXERCICES POUR APPROFONDIR")
    print("="*40)
    
    exercises = [
        "1. Modifier les récompenses et observer l'impact sur l'apprentissage",
        "2. Changer la taille de la grille (3x3, 7x7) et comparer",
        "3. Tester différents paramètres (α, γ, ε) et analyser",
        "4. Ajouter plus d'obstacles et voir comment l'agent s'adapte",
        "5. Implémenter SARSA au lieu de Q-Learning",
        "6. Ajouter des états partiellement observables",
        "7. Créer un environnement plus complexe (labyrinthe)",
        "8. Implémenter Double Q-Learning pour éviter la surestimation"
    ]
    
    for exercise in exercises:
        print(f"   {exercise}")
    
    print(f"\n💡 Conseils :")
    print("   - Commencez par modifier un paramètre à la fois")
    print("   - Observez les courbes d'apprentissage")  
    print("   - Testez la robustesse avec différents seeds aléatoires")

# =========================================
# EXÉCUTION DU PROGRAMME PRINCIPAL
# =========================================

if __name__ == "__main__":
    # Fixer la graine pour la reproductibilité
    random.seed(42)
    np.random.seed(42)
    
    # Lancer la démonstration
    main_demonstration()
    
    # Proposer des exercices
    # exercices_supplementaires()
    
    print(f"\n🎉 DÉMONSTRATION TERMINÉE !")
    print("="*30)