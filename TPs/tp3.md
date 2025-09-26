# TP RNN et LSTM 

---

## Partie 1 : RNN simple - Calculs de base

### 1.1 Architecture RNN vanilla

**Équations du RNN simple :**
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

#### **Exercice 1.1**

**Données :**
- Séquence d'entrée : x = [0.5, 0.8, 0.2]
- État initial : h_0 = [0.0, 0.0]
- Dimensions : input_size = 1, hidden_size = 2, output_size = 1

**Matrices de poids :**
```
W_xh = [0.3]    W_hh = [0.2  0.1]    b_h = [0.1]
       [0.4]           [-0.1 0.3]           [0.0]
W_xh = [[0.3], [0.4]]
W_hy = [0.5  -0.2]    b_y = [0.1]
```

**Questions :**
1. Calculez h_1, h_2, h_3 étape par étape
2. Calculez y_1, y_2, y_3
3. Tracez l'évolution des valeurs de h_t

**t=1, x_1 = 0.5:**
```
z_1 = W_hh × h_0 + W_xh × x_1 + b_h
z_1 = [[0.2, 0.1], [-0.1, 0.3]] × [0.0, 0.0] + [[0.3], [0.4]] × 0.5 + [[0.1], [0.0]]
z_1 = [0.0, 0.0] + [0.15, 0.2] + [0.1, 0.0] = [0.25, 0.2]

h_1 = tanh([0.25, 0.2]) = [0.2449, 0.1974]

y_1 = W_hy × h_1 + b_y = [0.5, -0.2] × [0.2449, 0.1974] + 0.1
y_1 = 0.1224 - 0.0395 + 0.1 = 0.1829
```

**t=2, x_2 = 0.8:**
```
z_2 = W_hh × h_1 + W_xh × x_2 + b_h
z_2 = [[0.2, 0.1], [-0.1, 0.3]] × [0.2449, 0.1974] + [[0.3], [0.4]] × 0.8 + [[0.1], [0.0]]
z_2 = [0.0688, 0.0348] + [0.24, 0.32] + [0.1, 0.0] = [0.4088, 0.3548]

h_2 = tanh([0.4088, 0.3548]) = [0.3879, 0.3402]

y_2 = [0.5, -0.2] × [0.3879, 0.3402] + 0.1 = 0.2259
  
```

**t=3, x_3 = 0.2:**
```
z_3 = [[0.2, 0.1], [-0.1, 0.3]] × [0.3879, 0.3402] + [[0.3], [0.4]] × 0.2 + [[0.1], [0.0]]
z_3 = [0.1116, 0.0635] + [0.06, 0.08] + [0.1, 0.0] = [0.2716, 0.1435]

h_3 = tanh([0.2716, 0.1435]) = [0.2654, 0.1425]

y_3 = [0.5, -0.2] × [0.2654, 0.1425] + 0.1 = 0.2042
```

**Résultats :**
- h_1 = [0.245, 0.197], y_1 = 0.183
- h_2 = [0.388, 0.340], y_2 = 0.226  
- h_3 = [0.265, 0.143], y_3 = 0.204

### 1.2 Problème du gradient qui disparaît

#### **Exercice 1.2 : Analyse quantitative**

**Calcul du gradient à travers le temps :**
```
∂L/∂h_0 = ∂L/∂h_3 × (∂h_3/∂h_2) × (∂h_2/∂h_1) × (∂h_1/∂h_0)
```

**Avec :**
```
∂h_t/∂h_{t-1} = W_hh × diag(1 - tanh²(z_t))
```

**Questions :**
1. Si W_hh = [[0.8, 0.1], [-0.1, 0.8]] et tanh'(z) ≈ 0.9 en moyenne, calculez le gradient après 3, 5, 10 pas

Avec W_hh = [[0.8, 0.1], [-0.1, 0.8]] et tanh'(z) ≈ 0.9 :

```
∂h_t/∂h_{t-1} = W_hh × diag(tanh'(z_t)) ≈ [[0.8, 0.1], [-0.1, 0.8]] × 0.9
```

```
Norme du gradient ≈ (0.9)³ × ||W_hh||³ ≈ 0.729 × (0.8)³ ≈ 0.373
```

**Après 5 pas :** ≈ 0.196  
**Après 10 pas :** ≈ 0.039


2. À partir de combien de pas le gradient devient-il < 0.01 ?
3. Que se passe-t-il si les valeurs propres de W_hh sont > 1 ?

---

## Partie 2 : Architecture LSTM

### 2.1 Équations complètes du LSTM

**Les 4 portes et mises à jour :**

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    # Input gate  
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C) # Candidate values
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    # Output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Cell state update
h_t = o_t ⊙ tanh(C_t)                   # Hidden state update
```

### 2.2 Exercice calcul manuel LSTM

#### **Exercice 2.1 : Mini-LSTM (1 neurone)**

**Configuration simplifiée :**
- input_size = 1, hidden_size = 1, 1 seul neurone LSTM
- Séquence : x = [1.0, 0.5, -0.3]
- États initiaux : h_0 = 0.0, C_0 = 0.0

**Paramètres :**
```python
# Chaque matrice W a 2 colonnes : [h_{t-1}, x_t]
W_f = [0.1, 0.2]    b_f = 0.0
W_i = [0.3, -0.1]   b_i = 0.1  
W_C = [0.2, 0.4]    b_C = 0.0
W_o = [0.1, 0.3]    b_o = 0.0
```

**Questions à résoudre manuellement :**

**Pas de temps t=1 (x_1 = 1.0) :**

1. **Calculez f_1 :**
   ```
   Entrée concatenée : [h_0, x_1] = [0.0, 1.0]
   z_f = W_f × [0.0, 1.0] + b_f = ?
   f_1 = σ(z_f) = ?
   ```

2. **Calculez i_1 :**
   ```
   z_i = W_i × [0.0, 1.0] + b_i = ?
   i_1 = σ(z_i) = ?
   ```

3. **Calculez C̃_1 :**
   ```
   z_C = W_C × [0.0, 1.0] + b_C = ?
   C̃_1 = tanh(z_C) = ?
   ```

4. **Calculez C_1 :**
   ```
   C_1 = f_1 ⊙ C_0 + i_1 ⊙ C̃_1 = ?
   ```

5. **Calculez o_1 et h_1 :**
   ```
   z_o = W_o × [0.0, 1.0] + b_o = ?
   o_1 = σ(z_o) = ?
   h_1 = o_1 ⊙ tanh(C_1) = ?
   ```

**Répétez pour t=2 et t=3**


##### Solutions 

**Configuration :**

- input_size = 1, hidden_size = 1, 1 seul neurone LSTM
- Séquence : x = [1.0, 0.5, -0.3]
- États initiaux : h_0 = 0.0, C_0 = 0.0


- W_f = [0.1, 0.2], b_f = 0.0
- W_i = [0.3, -0.1], b_i = 0.1
- W_C = [0.2, 0.4], b_C = 0.0  
- W_o = [0.1, 0.3], b_o = 0.0
- h_0 = 0.0, C_0 = 0.0

**Pas t=1 (x_1 = 1.0) :**

1. **Forget gate :**
```
z_f = [0.1, 0.2] × [0.0, 1.0] + 0.0 = 0.2
f_1 = σ(0.2) = 0.5498
```

2. **Input gate :**
```
z_i = [0.3, -0.1] × [0.0, 1.0] + 0.1 = -0.1 + 0.1 = 0.0
i_1 = σ(0.0) = 0.5
```

3. **Candidate :**
```
z_C = [0.2, 0.4] × [0.0, 1.0] + 0.0 = 0.4
C̃_1 = tanh(0.4) = 0.3799
```

4. **Cell state :**
```
C_1 = f_1 × C_0 + i_1 × C̃_1 = 0.5498 × 0.0 + 0.5 × 0.3799 = 0.1900
```

5. **Output gate et hidden state :**
```
z_o = [0.1, 0.3] × [0.0, 1.0] + 0.0 = 0.3
o_1 = σ(0.3) = 0.5744
h_1 = o_1 × tanh(C_1) = 0.5744 × tanh(0.1900) = 0.5744 × 0.1881 = 0.1080
```

**Pas t=2 (x_2 = 0.5) :**

```
Entrée : [h_1, x_2] = [0.1080, 0.5]

f_2 = σ(0.1 × 0.1080 + 0.2 × 0.5) = σ(0.1108) = 0.5277
i_2 = σ(0.3 × 0.1080 - 0.1 × 0.5 + 0.1) = σ(0.0824) = 0.5206
C̃_2 = tanh(0.2 × 0.1080 + 0.4 × 0.5) = tanh(0.2216) = 0.2182
o_2 = σ(0.1 × 0.1080 + 0.3 × 0.5) = σ(0.1608) = 0.5401

C_2 = 0.5277 × 0.1900 + 0.5206 × 0.2182 = 0.1003 + 0.1136 = 0.2139
h_2 = 0.5401 × tanh(0.2139) = 0.5401 × 0.2108 = 0.1138
```

**Pas t=3 (x_3 = -0.3) :**

```
Entrée : [h_2, x_3] = [0.1138, -0.3]

f_3 = σ(0.1 × 0.1138 + 0.2 × (-0.3)) = σ(-0.0486) = 0.4879
i_3 = σ(0.3 × 0.1138 - 0.1 × (-0.3) + 0.1) = σ(0.1641) = 0.5409
C̃_3 = tanh(0.2 × 0.1138 + 0.4 × (-0.3)) = tanh(-0.0973) = -0.0970
o_3 = σ(0.1 × 0.1138 + 0.3 × (-0.3)) = σ(-0.0786) = 0.4804

C_3 = 0.4879 × 0.2139 + 0.5409 × (-0.0970) = 0.1043 - 0.0525 = 0.0518
h_3 = 0.4804 × tanh(0.0518) = 0.4804 × 0.0517 = 0.0248
```

#### **Exercice 2.2 : LSTM 2 neurones**

**Configuration :**
- input_size = 1, hidden_size = 2
- Séquence : x = [0.8, -0.2]
- États initiaux : h_0 = [0.1, -0.1], C_0 = [0.2, 0.0]

**Matrices 2×3 (pour [h1, h2, x]) :**
```python
W_f = [[0.1, 0.2, 0.3],
       [0.0, 0.1, -0.2]]
b_f = [0.1, 0.0]

W_i = [[0.2, -0.1, 0.4],
       [0.1, 0.3, 0.1]]  
b_i = [0.0, 0.1]

W_C = [[0.3, 0.1, -0.1],
       [-0.1, 0.2, 0.3]]
b_C = [0.05, 0.0]

W_o = [[0.1, 0.4, 0.2],
       [0.2, 0.0, 0.3]]
b_o = [0.1, 0.0]
```

**Tâche :** Calculez toutes les valeurs pour t=1

### 2.3 Analyse mathématique des gradients LSTM

#### **Exercice 2.3 : Gradient flow analysis**

**Gradient de l'état cellulaire :**
```
∂C_t/∂C_{t-1} = f_t
```

**Questions théoriques :**
1. Pourquoi ce gradient ne disparaît-il pas comme dans les RNN ?
2. Si f_t = 0.9 constamment, que devient le gradient après 10 pas ?
3. Quelle est la condition pour éviter l'explosion du gradient ?

**Gradient complet à travers les portes :**
```
∂L/∂W_f = Σ_t (∂L/∂f_t × σ'(z_f_t) × [h_{t-1}, x_t])
```

**Calculez numériquement :**
Avec les valeurs de l'exercice 2.1, si ∂L/∂h_1 = 0.1, calculez ∂L/∂W_f

## Partie 3 : Implémentation from scratch 

### 3.1 LSTM minimal en Python pur

#### **Exercice 3.1 : Implémentation step by step**

```python
import numpy as np

class SimpleLSTM:
    """LSTM minimal avec calculs explicites"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: Initialisez les poids et biais
        # Utilisez une initialisation Xavier : limite = sqrt(6/(n_in + n_out))
        
    def sigmoid(self, x):
        """Fonction sigmoid avec protection overflow"""
        # TODO: Implémentez avec clipping pour éviter overflow
        pass
    
    def tanh(self, x):
        """Fonction tanh"""
        # TODO: Implémentez
        pass
    
    def forward_step(self, x_t, h_prev, C_prev):
        """Un pas forward du LSTM"""
        
        # TODO: 
        # 1. Concaténer [h_prev, x_t]
        # 2. Calculer les 4 portes
        # 3. Mettre à jour C_t et h_t
        # 4. Retourner h_t, C_t, et toutes les valeurs intermédiaires
        
        return h_t, C_t, gates_info
    
    def forward_sequence(self, X):
        """Forward sur séquence complète"""
        
        batch_size, seq_len, _ = X.shape
        
        # États initiaux
        h_t = np.zeros((batch_size, self.hidden_size))
        C_t = np.zeros((batch_size, self.hidden_size))
        
        # Stockage pour analyse
        all_states = []
        
        for t in range(seq_len):
            h_t, C_t, gate_info = self.forward_step(X[:, t, :], h_t, C_t)
            all_states.append({
                'h': h_t.copy(),
                'C': C_t.copy(), 
                'gates': gate_info
            })
        
        return all_states

# Test de l'implémentation
lstm = SimpleLSTM(input_size=1, hidden_size=2)

# Données de test (exercice 2.2)
X_test = np.array([[[0.8], [-0.2]]]).reshape(1, 2, 1)
states = lstm.forward_sequence(X_test)

print("États calculés:")
for t, state in enumerate(states):
    print(f"t={t+1}: h={state['h']}, C={state['C']}")
```

**Questions de validation :**
1. Vos résultats correspondent-ils aux calculs manuels ?
2. Que se passe-t-il si vous modifiez l'initialisation des poids ?
3. Comment évolue l'état cellulaire C_t par rapport à h_t ?

### 3.2 Analyse de sensibilité des paramètres

#### **Exercice 3.2 : Impact des poids**

**Testez différentes configurations :**

```python
def analyze_weight_sensitivity():
    """Analyser l'impact des poids sur le comportement LSTM"""
    
    configurations = {
        'conservative': {
            'W_f': 'petites valeurs (0.1-0.3)',
            'effect': 'Oublie peu, mémoire longue'
        },
        'aggressive': {
            'W_f': 'grandes valeurs (0.7-0.9)', 
            'effect': 'Oublie beaucoup, mémoire courte'
        },
        'selective_input': {
            'W_i': 'valeurs contrastées',
            'effect': 'Sélectif sur nouvelles informations'
        }
    }
    
    # TODO: Implémentez chaque configuration
    # Testez sur la même séquence
    # Comparez les états cellulaires C_t
```

**Questions d'analyse :**
1. Comment la forget gate influence-t-elle la persistance de l'information ?
2. Quelle configuration retient le mieux l'information du début de séquence ?
3. Comment équilibrer mémoire vs adaptabilité ?


## Partie 4 : Applications mathématiques

### 4.1 Prédiction de séquence arithmétique

```python
def create_arithmetic_dataset():
    """Créer dataset pour séquences arithmétiques"""
    
    sequences = []
    targets = []
    
    # Séquences arithmétiques
    for start in range(1, 20, 2):
        for step in [2, 3, 5]:
            seq = [start + i*step for i in range(7)]
            
            # Fenêtres glissantes: 3 input → 1 output
            for i in range(len(seq) - 3):
                sequences.append(seq[i:i+3])
                targets.append(seq[i+3])
    
    # Normalisation
    all_values = [val for seq in sequences for val in seq] + targets
    min_val, max_val = min(all_values), max(all_values)
    
    def normalize(x):
        return (x - min_val) / (max_val - min_val)
    
    X = np.array([[normalize(val) for val in seq] for seq in sequences])
    y = np.array([normalize(val) for val in targets])
    
    return X, y, (min_val, max_val)

def test_arithmetic_prediction():
    """Test prédiction séquences arithmétiques"""
    
    X, y, norm_params = create_arithmetic_dataset()
    
    # LSTM simple pour test
    lstm = SimpleLSTM(input_size=1, hidden_size=4)
    
    print("Dataset séquences arithmétiques:")
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")
    print(f"Exemple: {X[0]} → {y[0]}")
    
    # Test sur une séquence [2, 4, 6] → 8
    test_seq = np.array([2, 4, 6]).reshape(-1, 1)
    min_val, max_val = norm_params
    test_seq_norm = (test_seq - min_val) / (max_val - min_val)
    
    states = lstm.forward_sequence(test_seq_norm)
    
    print(f"\nTest [2, 4, 6] → ?:")
    print(f"États cachés finaux: {states[-1]['h']}")
    print("Évolution mémoire cellulaire:")
    for t, state in enumerate(states):
        print(f"  t={t+1}: C_t = {state['C'][:2]}...")  # Premiers éléments
```

### 4.2 Reconnaissance de motifs

```python
def pattern_recognition_test():
    """Test reconnaissance motif ABC"""
    
    # Encodage: A=1, B=2, C=3, etc.
    def encode_string(s):
        return [ord(c) - ord('A') + 1 for c in s]
    
    sequences = [
        ("ABCDEF", [0, 0, 1, 0, 0, 0]),    # ABC détecté en position 3
        ("XYZABC", [0, 0, 0, 0, 0, 1]),    # ABC détecté en position 6  
        ("ABCXABC", [0, 0, 1, 0, 0, 0, 1]), # ABC détecté positions 3,7
        ("ABXABC", [0, 0, 0, 0, 0, 1]),    # ABC détecté position 6
        ("ACBACB", [0, 0, 0, 0, 0, 0]),    # Pas de motif
    ]
    
    lstm = SimpleLSTM(input_size=1, hidden_size=3)
    
    print("Test reconnaissance motif ABC:")
    print("-" * 40)
    
    for seq_str, expected in sequences:
        # Encoder séquence
        seq_encoded = encode_string(seq_str)
        X = np.array(seq_encoded).reshape(-1, 1) / 26.0  # Normalisation
        
        # Forward pass
        states = lstm.forward_sequence(X)
        
        print(f"\nSéquence: {seq_str}")
        print(f"Encodée: {seq_encoded}")
        print("Évolution état cellulaire:")
        
        for t, state in enumerate(states):
            char = seq_str[t]
            print(f"  {char} (t={t+1}): C_t = {state['C']:.3f}")
            
            # Analyser si pattern "AB" en mémoire
            if char == 'C' and t >= 2:
                prev_chars = seq_str[t-2:t]
                if prev_chars == "AB":
                    print(f"    → Motif ABC détecté! Mémoire: {state['C'][0]:.3f}")

if __name__ == "__main__":
    test_arithmetic_prediction()
    pattern_recognition_test()
```