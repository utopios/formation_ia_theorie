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