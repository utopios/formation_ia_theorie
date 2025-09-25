# TP : Analyse d'un algorithme MLP - Multi-Layer Perceptron

---

## Partie 1 : Analyse théorique de l'architecture

### 1.1 Étude du modèle proposé

Vous disposez d'un MLP avec l'architecture suivante :
- **Couche d'entrée** : 4 neurones (variables x₁, x₂, x₃, x₄)
- **Couche cachée 1** : 6 neurones avec fonction d'activation ReLU
- **Couche cachée 2** : 4 neurones avec fonction d'activation ReLU  
- **Couche de sortie** : 3 neurones avec fonction d'activation Softmax

**Questions d'analyse :**

1. **Calcul de complexité**
   - Calculez le nombre total de paramètres (poids + biais)
    1. INPUT -> L1 (4x6 + 6) = 30
    2. L1 -> L2 (6X4 + 4) = 28
    3. L2 -> OUTPUT (4x3 + 3) = 15
        - Total : 73
   - Déterminez la complexité computationnelle pour une propagation avant
        O(p)
   - Analysez l'impact de chaque couche sur la complexité globale

2. **Architecture et capacité d'apprentissage**
   - Justifiez le choix des fonctions d'activation pour chaque couche
   - Que représentent les 3 sorties avec Softmax ? 
   - Cette architecture convient-elle à un problème de classification multiclasse ?

### 1.2 Analyse mathématique des transformations

Pour chaque couche, formalisez :
- Les équations de propagation avant
- Les dimensions des matrices de poids
- Les transformations géométriques appliquées aux données

1. INPUT -> L1
    - Transformation Linéaire: z1 = W1 X x + b1
    - Activation h1 = ReLu(z1) = max(0, z1)

    W1: (6 x 4) - matrice des poids
    x: (4 x 1) - vecteur d'entrée
    b1: (6 x 1) - vecteur de biais
    z1: (6 x 1) - sortie linéaire
    h1: (6 x 1) - sortie après activation

2. L2 -> L2
    - Transformation Linéaire: z2 = W2 X h1 + b2
    - Activation h2 = ReLu(z2) = max(0, z2)

    W2: (4 x 6) - matrice des poids
    h1: (6 x 1) - vecteur d'entrée
    b2: (4 x 1) - vecteur de biais
    z2: (4 x 1) - sortie linéaire
    h2: (4 x 1) - sortie après activation

3. L2 -> OUTPUT
    - Transformation Linéaire: z3 = W3 X h2 + b3
    - Activation y = Softmax(z3)


    W3: (3 x 4) - matrice des poids
    h2: (4 x 1) - vecteur d'entrée
    b3: (3 x 1) - vecteur de biais
    z3: (3 x 1) - sortie linéaire
    y: (3 x 1) - sortie après activation

---

## Partie 2 : Simulation manuelle de la propagation

### 2.1 Propagation avant pas à pas

**Données d'exemple :**
- Vecteur d'entrée : x = [0.5, -0.2, 0.8, 1.1]
- Poids et biais fournis dans les matrices ci-dessous

**Matrices de poids simplifiées (pour simulation) :**

**W₁** (entrée → couche cachée 1) - 4×6 :
```
     n₁   n₂   n₃   n₄   n₅   n₆
x₁ [ 0.3  0.1 -0.2  0.4  0.2  0.1]
x₂ [-0.1  0.5  0.3 -0.2  0.1  0.4]
x₃ [ 0.2 -0.3  0.1  0.5  0.3 -0.1]
x₄ [ 0.4  0.2 -0.1  0.3 -0.2  0.5]
```

**Biais b₁** : [0.1, -0.05, 0.2, 0.0, 0.15, -0.1]

**Travail demandé :**
1. Calculez les activations de la couche cachée 1
2. Appliquez la fonction ReLU
3. Continuez la propagation jusqu'à la sortie
4. Interprétez le résultat final (probabilités de classification)


**Données :**
- x = [0.5, -0.2, 0.8, 1.1]
- W₁ et b₁ comme donnés dans l'énoncé

**Étape 1 : Couche d'entrée → Couche cachée 1**

Calcul de z₁ = W₁ᵀ × x + b₁ :

```
z₁₁ = 0.3×0.5 + (-0.1)×(-0.2) + 0.2×0.8 + 0.4×1.1 + 0.1
    = 0.15 + 0.02 + 0.16 + 0.44 + 0.1 = 0.87

z₁₂ = 0.1×0.5 + 0.5×(-0.2) + (-0.3)×0.8 + 0.2×1.1 + (-0.05)
    = 0.05 - 0.1 - 0.24 + 0.22 - 0.05 = -0.12

z₁₃ = (-0.2)×0.5 + 0.3×(-0.2) + 0.1×0.8 + (-0.1)×1.1 + 0.2
    = -0.1 - 0.06 + 0.08 - 0.11 + 0.2 = 0.01

z₁₄ = 0.4×0.5 + (-0.2)×(-0.2) + 0.5×0.8 + 0.3×1.1 + 0.0
    = 0.2 + 0.04 + 0.4 + 0.33 = 0.97

z₁₅ = 0.2×0.5 + 0.1×(-0.2) + 0.3×0.8 + (-0.2)×1.1 + 0.15
    = 0.1 - 0.02 + 0.24 - 0.22 + 0.15 = 0.25

z₁₆ = 0.1×0.5 + 0.4×(-0.2) + (-0.1)×0.8 + 0.5×1.1 + (-0.1)
    = 0.05 - 0.08 - 0.08 + 0.55 - 0.1 = 0.34
```

h₁ = ReLU(z₁) = [0.87, 0, 0.01, 0.97, 0.25, 0.34]

**Étape 2 : Couche cachée 1 → Couche cachée 2**

Supposons W₂ (6×4) et b₂ :
```
W₂ = [0.2  -0.1   0.3   0.1]
     [0.1   0.4  -0.2   0.3]
     [-0.1  0.2   0.1  -0.3]
     [0.3  -0.2   0.4   0.2]
     [0.1   0.3  -0.1   0.4]
     [-0.2  0.1   0.3  -0.1]

b₂ = [0.05, -0.1, 0.15, 0.02]
```

z₂ = W₂ᵀ × a₁ + b₂ :

```
z₂₁ = 0.2×0.87 + 0.1×0 + (-0.1)×0.01 + 0.3×0.97 + 0.1×0.25 + (-0.2)×0.34 + 0.05
    = 0.174 + 0 - 0.001 + 0.291 + 0.025 - 0.068 + 0.05 = 0.471

z₂₂ = (-0.1)×0.87 + 0.4×0 + 0.2×0.01 + (-0.2)×0.97 + 0.3×0.25 + 0.1×0.34 + (-0.1)
    = -0.087 + 0 + 0.002 - 0.194 + 0.075 + 0.034 - 0.1 = -0.27

z₂₃ = 0.3×0.87 + (-0.2)×0 + 0.1×0.01 + 0.4×0.97 + (-0.1)×0.25 + 0.3×0.34 + 0.15
    = 0.261 + 0 + 0.001 + 0.388 - 0.025 + 0.102 + 0.15 = 0.877

z₂₄ = 0.1×0.87 + 0.3×0 + (-0.3)×0.01 + 0.2×0.97 + 0.4×0.25 + (-0.1)×0.34 + 0.02
    = 0.087 + 0 - 0.003 + 0.194 + 0.1 - 0.034 + 0.02 = 0.364
```

**Application de ReLU :**
h₂ = ReLU(z₂) = [0.471, 0, 0.877, 0.364]

**Étape 3 : Couche cachée 2 → Couche de sortie**

Supposons W₃ (4×3) et b₃ :
```
W₃ = [0.4  -0.2   0.3]
     [0.1   0.5  -0.1]
     [-0.2  0.3   0.4]
     [0.3  -0.1   0.2]

b₃ = [0.1, 0.05, -0.05]
```

z₃ = W₃ᵀ × a₂ + b₃ :

```
z₃₁ = 0.4×0.471 + 0.1×0 + (-0.2)×0.877 + 0.3×0.364 + 0.1
    = 0.188 + 0 - 0.175 + 0.109 + 0.1 = 0.222

z₃₂ = (-0.2)×0.471 + 0.5×0 + 0.3×0.877 + (-0.1)×0.364 + 0.05
    = -0.094 + 0 + 0.263 - 0.036 + 0.05 = 0.183

z₃₃ = 0.3×0.471 + (-0.1)×0 + 0.4×0.877 + 0.2×0.364 + (-0.05)
    = 0.141 + 0 + 0.351 + 0.073 - 0.05 = 0.515
```

**Application de Softmax :**
```
exp(z₃) = [exp(0.222), exp(0.183), exp(0.515)]
        = [1.249, 1.201, 1.674]

Sum = 1.249 + 1.201 + 1.674 = 4.124

Softmax(z₃) = [1.249/4.124, 1.201/4.124, 1.674/4.124]
             = [0.303, 0.291, 0.406]
```


### 2.2 Analyse de sensibilité

Modifiez légèrement le vecteur d'entrée et observez :
- L'impact sur les activations intermédiaires
- La robustesse de la prédiction finale
- Les neurones les plus sensibles aux variations

Cibe: y = [0, 1, 0]
L = -Σᵢ yᵢ log(ŷᵢ) = -log(0.291) = 1.234
∂L/∂z₃ = [0.303, 0.291, 0.406] - [0, 1, 0] = [0.303, -0.709, 0.406]

**Gradient des poids W₃ :**
```
∂L/∂W₃ = h₂ × (∂L/∂z₃)ᵀ

∂L/∂W₃ = [0.471]  × [0.303, -0.709, 0.406]
         [0    ]
         [0.877]
         [0.364]

= [0.143  -0.334   0.191]
  [0      0       0    ]
  [0.266  -0.622   0.356]
  [0.110  -0.258   0.148]
```

