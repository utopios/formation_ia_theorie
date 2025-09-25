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
   - Déterminez la complexité computationnelle pour une propagation avant
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

### 2.2 Analyse de sensibilité

Modifiez légèrement le vecteur d'entrée et observez :
- L'impact sur les activations intermédiaires
- La robustesse de la prédiction finale
- Les neurones les plus sensibles aux variations