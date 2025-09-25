# TP : Analyse et Construction de Réseaux de Neurones Convolutifs (CNN)

---

## Partie 1 : Blocs de construction des CNN

### 1.1 Analyse d'une couche de convolution (CONV)

**Paramètres donnés :**
- Image d'entrée : 32×32×3 (RGB)
- Filtre : 5×5×3 (même profondeur que l'entrée)
- Nombre de filtres : 16
- Stride : 1
- Padding : 2

#### Questions d'analyse :

**1.1.1 Calcul des dimensions de sortie**
```
Formule : Output_size = (Input_size + 2×Padding - Filter_size) / Stride + 1
```

Calculez :
- Largeur de sortie
- Hauteur de sortie  
- Profondeur de sortie
- **Réponse attendue :** (32 + 2×2 - 5)/1 + 1 = 32×32×16

**1.1.2 Nombre de paramètres**
- Paramètres par filtre : 5×5×3 + 1 (biais) = 76
- Total pour 16 filtres : 76×16 = 1,216 paramètres
- **Justifiez pourquoi chaque filtre partage ses poids sur toute l'image**

**1.1.3 Champ récepteur (Receptive Field)**
Calculez le champ récepteur de cette couche et expliquez son impact sur la détection de motifs.

### 1.2 Couche de Pooling (POOL)

**Configuration Max Pooling :**
- Fenêtre : 2×2
- Stride : 2
- Entrée : 32×32×16 (sortie de la convolution précédente)

#### Analyse demandée :

**1.2.1 Impact sur les dimensions**
- Sortie attendue : 16×16×16
- **Pourquoi la profondeur reste inchangée ?**

**1.2.2 Avantages et inconvénients**
Comparez Max Pooling vs Average Pooling :
```
Max Pooling : Préserve les caractéristiques dominantes
Average Pooling : Lisse les caractéristiques, réduit le bruit
```

**1.2.3 Alternative moderne : Strided Convolution**
Proposez une couche de convolution équivalente au Max Pooling 2×2 avec stride 2.

### 1.3 Couches d'activation

**Analyse comparative :**

#### ReLU vs Sigmoid vs Tanh

**Testez manuellement :** Appliquez chaque fonction sur le vecteur [-2, -0.5, 0, 0.5, 2]

```
ReLU(x) = max(0, x)
Sigmoid(x) = 1 / (1 + e^(-x))
Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Questions :**
1. Laquelle évite le problème du gradient qui disparaît ?
2. Laquelle normalise les sorties entre 0 et 1 ?
3. Pourquoi ReLU est-elle privilégiée dans les couches cachées ?

### 1.4 Architecture CNN complète - Analyse

**Modèle proposé :**
```
INPUT → CONV1 → ReLU → POOL1 → CONV2 → ReLU → POOL2 → FC1 → ReLU → FC2 → SOFTMAX
32×32×3   32×32×16      16×16×16   16×16×32      8×8×32    2048    512     10
```

**Détail des couches :**
- **CONV1 :** 16 filtres 5×5×3, stride=1, padding=2
- **POOL1 :** Max pooling 2×2, stride=2
- **CONV2 :** 32 filtres 5×5×16, stride=1, padding=2
- **POOL2 :** Max pooling 2×2, stride=2
- **FC1 :** 8×8×32 → 512 neurones
- **FC2 :** 512 → 10 neurones (classification 10 classes)

#### Calculs demandés :

**1.4.1 Vérification des dimensions**
Tracez le passage des dimensions à travers chaque couche.

**1.4.2 Nombre total de paramètres**
- CONV1 : (5×5×3+1)×16 = 1,216
- CONV2 : (5×5×16+1)×32 = 12,832
- FC1 : (8×8×32)×512 + 512 = 1,049,088
- FC2 : 512×10 + 10 = 5,130
- **Total :** 1,068,266 paramètres

**1.4.3 Complexité computationnelle**
Estimez le nombre d'opérations (FLOPS) pour une image.