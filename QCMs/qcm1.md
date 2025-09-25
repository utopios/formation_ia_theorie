# **Chapitre 1 : Réseaux de neurones – Concepts fondamentaux**

1. Quelle est la fonction principale d’un neurone artificiel ?
   A. Classer des images
   B. Combiner et transformer des entrées en sortie
   C. Stocker des données
   D. Transférer uniquement des signaux binaires

2. Une fonction d’activation sert à :
   A. Normaliser les données d’entrée
   B. Introduire de la non-linéarité
   C. Calculer les poids optimaux
   D. Remplacer la fonction de combinaison

3. Vrai ou Faux : un réseau sans fonction d’activation équivaut à une simple combinaison linéaire.

4. Dans la propagation avant (forward pass), les entrées :
   A. Sont directement la sortie
   B. Passent par des couches pondérées et activées
   C. Sont inversées
   D. Restent inchangées

5. La rétropropagation sert principalement à :
   A. Calculer la sortie finale
   B. Ajuster les poids du réseau
   C. Ajouter de nouvelles couches
   D. Supprimer les biais

6. La structure d’un réseau est définie par :
   A. Le nombre de couches et de neurones par couche
   B. Le type de fonction de coût uniquement
   C. Le choix des données d’entrée
   D. La taille du batch

7. Exemple de fonction d’activation non linéaire :
   A. Sigmoïde
   B. Identité
   C. Moyenne
   D. Somme

8. Dans un problème de classification binaire, la fonction de sortie est souvent :
   A. ReLU
   B. Softmax
   C. Sigmoïde
   D. Tangente hyperbolique

9. Le biais dans un neurone permet :
   A. De réduire les calculs
   B. De décaler la fonction d’activation
   C. De supprimer la propagation
   D. D’ajouter du bruit

10. Le rôle d’une fonction de perte est :
    A. Minimiser l’erreur entre sortie prédite et sortie réelle
    B. Maximiser la précision du CPU
    C. Normaliser les données
    D. Générer des poids aléatoires

11. Vrai ou Faux : dans un problème de régression, la fonction Softmax est utilisée en sortie.

12. Le gradient est :
    A. Une dérivée partielle par rapport aux poids
    B. Un vecteur aléatoire
    C. Une métrique de précision
    D. Un biais

13. Quel problème est associé au gradient dans les réseaux profonds ?
    A. Explosion ou disparition du gradient
    B. Manque de données
    C. Surcharge mémoire uniquement
    D. Trop de classes

14. Le perceptron de Rosenblatt peut résoudre :
    A. Les problèmes non-linéaires complexes
    B. Uniquement les problèmes linéairement séparables
    C. Tous les types de problèmes
    D. Aucun problème

15. La normalisation des données sert à :
    A. Accélérer et stabiliser l’apprentissage
    B. Supprimer la rétropropagation
    C. Remplacer la fonction de perte
    D. Réduire la taille du réseau

16. Un neurone artificiel calcule :
    A. Une moyenne des entrées
    B. Une somme pondérée suivie d’une activation
    C. Une permutation aléatoire
    D. Une dérivée directe

17. Vrai ou Faux : plus il y a de couches, plus le réseau est capable de représenter des relations complexes.

18. Les réseaux denses (fully connected) :
    A. Connectent chaque neurone d’une couche à tous ceux de la couche suivante
    B. Utilisent toujours ReLU
    C. Ne possèdent pas de biais
    D. Sont réservés aux images

19. Le surapprentissage (overfitting) survient lorsque :
    A. Le réseau est trop petit
    B. Le réseau apprend trop bien les données d’entraînement
    C. La fonction d’activation est linéaire
    D. Le taux d’apprentissage est faible

20. Vrai ou Faux : la propagation avant et la rétropropagation constituent ensemble le cycle d’apprentissage.

---

# **Chapitre 2 : MLP (Multi-Layer Perceptron)**

1. Un MLP est :
   A. Un perceptron simple
   B. Un réseau de neurones à plusieurs couches
   C. Un réseau récurrent
   D. Un autoencodeur

2. La différence principale entre perceptron simple et MLP est :
   A. Le nombre de classes
   B. L’ajout de couches cachées
   C. L’utilisation de ReLU uniquement
   D. Le type de perte

3. La propagation avant dans un MLP signifie :
   A. Le calcul des poids
   B. Le passage des données d’entrée à travers toutes les couches
   C. La mise à jour des biais
   D. Le calcul de la matrice de covariance

4. L’apprentissage d’un MLP repose sur :
   A. La descente de gradient
   B. La multiplication matricielle aléatoire
   C. L’approximation de Monte Carlo
   D. La réduction dimensionnelle

5. Vrai ou Faux : un MLP sans fonction d’activation équivaut à une régression linéaire.

6. Les poids d’un MLP sont initialisés :
   A. Tous à zéro
   B. Avec des valeurs aléatoires
   C. Avec des valeurs infinies
   D. Avec les moyennes des entrées

7. Quel algorithme est utilisé pour ajuster les poids dans un MLP ?
   A. Régression logistique
   B. Rétropropagation
   C. Softmax
   D. Méthode de Newton

8. La complexité d’un MLP dépend de :
   A. La profondeur et du nombre de neurones par couche
   B. La fonction d’activation uniquement
   C. La taille du batch uniquement
   D. La normalisation des entrées

9. Un MLP peut résoudre :
   A. Des problèmes linéaires uniquement
   B. Des problèmes non linéaires complexes
   C. Aucun problème pratique
   D. Uniquement des régressions

10. Vrai ou Faux : un MLP est universellement approximant.

11. Les couches cachées servent à :
    A. Simplifier les calculs
    B. Extraire des représentations intermédiaires
    C. Éliminer la perte
    D. Réduire la taille du dataset

12. La fonction Softmax est utilisée :
    A. Pour transformer des scores en probabilités
    B. Pour réduire le surapprentissage
    C. Pour augmenter le taux d’apprentissage
    D. Pour remplacer la rétropropagation

13. La fonction de coût classique pour un problème de classification est :
    A. MSE
    B. Cross-Entropy
    C. ReLU
    D. L1

14. Quel est l’avantage d’un MLP par rapport à un perceptron simple ?
    A. Plus rapide
    B. Capacité à modéliser des relations non linéaires
    C. Toujours moins de paramètres
    D. Pas besoin d’entraînement

15. L’algorithme de descente de gradient peut être :
    A. Stochastique, batch ou mini-batch
    B. Exact uniquement
    C. Aléatoire uniquement
    D. Sans fonction de coût

16. Vrai ou Faux : plus il y a de neurones dans un MLP, meilleure sera toujours la généralisation.

17. Le taux d’apprentissage influence :
    A. La rapidité et la stabilité de la convergence
    B. La taille du dataset
    C. La complexité du réseau
    D. Le nombre de classes

18. Un MLP est particulièrement adapté pour :
    A. La vision par ordinateur uniquement
    B. Des données tabulaires et séquentielles simples
    C. Les signaux audio uniquement
    D. Les réseaux bayésiens

19. L’activation ReLU est préférée car :
    A. Elle est non linéaire et évite le problème du gradient qui disparaît
    B. Elle est symétrique
    C. Elle n’a pas besoin de biais
    D. Elle supprime la normalisation

20. Vrai ou Faux : un MLP peut servir de base à des réseaux plus complexes comme les CNN ou RNN.