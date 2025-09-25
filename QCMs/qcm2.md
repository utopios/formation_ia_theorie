# **Chapitre 3 : Convolutional Neural Networks (CNN)**

1. Une couche de convolution (CONV) sert à :
   A. Appliquer une transformation linéaire complète
   B. Extraire des motifs locaux
   C. Remplacer la fonction de perte
   D. Réduire la taille du dataset

2. Le filtre (kernel) est :
   A. Une matrice de poids partagée
   B. Une fonction d’activation
   C. Un biais fixe
   D. Une normalisation

3. Vrai ou Faux : un même filtre est appliqué à toutes les régions d’une image.

4. Le pooling sert à :
   A. Augmenter la dimensionnalité
   B. Réduire la dimensionnalité et extraire les caractéristiques robustes
   C. Supprimer les biais
   D. Normaliser les entrées

5. ReLU est souvent utilisée dans les CNN car :
   A. Elle est toujours convexe
   B. Elle accélère l’apprentissage et évite le problème du gradient qui disparaît
   C. Elle réduit automatiquement la dimensionnalité
   D. Elle remplace le pooling

6. Une couche entièrement connectée (FC) en fin de CNN sert à :
   A. Extraire les motifs
   B. Décider de la sortie finale (classe, score, etc.)
   C. Remplacer les filtres
   D. Supprimer la normalisation

7. Vrai ou Faux : le surapprentissage peut être réduit par la technique du dropout.

8. Le choix du nombre de filtres dans un CNN dépend :
   A. Du nombre de classes uniquement
   B. De la complexité des motifs à apprendre
   C. De la taille du batch
   D. Du taux d’apprentissage

9. Le max pooling conserve :
   A. La valeur maximale d’une région
   B. La valeur moyenne
   C. Le biais
   D. Le gradient

10. Exemple de régularisation explicite :
    A. Dropout
    B. Weight decay
    C. Data augmentation
    D. Normalisation batch

11. Vrai ou Faux : augmenter artificiellement les données améliore la généralisation.

12. La couche de perte (loss) mesure :
    A. L’écart entre prédictions et vraies valeurs
    B. Le nombre de couches
    C. La vitesse d’exécution
    D. La taille de l’image

13. Un CNN est particulièrement adapté pour :
    A. Les images et signaux structurés
    B. Les tableaux de données simples
    C. Les graphes sociaux
    D. Les systèmes bayésiens

14. L’augmentation de la taille du réseau (plus de filtres, couches) entraîne :
    A. Toujours une meilleure généralisation
    B. Plus de capacité d’apprentissage mais aussi risque d’overfitting
    C. Une suppression de la perte
    D. Une simplification des données

15. Vrai ou Faux : un CNN est invariant aux translations grâce aux convolutions.

16. Un filtre de Sobel est utilisé pour :
    A. Détecter des contours
    B. Réduire la dimension
    C. Supprimer les biais
    D. Augmenter la précision

17. La régularisation L2 est équivalente à :
    A. Dropout
    B. Weight decay
    C. Batch normalization
    D. Max pooling

18. Le choix de la taille du filtre dépend :
    A. De la complexité des motifs à capter
    B. Du nombre de classes
    C. Du type de fonction de perte
    D. Du biais

19. La normalisation batch sert à :
    A. Stabiliser l’entraînement
    B. Supprimer la rétropropagation
    C. Réduire la taille du réseau
    D. Remplacer la perte

20. Vrai ou Faux : les CNN modernes utilisent souvent des blocs empilés (VGG, ResNet).
