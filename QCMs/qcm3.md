# **Chapitre 4 : Réseaux récurrents (RNN, LSTM)**

1. Un réseau récurrent est adapté pour :
   A. Les images fixes
   B. Les données séquentielles
   C. Les graphes
   D. Les données tabulaires uniquement

2. La particularité d’un RNN est :
   A. Il n’utilise pas de poids
   B. Il garde une mémoire des états précédents
   C. Il fonctionne sans activation
   D. Il ne fait pas de rétropropagation

3. Vrai ou Faux : les RNN classiques souffrent du problème du gradient qui disparaît.

4. LSTM signifie :
   A. Long Short-Term Memory
   B. Large Scale Training Model
   C. Linear Sequence Time Model
   D. Logistic Sequential Training Mechanism

5. L’architecture LSTM inclut :
   A. Des portes (input, forget, output)
   B. Uniquement des convolutions
   C. Un autoencodeur
   D. Des perceptrons uniquement

6. La porte d’oubli (forget gate) sert à :
   A. Supprimer toujours l’information
   B. Contrôler quelles informations du passé garder ou oublier
   C. Ajouter du bruit
   D. Normaliser la séquence

7. La porte d’entrée (input gate) décide :
   A. Quelle nouvelle information ajouter à l’état mémoire
   B. Quelle sortie donner directement
   C. Quels poids supprimer
   D. Quelle séquence ignorer

8. La porte de sortie (output gate) contrôle :
   A. Le gradient
   B. La sortie de l’état caché
   C. La normalisation
   D. La taille du batch

9. Vrai ou Faux : un LSTM peut gérer des dépendances longues.

10. L’entraînement d’un LSTM utilise :
    A. Backpropagation Through Time (BPTT)
    B. Rétropropagation simple
    C. Pas d’entraînement
    D. Uniquement des heuristiques

11. Les applications typiques d’un LSTM incluent :
    A. La traduction automatique
    B. La classification d’images fixes
    C. La compression vidéo
    D. La détection de contours

12. Le GRU est :
    A. Une variante simplifiée du LSTM
    B. Un CNN
    C. Un autoencodeur
    D. Une fonction d’activation

13. Vrai ou Faux : un LSTM est plus rapide qu’un GRU.

14. Le problème du gradient explosif est traité par :
    A. Gradient clipping
    B. Dropout
    C. Augmentation de données
    D. Max pooling

15. Un état caché (hidden state) :
    A. Contient une représentation des séquences précédentes
    B. Est identique à l’entrée
    C. Est inutile pour la sortie
    D. Est remplacé par le biais

16. LSTM est particulièrement utile pour :
    A. Les séries temporelles
    B. Les images fixes
    C. Les graphes
    D. Les données tabulaires

17. La sortie finale d’un LSTM peut être :
    A. Une valeur unique ou une séquence
    B. Toujours une valeur unique
    C. Toujours une probabilité
    D. Toujours un vecteur

18. Vrai ou Faux : un LSTM peut être combiné avec un CNN.

19. Un bi-LSTM lit :
    A. Les séquences uniquement en avant
    B. Les séquences dans les deux sens
    C. Les images
    D. Les graphes

20. Le mécanisme d’attention peut être combiné avec LSTM pour :
    A. Se concentrer sur les parties importantes d’une séquence
    B. Remplacer les fonctions d’activation
    C. Supprimer la rétropropagation
    D. Réduire la taille des filtres