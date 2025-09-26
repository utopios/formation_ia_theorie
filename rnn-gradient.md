* Longueur de séquence $T = 5$,
* $W = 0.9$,
* Fonction d’activation = tanh → dérivée typique $\sigma'(z) = 0.5$.

Alors :

$$
\frac{\partial h^{(2)}}{\partial h^{(1)}} = 0.9 \times 0.5 = 0.45
$$

$$
\frac{\partial h^{(3)}}{\partial h^{(2)}} = 0.9 \times 0.5 = 0.45
$$

Pareil pour les autres.

Le produit final :

$$
\prod_{k=2}^{5} 0.45 = 0.45 \times 0.45 \times 0.45 \times 0.45 = 0.041
$$

👉 Le gradient qui remonte du pas 5 au pas 1 est déjà divisé par 25 environ → il **s’éteint**.