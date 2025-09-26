import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ========================================
# PARTIE 1 : PRÉPARATION DES DONNÉES
# ========================================

# Corpus d'exemple simple en français
corpus = [
    ["le", "chat", "mange", "du", "poisson"],
    ["le", "chien", "mange", "de", "la", "viande"],
    ["les", "animaux", "domestiques", "sont", "adorables"],
    ["le", "poisson", "nage", "dans", "l'eau"],
    ["le", "chat", "et", "le", "chien", "sont", "des", "animaux"],
    ["la", "nourriture", "pour", "animaux", "est", "importante"],
    ["le", "chat", "dort", "sur", "le", "canapé"],
    ["le", "chien", "joue", "dans", "le", "jardin"],
    ["les", "poissons", "vivent", "dans", "l'aquarium"],
    ["nourrir", "les", "animaux", "chaque", "jour"]
]

# print("Corpus d'entraînement créé avec", len(corpus), "phrases")


model = Word2Vec(sentences=corpus, vector_size=50, window=3, min_count=1, workers=4, sg=0)
# print(f"Modèle Word2Vec entraîné. Vocabulaire de taille : {len(model.wv.key_to_index)} mots")

# print((f"Exemple de vecteur pour le mot 'dort' : {model.wv['dort']}"))


def calculer_similarite(mot1, mot2, model):
    if mot1 in model.wv and mot2 in model.wv:
        similarite = model.wv.similarity(mot1, mot2)
        return similarite
    else:
        return None

print(calculer_similarite("poisson", "poissons", model))

def mots_similaires(mot, model, top_n=3):
    """Trouve les mots les plus similaires à un mot donné"""
    if mot in model.wv:
        similaires = model.wv.most_similar(mot, topn=top_n)
        return similaires
    else:
        return None

print(f"\n MOTS LES PLUS SIMILAIRES")
print("=" * 30)

mots_test = ["chat", "mange", "animaux"]
for mot in mots_test:
    similaires = mots_similaires(mot, model)
    if similaires:
        print(f"\nMots similaires à '{mot}':")
        for mot_sim, score in similaires:
            print(f"  - {mot_sim}: {score:.3f}")

print(f"\n EXERCICES PRATIQUES")
print("=" * 25)

print("\n1. Complétez cette analogie : 'chat' est à 'chien' comme 'poisson' est à... ?")

print(f"\n2. Classez ces mots par similarité avec 'animal':")