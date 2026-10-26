# ==============================================================================
# cbow.py : Modèle CBOW (Continuous Bag of Words) codé à la main "from scratch"
# ==============================================================================
# Basé sur le document de cours : "Calcul du gradient dans le modèle CBOW"
# - Couche Softmax et fonction de coût Cross-Entropy
# - Pas de bibliothèques externes de Machine Learning (ni NumPy, ni PyTorch, ni TensorFlow)
# - Tout est implémenté en Python pur (listes, boucles et opérations élémentaires)
# - La taille du contexte 'k' (demi-fenêtre) est choisie dynamiquement par l'utilisateur
# ==============================================================================

import math
import random

# Importation du prétraitement et de la gestion du vocabulaire
from prepare import tokeniser_corpus
from vocabulary import construire_vocabulaire, creer_dictionnaires, tokens_vers_indices


# ------------------------------------------------------------------------------
# 1. FONCTIONS MATHÉMATIQUES ET ALGÉBRIQUES ÉLÉMENTAIRES (SANS BIBLIOTHÈQUE)
# ------------------------------------------------------------------------------
def initialiser_matrice(lignes, colonnes, echelle=0.01):
    """
    Crée une matrice de dimensions (lignes x colonnes) avec de petites valeurs
    aléatoires uniformes centrées autour de 0.
    """
    matrice = []
    for _ in range(lignes):
        ligne = []
        for _ in range(colonnes):
            valeur = (random.random() - 0.5) * 2 * echelle
            ligne.append(valeur)
        matrice.append(ligne)
    return matrice


def exponentielle(x):
    """Calcule exp(x) avec seuillage pour éviter les débordements numériques (overflow)."""
    if x > 50.0:
        x = 50.0
    elif x < -50.0:
        x = -50.0
    return math.exp(x)


def fonction_softmax(scores):
    """
    Calcule la distribution de probabilité Softmax :
        y_hat[j] = exp(y[j]) / sum(exp(y[k]))
    Avec soustraction du maximum max(y) pour garantir la stabilité numérique.
    """
    # Recherche du maximum à la main
    max_val = scores[0]
    for s in scores:
        if s > max_val:
            max_val = s

    # Calcul des exponentielles décalées
    exp_scores = []
    somme_exp = 0.0
    for s in scores:
        e = exponentielle(s - max_val)
        exp_scores.append(e)
        somme_exp = somme_exp + e

    # Normalisation
    y_chapeau = []
    for e in exp_scores:
        y_chapeau.append(e / somme_exp)

    return y_chapeau


def produit_scalaire(v1, v2):
    """Calcule le produit scalaire entre deux vecteurs de même taille."""
    total = 0.0
    for i in range(len(v1)):
        total = total + v1[i] * v2[i]
    return total


def similarite_cosinus(v1, v2):
    """Calcule la similarité cosinus entre deux vecteurs : (v1 . v2) / (||v1|| * ||v2||)."""
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(len(v1)):
        dot = dot + v1[i] * v2[i]
        norm1 = norm1 + v1[i] * v1[i]
        norm2 = norm2 + v2[i] * v2[i]
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


# ------------------------------------------------------------------------------
# 2. GÉNÉRATION DES DONNÉES D'ENTRAÎNEMENT CBOW AVEC FENÊTRE 'k'
# ------------------------------------------------------------------------------
def generer_donnees_cbow(corpus_indices, k):
    """
    Génère l'ensemble des paires d'entraînement (contexte, cible) :
    - Pour chaque position t d'une phrase, le mot cible est w_t.
    - Le contexte est constitué des mots dans la fenêtre [t - k, t + k], à l'exclusion de w_t.
    - C = nombre de mots effectifs dans le contexte (peut être inférieur à 2*k sur les bords).
    
    Retour :
        donnees (list) : Liste de tuples ([indice_c1, indice_c2, ...], indice_cible).
    """
    donnees = []
    for phrase in corpus_indices:
        longueur = len(phrase)
        if longueur < 2:
            continue
        for t in range(longueur):
            cible = phrase[t]
            contexte = []

            # Mots à gauche du mot cible : de t-k à t-1
            debut = t - k
            if debut < 0:
                debut = 0
            for i in range(debut, t):
                contexte.append(phrase[i])

            # Mots à droite du mot cible : de t+1 à t+k
            fin = t + k + 1
            if fin > longueur:
                fin = longueur
            for i in range(t + 1, fin):
                contexte.append(phrase[i])

            # Si le contexte contient au moins un mot
            if len(contexte) > 0:
                donnees.append((contexte, cible))

    return donnees


# ------------------------------------------------------------------------------
# 3. CLASSE DU MODÈLE CBOW (CONFORME AUX ÉQUATIONS DU COURS)
# ------------------------------------------------------------------------------
class ModeleCBOW:
    """
    Implémentation mathématique exacte du document :
    - Matrice d'entrée (embeddings) : W  de dimension (V x d)
    - Matrice de sortie             : W' de dimension (d x V)
    - Contexte                      : C mots {w_1, ..., w_C}
    - Vecteur caché                 : h = (1/C) * sum(h_i) = (1/C) * sum(W^T * x_i)
    - Scores                        : y = W'^T * h  (vecteur de taille V)
    - Prédiction Softmax            : y_hat = softmax(y)
    - Coût Cross-Entropy            : L = - log(y_hat[cible])
    """

    def __init__(self, taille_vocabulaire, dimension_embedding=20, taux_apprentissage=0.025):
        self.V = taille_vocabulaire
        self.d = dimension_embedding
        self.mu = taux_apprentissage

        # 1. Matrice W de taille (V x d) : chaque ligne W[i] est l'embedding du mot i
        self.W = initialiser_matrice(self.V, self.d, echelle=0.1)

        # 2. Matrice W' de taille (d x V) : correspond exactement à la notation W' de cours
        self.W_prime = initialiser_matrice(self.d, self.V, echelle=0.1)

    def forward(self, contexte_indices):
        """
        Passe avant (Forward Pass) :
        1. h_i = W^T * x_i  => comme x_i est one-hot, h_i est simplement la ligne W[c_i].
        2. h = (1 / C) * sum_{i=1}^C h_i
        3. y = W'^T * h     => y[j] = sum_{m=0}^{d-1} W_prime[m][j] * h[m]
        4. y_hat = softmax(y)
        """
        C = len(contexte_indices)

        # Étape 1 & 2 : Calcul du vecteur caché h (moyenne des embeddings du contexte)
        h = [0.0] * self.d
        for idx in contexte_indices:
            vecteur_mot = self.W[idx]
            for m in range(self.d):
                h[m] = h[m] + vecteur_mot[m]

        for m in range(self.d):
            h[m] = h[m] / C

        # Étape 3 : Calcul des scores de sortie y = W'^T * h
        # Pour chaque mot j du vocabulaire (j de 0 à V-1)
        y = [0.0] * self.V
        for j in range(self.V):
            score_j = 0.0
            for m in range(self.d):
                score_j = score_j + self.W_prime[m][j] * h[m]
            y[j] = score_j

        # Étape 4 : Probabilités Softmax
        y_hat = fonction_softmax(y)

        return h, y_hat

    def backward_et_mise_a_jour(self, contexte_indices, indice_cible, h, y_hat):
        """
        Rétropropagation du gradient et mise à jour des poids selon le tableau du cours :
        1. Gradient des scores          : dL/dy = y_hat - u
        2. Gradient de W'              : dL/dW' = h * (y_hat - u)^T
        3. Gradient de h               : dL/dh  = W' * (y_hat - u)
        4. Gradient de chaque embedding: dL/dh_i = (1/C) * W' * (y_hat - u)
        5. Gradient de W               : dL/dW  = (1/C) * sum_{i=1}^C x_i * [W' * (y_hat - u)]^T
        6. Mises à jour                : W' <- W' - mu * (dL/dW')
                                         W  <- W  - mu * (dL/dW)
        """
        C = len(contexte_indices)

        # 1. Gradient des scores : e[j] = y_hat[j] - u[j]
        # Comme u est un vecteur one-hot : u[cible] = 1, et u[j] = 0 pour j != cible
        e = [0.0] * self.V
        for j in range(self.V):
            e[j] = y_hat[j]
        e[indice_cible] = e[indice_cible] - 1.0

        # 2. Gradient de h : dL/dh = W' * (y_hat - u)
        # Vecteur de dimension d : dL_dh[m] = sum_{j=0}^{V-1} W_prime[m][j] * e[j]
        dL_dh = [0.0] * self.d
        for m in range(self.d):
            val = 0.0
            for j in range(self.V):
                val = val + self.W_prime[m][j] * e[j]
            dL_dh[m] = val

        # 3. Mise à jour de W' : W' <- W' - mu * (h * e^T)
        # Pour chaque m dans 0..d-1 et chaque j dans 0..V-1
        for m in range(self.d):
            h_m = h[m]
            for j in range(self.V):
                grad_W_prime = h_m * e[j]
                self.W_prime[m][j] = self.W_prime[m][j] - self.mu * grad_W_prime

        # 4. Mise à jour de W : W[c_i] <- W[c_i] - mu * (1/C) * dL/dh
        # Pour chaque mot du contexte c_i
        coeff = self.mu / C
        for idx in contexte_indices:
            for m in range(self.d):
                self.W[idx][m] = self.W[idx][m] - coeff * dL_dh[m]

        # Calcul du coût (Cross-Entropy) : L = -log(y_hat[cible])
        proba_cible = y_hat[indice_cible]
        if proba_cible < 1e-12:
            proba_cible = 1e-12
        loss = -math.log(proba_cible)

        return loss

    def entrainer(self, donnees, epochs=20):
        """Boucle d'entraînement sur le jeu de données généré."""
        print(f"\nDébut de l'entraînement sur {epochs} époques ({len(donnees)} exemples par époque)...")
        for ep in range(1, epochs + 1):
            perte_totale = 0.0
            # Mélange aléatoire des exemples à la main (Fisher-Yates shuffle)
            n_exemples = len(donnees)
            for i in range(n_exemples - 1, 0, -1):
                j = int(random.random() * (i + 1))
                donnees[i], donnees[j] = donnees[j], donnees[i]

            for contexte, cible in donnees:
                # 1. Forward
                h, y_hat = self.forward(contexte)
                # 2. Backward & Update
                loss = self.backward_et_mise_a_jour(contexte, cible, h, y_hat)
                perte_totale = perte_totale + loss

            perte_moyenne = perte_totale / len(donnees)
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                print(f"  Époque {ep:2d}/{epochs} | Perte moyenne (Loss) : {perte_moyenne:.4f}")

    def get_embedding(self, indice_mot):
        """Renvoie le vecteur plongement (embedding) associé à un mot."""
        return self.W[indice_mot]

    def mots_les_plus_proches(self, mot_requete, mot_vers_indice, indice_vers_mot, top_n=5):
        """Recherche les mots les plus similaires par similarité cosinus."""
        if mot_requete not in mot_vers_indice:
            return []

        idx_ref = mot_vers_indice[mot_requete]
        v_ref = self.get_embedding(idx_ref)

        scores = []
        for idx in range(self.V):
            if idx == idx_ref:
                continue
            v_cand = self.get_embedding(idx)
            sim = similarite_cosinus(v_ref, v_cand)
            scores.append((sim, indice_vers_mot[idx]))

        # Tri décroissant à la main (tri par sélection / insertion sans bibliothèque)
        for i in range(len(scores)):
            max_idx = i
            for j in range(i + 1, len(scores)):
                if scores[j][0] > scores[max_idx][0]:
                    max_idx = j
            scores[i], scores[max_idx] = scores[max_idx], scores[i]

        return scores[:top_n]


# ------------------------------------------------------------------------------
# 4. PROGRAMME PRINCIPAL ET INTERACTION UTILISATEUR
# ------------------------------------------------------------------------------
def main():
    print("====================================================================")
    print("  IMPLÉMENTATION COMPLÈTE DU MODÈLE CBOW (FROM SCRATCH)")
    print("  Calcul du Gradient, Softmax et Cross-Entropy")
    print("====================================================================")

    # 1. Choix dynamique de 'k' (demi-fenêtre de contexte) par l'utilisateur
    print("\n[Étape 1] Configuration de la fenêtre de contexte 'k' :")
    valeur_k_valide = False
    k = 2  # Valeur par défaut de repli

    while not valeur_k_valide:
        try:
            entree = input(">> Veuillez entrer la valeur de k (ex: 2, 3, 4) [Défaut = 2] : ").strip()
            if entree == "":
                k = 2
                valeur_k_valide = True
            else:
                k = int(entree)
                if k >= 1:
                    valeur_k_valide = True
                else:
                    print("   Erreur : k doit être un entier supérieur ou égal à 1.")
        except (ValueError, EOFError):
            print("   Entrée invalide ou non interactive. Utilisation de la valeur par défaut k = 2.")
            k = 2
            valeur_k_valide = True

    print(f"--> Demi-fenêtre de contexte sélectionnée : k = {k} (contexte maximum C = {2 * k} mots)")

    # 2. Prétraitement et chargement du corpus
    print("\n[Étape 2] Chargement et prétraitement du corpus 'Campus_Aprentissage.txt'...")
    corpus = tokeniser_corpus("Campus_Aprentissage.txt")
    print(f"--> Nombre de phrases extraites : {len(corpus)}")

    # 3. Construction du vocabulaire
    vocabulaire = construire_vocabulaire(corpus)
    mot_vers_indice, indice_vers_mot = creer_dictionnaires(vocabulaire)
    V = len(vocabulaire)
    print(f"--> Taille du vocabulaire unique V : {V} mots")

    # 4. Conversion des phrases en indices
    corpus_indices = []
    for phrase_tokens in corpus:
        corpus_indices.append(tokens_vers_indices(phrase_tokens, mot_vers_indice))

    # 5. Génération des paires (contexte, cible) selon 'k'
    print(f"\n[Étape 3] Génération des exemples CBOW avec demi-fenêtre k = {k}...")
    donnees = generer_donnees_cbow(corpus_indices, k=k)
    print(f"--> Nombre total d'exemples d'apprentissage (contexte, cible) : {len(donnees)}")

    if len(donnees) > 0:
        contexte_demo, cible_demo = donnees[0]
        mots_ctx = [indice_vers_mot[i] for i in contexte_demo]
        print(f"--> Exemple 1 : Contexte {mots_ctx} ==> Cible : '{indice_vers_mot[cible_demo]}'")

    # 6. Initialisation du modèle CBOW
    d = 25          # Dimension des embeddings
    lr = 0.05       # Taux d'apprentissage mu
    epochs = 25     # Nombre d'itérations
    print(f"\n[Étape 4] Initialisation du modèle CBOW :")
    print(f"    - Taille vocabulaire V = {V}")
    print(f"    - Dimension embedding d = {d}")
    print(f"    - Matrice W  : ({V} x {d})")
    print(f"    - Matrice W' : ({d} x {V})")
    print(f"    - Taux d'apprentissage mu = {lr}")

    modele = ModeleCBOW(taille_vocabulaire=V, dimension_embedding=d, taux_apprentissage=lr)

    # 7. Entraînement
    modele.entrainer(donnees, epochs=epochs)

    # 8. Test sémantique des embeddings appris
    print("\n====================================================================")
    print("  TEST DES PLONGEMENTS SÉMANTIQUES (MOTS LES PLUS PROCHES)")
    print("====================================================================")
    mots_a_tester = ["université", "étudiants", "professeur", "recherche", "informatique", "maroc"]

    for mot in mots_a_tester:
        if mot in mot_vers_indice:
            voisins = modele.mots_les_plus_proches(mot, mot_vers_indice, indice_vers_mot, top_n=4)
            voisins_str = ", ".join([f"{m} ({sim:.3f})" for sim, m in voisins])
            print(f"  Mots les plus proches de '{mot}' : {voisins_str}")


if __name__ == "__main__":
    main()

