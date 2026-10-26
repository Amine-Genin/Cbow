# ==============================================================================
# vocabulary.py (Construction du Vocabulaire et Encodage Numérique TALN)
# ==============================================================================
# Rôle :
#   En Traitement Automatique du Langage Naturel (TALN), les modèles
#   mathématiques (tels que Word2Vec, CBOW, Skip-Gram ou réseaux de neurones)
#   ne peuvent pas traiter directement des chaînes de caractères.
#   
#   Ce script intervient après la phase de prétraitement (prepare.py) :
#   - Il reçoit le corpus tokenisé, issu d'un grand texte brut (paragraphes)
#     débarrassé de tous ses bruits et caractères spéciaux (/[=-"|~@#$%&^*()_+-{; etc.).
#   - Il extrait le vocabulaire unique V (ensemble des mots réels distincts).
#   - Il crée les tables bijectives mot <-> indice (pour l'entrée et la sortie du modèle).
#   - Il encode les séquences de mots en vecteurs d'indices numériques.
# ==============================================================================

# Importation de la fonction de tokenisation développée dans prepare.py
from prepare import tokeniser_corpus


# ------------------------------------------------------------------------------
# 1. EXTRACTION DU VOCABULAIRE UNIQUE
# ------------------------------------------------------------------------------
def construire_vocabulaire(corpus_tokenise):
    """
    Extrait l'ensemble des mots uniques apparaissant dans le corpus nettoyé,
    en conservant leur ordre de première apparition.
    
    Principe algorithmique :
    - On parcourt chaque mot de chaque phrase tokenisée.
    - On maintient une table de hachage 'deja_vu' (dictionnaire) pour tester
      l'existence d'un mot en temps constant O(1).
    - Si le mot n'a jamais été rencontré, on l'ajoute à la liste 'vocabulaire'
      et on le marque comme vu dans 'deja_vu'.
    
    Paramètre :
        corpus_tokenise (list[list[str]]) : Corpus sous forme de liste de phrases tokenisées.
    Retour :
        vocabulaire (list[str]) : Liste ordonnée de tous les mots distincts du corpus.
    """
    vocabulaire = []
    deja_vu = {}  # Table de hachage servant de filtre rapide pour éviter les doublons

    for phrase in corpus_tokenise:
        for mot in phrase:
            if mot not in deja_vu:
                deja_vu[mot] = True        # Marquer le mot comme enregistré
                vocabulaire.append(mot)    # Ajouter au vocabulaire ordonné
                
    return vocabulaire


# ------------------------------------------------------------------------------
# 2. CRÉATION DES DICTIONNAIRES D'INDEXATION
# ------------------------------------------------------------------------------
def creer_dictionnaires(vocabulaire):
    """
    Construit les deux tables de correspondance bijectives nécessaires au modèle.
    
    1. mot_vers_indice (dict: str -> int) :
       Permet d'encoder les mots textuels en identifiants numériques
       pour alimenter les couches d'entrée ou matrices d'embeddings.
       Exemple : {"université": 0, "mohammed": 1, ...}
       
    2. indice_vers_mot (dict: int -> str) :
       Permet de décoder les prédictions du modèle (indices) pour retrouver
       le mot textuel correspondant (ex: après un softmax ou argmax).
       Exemple : {0: "université", 1: "mohammed", ...}
    
    Paramètre :
        vocabulaire (list[str]) : Liste des mots uniques.
    Retour :
        tuple(dict, dict) : (mot_vers_indice, indice_vers_mot)
    """
    mot_vers_indice = {}
    indice_vers_mot = {}
    indice = 0

    for mot in vocabulaire:
        mot_vers_indice[mot] = indice
        indice_vers_mot[indice] = mot
        indice = indice + 1

    return mot_vers_indice, indice_vers_mot


# ------------------------------------------------------------------------------
# 3. ENCODAGE : CONVERSION DE TOKENS EN INDICES
# ------------------------------------------------------------------------------
def tokens_vers_indices(tokens, mot_vers_indice):
    """
    Convertit une liste de tokens (mots) en une séquence d'indices entiers.
    
    Exemple :
        tokens          = ["oujda", "est", "une", "ville"]
        mot_vers_indice = {"oujda": 10, "est": 11, "une": 12, "ville": 13}
        -> indices      = [10, 11, 12, 13]
    
    Paramètres :
        tokens (list[str]) : Liste de mots d'une phrase.
        mot_vers_indice (dict) : Dictionnaire associant chaque mot à son entier.
    Retour :
        indices (list[int]) : Liste des entiers correspondants.
    """
    indices = []
    for mot in tokens:
        indices.append(mot_vers_indice[mot])
    return indices


# ------------------------------------------------------------------------------
# 4. BLOC DE TEST ET EXÉCUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Étape 1 : Chargement, nettoyage des erreurs, segmentation et tokenisation
    print("Chargement et traitement du corpus brut 'Campus_Aprentissage.txt'...")
    corpus = tokeniser_corpus("Campus_Aprentissage.txt")
    
    # Étape 2 : Construction du vocabulaire unique
    vocabulaire = construire_vocabulaire(corpus)
    
    # Étape 3 : Création des tables de mappage direct et inverse
    mot_vers_indice, indice_vers_mot = creer_dictionnaires(vocabulaire)

    # Affichage des métriques et vérifications
    print("\n=== Analyse du Vocabulaire du Corpus ===")
    print("Nombre de phrases extraites :", len(corpus))
    print("Taille totale du vocabulaire (mots uniques) :", len(vocabulaire))
    print("Les 10 premiers mots du vocabulaire :", vocabulaire[0:10])
    print()
    
    print("=== Exemples d'accès aux dictionnaires ===")
    mots_tests = ["université", "oujda", "maroc", "étudiants", "professeur"]
    for m in mots_tests:
        if m in mot_vers_indice:
            print(f"mot_vers_indice['{m}'] = {mot_vers_indice[m]}")
    print()
    print("Mot correspondant à l'indice 0 :", indice_vers_mot[0])
    print()

    # Démonstration de l'encodage numérique sur la première phrase
    print("=== Démonstration de l'encodage numérique sur la 1ère phrase ===")
    if len(corpus) > 0:
        premiere_phrase_tokens = corpus[0]
        premiere_phrase_indices = tokens_vers_indices(premiere_phrase_tokens, mot_vers_indice)
        print("Phrase 1 (tokens)  :", premiere_phrase_tokens)
        print("Phrase 1 (indices) :", premiere_phrase_indices)
