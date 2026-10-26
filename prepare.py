# ==============================================================================
# prepare.py (Nettoyage de Texte Brut Bruit頻, Filtrage et Tokenisation TALN)
# ==============================================================================
# Rôle :
#   Ce script prend en charge le traitement d'un corpus brut fortement bruité,
#   contenant du texte réel mélangé à des caractères spéciaux et des artefacts
#   de balisage ou de formatage :
#     /[=-"|~@#$%&^*()_+-{;...... \ < > ] [ } etc.
#
#   Pipeline de traitement :
#   1. Définition des tables de caractères valides (lettres françaises accentuées).
#   2. Filtrage et élimination de tous les symboles et bruits typographiques.
#   3. Normalisation des espaces et de la ponctuation de fin de phrase (. ! ?).
#   4. Segmentation intelligente du texte brut en phrases cohérentes.
#   5. Tokenisation stricte ne conservant que les mots réels (lettres uniquement).
#   6. Exclusion automatique des lignes de séparation (===, ---, ***, etc.)
#      et des identifiants techniques ([SECTION-1], #ID, etc.).
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. TABLES DE CARACTÈRES (LETTRES VALIDES ET BRUITS)
# ------------------------------------------------------------------------------
MAJUSCULES = "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ"
MINUSCULES = "abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüç"

# Ponctuations fortes marquant la fin d'une phrase
PONCTUATION_FIN = ".!?"

# Caractères de bruit à filtrer / remplacer par des espaces :
# Comprend tous les symboles demandés : /[=-"|~@#$%&^*()_+-{; ainsi que \], }, >, <, etc.
CARACTERES_BRUIT = "/[=-" + '"' + "|~@#$%&^*()_+-{;}\\]<>:`«»"


# ------------------------------------------------------------------------------
# 2. FONCTIONS DE VALIDATION ET NORMALISATION DE BASE
# ------------------------------------------------------------------------------
def est_lettre(c):
    """Vérifie si un caractère est une lettre de l'alphabet (avec accents français)."""
    return (c in MINUSCULES) or (c in MAJUSCULES)


def mettre_en_minuscules(texte):
    """
    Transforme chaque majuscule en minuscule (support complet des accents).
    Tout autre caractère reste inchangé.
    """
    resultat = ""
    for c in texte:
        i = 0
        trouve = False
        while i < len(MAJUSCULES) and not trouve:
            if c == MAJUSCULES[i]:
                resultat = resultat + MINUSCULES[i]
                trouve = True
            i = i + 1
        if not trouve:
            resultat = resultat + c
    return resultat


# ------------------------------------------------------------------------------
# 3. ÉLIMINATION DU BRUIT ET NORMALISATION DU TEXTE BRUT
# ------------------------------------------------------------------------------
def nettoyer_bruit_texte(texte_brut):
    """
    Nettoie le texte brut en éliminant les caractères parasites et en normalisant
    les délimiteurs de phrases.
    
    Opérations :
    - Remplace les apostrophes alternatives (’ ‘ `) par l'apostrophe standard '.
    - Remplace les caractères de bruit (/[=-"|~@#$%&^*()_+-{; etc.) par un espace.
    - Condense les séquences de points répétés (......) en un seul point de fin de phrase.
    - Élimine les espaces multiples.
    """
    # Étape 1 : Normalisation des apostrophes
    texte_norm = ""
    for c in texte_brut:
        if c == "’" or c == "‘" or c == "`":
            texte_norm = texte_norm + "'"
        else:
            texte_norm = texte_norm + c

    # Étape 2 : Remplacement des caractères de bruit par un espace
    # On préserve uniquement : les lettres, les chiffres, l'apostrophe, les fins de phrase (. ! ?) et les sauts de ligne
    sans_bruit = ""
    for c in texte_norm:
        if est_lettre(c) or c == "'" or c in PONCTUATION_FIN or c == "\n" or c == " " or c == ",":
            sans_bruit = sans_bruit + c
        else:
            # Tout autre symbole (bruit, chiffres résiduels, crochets, etc.) devient un espace
            sans_bruit = sans_bruit + " "

    # Étape 3 : Réduction des ponctuations répétées (ex: "......" ou "!!!" -> "." ou "!")
    ponct_propre = ""
    n = len(sans_bruit)
    i = 0
    while i < n:
        c = sans_bruit[i]
        if c in PONCTUATION_FIN:
            # Si le caractère suivant est identique, on avance jusqu'au dernier
            while i + 1 < n and sans_bruit[i + 1] in PONCTUATION_FIN:
                i = i + 1
            ponct_propre = ponct_propre + c + " "
        else:
            ponct_propre = ponct_propre + c
        i = i + 1

    # Étape 4 : Réduction des espaces et tabulations multiples
    final = ""
    dernier_espace = False
    for c in ponct_propre:
        if c == " " or c == "\t" or c == "\r":
            if not dernier_espace:
                final = final + " "
                dernier_espace = True
        else:
            final = final + c
            dernier_espace = False

    return final


# ------------------------------------------------------------------------------
# 4. SEGMENTATION DU TEXTE BRUT EN PHRASES COHÉRENTES
# ------------------------------------------------------------------------------
def segmenter_en_phrases(texte_brut):
    """
    Découpe le texte nettoyé en une liste de phrases signifiantes.
    
    Critères :
    - Découpe sur les ponctuations de fin (. ! ?).
    - Découpe sur les sauts de ligne de paragraphes.
    - Ignore les fragments ne contenant pas au moins 2 vrais mots (ex: lignes de bruit vides).
    """
    texte_nettoye = nettoyer_bruit_texte(texte_brut)

    phrases = []
    phrase_en_cours = ""

    for c in texte_nettoye:
        phrase_en_cours = phrase_en_cours + c

        # Fin de phrase détectée
        if c in PONCTUATION_FIN or c == "\n":
            segment = phrase_en_cours.strip()
            # On vérifie si ce segment contient des mots valides (au moins 5 lettres au total)
            nb_lettres = sum(1 for car in segment if est_lettre(car))
            if nb_lettres >= 5:
                phrases.append(segment)
            phrase_en_cours = ""

    # Reliquat final éventuel
    reste = phrase_en_cours.strip()
    if sum(1 for car in reste if est_lettre(car)) >= 5:
        phrases.append(reste)

    return phrases


# ------------------------------------------------------------------------------
# 5. TOKENISATION D'UNE PHRASE EN MOTS VALIDES
# ------------------------------------------------------------------------------
def tokeniser_phrase(phrase):
    """
    Extrait les tokens (mots) d'une phrase nettoyée.
    
    Règles :
    - Mise en minuscules systématique.
    - Découpe sur tout caractère qui n'est pas une lettre (espaces, apostrophes, ponctuation).
    - L'apostrophe sépare le mot (ex: "l'université" -> ["l", "université"]).
    - Ne conserve que les tokens composés exclusivement de lettres valides.
    """
    phrase_min = mettre_en_minuscules(phrase)
    tokens = []
    mot = ""

    for c in phrase_min:
        if est_lettre(c):
            mot = mot + c
        else:
            # Dès qu'on rencontre un séparateur (espace, apostrophe, virgule, point...)
            if mot != "":
                tokens.append(mot)
                mot = ""

    # Dernier mot
    if mot != "":
        tokens.append(mot)

    return tokens


# ------------------------------------------------------------------------------
# 6. PIPELINE COMPLET POUR LE CORPUS BRUT
# ------------------------------------------------------------------------------
def lire_texte_brut(nom_fichier):
    """Ouvre et lit l'intégralité du fichier texte brut en UTF-8."""
    fichier = open(nom_fichier, "r", encoding="utf-8")
    texte = fichier.read()
    fichier.close()
    return texte


def tokeniser_corpus(nom_fichier):
    """
    Pipeline principal complet :
    1. Lecture du fichier brut contenant du texte et des bruits (/[=-"|~@#$%&^*()_+-{;...).
    2. Segmentation en phrases nettoyées.
    3. Tokenisation de chaque phrase en mots propres.
    4. Renvoie la liste des phrases tokenisées : list[list[str]].
    """
    texte_brut = lire_texte_brut(nom_fichier)
    phrases = segmenter_en_phrases(texte_brut)

    corpus_tokenise = []
    for phrase in phrases:
        tokens = tokeniser_phrase(phrase)
        if len(tokens) > 0:
            corpus_tokenise.append(tokens)

    return corpus_tokenise


# ------------------------------------------------------------------------------
# 7. BLOC DE TEST ET DÉMONSTRATION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("====================================================================")
    print("TEST DE NETTOYAGE DE TEXTE BRUT BRUITÉ (AVEC CARACTÈRES SPÉCIAUX)")
    print("====================================================================")

    # Exemple de texte brut fortement bruité avec les caractères demandés
    texte_test = (
        "### [SECTION-1] ==> L'Université Mohammed Premier @ Oujda /// {Région: Oriental | Maroc} ~~~ "
        "Les étudiants (M2I/S3) apprennent le TALN & l'Intelligence Artificielle... "
        "/* Note: ID #42 */ Le professeur explique la théorie des algorithmes! "
        "===================================================================="
    )

    print("\n[1] Texte brut en entrée (très bruité) :")
    print(texte_test)

    phrases = segmenter_en_phrases(texte_test)
    print(f"\n[2] Phrases extraites après élimination du bruit ({len(phrases)} phrases) :")
    for idx, p in enumerate(phrases, start=1):
        print(f"    Phrase {idx} : {p}")
        tokens = tokeniser_phrase(p)
        print(f"    Tokens   : {tokens}")

    print("\n[3] Traitement complet de 'Campus_Aprentissage.txt' :")
    corpus = tokeniser_corpus("Campus_Aprentissage.txt")
    print(f"    Nombre total de phrases retenues : {len(corpus)}")
    print(f"    Nombre total de mots (tokens)    : {sum(len(p) for p in corpus)}")
    if len(corpus) > 0:
        print("\n    Première phrase tokenisée :")
        print("   ", corpus[0])
        print("\n    Deuxième phrase tokenisée :")
        print("   ", corpus[1])
