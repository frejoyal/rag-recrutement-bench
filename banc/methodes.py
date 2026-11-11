"""Les techniques de recherche comparées par le banc d'essai.

Chaque méthode reçoit une question et retourne les candidats classés, du plus pertinent au moins pertinent,
avec un score. Le score du premier candidat sert ensuite à la barrière « aucun candidat » (voir evaluer.py).
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .textes import DECOUPAGES, jetons

K_RRF = 60            # constante habituelle de la fusion par rangs réciproques
PASSAGES_RECLASSES = 30


class Index:
    """Les passages d'un découpage, avec leurs vecteurs et leur index BM25."""

    def __init__(self, cvs, decoupage, modeles, contextuel=False):
        self.passages, self.candidats = [], []
        for cv in cvs:
            for morceau in DECOUPAGES[decoupage](cv):
                if contextuel:
                    morceau = prefixe_contextuel(cv, morceau, modeles) + '\n\n' + morceau
                self.passages.append(morceau)
                self.candidats.append(cv['id'])
        self.modeles = modeles
        self.vecteurs = modeles.vecteurs(self.passages)
        self.bm25 = BM25Okapi([jetons(p) for p in self.passages])

    # Scores arrondis à 6 décimales : le classement reste le même d'une machine à l'autre (CI comprise).
    def scores_vecteur(self, question):
        return np.round((self.vecteurs @ self.modeles.vecteurs([question])[0]).astype(np.float64), 6)

    def scores_bm25(self, question):
        return np.round(np.asarray(self.bm25.get_scores(jetons(question)), dtype=np.float64), 6)


def prefixe_contextuel(cv, morceau, modeles):
    """Contextual Retrieval : une ou deux phrases qui situent le passage dans le CV, écrites par le modèle local."""
    invite = (
        f"Voici un CV complet :\n<cv>\n{cv['cv_texte']}\n</cv>\n\nVoici un passage de ce CV :\n<passage>\n{morceau}\n</passage>\n\n"
        "Écris une ou deux phrases courtes, en français, qui situent ce passage dans le CV pour améliorer sa recherche : "
        "nomme la personne, son domaine, sa ville et le sujet du passage. Réponds seulement avec ces phrases."
    )
    return modeles.generer(invite)


def par_candidat(index, scores):
    """Classe les candidats selon le meilleur score de leurs passages. Égalités départagées par l'identifiant."""
    meilleurs = {}
    for c, s in zip(index.candidats, scores):
        if c not in meilleurs or s > meilleurs[c]:
            meilleurs[c] = float(s)
    return sorted(meilleurs.items(), key=lambda x: (-x[1], x[0]))


def rangs(scores):
    """Rang de chaque passage (0 = meilleur), égalités départagées par la position."""
    ordre = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    r = np.empty(len(scores), dtype=int)
    r[ordre] = np.arange(len(scores))
    return r


def fusion_rrf(*listes_de_scores):
    """Fusion par rangs réciproques : chaque liste donne 1 / (60 + rang) à chaque passage."""
    return sum(1.0 / (K_RRF + 1 + rangs(s)) for s in listes_de_scores)


# ---------- Méthodes ----------
def vecteur(index, question):
    return par_candidat(index, index.scores_vecteur(question))


def bm25(index, question):
    return par_candidat(index, index.scores_bm25(question))


def hybride(index, question):
    return par_candidat(index, fusion_rrf(index.scores_vecteur(question), index.scores_bm25(question)))


def hybride_reclasse(index, question):
    """Hybride, puis le modèle de reclassement relit les 30 meilleurs passages face à la question."""
    fusion = fusion_rrf(index.scores_vecteur(question), index.scores_bm25(question))
    meilleurs = np.argsort(-fusion, kind='stable')[:PASSAGES_RECLASSES]
    scores = index.modeles.reclasser(question, [index.passages[i] for i in meilleurs])
    classement = {}
    for i, s in zip(meilleurs, scores):
        c = index.candidats[i]
        if c not in classement or s > classement[c]:
            classement[c] = s
    return sorted(classement.items(), key=lambda x: (-x[1], x[0]))


def sous_questions(question, modeles):
    """RAG agentique : le modèle local découpe une question à plusieurs critères en sous-questions."""
    invite = (
        "Tu aides un recruteur à chercher des candidats dans des CV. Si la demande ci-dessous contient plusieurs critères "
        "indépendants (par exemple une ville et une expérience), découpe-la en deux ou trois sous-questions courtes, une par critère. "
        "Sinon, garde la demande telle quelle. Réponds seulement avec une liste JSON de chaînes.\n\n"
        f"Demande : {question}"
    )
    texte = modeles.generer(invite)
    trouve = re.search(r'\[.*\]', texte, flags=re.S)
    try:
        liste = [q.strip() for q in json.loads(trouve.group(0)) if isinstance(q, str) and q.strip()] if trouve else []
    except json.JSONDecodeError:
        liste = []
    return liste[:3] or [question]


def agentique(index, question, avec_question=False):
    """Chaque sous-question passe par l'hybride reclassé ; les classements sont fusionnés par rangs réciproques.
    Avec avec_question, la question entière fait aussi partie des recherches fusionnées.
    Le score final reste celui du reclassement pour la question entière, ce qui garde la barrière comparable."""
    parties = sous_questions(question, index.modeles)
    if avec_question:
        parties = [question] + [p for p in parties if p != question]
    if len(parties) == 1:
        return hybride_reclasse(index, question)
    fusion = {}
    for partie in parties:
        for rang, (c, _) in enumerate(hybride_reclasse(index, partie)):
            fusion[c] = fusion.get(c, 0.0) + 1.0 / (K_RRF + 1 + rang)
    entiere = dict(hybride_reclasse(index, question))
    ordre = sorted(fusion.items(), key=lambda x: (-x[1], x[0]))
    return [(c, entiere.get(c, min(entiere.values()) if entiere else 0.0)) for c, _ in ordre]


def agentique_avec_question(index, question):
    return agentique(index, question, avec_question=True)


# Technique, découpage, préfixes contextuels, fonction
TECHNIQUES = [
    ('Vecteurs', vecteur),
    ('BM25', bm25),
    ('Hybride (vecteurs + BM25)', hybride),
    ('Hybride + reclassement', hybride_reclasse),
    ('Agentique, sous-questions seules', agentique),
    ('Agentique, question + sous-questions', agentique_avec_question),
]
METHODES = [
    ('Vecteurs', 'cv-entier', False, vecteur),
    ('Vecteurs', 'taille-fixe', False, vecteur),
] + [(nom, 'sections', contextuel, f) for contextuel in (False, True) for nom, f in TECHNIQUES]
