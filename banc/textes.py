"""Découpage des CV et préparation du texte pour BM25."""
import re
import unicodedata

# Mots vides courants : ils n'aident pas BM25 à distinguer les CV.
MOTS_VIDES = set("""
a au aux avec ce ces cet cette d dans de des du elle en et eux il ils je j l la le les leur leurs lui m ma mais me meme mes moi mon
n ne nos notre nous on ou par pas pour qu que qui s sa se ses son sur t ta te tes toi ton tu un une vos votre vous y ete etre ai
as avait avons ont est sont qui quoi quel quelle quels quelles dont the of and for with in on at to an or by is are
""".split())


def sans_accents(texte):
    return ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')


def jetons(texte):
    """Jetons pour BM25 : minuscules, sans accents, sans mots vides. Les sigles (AZ-700, PL-300) restent entiers."""
    mots = re.findall(r'[a-z0-9]+(?:[-/][a-z0-9]+)*', sans_accents(texte.lower()))
    return [m for m in mots if m not in MOTS_VIDES and len(m) > 1]


def cv_entier(cv):
    return [cv['cv_texte'].strip()]


def taille_fixe(cv, taille=200, chevauchement=40):
    """Fenêtres de 200 mots qui se chevauchent de 40 mots, sans tenir compte de la structure."""
    mots = cv['cv_texte'].split()
    pas = taille - chevauchement
    return [' '.join(mots[i:i + taille]) for i in range(0, max(len(mots) - chevauchement, 1), pas)]


def sections(cv):
    """Une section par titre « ## » du CV ; le titre reste au début de la section."""
    morceaux = re.split(r'\n(?=## )', cv['cv_texte'].strip())
    return [m.strip() for m in morceaux if len(m.split()) >= 5]


DECOUPAGES = {'cv-entier': cv_entier, 'taille-fixe': taille_fixe, 'sections': sections}
