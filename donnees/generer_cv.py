"""Génère les 50 CV longs de Recrutement Élite inc. avec un modèle local (Ollama).

Usage : python donnees/generer_cv.py [--modele gemma4:e4b]

Chaque CV doit contenir mot pour mot les trois détails du candidat (voir profils.py), et aucun autre CV ne doit les contenir.
Un CV qui échoue aux contrôles est régénéré avec un autre germe, au plus cinq fois.
Les CV déjà valides ne sont pas régénérés : le script peut reprendre là où il s'est arrêté.
Résultat : donnees/cv/cXXX.md et donnees/cvs.json.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profils import CANDIDATS, EMPLOYEURS, SPECIALITES  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ICI = Path(__file__).resolve().parent
DOSSIER = ICI / 'cv'
OLLAMA = 'http://127.0.0.1:11434/api/generate'
# Vraies entreprises qui ne doivent pas apparaître dans les CV fictifs.
INTERDITS = ['Desjardins', 'Banque Nationale', 'CGI', 'Hydro-Québec', 'Bell', 'Vidéotron', 'Videotron', 'Bombardier', 'Air Canada',
             'Ubisoft', 'Shopify', 'Coveo', 'Lightspeed', 'Couche-Tard', 'SAQ', 'Loto-Québec', 'Mila', 'RBC', 'Banque Royale', 'BMO',
             'Banque Scotia', 'Intact', 'Beneva', 'Sun Life', 'Manuvie', 'Rogers', 'Telus', 'Saputo', 'Cascades', 'Rio Tinto', 'Alcoa',
             'Morgan Stanley', 'Ericsson', 'Jean Coutu', 'Familiprix', 'Metro inc', 'Behaviour', 'Moment Factory', 'Banque Laurentienne',
             'Industrielle Alliance', 'Postes Canada', 'Via Rail', 'Énergir', 'Cirque du Soleil', 'STM']


def employeurs_de(ident):
    return random.Random(ident).sample(EMPLOYEURS, 3)


def invite(c):
    ident, nom, spec, annees, bilingue, ville, certification, projet, engagement = c
    e1, e2, e3 = employeurs_de(ident)
    langues = 'Il ou elle est bilingue, français et anglais.' if bilingue else 'Il ou elle travaille en français ; son anglais est fonctionnel, sans plus.'
    return f"""Rédige en français du Québec le CV détaillé et réaliste d'une personne fictive, pour un cabinet de recrutement.

Profil :
- Nom : {nom}
- Domaine : {SPECIALITES[spec]}
- Expérience : {annees} ans
- Ville : {ville}
- {langues}

Structure obligatoire, en Markdown :
# {nom}
Une ligne : la ville, le domaine et les années d'expérience.
## Profil (un paragraphe)
## Expérience professionnelle (trois postes, du plus récent au plus ancien, chacun avec des dates et quatre à six réalisations concrètes)
## Projets marquants (trois projets décrits en un paragraphe chacun)
## Formation
## Certifications
## Compétences techniques
## Langues
## Engagement et loisirs

Contraintes strictes :
1. Les trois employeurs sont, dans l'ordre : {e1}, {e2} et {e3}. N'invente aucune autre entreprise et ne nomme aucune vraie entreprise.
2. La section Projets marquants contient cette phrase, mot pour mot : « J'ai dirigé {projet}. »
3. La section Certifications contient une seule certification, écrite mot pour mot : {certification}
4. La section Engagement et loisirs contient cette phrase, mot pour mot : « En dehors du travail, je {engagement}. »
5. Aucun courriel, numéro de téléphone ni adresse.
6. Entre 1300 et 1900 mots. Un ton professionnel, à la première personne.
7. Réponds uniquement avec le CV, sans commentaire avant ni après."""


def nettoyer(texte):
    texte = texte.strip()
    texte = re.sub(r'^```(?:markdown)?\s*|\s*```$', '', texte).replace('’', "'")
    return texte.replace('« ', '« ').replace(' »', ' »').strip() + '\n'


def problemes(c, texte):
    ident, nom, *_, certification, projet, engagement = c
    bas = texte.lower()
    erreurs = []
    for libelle, detail in (('certification', certification), ('projet', projet), ('engagement', engagement)):
        if detail.lower() not in bas:
            erreurs.append(f'{libelle} absent')
    sans_nom = texte.replace(nom, '')
    for terme in INTERDITS:
        if re.search(r'(?<![\w-])' + re.escape(terme) + r'(?![\w-])', sans_nom):
            erreurs.append(f'vraie entreprise : {terme}')
    if re.search(r'[\w.+-]+@[\w-]+\.\w+|\(?\d{3}\)?[ .-]\d{3}-\d{4}', texte):
        erreurs.append('coordonnées')
    mots = len(texte.split())
    if not 1100 <= mots <= 2300:
        erreurs.append(f'{mots} mots')
    return erreurs


def generer(c, modele, germe):
    reponse = requests.post(OLLAMA, json={
        'model': modele, 'prompt': invite(c), 'stream': False, 'think': False,
        'options': {'temperature': 0.8, 'seed': germe, 'num_ctx': 8192, 'num_predict': 4500},
    }, timeout=900)
    reponse.raise_for_status()
    return nettoyer(reponse.json()['response'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modele', default='gemma4:e4b')
    args = parser.parse_args()
    DOSSIER.mkdir(exist_ok=True)
    debut = time.time()
    for i, c in enumerate(CANDIDATS, 1):
        fichier = DOSSIER / f'{c[0]}.md'
        if fichier.exists() and not problemes(c, fichier.read_text(encoding='utf-8')):
            continue
        for essai in range(5):
            t = time.time()
            texte = generer(c, args.modele, 1000 + int(c[0][1:]) * 10 + essai)
            erreurs = problemes(c, texte)
            print(f'[{i}/50] {c[0]} essai {essai + 1} : {len(texte.split())} mots, {time.time() - t:.0f} s'
                  + (f', à refaire ({", ".join(erreurs)})' if erreurs else ', valide'), flush=True)
            if not erreurs:
                fichier.write_text(texte, encoding='utf-8', newline='\n')
                break
        else:
            print(f'ERREUR {c[0]} : aucun essai valide', flush=True)

    # Unicité : chaque détail n'apparaît que dans le CV de son candidat.
    textes = {c[0]: (DOSSIER / f'{c[0]}.md').read_text(encoding='utf-8').lower() for c in CANDIDATS if (DOSSIER / f'{c[0]}.md').exists()}
    doublons = []
    for c in CANDIDATS:
        for detail in c[6:9]:
            ailleurs = [k for k, t in textes.items() if k != c[0] and detail.lower() in t]
            if ailleurs:
                doublons.append(f'{c[0]} « {detail} » aussi dans {ailleurs}')
    print('\n'.join(doublons) if doublons else 'Unicité : chaque détail n\'apparaît que dans son CV.')

    cvs = [{'id': c[0], 'nom': c[1], 'specialite': c[2], 'annees_exp': c[3], 'bilingue': c[4], 'localisation': c[5],
            'cv_texte': textes.get(c[0]) and (DOSSIER / f'{c[0]}.md').read_text(encoding='utf-8')} for c in CANDIDATS]
    (ICI / 'cvs.json').write_text(json.dumps(cvs, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(textes)} CV valides sur 50, en {(time.time() - debut) / 60:.0f} min')


if __name__ == '__main__':
    main()
