"""Banc d'essai : mesure chaque technique de recherche sur les 60 questions.

Usage :
    python -m banc.evaluer --mode direct     # calcule avec les modèles locaux et remplit le cache
    python -m banc.evaluer                   # recalcule à partir du cache, sans modèle
    python -m banc.evaluer --verifier        # s'arrête en erreur si les résultats ne sont plus ceux publiés (utilisé par la CI)

Protocole :
- Recherche : premier rang, trois premiers et rang réciproque moyen, sur les 45 questions qui ont une réponse.
  Ces mesures ne dépendent d'aucun réglage.
- Barrière « aucun candidat » : chaque méthode refuse de répondre quand le score de son premier candidat est sous un seuil.
  Le seuil est calibré sur les questions impaires (23 avec réponse, 7 pièges) et mesuré sur les questions paires
  (22 avec réponse, 8 pièges), que la calibration n'a jamais vues.
- Coût : appels au modèle de langage et passages relus par le modèle de reclassement, en moyenne par question.
"""
import argparse
import json
import sys
from pathlib import Path

from .methodes import METHODES, Index
from .modeles import RACINE, Modeles

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
RESULTATS = RACINE / 'resultats'
NOMS_DECOUPAGES = {'cv-entier': 'CV entier', 'taille-fixe': 'fenêtres de 200 mots', 'sections': 'sections du CV'}


def charger():
    cvs = json.loads((RACINE / 'donnees' / 'cvs.json').read_text(encoding='utf-8'))
    questions = json.loads((RACINE / 'evaluation' / 'questions.json').read_text(encoding='utf-8'))
    for q in questions:
        q['partie'] = 'calibration' if int(q['id'][1:]) % 2 else 'mesure'
    return cvs, questions


def refuse(classement, seuil):
    return not classement or classement[0][1] < seuil


def exactitude(lignes, seuil):
    """Bonne décision : bon candidat en premier sans refus, ou refus pour une question piège."""
    bons = 0
    for q, classement in lignes:
        if q['attendus']:
            bons += (not refuse(classement, seuil)) and classement[0][0] in q['attendus']
        else:
            bons += refuse(classement, seuil)
    return bons / len(lignes)


def calibrer(lignes):
    """Seuil qui maximise l'exactitude sur les questions de calibration ; à égalité, le plus bas."""
    tops = sorted({c[0][1] for _, c in lignes if c})
    candidats = [float('-inf')] + [(a + b) / 2 for a, b in zip(tops, tops[1:])] + [tops[-1] + 1e-9]
    return max(candidats, key=lambda s: (exactitude(lignes, s), -s))


def rang(q, classement):
    return next((i for i, (c, _) in enumerate(classement) if c in q['attendus']), None)


def recherche(lignes):
    r = [rang(q, c) for q, c in lignes]
    n = len(r)
    return {
        'premier_rang': sum(x == 0 for x in r) / n,
        'trois_premiers': sum(x is not None and x < 3 for x in r) / n,
        'rang_reciproque': sum(1 / (x + 1) for x in r if x is not None) / n,
    }


def barriere(lignes, seuil):
    repondables = [c for q, c in lignes if q['attendus']]
    pieges = [c for q, c in lignes if not q['attendus']]
    return {
        'pieges_refuses': sum(refuse(c, seuil) for c in pieges) / len(pieges),
        'refus_a_tort': sum(refuse(c, seuil) for c in repondables) / len(repondables),
        'exactitude': exactitude(lignes, seuil),
    }


def evaluer(mode):
    cvs, questions = charger()
    modeles = Modeles(mode)
    index, resultats = {}, []
    for technique, decoupage, contextuel, methode in METHODES:
        cle_index = (decoupage, contextuel)
        if cle_index not in index:
            print(f'Index : {NOMS_DECOUPAGES[decoupage]}{", préfixes contextuels" if contextuel else ""}…', flush=True)
            index[cle_index] = Index(cvs, decoupage, modeles, contextuel)
            modeles.enregistrer()
        modeles.appels_llm = modeles.passages_reclasses = 0
        lignes = [(q, methode(index[cle_index], q['question'])) for q in questions]
        modeles.enregistrer()
        seuil = calibrer([(q, c) for q, c in lignes if q['partie'] == 'calibration'])
        repondables = [(q, c) for q, c in lignes if q['attendus']]
        par_type = {}
        for t in sorted({q['type'] for q, _ in repondables}):
            l = [(q, c) for q, c in repondables if q['type'] == t]
            par_type[t] = round(sum(rang(q, c) == 0 for q, c in l) / len(l), 3)
        resultats.append({
            'technique': technique,
            'decoupage': NOMS_DECOUPAGES[decoupage],
            'prefixes_contextuels': contextuel,
            'passages': len(index[cle_index].passages),
            'recherche': {k: round(v, 3) for k, v in recherche(repondables).items()},
            'premier_rang_par_type': par_type,
            'seuil': None if seuil == float('-inf') else round(seuil, 5),
            'barriere': {k: round(v, 3) for k, v in barriere([(q, c) for q, c in lignes if q['partie'] == 'mesure'], seuil).items()},
            'cout_par_question': {
                'appels_llm': round(modeles.appels_llm / len(questions), 2),
                'passages_reclasses': round(modeles.passages_reclasses / len(questions), 1),
            },
            'trois_premiers_par_question': {q['id']: [x[0] for x in c[:3]] for q, c in lignes},
        })
        r = resultats[-1]
        print(f"{technique} ({r['decoupage']}{', contexte' if contextuel else ''}) : "
              f"premier rang {r['recherche']['premier_rang']:.0%}, trois premiers {r['recherche']['trois_premiers']:.0%}, "
              f"pièges refusés {r['barriere']['pieges_refuses']:.0%}, refus à tort {r['barriere']['refus_a_tort']:.0%}, "
              f"exactitude {r['barriere']['exactitude']:.0%}", flush=True)
    return resultats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cache', 'direct'], default='cache')
    parser.add_argument('--verifier', action='store_true')
    args = parser.parse_args()
    publie = RESULTATS / 'resultats.json'
    resultats = evaluer(args.mode)
    if args.verifier:
        identique = resultats == json.loads(publie.read_text(encoding='utf-8'))
        print('Résultats identiques à ceux publiés.' if identique else 'ERREUR : les résultats diffèrent de ceux publiés.')
        sys.exit(0 if identique else 1)
    publie.write_text(json.dumps(resultats, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Résultats écrits dans {publie.relative_to(RACINE)}')


if __name__ == '__main__':
    main()
