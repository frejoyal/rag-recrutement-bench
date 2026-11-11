"""Trace le graphique des résultats publié dans le README, à partir de resultats.json.

Usage : python resultats/graphique.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ICI = Path(__file__).resolve().parent
SURFACE, TEXTE, TEXTE_2, GRILLE = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIE_1, SERIE_2 = '#2a78d6', '#eb6834'


def etiquette(r):
    contexte = ' + contexte' if r['prefixes_contextuels'] else ''
    return f"{r['technique']}  ·  {r['decoupage']}{contexte}"


def panneau(ax, lignes, cles, noms, titre):
    y = list(range(len(lignes)))[::-1]
    for i, yi in enumerate(y):
        valeurs = [100 * lignes[i][1][k] for k in cles]
        ax.plot([min(valeurs), max(valeurs)], [yi, yi], color=GRILLE, linewidth=2, zorder=1)
    for k, nom, couleur in zip(cles, noms, (SERIE_1, SERIE_2)):
        ax.scatter([100 * l[1][k] for l in lignes], y, s=64, color=couleur, edgecolor=SURFACE, linewidth=2, zorder=2, label=nom)
    ax.set_xlim(-2, 102)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f'{x} %' for x in range(0, 101, 20)])
    ax.grid(axis='x', color=GRILLE, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(titre, loc='left', color=TEXTE, fontsize=11, fontweight='bold', pad=26)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, handletextpad=0.3,
              columnspacing=1.2, labelcolor=TEXTE_2, borderaxespad=0.2)
    for cote in ('top', 'right', 'left'):
        ax.spines[cote].set_visible(False)
    ax.spines['bottom'].set_color(GRILLE)
    ax.tick_params(colors=TEXTE_2, length=0, labelsize=9)
    return y


def main():
    resultats = json.loads((ICI / 'resultats.json').read_text(encoding='utf-8'))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True, facecolor=SURFACE,
                                         gridspec_kw={'wspace': 0.08})
    for ax in (gauche, droite):
        ax.set_facecolor(SURFACE)
    y = panneau(gauche, [(r, r['recherche']) for r in resultats], ['premier_rang', 'trois_premiers'],
                ['Bon candidat en 1er', 'Bon candidat dans les 3 premiers'], 'Recherche (45 questions avec réponse)')
    panneau(droite, [(r, r['barriere']) for r in resultats], ['pieges_refuses', 'refus_a_tort'],
            ['Pièges refusés (plus haut = mieux)', 'Refus à tort (plus bas = mieux)'],
            'Barrière « aucun candidat » (30 questions jamais vues)')
    gauche.set_yticks(y)
    gauche.set_yticklabels([etiquette(r) for r in resultats], color=TEXTE, fontsize=9)
    fig.text(0.01, 0.01, 'Banc d\'essai RAG sur 50 CV fictifs. Modèles locaux : bge-m3, gemma4:e4b, mmarco-mMiniLMv2.',
             color=TEXTE_2, fontsize=8)
    fig.savefig(ICI / 'graphique-resultats.png', dpi=150, bbox_inches='tight', facecolor=SURFACE)
    print('graphique-resultats.png écrit')


if __name__ == '__main__':
    main()
