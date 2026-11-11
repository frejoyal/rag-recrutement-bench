"""Appels aux modèles locaux, avec un cache publié dans le dépôt.

En mode « direct », les modèles tournent sur le poste (Ollama et sentence-transformers) et le cache se remplit.
En mode « cache », rien n'est calculé : tout vient du cache. C'est ce que fait la CI, sans modèle ni réseau,
pour vérifier que les résultats publiés se recalculent à l'identique.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / 'resultats' / 'cache'
OLLAMA = 'http://127.0.0.1:11434'
MODELE_EMBEDDING = 'bge-m3'
MODELE_LLM = 'gemma4:e4b'
MODELE_RECLASSEMENT = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def cle(*parties):
    return hashlib.sha1('\x1f'.join(parties).encode('utf-8')).hexdigest()


class Modeles:
    def __init__(self, mode='cache'):
        if mode not in ('cache', 'direct'):
            raise ValueError(mode)
        self.mode = mode
        CACHE.mkdir(parents=True, exist_ok=True)
        self._vecteurs = {}
        fichier = CACHE / 'embeddings.npz'
        if fichier.exists():
            donnees = np.load(fichier)
            self._vecteurs = dict(zip(donnees['cles'].tolist(), donnees['vecteurs']))
        self._textes = self._lire('generations.json')
        self._scores = self._lire('reclassement.json')
        self._reclasseur = None
        self.modifie = False
        # Coût d'une méthode, compté de la même façon que la réponse vienne du cache ou du modèle.
        self.appels_llm = 0
        self.passages_reclasses = 0

    @staticmethod
    def _lire(nom):
        fichier = CACHE / nom
        return json.loads(fichier.read_text(encoding='utf-8')) if fichier.exists() else {}

    def _manque(self, quoi):
        raise RuntimeError(f'{quoi} absent du cache : relancez en mode direct, avec Ollama démarré.')

    # ---------- Embeddings ----------
    def vecteurs(self, textes):
        cles = [cle(MODELE_EMBEDDING, t) for t in textes]
        manquants = [(k, t) for k, t in zip(cles, textes) if k not in self._vecteurs]
        if manquants:
            if self.mode == 'cache':
                self._manque('Embedding')
            import requests
            for i in range(0, len(manquants), 16):
                lot = manquants[i:i + 16]
                r = requests.post(f'{OLLAMA}/api/embed', json={'model': MODELE_EMBEDDING, 'input': [t for _, t in lot]}, timeout=600)
                r.raise_for_status()
                for (k, _), v in zip(lot, r.json()['embeddings']):
                    self._vecteurs[k] = np.asarray(v, dtype=np.float16)
            self.modifie = True
        matrice = np.stack([self._vecteurs[k].astype(np.float32) for k in cles])
        return matrice / np.linalg.norm(matrice, axis=1, keepdims=True)

    # ---------- Modèle de langage ----------
    def generer(self, invite):
        k = cle(MODELE_LLM, invite)
        self.appels_llm += 1
        if k not in self._textes:
            if self.mode == 'cache':
                self._manque('Réponse du modèle de langage')
            import requests
            r = requests.post(f'{OLLAMA}/api/generate', json={
                'model': MODELE_LLM, 'prompt': invite, 'stream': False, 'think': False,
                'options': {'temperature': 0, 'seed': 7, 'num_ctx': 8192, 'num_predict': 400}}, timeout=600)
            r.raise_for_status()
            self._textes[k] = r.json()['response'].strip()
            self.modifie = True
        return self._textes[k]

    # ---------- Reclassement ----------
    def reclasser(self, question, passages):
        cles = [cle(MODELE_RECLASSEMENT, question, p) for p in passages]
        self.passages_reclasses += len(passages)
        manquants = [(k, p) for k, p in zip(cles, passages) if k not in self._scores]
        if manquants:
            if self.mode == 'cache':
                self._manque('Score de reclassement')
            if self._reclasseur is None:
                import os
                os.environ.setdefault('USE_TF', '0')  # le reclassement n'utilise que PyTorch
                from sentence_transformers import CrossEncoder
                self._reclasseur = CrossEncoder(MODELE_RECLASSEMENT, max_length=512)
            scores = self._reclasseur.predict([(question, p) for _, p in manquants], batch_size=16)
            for (k, _), s in zip(manquants, scores):
                self._scores[k] = round(float(s), 5)
            self.modifie = True
        return [self._scores[k] for k in cles]

    def enregistrer(self):
        if not self.modifie:
            return
        cles = sorted(self._vecteurs)
        np.savez_compressed(CACHE / 'embeddings.npz', cles=np.array(cles), vecteurs=np.stack([self._vecteurs[k] for k in cles]))
        for nom, contenu in (('generations.json', self._textes), ('reclassement.json', self._scores)):
            (CACHE / nom).write_text(json.dumps(dict(sorted(contenu.items())), ensure_ascii=False, indent=0) + '\n', encoding='utf-8')
        self.modifie = False
