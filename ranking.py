"""
ranking.py — classement "head-to-head" par distance, à partir de race_data.py

Principe
--------
Chaque course donne des duels : si A finit devant B, A bat B.
On ajuste un modèle de Bradley-Terry (comme Elo, mais calculé d'un seul coup
sur tout l'historique) : P(A bat B) = 1 / (1 + 10^((rB - rA) / 400)).

Pour classer à une distance D, chaque duel est pondéré par :
  - la proximité de distance : exp(-0.5 * (ln(d_course / D) / SIGMA)^2)
        -> un 100 km compte à fond pour un classement 100 km,
           beaucoup moins pour un classement 42 km
  - la fraîcheur : 0.5 ** (âge_en_années / DEMI_VIE)
  - 1/(n-1) par course, pour qu'une course à 50 coureurs ne pèse pas
    plus qu'une à 20 (chaque coureur "pèse" ~1 par course)

Un coureur "DNF" perd contre tous les classés de la course (poids réduit).
Une régularisation (un adversaire fictif moyen) évite les notes extrêmes
pour ceux qui n'ont que 1-2 courses.

Usage
-----
  Double-clic / bouton "Run" : un menu s'ouvre et pose les questions.
  Ou, en ligne de commande :
  python ranking.py 100                       # top 25 à 100 km
  python ranking.py 50 --top 40
  python ranking.py 100 --vs "Jim Walmsley"   # qui il bat / contre qui il perd
  python ranking.py 100 --duel "Jim Walmsley" "Zach Miller"
  python ranking.py --all                     # 30 / 50 / 100 / 170 km côte à côte
"""
import argparse
import math
import re
import unicodedata
from collections import defaultdict

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from race_data import RACES_DATA, NAME_ALIASES
except ImportError:
    input("Je ne trouve pas race_data.py : mets-le dans le même dossier que ranking.py.\n"
          "(Entrée pour fermer)")
    raise SystemExit

# ----------------------------- paramètres ----------------------------------
SIGMA = 0.40          # largeur du filtre de distance (en log). 0.4 : 50k↔100k pèse ~0.2
HALF_LIFE = 2.0       # demi-vie de la fraîcheur, en années
REF_DATE = (2026, 10) # "aujourd'hui" (année, mois)
DNF_WEIGHT = 0.5      # poids d'un duel perdu par abandon
PRIOR = 0.5           # force de la régularisation (plus grand = notes plus resserrées)
MIN_RELEVANT = 2      # nb mini de courses "pertinentes" pour apparaître
RELEVANT_W = 0.25     # poids de distance mini pour qu'une course soit "pertinente"
SCALE = 400 / math.log(10)   # échelle Elo


# ----------------------------- noms ----------------------------------------
def _strip(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def key(name):
    """Clé de comparaison : sans accents, minuscules, indépendante de l'ordre des mots."""
    toks = re.findall(r"[a-z0-9]+", _strip(name).lower())
    return " ".join(sorted(set(toks)))   # set : "ADRIEN BRIFFOD Adrien" -> {adrien, briffod}


ALIAS_KEYS = {key(a): key(b) for a, b in NAME_ALIASES.items()}
ALIAS_DISPLAY = {key(a): b for a, b in NAME_ALIASES.items()}


def canon(name):
    k = key(name)
    return ALIAS_KEYS.get(k, k)


def display(name):
    """'WALMSLEY Jim' -> 'Jim Walmsley' ; 'JOYEUX BOUILLON Arthur' -> 'Arthur Joyeux Bouillon'."""
    toks, seen = [], set()
    for t in name.split():            # retire les mots en double (source : "ADRIEN BRIFFOD Adrien")
        if t.lower() not in seen:
            seen.add(t.lower()); toks.append(t)
    if not toks:
        return name
    up = [t for t in toks if t.isupper() and len(t) > 1]
    if len(up) == len(toks):          # tout en majuscules : 1er mot = nom
        sur, giv = toks[:1], toks[1:]
    else:
        i = 0
        while i < len(toks) and toks[i].isupper() and len(toks[i]) > 1:
            i += 1
        sur, giv = toks[:i], toks[i:]
        if not sur:                   # déjà "Prénom Nom"
            return " ".join(t.capitalize() if t.isupper() else t for t in toks)
    return " ".join([t.capitalize() for t in giv] + [t.capitalize() for t in sur])


# ----------------------------- lecture des courses -------------------------
def parse_race(r):
    """Renvoie [(rang, nom_affiché)] pour les max_n premières lignes."""
    out = []
    lines = [l for l in r["results"].strip().splitlines() if l.strip()]
    for l in lines[: r["max_n"]]:
        if "\t" in l:
            f = [x.strip() for x in l.split("\t")]
            if r["format"] == "first_last_table":
                name = f"{f[1]} {f[2]}"
            else:
                name = f[1]
        else:
            m = re.match(r"\s*(\d+)\s+(.*)", l)
            if not m:
                continue
            f = [m.group(1), m.group(2).strip()]
            name = f[1]
        try:
            rank = int(f[0])
        except ValueError:
            continue
        if not name or name.lower().startswith("unknown"):
            continue
        out.append((rank, name))
    return out


def race_age_years(date):
    y, m = int(date[:4]), int(date[5:7])
    return max(0.0, ((REF_DATE[0] - y) * 12 + (REF_DATE[1] - m)) / 12)


def build_duels():
    """Liste de duels bruts : (gagnant, perdant, nul?, poids_base, distance, date, course)."""
    duels, names = [], {}
    for r in RACES_DATA:
        res = parse_race(r)
        n = len(res)
        if n < 2:
            continue
        base = 1.0 / (n - 1)
        age_w = 0.5 ** (race_age_years(r["date"]) / HALF_LIFE)
        ids = []
        for rank, nm in res:
            c = canon(nm)
            names.setdefault(c, ALIAS_DISPLAY.get(key(nm), display(nm)))
            ids.append((rank, c))
        for i in range(n):
            for j in range(i + 1, n):
                (ra, a), (rb, b) = ids[i], ids[j]
                if a == b:
                    continue
                if ra == rb:
                    duels.append((a, b, True, base, r["distance_km"], age_w, r["name"]))
                else:
                    duels.append((a, b, False, base, r["distance_km"], age_w, r["name"]))
        # abandons : perdent contre tous les classés
        listed = {c for _, c in ids}
        for d in r["dnf"]:
            cd = canon(d)
            if cd in listed:
                continue
            names.setdefault(cd, d)
            for _, c in ids:
                duels.append((c, cd, False, base * DNF_WEIGHT, r["distance_km"], age_w, r["name"]))
    return duels, names


# ----------------------------- modèle --------------------------------------
def dist_weight(race_km, target_km):
    return math.exp(-0.5 * (math.log(race_km / target_km) / SIGMA) ** 2)


def fit(duels, names, target_km, iters=300):
    """Bradley-Terry par algorithme MM, pondéré par distance/fraîcheur."""
    idx = {c: i for i, c in enumerate(names)}
    W, L, w, rel = [], [], [], defaultdict(set)
    for a, b, tie, base, km, agew, rname in duels:
        wt = base * agew * dist_weight(km, target_km)
        if wt < 1e-4:
            continue
        W.append(idx[a]); L.append(idx[b])
        w.append(wt * (0.5 if tie else 1.0))
        if tie:  # un nul = 0.5 victoire de chaque côté
            W.append(idx[b]); L.append(idx[a]); w.append(wt * 0.5)
        if dist_weight(km, target_km) >= RELEVANT_W:
            rel[a].add(rname); rel[b].add(rname)
    W, L, w = np.array(W), np.array(L), np.array(w)
    n = len(idx)
    wins = np.bincount(W, weights=w, minlength=n)
    games = np.bincount(W, weights=w, minlength=n) + np.bincount(L, weights=w, minlength=n)
    g = np.ones(n)
    for _ in range(iters):
        denom = np.bincount(W, weights=w / (g[W] + g[L]), minlength=n) \
              + np.bincount(L, weights=w / (g[W] + g[L]), minlength=n)
        # adversaire fictif de force 1 : une demi-victoire + une demi-défaite
        denom = denom + PRIOR / (g + 1.0)
        new = (wins + PRIOR / 2) / np.maximum(denom, 1e-12)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(g))) < 1e-7:
            g = new
            break
        g = new
    rating = 1500 + SCALE * np.log(g)
    return idx, rating, wins, games - wins, rel


def rank_table(duels, names, km):
    idx, rating, wins, losses, rel = fit(duels, names, km)
    rows = []
    for c, i in idx.items():
        k = len(rel.get(c, ()))
        if k >= MIN_RELEVANT:
            rows.append((rating[i], names[c], k, wins[i], losses[i], c))
    rows.sort(reverse=True)
    return rows, idx, rating


def find(names, query):
    q = canon(query)
    if q in names:
        return q
    hits = [c for c, n in names.items() if q in c or q in key(n)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"Coureur introuvable : {query!r}")
    raise SystemExit("Plusieurs correspondances : " + ", ".join(names[h] for h in hits[:8]))


# ----------------------------- affichage -----------------------------------
def show_top(duels, names, km, top):
    rows, *_ = rank_table(duels, names, km)
    print(f"\n=== Classement à {km:g} km  (σ={SIGMA}, demi-vie {HALF_LIFE:g} ans) ===")
    print(f"{'#':>3}  {'Coureur':<28}{'Note':>6}  {'Courses':>7}  {'Victoires':>9}  {'Défaites':>8}")
    for i, (r, n, k, w, l, _) in enumerate(rows[:top], 1):
        print(f"{i:>3}  {n:<28}{r:>6.0f}  {k:>7}  {w:>9.1f}  {l:>8.1f}")


def show_all(duels, names, top):
    dists = [30, 50, 100, 170]
    tables = {d: rank_table(duels, names, d)[0][:top] for d in dists}
    print(f"\n{'#':>3}  " + "".join(f"{str(d)+' km':<30}" for d in dists))
    for i in range(top):
        print(f"{i+1:>3}  " + "".join(
            f"{(tables[d][i][1][:20] + f' ({tables[d][i][0]:.0f})') if i < len(tables[d]) else '':<30}"
            for d in dists))


def show_vs(duels, names, km, who, top):
    c = find(names, who)
    rows, idx, rating = rank_table(duels, names, km)
    pos = next((i for i, r in enumerate(rows, 1) if r[5] == c), None)
    print(f"\n=== {names[c]} à {km:g} km : note {rating[idx[c]]:.0f}"
          + (f", #{pos}" if pos else " (pas assez de courses pertinentes)") + " ===")
    beat, lost = defaultdict(float), defaultdict(float)
    nb, nl = defaultdict(int), defaultdict(int)     # nb de duels bruts
    together = defaultdict(set)
    for a, b, tie, base, dkm, agew, rname in duels:
        wt = base * agew * dist_weight(dkm, km)
        if wt < 0.01 * base:       # course trop éloignée en distance : ignorée
            continue
        if a == c:
            beat[b] += wt * (0.5 if tie else 1); together[b].add(rname)
            nb[b] += 0 if tie else 1
            if tie: lost[b] += wt * 0.5
        elif b == c:
            lost[a] += wt * (0.5 if tie else 1); together[a].add(rname)
            nl[a] += 0 if tie else 1
            if tie: beat[a] += wt * 0.5

    def block(title, d, sign):
        print(f"\n{title}")
        items = sorted(d.items(), key=lambda kv: -kv[1])[:top]
        for o, wgt in items:
            net = beat[o] - lost[o]
            if sign * net <= 0:
                continue
            print(f"  {names[o]:<28} {nb[o]}-{nl[o]}  en {len(together[o])} course(s)"
                  f"   (note {rating[idx[o]]:.0f})")

    block("A battu (les plus pertinents pour cette distance en premier) :", beat, +1)
    block("A perdu contre :", lost, -1)


def show_duel(duels, names, km, a, b):
    ca, cb = find(names, a), find(names, b)
    _, idx, rating = rank_table(duels, names, km)
    ra, rb = rating[idx[ca]], rating[idx[cb]]
    p = 1 / (1 + 10 ** ((rb - ra) / 400))
    print(f"\nÀ {km:g} km : {names[ca]} ({ra:.0f}) vs {names[cb]} ({rb:.0f})")
    print(f"  P({names[ca]} devant) = {p:.0%}")
    print("  Confrontations directes :")
    found = False
    for x, y, tie, base, dkm, agew, rname in duels:
        if {x, y} == {ca, cb}:
            found = True
            win = names[x] if not tie else "égalité"
            print(f"    {rname:<34} {dkm:>4} km  -> {win}")
    if not found:
        print("    (jamais classés ensemble : la comparaison passe par des adversaires communs)")


# ----------------------------- mode menu (sans terminal) -------------------
def ask_km(default=100):
    while True:
        t = input(f"Distance en km ? (Entrée = {default}) > ").strip().lower().replace("km", "").replace(",", ".")
        if not t:
            return float(default)
        try:
            v = float(t)
            if v > 0:
                return v
        except ValueError:
            pass
        print("  -> Écris un nombre, par exemple 50 ou 100.")


def ask_top(default=25):
    t = input(f"Combien de coureurs afficher ? (Entrée = {default}) > ").strip()
    return int(t) if t.isdigit() and int(t) > 0 else default


def ask_runner(names, counts, label="Nom du coureur"):
    """Demande un nom (même partiel) ; propose une liste si plusieurs correspondent."""
    while True:
        q = input(f"{label} (ou une partie du nom, ex: walmsley) > ").strip()
        if not q:
            return None
        toks = re.findall(r"[a-z0-9]+", _strip(q).lower())
        hits = [c for c in names if all(any(t in w for w in c.split()) for t in toks)]
        if not hits:
            print("  -> Personne trouvé, essaie une autre orthographe.")
            continue
        if len(hits) == 1:
            return hits[0]
        hits.sort(key=lambda c: -counts.get(c, 0))
        hits = hits[:15]
        print("  Plusieurs correspondances :")
        for n, c in enumerate(hits, 1):
            print(f"   {n:>2}. {names[c]}  ({counts.get(c, 0)} courses)")
        ch = input("  Numéro (ou Entrée pour recommencer) > ").strip()
        if ch.isdigit() and 1 <= int(ch) <= len(hits):
            return hits[int(ch) - 1]


MENU = """
==================== RANKING TRAIL ====================
  1. Classement à une distance donnée
  2. Classements 30 / 50 / 100 / 170 km côte à côte
  3. Victoires et défaites d'un coureur
  4. Duel entre deux coureurs
  5. Doublons possibles dans les noms
  0. Quitter
"""


def interactive():
    print("Chargement des courses...")
    duels, names = build_duels()
    counts = defaultdict(set)
    for a, b, _, _, _, _, rn in duels:
        counts[a].add(rn); counts[b].add(rn)
    counts = {c: len(v) for c, v in counts.items()}
    print(f"{len(RACES_DATA)} courses, {len(names)} coureurs.")
    while True:
        print(MENU)
        ch = input("Ton choix > ").strip()
        print()
        try:
            if ch == "1":
                show_top(duels, names, ask_km(), ask_top())
            elif ch == "2":
                show_all(duels, names, ask_top(15))
            elif ch == "3":
                c = ask_runner(names, counts)
                if c:
                    show_vs(duels, names, ask_km(), names[c], ask_top(10))
            elif ch == "4":
                a = ask_runner(names, counts, "Premier coureur")
                b = ask_runner(names, counts, "Deuxième coureur") if a else None
                if a and b:
                    show_duel(duels, names, ask_km(), names[a], names[b])
            elif ch == "5":
                show_aliases(names)
            elif ch in ("0", "q", "quit", ""):
                break
            else:
                print("Choix inconnu : tape un chiffre du menu.")
        except SystemExit as e:           # erreurs de recherche de nom
            print(e)
        input("\n(Entrée pour revenir au menu)")


def show_aliases(names):
    ks = {c: set(c.split()) for c in names}
    found = False
    for x in ks:
        for y in ks:
            if x < y and len(ks[x]) >= 2 and len(ks[y]) >= 2 and (ks[x] < ks[y] or ks[y] < ks[x]):
                print(f'    "{names[x]}": "{names[y]}",   # ou l\'inverse')
                found = True
    if not found:
        print("Aucun doublon évident.")
    print("\nPour fusionner deux noms : ajoute la ligne dans NAME_ALIASES (race_data.py).")


def main():
    if len(sys.argv) == 1:
        return interactive()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("km", nargs="?", type=float, help="distance cible en km")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--vs", metavar="COUREUR", help="détail : qui il bat / contre qui il perd")
    ap.add_argument("--duel", nargs=2, metavar=("A", "B"))
    ap.add_argument("--aliases", action="store_true", help="liste les noms qui ressemblent à des doublons")
    ap.add_argument("--all", action="store_true", help="30/50/100/170 km côte à côte")
    a = ap.parse_args()

    duels, names = build_duels()
    if a.aliases:
        return show_aliases(names)
    if a.all:
        return show_all(duels, names, a.top)
    if a.km is None:
        ap.error("donne une distance (ex: 100) ou --all")
    if a.vs:
        return show_vs(duels, names, a.km, a.vs, a.top)
    if a.duel:
        return show_duel(duels, names, a.km, *a.duel)
    show_top(duels, names, a.km, a.top)


if __name__ == "__main__":
    main()
