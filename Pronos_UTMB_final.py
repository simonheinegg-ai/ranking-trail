"""
Classement des coureurs de trail via un modèle Plackett-Luce (bayésien).
"""

import re
import sys
import unicodedata

import numpy as np


def _fit_bradley_terry(n_items, pairs, weights=None, alpha=1.0,
                       max_iter=100, tol=1e-10):
    """Ajuste un Bradley-Terry PONDÉRÉ avec prior gaussien (régularisation L2
    de force `alpha`) et renvoie (theta_MAP, covariance de Laplace).

    Implémentation numpy pure (Newton + recherche linéaire d'Armijo) :
      - aucune dépendance à `choix` ni à `scipy` ;
      - le poids w_k de chaque duel k entre directement dans la
        log-vraisemblance et dans le hessien (vrai poids continu) ;
      - le problème est strictement convexe (alpha > 0) : le Newton converge
        en quelques itérations vers l'unique optimum.

    Pourquoi ne plus utiliser le package `choix` : il ne gère pas les poids
    par duel, donc chaque paire était répliquée jusqu'à ~10 fois puis
    traitée une par une par son EP en Python pur -> le script ne
    terminait pratiquement jamais dès que `choix` était installé."""
    alpha = max(float(alpha), 1e-12)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(n_items), np.eye(n_items) / alpha

    if weights is None:
        w = np.ones(len(pairs), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(pairs):
            raise ValueError(
                f"weights ({w.shape[0]}) et pairs ({len(pairs)}) "
                "doivent avoir la même longueur."
            )

    win, lose = pairs[:, 0], pairs[:, 1]
    n = n_items

    def neg_log_posterior(theta):
        d = theta[win] - theta[lose]
        return float((w * np.logaddexp(0.0, -d)).sum() + 0.5 * alpha * theta @ theta)

    def grad_and_hessian(theta):
        d = theta[win] - theta[lose]
        p = 1.0 / (1.0 + np.exp(-np.clip(d, -60.0, 60.0)))   # P(gagnant bat perdant)
        miss = w * (1.0 - p)
        g = (alpha * theta
             - np.bincount(win, weights=miss, minlength=n)
             + np.bincount(lose, weights=miss, minlength=n))
        c = w * p * (1.0 - p)
        H = np.zeros(n * n)
        H -= np.bincount(win * n + lose, weights=c, minlength=n * n)
        H -= np.bincount(lose * n + win, weights=c, minlength=n * n)
        H = H.reshape(n, n)
        H[np.diag_indices(n)] += (alpha
                                  + np.bincount(win, weights=c, minlength=n)
                                  + np.bincount(lose, weights=c, minlength=n))
        return g, H

    theta = np.zeros(n)
    f = neg_log_posterior(theta)
    for _ in range(max_iter):
        g, H = grad_and_hessian(theta)
        step = np.linalg.solve(H, g)
        decrement = float(g @ step)          # décrément de Newton (>= 0)
        if decrement < tol:
            break
        t = 1.0
        while t > 1e-12:
            candidate = theta - t * step
            f_new = neg_log_posterior(candidate)
            if f_new <= f - 1e-4 * t * decrement:
                break
            t *= 0.5
        else:
            break                             # plus de progrès possible
        theta, f = candidate, f_new

    _, H = grad_and_hessian(theta)
    cov = np.linalg.inv(H)
    return theta, cov

from collections import defaultdict
from difflib import SequenceMatcher

from race_data import NAME_ALIASES, RACES_DATA

MAX_POSITION = 30  

# Largeur du noyau de pertinence en écart RELATIF de distance (log-ratio),
# plutôt qu'en km absolus. Avec un bandwidth fixe en km, la fenêtre de
# pertinence était disproportionnée : trop large pour les courtes distances
# (ex: CCC 101 km aspirait des courses de 42 km) et trop étroite pour les
# longues (ex: UTMB 176 km excluait la TDS 140 km ou Lavaredo 120 km).
# En log-ratio, un écart de ±41% (~exp(0.35)) pèse pareil quelle que soit
# la distance cible.
#
# ÉLARGI de 0.35 à 0.5 (piste distance) : à 0.35, un résultat à 100-120 km
# ne pesait plus que 0.27-0.55 pour une cible UTMB (176 km) -- assez pour
# ne pas fausser le modèle, mais assez peu pour qu'une victoire nette à
# 100 km (ex: Puppi à The Canyons/CCC 2025, poids ~0.22 chacune) soit
# quasi effacée alors qu'elle reste un signal fort de niveau general. À
# 0.5, le poids distance seul passe à ~0.45-0.72 pour ces mêmes courses :
# ça laisse encore les courses très éloignées (marathon, courses
# techniques courtes type Zegama/Sierre-Zinal) largement hors-jeu, mais
# arrête d'écraser les ultras "voisins" (100-140 km) d'une cible longue.
DISTANCE_BANDWIDTH_LOG = 0.5

# En dessous de ce poids combiné (récence x distance), la course est
# purement ignorée pour cette distance cible (pas de plancher à 1 duel).
MIN_RELEVANCE_WEIGHT = 0.10

# Poids alpha (régularisation L2 du prior sur les ratings) utilisé
# PARTOUT dans le pipeline. Avant, `predict_race` appelait
# `compute_ratings_bayesian(..., alpha=3.0)` alors que la fonction elle-même
# avait un défaut de 1.0 : deux valeurs différentes cohabitaient sans
# qu'aucune ne soit documentée. On centralise ici la valeur réellement
# utilisée en production, pour que le défaut de la fonction et l'appel en
# production soient toujours identiques.
#
# RÉDUIT de 3.0 à 2.5 (piste régularisation) : à 3.0, un coureur avec
# seulement 2-3 courses pertinentes (même de bons résultats face à un
# plateau relevé, ex: Cardin 4e à Western States 2026 + 1er à Chianti
# 120k 2026) était ramené très près de la moyenne du plateau (1500) --
# le prior dominait la faible quantité de données observées. À 2.5, la
# régularisation reste réelle (les coureurs peu observés ne s'envolent
# pas artificiellement en tête), mais laisse un peu plus de place aux
# victoires/podiums nets pour faire bouger le rating, même avec peu de
# courses.
DEFAULT_ALPHA = 2.5

def distance_weight(distance_km, target_distance_km, bandwidth=DISTANCE_BANDWIDTH_LOG):
    """Noyau gaussien en échelle log : 1.0 si distance = cible, décroît
    doucement en fonction de l'écart RELATIF (ratio) de distance, pas de
    l'écart absolu en km. Un coureur qui a couru à 0.7x ou 1.4x la distance
    cible reçoit un poids comparable, quelle que soit la distance cible."""
    if (
        distance_km is None
        or target_distance_km is None
        or distance_km <= 0
        or target_distance_km <= 0
    ):
        return 0.5
    diff_log = np.log(distance_km / target_distance_km)
    return float(np.exp(-0.5 * (diff_log / bandwidth) ** 2))


def race_relevance_weight(race, target_distance_km):
    """Poids combiné récence x pertinence de distance pour une course donnée."""
    return race_weight(race) * distance_weight(race.get("distance_km"), target_distance_km)   


def relevant_races_for_distance(races, target_distance_km):
    """Ne garde que les courses dont le poids combiné (récence x distance)
    dépasse le seuil — évite de gonfler l'index avec des coureurs qui
    n'auront de toute façon aucun duel effectif pour cette distance."""
    return [
        r for r in races
        if race_relevance_weight(r, target_distance_km) >= MIN_RELEVANCE_WEIGHT
    ]
# ---------------------------------------------------------------------------
# Incertitude du rating et variabilité le jour de la course
# ---------------------------------------------------------------------------

# Plancher de l'incertitude du rating lorsque les données sont rares.
#
# CORRIGÉ (piste A) : à 180, sigma_data = SIGMA_SAMPLE/sqrt(n_eff+PRIOR_N)
# dominait systématiquement sigma_model (issu du Hessian du fit
# Bradley-Terry) pour 100% des coureurs de la base, y compris les plus
# observés. Résultat : sigma_model, qui reflète correctement le fait qu'un
# duel contre un plateau fort et resserré est plus informatif qu'un duel
# contre un plateau faible et homogène, n'avait jamais l'occasion de jouer
# -- seul le comptage brut de courses pondérées (n_eff) décidait de
# l'incertitude finale. Concrètement : un coureur qui ne dispute que 2-3
# grandes courses par an mais avec des duels très informatifs (contre des
# adversaires de niveau proche) se retrouvait avec la même pénalité
# d'incertitude qu'un débutant, simplement parce que n_eff était faible --
# indépendamment de la qualité de l'information contenue dans ces duels.
#
# Vérifié empiriquement (UTMB, cible 176 km) : à 110, sigma_model prend le
# relais de sigma_data dès que n_eff dépasse ~4.5 (spécialistes établis des
# grandes courses), tout en laissant sigma_data agir comme plancher pour la
# majorité des coureurs moins observés. Les coureurs quasi sans données
# (n_eff proche de 0) restent protégés dans tous les cas -- pas par
# sigma_data, mais par sigma_model lui-même, dont le prior L2 (alpha) donne
# déjà ~85-95 Elo d'incertitude pour un nœud quasi isolé du graphe de duels.
SIGMA_SAMPLE = 110
PRIOR_N = 2.0
MIN_SIGMA_SKILL = 10.0

# A priori de variabilité d'un coureur d'une course à l'autre.
# Cette composante représente le comportement réel du coureur, pas le manque
# de données sur son niveau.
SIGMA_DAYOF_PRIOR = 55.0
REGULARITY_PRIOR_N = 2.0
MIN_SIGMA_DAYOF = 25.0
MAX_SIGMA_DAYOF = 220.0

# Degrés de liberté de la loi de Student utilisée pour les tirages : plus
# c'est petit, plus la queue est épaisse (plus de "mauvais jours" extrêmes
# possibles), donc plus les favoris peuvent être surpris par un outsider.
# df=4 est un choix raisonnable ; df→∞ redonnerait une loi normale.
STUDENT_T_DF = 4

CURRENT_YEAR = 2026
DECAY_PER_YEAR = 0.80
MIN_YEAR_WEIGHT = 0.15
DEFAULT_YEAR_WEIGHT = 1.0
REPEAT_SCALE = 10

def race_year(race):
    m = re.match(r'(\d{4})', race.get("date", ""))
    return int(m.group(1)) if m else None


def race_weight(race):
    year = race_year(race)
    if year is None:
        return DEFAULT_YEAR_WEIGHT
    age = CURRENT_YEAR - year
    return max(MIN_YEAR_WEIGHT, DECAY_PER_YEAR ** age)


def race_repeat_count(race, target_distance_km):
    """Conservé pour compatibilité / affichage (ex: explain_athlete), mais
    n'est PLUS utilisé pour construire les duels (voir rankings_to_pairs,
    qui utilise désormais un poids continu au lieu d'une répétition
    entière arrondie)."""
    w = race_relevance_weight(race, target_distance_km)
    if w < MIN_RELEVANCE_WEIGHT:
        return 0  # course non pertinente pour cette distance : on l'ignore
    return max(1, round(w * REPEAT_SCALE))


# ---------------------------------------------------------------------------
# DNF (abandons)
# ---------------------------------------------------------------------------
# Volontairement DÉCOUPLÉ du rating de niveau (Bradley-Terry/Plackett-Luce) :
# un abandon (blessure, digestion, cut-off raté à cause de la météo...) n'est
# pas un signal fiable sur le niveau de vitesse pure d'un coureur face au
# plateau. On modélise donc une probabilité de DNF à part, utilisée
# uniquement au moment du Monte-Carlo (voir simulate_race_probabilities).
#
# Prior volontairement FAIBLE (peu de "pseudo-observations") : les DNF ne
# sont renseignés qu'au coup par coup pour certains favoris sur les courses
# récentes (voir add_dnf), pas pour les 87 courses de la base. Avec un prior
# fort, un ou deux DNF récents seraient noyés et ne feraient presque pas
# bouger la probabilité — ce qu'on veut éviter.
DNF_PRIOR_ALPHA = 1.0   # ~1 pseudo-DNF
DNF_PRIOR_BETA = 4.0    # ~4 pseudo-finishes -> prior moyen 20%
DEFAULT_DNF_RATE = DNF_PRIOR_ALPHA / (DNF_PRIOR_ALPHA + DNF_PRIOR_BETA)


def compute_dnf_rate(races, name, target_distance_km,
                      prior_alpha=DNF_PRIOR_ALPHA, prior_beta=DNF_PRIOR_BETA):
    """Probabilité (Beta-Binomiale) qu'un coureur abandonne, pour une
    distance cible donnée.

    Chaque course pèse par `race_relevance_weight` (récence x pertinence de
    distance), exactement comme pour le rating : un DNF sur l'UTMB (176 km)
    ne compte presque pas pour l'OCC (56 km), et pèse plein pot pour l'UTMB
    lui-même. Un DNF récent, avec un prior faible, fait donc bouger la
    probabilité de façon nette.

    NB : l'absence d'un coureur dans race["dnf"] n'est PAS interprétée comme
    "a fini" — seule sa présence effective dans race["results"] compte comme
    finish. Pour toutes les courses où les DNF n'ont pas été renseignés
    (l'immense majorité de la base), ni n_dnf ni n_finish ne bougent : seul
    le prior s'applique."""
    n_dnf, n_finish = 0.0, 0.0
    for race in races:
        weight = race_relevance_weight(race, target_distance_km)
        if weight < MIN_RELEVANCE_WEIGHT:
            continue
        if name in race.get("dnf", ()):
            n_dnf += weight
        elif name in race["results"]:
            n_finish += weight
    return (prior_alpha + n_dnf) / (prior_alpha + prior_beta + n_dnf + n_finish)


def compute_dnf_rates(races, names, target_distance_km):
    """Dict {nom: p_dnf} pour une liste de noms (ex: la startlist simulée)."""
    return {name: compute_dnf_rate(races, name, target_distance_km) for name in names}


# Force du lien entre p(DNF) et la variabilité jour de course (sigma_dayof).
# Intuition : un coureur qui joue le podium prend plus de risques (allure
# tendue, gestion nutrition serrée) qu'un coureur qui vise le top10 — donc
# plus de variabilité de performance ET plus de risque d'abandon, même sans
# DNF déjà observé. gamma faible (0.4) : effet modeste qui vient nuancer le
# taux de DNF historique (compute_dnf_rate), sans jamais le dominer.
DNF_SIGMA_DAYOF_GAMMA = 0.4
DNF_RATE_MIN = 0.01
DNF_RATE_MAX = 0.95


def adjust_dnf_by_dayof_sigma(dnf_prob, dayof_sigma_map, global_sigma,
                               gamma=DNF_SIGMA_DAYOF_GAMMA):
    """Ajuste p(DNF) à la hausse pour les coureurs plus erratiques que la
    moyenne (sigma_dayof > global_sigma) et à la baisse pour les plus
    réguliers, via un multiplicateur (sigma_dayof / global_sigma) ** gamma.

    Depuis que compute_dayof_residuals ne mélange plus les DNF dans le
    calcul de sigma_dayof (sigma_dayof est désormais une variance purement
    CONDITIONNELLE au fait de terminer), ce multiplicateur est le SEUL
    canal qui relie variabilité de performance et risque d'abandon pour
    les coureurs sans DNF observé : un coureur irrégulier sur ses courses
    terminées (grosse dispersion de classement) est aussi supposé plus
    à risque d'abandon, même s'il n'a encore jamais craqué. Ce lien ne
    joue que dans un sens (sigma erratique -> p(DNF) ajustée) ; l'inverse
    (un DNF observé -> sigma_dayof gonflé) a été retiré car il confondait
    deux profils très différents ("tout ou rien" vs simplement irrégulier).

    Le résultat reste borné à [DNF_RATE_MIN, DNF_RATE_MAX] pour éviter les
    valeurs dégénérées (proba de DNF garantie ou nulle)."""
    if global_sigma <= 0:
        return dict(dnf_prob)
    adjusted = {}
    for name, p in dnf_prob.items():
        sigma = dayof_sigma_map.get(name, {}).get("sigma_dayof", global_sigma)
        ratio = sigma / global_sigma
        p_adj = p * (ratio ** gamma)
        adjusted[name] = float(min(max(p_adj, DNF_RATE_MIN), DNF_RATE_MAX))
    return adjusted


def add_dnf(races, race_name, dnf_names):
    """Enregistre des DNF sur une course déjà créée via add_race, sans
    devoir toucher à l'appel add_race original ni ressaisir le classement.
    Pensé pour renseigner ponctuellement les favoris sur les courses
    récentes, sans devoir le faire pour les 87 courses de la base.

    `race_name` doit correspondre exactement au champ "name" utilisé dans
    l'appel add_race correspondant (ex: "UTMB 2025"). Si le coureur était
    par erreur déjà présent dans les finishers de cette course, il est
    retiré de `results`/`ranks` et basculé en DNF (un coureur ne peut pas
    être les deux à la fois).

    IMPORTANT sur le format des noms : `dnf_names` doit utiliser le même
    format normalisé "Prénom Nom" que celui utilisé dans les startlists
    (STARTLIST_UTMB_2026 etc.), PAS le format brut "NOM Prénom" des
    résultats collés. C'est ce format normalisé qui sert de clé partout
    dans le pipeline (results, startlist, filter_to_startlist...)."""
    dnf_names = [canonical_name(n) for n in dnf_names]
    matched = False
    for race in races:
        if race["name"] != race_name:
            continue
        matched = True
        existing = set(race.get("dnf", []))
        existing.update(dnf_names)

        for n in dnf_names:
            if n in race["results"]:
                idx = race["results"].index(n)
                old_ranks = race.get("ranks") or [None] * len(race["results"])
                race["results"] = [x for i, x in enumerate(race["results"]) if i != idx]
                race["ranks"] = [r for i, r in enumerate(old_ranks) if i != idx]
                print(f"⚠️ '{n}' était dans les finishers de '{race_name}' : "
                      f"retiré et basculé en DNF.")

        race["dnf"] = sorted(existing)

    if not matched:
        print(f"⚠️ add_dnf : aucune course nommée '{race_name}' trouvée dans races.")
    return races


def _extract_rank_and_name_from_line(line):
    """Comme _extract_name_from_line, mais retourne aussi le rang affiché
    (numéro en tête de ligne) quand il est présent, pour permettre de
    détecter les ex-aequo (rangs identiques). Retourne (rank_or_None, name)
    ou None si la ligne ne contient pas de nom exploitable."""
    raw = line.strip()
    if not raw:
        return None

    rank = None
    m_rank = re.match(r'\s*#?(\d{1,4})\s*[\.\)\-–]?\s+', raw)
    if m_rank:
        rank = int(m_rank.group(1))

    name = _extract_name_from_line(line)
    if name is None:
        return None
    return (rank, name)


def _extract_name_from_line(line):
    line = line.strip()
    if not line:
        return None

    line = re.sub(r'^\s*#?\d{1,4}\s*[\.\)\-–]?\s+', '', line)

    m = re.search(r'\d{1,3}:\d{2}(:\d{2})?', line)
    if m:
        line = line[:m.start()]

    line = re.sub(r'\([A-Za-z]{2,3}\)', '', line)

    # BUG CORRIGÉ : ce retrait d'un "code pays" de 3 majuscules en fin de
    # ligne mangeait aussi de vrais noms de famille ("CHEN LIN" -> "Chen",
    # "DOMENECH SAU" -> "Domenech"). Quand la ligne contient une heure, tout
    # ce qui suit l'heure (dont le pays) est déjà coupé ci-dessus : on ne
    # retire donc le code pays que pour les lignes SANS heure.
    if not m:
        line = re.sub(r'\b[A-Z]{3}\b\s*$', '', line)

    line = re.sub(r'\b\d+\b', '', line)

    line = re.sub(r'\s{2,}', ' ', line).strip(" -–.,\t")

    if len(line) < 3 or not re.search(r'[A-Za-zÀ-ÿ]', line):
        return None

    # BUG CORRIGÉ : seuls les noms EXACTEMENT "unknown"/"anonyme"... étaient
    # rejetés. "Anonymous 393891" (-> "Anonymous") et "UNKNOWN Anonymous"
    # passaient, et tous les anonymes de toutes les courses fusionnaient en
    # UN SEUL faux coureur "Anonymous", relié artificiellement à des duels
    # dans plusieurs courses.
    if re.search(r'\b(unknown|inconnu|anonymous|anonyme)\b', line, re.IGNORECASE):
        return None

    reordered = _reorder_upper_surname(line)
    if reordered:
        return reordered

    if line.isupper() or line.islower():
        line = line.title()

    return line


def _reorder_upper_surname(line):
    tokens = line.split()
    if len(tokens) < 2:
        return None
    upper_tokens = [t for t in tokens if t.isupper() and len(t) > 1]
    other_tokens = [t for t in tokens if t not in upper_tokens]
    if not upper_tokens or not other_tokens:
        return None
    given = " ".join(t.capitalize() for t in other_tokens)
    family = " ".join(t.capitalize() for t in upper_tokens)
    return f"{given} {family}"


def parse_paste_with_ranks(text, max_n=None):
    """Comme parse_paste, mais conserve le rang affiché à côté de chaque nom
    (None si aucun rang détecté sur la ligne). Sert à détecter les ex-aequo :
    quand deux lignes consécutives partagent le même rang affiché, ce ne
    sont pas deux positions distinctes mais un vrai ex-aequo, et aucun duel
    ne doit être créé entre ces deux coureurs."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    entries = []
    for line in lines:
        parsed = _extract_rank_and_name_from_line(line)
        if parsed:
            entries.append(parsed)
        if max_n and len(entries) >= max_n:
            break
    return entries


def first_last_table_to_text(raw):
    """Convertit un tableau tabulé du type
        rang <TAB> Prénom <TAB> Nom <TAB> ville <TAB> état <TAB> âge <TAB> ...
    en lignes "rang Prénom Nom" que parse_paste_with_ranks sait lire.

    BUG CORRIGÉ (George Waterfall 100k 2026) : ce tableau, où prénom et nom
    sont dans deux colonnes SÉPARÉES suivies de la ville et de l'état, était
    collé tel quel dans add_race. Le parseur générique gardait alors ville,
    état et sexe dans le nom ("Jeshurun Small M Co", "Brayden Mills
    Vancouver M Bc"...) : ces 20 coureurs devenaient des inconnus, jamais
    reliés à leurs autres courses ni aux startlists (ex: Drew Holmen,
    Jeshurun Small)."""
    lines = []
    for row in raw.strip().split("\n"):
        cols = [c.strip() for c in row.split("\t")]
        if len(cols) >= 3 and cols[0].isdigit() and cols[1] and cols[2]:
            lines.append(f"{cols[0]} {cols[1]} {cols[2]}")
    return "\n".join(lines)


def canonical_name(name):
    return NAME_ALIASES.get(name, name)


def add_race(races, name, date, names_or_text, max_n=None, distance_km=None):
    if isinstance(names_or_text, str):
        entries = parse_paste_with_ranks(names_or_text, max_n=max_n)
        results = [canonical_name(n) for _, n in entries]
        # Rang affiché par coureur (None si non détecté). Utilisé uniquement
        # pour la détection des ex-aequo lors de la construction des duels.
        ranks = [r for r, _ in entries]
    else:
        results = [canonical_name(n) for n in names_or_text]
        if max_n:
            results = results[:max_n]
        ranks = [None] * len(results)

    races.append({
        "name": name,
        "date": date,
        "results": results,
        "ranks": ranks,
        "dnf": [],  # renseigné a posteriori au cas par cas via add_dnf()
        "distance_km": distance_km,
    })
    return races


def compute_ratings_bayesian(races, target_distance_km, alpha=DEFAULT_ALPHA, k=1.0, model="logit"):
    """Calcule les ratings et leur incertitude pour une distance cible,
    en pondérant chaque course par récence x pertinence de distance, et en
    faisant régresser le niveau estimé vers 1500 en cas d'inactivité."""
    races = relevant_races_for_distance(races, target_distance_km)
    names, name_to_idx = build_index(races)
    pairs, weights = rankings_to_pairs(races, name_to_idx, target_distance_km)

    mean, cov = _fit_ratings(len(names), pairs, weights, alpha=alpha, model=model)
    sigma_raw = np.sqrt(np.maximum(np.diag(cov), 0.0))

    scale = 400 / np.log(10)
    centered = mean - np.mean(mean)
    mu_elo = 1500 + scale * centered
    sigma_elo_raw = scale * sigma_raw

    effective_n = compute_effective_appearances(races, target_distance_km)
    race_counts = count_races_per_runner(races)          # <-- AJOUT

    results = []
    for name, mu, sigma_model in zip(names, mu_elo, sigma_elo_raw):
        n_eff = effective_n.get(name, 0.0)
        n_races = race_counts.get(name, 0)                # <-- AJOUT
        sigma_data = SIGMA_SAMPLE / np.sqrt(n_eff + PRIOR_N)
        sigma = max(sigma_model, sigma_data, MIN_SIGMA_SKILL)

        last_year = last_relevant_race_year(races, name, target_distance_km)
        years_inactive = 5.0 if last_year is None else max(0.0, CURRENT_YEAR - last_year)
        shrink = 0.5 ** (years_inactive / INACTIVITY_HALF_LIFE_YEARS)
        mu_adjusted = 1500.0 + (mu - 1500.0) * shrink

        conservative = mu_adjusted - k * sigma
        results.append((name, mu_adjusted, sigma, conservative, n_eff, n_races))  # <-- champ ajouté à la fin

    results.sort(key=lambda x: -x[3])
    return results


def _fit_ratings(n_items, pairs, weights, alpha=DEFAULT_ALPHA, model="logit"):
    """Ajuste le modèle Bradley-Terry/Plackett-Luce avec un poids CONTINU
    par duel (récence x pertinence de distance), sans répliquer les paires.
    Voir _fit_bradley_terry."""
    if model != "logit":
        raise ValueError("Seul model='logit' est supporté.")
    return _fit_bradley_terry(n_items, pairs, weights, alpha=alpha)


def build_index(races):
    names = sorted({name for race in races for name in race["results"]})
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, name_to_idx


def rankings_to_pairs(races, name_to_idx, target_distance_km):
    """Construit la liste des duels (i, j) avec i devant j au classement,
    accompagnée d'un poids CONTINU par duel (récence x pertinence de
    distance de la course), au lieu de dupliquer physiquement les paires.

    Gestion des ex-aequo : quand deux coureurs partagent le même rang
    affiché dans les résultats bruts (même temps), aucun duel n'est créé
    entre eux — l'ordre dans la liste `results` n'est alors qu'un artefact
    de la façon dont le texte a été collé, pas un vrai résultat de course.
    """
    pairs = []
    weights = []
    for race in races:
        weight = race_relevance_weight(race, target_distance_km)
        if weight < MIN_RELEVANCE_WEIGHT:
            continue

        order = [name_to_idx[n] for n in race["results"]]
        ranks = race.get("ranks") or [None] * len(order)

        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                # Ex-aequo : même rang affiché et rang connu -> pas de duel.
                if ranks[i] is not None and ranks[i] == ranks[j]:
                    continue
                pairs.append((order[i], order[j]))
                weights.append(weight)

    return pairs, weights


def compute_effective_appearances(races, target_distance_km):
    """Nombre effectif de performances, pondéré par récence ET pertinence de distance."""
    effective_n = defaultdict(float)
    for race in races:
        weight = race_relevance_weight(race, target_distance_km)
        if weight < MIN_RELEVANCE_WEIGHT:
            continue
        for name in race["results"]:
            effective_n[name] += weight
    return effective_n


def count_races_per_runner(races):
    """Nombre BRUT (non pondéré) de courses jugées pertinentes pour la
    distance cible dans lesquelles le coureur apparaît. Contrairement à
    `compute_effective_appearances` (qui pondère par récence/distance),
    ce compteur est un simple entier, facile à lire dans le classement.

    NB : `races` doit déjà être filtré via `relevant_races_for_distance`
    (c'est le cas dans `compute_ratings_bayesian`)."""
    counts = defaultdict(int)
    for race in races:
        for name in race["results"]:
            counts[name] += 1
    return counts

# Nombre d'années pour revenir à mi-chemin de la moyenne du plateau (1500)
# si le coureur n'a plus disputé de course pertinente pour cette distance.
#
# CORRIGÉ (piste C) : à 1.5 ans, ce mécanisme punissait plus durement les
# spécialistes de grandes courses que n'importe quel autre facteur du
# pipeline. Exemple réel (UTMB, cible 176 km) : Jim Walmsley (vainqueur
# UTMB 2023 + Western States 2024 + Chianti 2025, mu BRUT le plus haut du
# plateau à 1858) et Tom Evans (vainqueur UTMB 2025, il y a tout juste un
# an) voyaient leur rating amputé de ~37% simplement parce que leur dernier
# résultat pertinent datait de plus de 12 mois -- une cadence pourtant
# parfaitement normale pour un coureur qui vise 1-2 courses majeures par an.
# Un coureur qui accumule des petites courses régionales tangentes (poids
# proche du seuil MIN_RELEVANCE_WEIGHT) rafraîchissait au contraire son
# horloge en continu et échappait totalement au shrink, quelle que soit la
# qualité de ses résultats.
#
# Par ailleurs ce mécanisme fait doublon avec DECAY_PER_YEAR, qui décote
# déjà chaque duel en continu selon son ancienneté DANS le fit lui-même --
# la récence est donc déjà prise en compte de façon principielle. Ce shrink
# reste utile comme filet de sécurité pour les carrières réellement à
# l'arrêt (un résultat brillant isolé qui ne disparaît jamais tout à fait
# du fit à cause du plancher MIN_YEAR_WEIGHT), mais sa fenêtre doit rester
# nettement plus longue qu'un simple cycle de saison.
INACTIVITY_HALF_LIFE_YEARS = 4.0

def last_relevant_race_year(races, name, target_distance_km):
    years = [
        race_year(r) for r in races
        if name in r["results"]
        and race_relevance_weight(r, target_distance_km) >= MIN_RELEVANCE_WEIGHT
        and race_year(r) is not None
    ]
    return max(years) if years else None


ELO_SCALE = 400 / np.log(10)

def expected_rank_from_rating(rating, opponent_mus):
    """
    Rang moyen attendu d'un coureur ayant un rating donné
    face aux niveaux opponent_mus.

    Modèle Bradley-Terry / Elo :
        P(i bat j) = sigmoid((rating_i - rating_j) / SCALE)
    """

    opponent_mus = np.asarray(opponent_mus, dtype=float)

    probs = 1.0 / (
        1.0 + np.exp(-(rating - opponent_mus) / ELO_SCALE)
    )

    # Nombre attendu de coureurs battus
    expected_beaten = np.sum(probs)

    # Rang = 1 + nombre attendu de coureurs devant
    expected_rank = len(opponent_mus) - expected_beaten

    return expected_rank


def race_performance_rating(rank, field_mus):

    """
    Transforme un classement réel dans une course en un
    rating Elo latent correspondant à ce classement.

    Exemple :
        si un coureur est 3e alors que son mu était 1500,
        on cherche le rating qui aurait produit environ
        une 3e place attendue face à ce plateau.
    """

    field_mus = np.asarray(field_mus, dtype=float)
    n = len(field_mus)

    if n <= 1:
        return np.nan

    target_rank = float(rank)

    # Bornes suffisamment larges en Elo
    low = field_mus.min() - 1000
    high = field_mus.max() + 1000

    # Recherche binaire du rating dont le rang attendu
    # correspond au rang réellement obtenu.
    for _ in range(60):

        mid = (low + high) / 2

        expected = expected_rank_from_rating(
            mid,
            field_mus
        )

        # Plus le rating est élevé, plus le rang attendu est petit
        if expected > target_rank:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def compute_dayof_sigma(
    races,
    bayes_ratings,
    target_distance_km,
    prior_sigma=SIGMA_DAYOF_PRIOR,
    prior_n=REGULARITY_PRIOR_N,
):
    """Estime, pour chaque coureur, l'écart-type de sa variabilité "jour de
    course" (résidu performance - rating), avec un shrinkage bayésien vers
    un a priori de population quand les données individuelles sont rares."""
    races = relevant_races_for_distance(races, target_distance_km)
    residuals = compute_dayof_residuals(races, bayes_ratings, target_distance_km)

    # A priori de population : variance moyenne des résidus, tous coureurs
    # confondus, pondérée par récence/pertinence. Sert de repli si un
    # coureur n'a aucun résidu exploitable.
    all_res, all_w = [], []
    for res_list in residuals.values():
        for r, w in res_list:
            all_res.append(r)
            all_w.append(w)

    if all_res:
        all_res = np.asarray(all_res)
        all_w = np.asarray(all_w)
        global_sigma = float(np.sqrt(np.average(all_res ** 2, weights=all_w)))
    else:
        global_sigma = prior_sigma

    global_sigma = float(min(max(global_sigma, MIN_SIGMA_DAYOF), MAX_SIGMA_DAYOF))

    dayof_sigma = {}
    for name, res_list in residuals.items():
        res = np.asarray([r for r, _ in res_list])
        w = np.asarray([wt for _, wt in res_list])
        n_eff = float(w.sum())

        sample_var = np.average(res ** 2, weights=w) if n_eff > 0 else prior_sigma ** 2

        # Shrinkage : mélange variance individuelle observée et a priori,
        # pondéré par prior_n "pseudo-courses".
        posterior_var = (
            (prior_n * prior_sigma ** 2) + (n_eff * sample_var)
        ) / (prior_n + n_eff)

        sigma = float(min(max(np.sqrt(posterior_var), MIN_SIGMA_DAYOF), MAX_SIGMA_DAYOF))

        dayof_sigma[name] = {"sigma_dayof": sigma, "n_eff": n_eff}

    return dayof_sigma, global_sigma


def compute_dayof_residuals(races, bayes_ratings, target_distance_km, censored_margin=3,
                             dnf_rank_margin=1):
    """Retourne les résidus performance-rating, pondérés par récence ET distance.

    Les `censored_margin` derniers rangs du classement AFFICHÉ de chaque
    course sont ignorés : comme les listes sont tronquées (max_n /
    MAX_POSITION), un coureur classé proche de la coupure n'a pas
    forcément fini dernier de la vraie course — il y avait sans doute des
    finishers non listés derrière lui. Sans cette censure, ces rangs de
    fin de liste génèrent des résidus artificiellement mauvais qui
    gonflent global_sigma et écrasent la différenciation individuelle
    entre coureurs réguliers et coureurs "à pic".

    BUG CORRIGÉ (double comptage du signal DNF) : les DNF étaient jusqu'ici
    injectés ICI comme résidu EXTRÊME (rang fictif = n_participants +
    dnf_rank_margin), en plus d'alimenter séparément compute_dnf_rate /
    compute_dnf_rates (probabilité de DNF, Beta-Binomiale). Résultat : un
    même abandon gonflait À LA FOIS p(DNF) ET sigma_dayof, et comme la
    plupart des coureurs n'ont que 3-6 courses pertinentes pour une
    distance donnée, UN SEUL DNF historique suffisait à faire exploser
    sigma_dayof jusqu'au plafond MAX_SIGMA_DAYOF (220) — y compris pour
    des coureurs par ailleurs très réguliers sur leurs courses terminées
    (ex: un DNF isolé en 2023 écrasait totalement le signal "constant
    dans le top 10" d'un coureur qui n'a plus jamais craqué depuis).

    sigma_dayof ne représente donc plus qu'une variance CONDITIONNELLE au
    fait de terminer la course ("étant donné qu'il finit, quelle est sa
    dispersion de performance ?"). Le risque de ne pas finir reste modélisé
    À PART, uniquement via p(DNF) (voir plus bas), ce qui correspond très
    exactement à la façon dont le Monte-Carlo tire les deux évènements
    (Bernoulli DNF, PUIS performance conditionnelle si le coureur finit) —
    voir simulate_race_probabilities. Cette séparation est aussi ce qui
    permet au modèle de distinguer deux profils qui, avant, se
    ressemblaient à tort une fois plafonnés à 220 :
      - un coureur "tout ou rien" (podium ou abandon, jamais 8e) aura un
        p(DNF) élevé MAIS un sigma_dayof bas, car ses courses terminées
        sont toutes resserrées près de son potentiel ;
      - un coureur abonné aux places d'honneur mais qui a craqué une fois
        gardera un p(DNF) modéré tout en retrouvant un sigma_dayof bas,
        cohérent avec sa régularité réelle sur ses courses terminées.
    """
    mu_by_name = {r[0]: r[1] for r in bayes_ratings}
    residuals = defaultdict(list)

    for race in races:
        weight = race_relevance_weight(race, target_distance_km)
        if weight < MIN_RELEVANCE_WEIGHT:
            continue

        results = race["results"]
        participants = [name for name in results if name in mu_by_name]

        if len(participants) < 3:
            continue

        field_mus = np.array([mu_by_name[name] for name in participants])
        n = len(participants)
        censored_from = max(1, n - censored_margin + 1)

        for rank, name in enumerate(participants, start=1):
            if rank >= censored_from:
                continue  # rang trop proche de la coupure : résidu non fiable
            race_rating = race_performance_rating(rank, field_mus)
            residual = race_rating - mu_by_name[name]
            residuals[name].append((residual, weight))

        # Les DNF ne sont plus injectés ici (voir docstring) : ils restent
        # gérés exclusivement par compute_dnf_rate / adjust_dnf_by_dayof_sigma.

    return residuals


def truncate_races(races, max_position):
    return [
        {
            "name": r["name"],
            "date": r["date"],
            "results": r["results"][:max_position],
            "ranks": (r.get("ranks") or [None] * len(r["results"]))[:max_position],
            "dnf": list(r.get("dnf", [])),  # jamais tronqué : pas lié à max_position
            "distance_km": r.get("distance_km"),
        }
        for r in races
    ]


# ---------------------------------------------------------------------------
# DONNÉES : courses, DNF, startlists, alias et cibles de prédiction vivent
# dans race_data.py (à éditer sans toucher à ce fichier). Ici on ne fait que
# les transformer en structure de travail.
# ---------------------------------------------------------------------------
def load_races(races_data=RACES_DATA):
    """Construit la liste de courses du modèle à partir de race_data.RACES_DATA
    (mêmes appels add_race puis add_dnf qu'avant, dans le même ordre)."""
    races = []
    for r in races_data:
        text = r["results"]
        if r.get("format") == "first_last_table":
            text = first_last_table_to_text(text)
        add_race(races, r["name"], r["date"], text,
                 max_n=r.get("max_n"), distance_km=r.get("distance_km"))
        if r.get("dnf"):
            add_dnf(races, r["name"], r["dnf"])
    return races


RACES = load_races()



def simulate_race_probabilities(bayes_ratings, dayof_sigma, dnf_prob=None,
                                 n_simulations=50000, seed=42):
    """
    Simule une course et calcule toutes les probabilités de classement.

    Pour chaque coureur, la dispersion simulée combine :

        1. sigma_skill : incertitude sur le rating estimé ;
        2. sigma_dayof : variabilité individuelle observée d'une course à l'autre.

    Les deux composantes sont ensuite utilisées directement par le Monte-Carlo.

    `dnf_prob` (optionnel) : dict {nom: p_dnf} (voir compute_dnf_rates). Si
    fourni, un tirage Bernoulli(p_dnf) est fait à CHAQUE simulation pour
    chaque coureur ; un coureur tiré DNF est retiré du classement de cette
    simulation-là (les autres remontent d'un rang), et compte dans p_dnf en
    sortie plutôt que dans p_top20/.../p_win. Si absent, comportement
    identique à avant (tout le monde finit).
    """
    rng = np.random.default_rng(seed)

    names = [r[0] for r in bayes_ratings]
    mus = np.asarray([r[1] for r in bayes_ratings], dtype=float)
    sigma_skill = np.asarray([r[2] for r in bayes_ratings], dtype=float)

    sigma_dayof = np.asarray([
        dayof_sigma.get(name, {}).get(
            "sigma_dayof",
            SIGMA_DAYOF_PRIOR,
        )
        for name in names
    ], dtype=float)

    sigma_total = np.sqrt(
        sigma_skill ** 2
        + sigma_dayof ** 2
    )

    p_dnf = np.asarray([
        (dnf_prob or {}).get(name, 0.0) for name in names
    ], dtype=float)
    has_dnf = bool(dnf_prob) and np.any(p_dnf > 0)

    counters = {
        name: {
            "top20": 0,
            "top10": 0,
            "top5": 0,
            "podium": 0,
            "win": 0,
            "dnf": 0,
        }
        for name in names
    }

    for _ in range(n_simulations):
        draws = (
            mus
            + sigma_total
            * rng.standard_t(
                df=STUDENT_T_DF,
                size=len(names),
            )
        )

        if has_dnf:
            finished = rng.random(len(names)) >= p_dnf
        else:
            finished = np.ones(len(names), dtype=bool)

        for idx in np.nonzero(~finished)[0]:
            counters[names[idx]]["dnf"] += 1

        finisher_idx = np.nonzero(finished)[0]
        order = finisher_idx[np.argsort(-draws[finisher_idx])]

        for rank, idx in enumerate(order, start=1):
            name = names[idx]

            if rank <= 20:
                counters[name]["top20"] += 1
            if rank <= 10:
                counters[name]["top10"] += 1
            if rank <= 5:
                counters[name]["top5"] += 1
            if rank <= 3:
                counters[name]["podium"] += 1
            if rank == 1:
                counters[name]["win"] += 1

    results = []
    for name, skill_sigma, day_sigma, total_sigma in zip(
        names,
        sigma_skill,
        sigma_dayof,
        sigma_total,
    ):
        c = counters[name]
        results.append({
            "name": name,
            "p_top20": c["top20"] / n_simulations,
            "p_top10": c["top10"] / n_simulations,
            "p_top5": c["top5"] / n_simulations,
            "p_podium": c["podium"] / n_simulations,
            "p_win": c["win"] / n_simulations,
            "p_dnf": c["dnf"] / n_simulations,
            "sigma_skill": float(skill_sigma),
            "sigma_dayof": float(day_sigma),
            "sigma_total": float(total_sigma),
        })

    results.sort(key=lambda x: -x["p_podium"])
    return results


def print_topn_probabilities(results, n=20, top_display=30):
    print(f"=== Probabilités Monte-Carlo — top {n} ===")
    header = (
        f"{'':>3} {'Coureur':<28} {'P(top20)':>10} "
        f"{'P(top10)':>10} {'P(top5)':>10} {'P(podium)':>11} {'P(victoire)':>12} {'P(DNF)':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(results[:top_display], start=1):
        print(
            f"{rank:>3}. {row['name']:<28} "
            f"{row['p_top20']:9.1%} "
            f"{row['p_top10']:9.1%} "
            f"{row['p_top5']:9.1%} "
            f"{row['p_podium']:10.1%} "
            f"{row['p_win']:11.1%} "
            f"{row.get('p_dnf', 0.0):8.1%}"
        )
    print()


def _normalize_name(name):
    """Normalisation pour faire correspondre les startlists aux résultats :
    insensible à la casse, aux accents ("Loïc" == "Loic"), à la ponctuation
    et aux tirets."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[\W_]+", " ", name.casefold())
    return re.sub(r"\s+", " ", name).strip()


def _sorted_tokens_key(normalized):
    """Clé insensible à l'ordre des mots ("Rueda Gabriel" == "Gabriel Rueda")."""
    return " ".join(sorted(normalized.split()))


def filter_to_startlist(bayes_cat, startlist, fuzzy_threshold=0.87):
    """Filtre le modèle sur la startlist. Correspondance en 3 niveaux :
    (1) nom normalisé identique, (2) mêmes mots dans un autre ordre,
    (3) REPLI fuzzy (SequenceMatcher) au-dessus de `fuzzy_threshold`, avec
    trace explicite pour vérification manuelle.

    BUG CORRIGÉ : le fuzzy matching était tenté coureur par coureur, DANS
    L'ORDRE de la startlist. Un nom approximatif pouvait donc "voler" la
    ligne d'un autre coureur dont le nom exact apparaissait PLUS BAS dans la
    startlist (ex: "Julien Chorier" vs "Julien Cormier", similarité 93%).
    Désormais toutes les correspondances exactes sont attribuées d'abord ;
    le fuzzy ne travaille que sur les coureurs restants. Les doublons de
    la startlist sont aussi ignorés proprement."""
    startlist = list(dict.fromkeys(startlist))   # dédoublonne, garde l'ordre

    by_normalized = {}
    by_sorted = {}
    for row in bayes_cat:
        key = _normalize_name(row[0])
        by_normalized.setdefault(key, row)
        by_sorted.setdefault(_sorted_tokens_key(key), row)

    matched = {}        # nom demandé -> ligne du modèle
    used = set()        # noms du modèle déjà attribués
    pending = []

    # Passes 1 et 2 : correspondances exactes (et ordre des mots inversé)
    for requested in startlist:
        key = _normalize_name(requested)
        row = by_normalized.get(key) or by_sorted.get(_sorted_tokens_key(key))
        if row is not None and row[0] not in used:
            matched[requested] = row
            used.add(row[0])
        elif row is None:
            pending.append(requested)

    # Passe 3 : fuzzy, uniquement parmi les noms encore libres
    fuzzy_matched = []  # (nom demandé, nom retenu, score) — à vérifier
    missing = []
    for requested in pending:
        key = _normalize_name(requested)
        best_row, best_ratio = None, 0.0
        for norm_name, candidate_row in by_normalized.items():
            if candidate_row[0] in used:
                continue
            ratio = SequenceMatcher(None, key, norm_name).ratio()
            if ratio > best_ratio:
                best_ratio, best_row = ratio, candidate_row
        if best_row is not None and best_ratio >= fuzzy_threshold:
            matched[requested] = best_row
            used.add(best_row[0])
            fuzzy_matched.append((requested, best_row[0], best_ratio))
        else:
            missing.append(requested)

    selected = [matched[r] for r in startlist if r in matched]

    if fuzzy_matched:
        print("ℹ️ Correspondances approximatives startlist -> modèle (à vérifier) :")
        for requested, matched_name, ratio in fuzzy_matched:
            print(f"   - '{requested}' -> '{matched_name}'   (similarité {ratio:.0%})")
        print()

    return selected, missing


def print_ranking(bayes, title, max_display=100):
    print(f"=== {title} (top {min(len(bayes), max_display)}) ===")
    header = (
        f"{'':>3} {'Coureur':<28} {'Niveau (μ)':>11} {'Incertitude (σ)':>17} "
        f"{'Score conserv.':>15} {'Courses eff.':>12} {'Courses':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, (name, mu, sigma, cons, n, n_races) in enumerate(bayes[:max_display], start=1):
        print(
            f"{rank:>3}. {name:<28} {mu:11.1f} {sigma:17.1f} {cons:15.1f} "
            f"{n:12.1f} {n_races:9d}"
        )
    print()


# Nombre de coureurs simulés ensemble pour le classement "ligne de départ
# unique". Au-delà, un coureur n'a de toute façon aucune chance réaliste de
# peser sur les probabilités de victoire/podium des favoris -- pas besoin de
# simuler les 600+ coureurs de la base à chaque fois.
STARTLINE_RANKING_POOL = 150


def rank_by_win_probability(races_model, target_distance_km, top_n_candidates=STARTLINE_RANKING_POOL,
                             n_simulations=50000, seed=42, alpha=DEFAULT_ALPHA, K=2.5):
    """Classement par probabilité de victoire dans une course HYPOTHÉTIQUE où
    tous les coureurs seraient sur la même ligne de départ, le même jour.

    À la différence de `compute_ratings_bayesian` trié par score conservateur
    (`mu - k*sigma`), qui favorise mécaniquement les coureurs à faible sigma
    -- donc ceux qui courent souvent, MÊME à un niveau moyen -- ce classement
    répond directement à la question "qui a le plus de chances de gagner si
    tout le monde course ensemble demain ?" via un Monte-Carlo : on tire pour
    chaque coureur une performance (rating + bruit d'incertitude + bruit
    jour de course, loi de Student pour les queues épaisses), on classe les
    tirages, et on compte les victoires/podiums sur `n_simulations` tirages.

    Un coureur avec un `mu` élevé mais une grosse incertitude peut très bien
    gagner souvent en simulation (quand il tire un bon jour) sans dominer le
    score conservateur -- c'est la différence attendue entre les deux
    classements.

    NB : réutilise exactement le même Monte-Carlo que `predict_race` (via
    `simulate_race_probabilities`), simplement appliqué à l'ensemble du
    plateau plutôt qu'à la startlist d'un événement précis.
    """
    bayes_cat = compute_ratings_bayesian(races_model, target_distance_km, alpha=alpha, k=K)
    if not bayes_cat:
        return [], bayes_cat

    candidates = bayes_cat[:top_n_candidates] if top_n_candidates else bayes_cat
    candidate_names = [row[0] for row in candidates]

    dayof_sigma, global_sigma = compute_dayof_sigma(races_model, bayes_cat, target_distance_km)
    dayof_map = {
        name: dayof_sigma.get(name, {"sigma_dayof": SIGMA_DAYOF_PRIOR})
        for name in candidate_names
    }

    dnf_prob = compute_dnf_rates(races_model, candidate_names, target_distance_km)
    dnf_prob = adjust_dnf_by_dayof_sigma(dnf_prob, dayof_map, global_sigma)

    results = simulate_race_probabilities(
        candidates,
        dayof_map,
        dnf_prob=dnf_prob,
        n_simulations=n_simulations,
        seed=seed,
    )
    return results, bayes_cat


def print_startline_ranking(results, title, max_display=100):
    print(f"=== {title} — tous sur la même ligne (top {min(len(results), max_display)}) ===")
    header = (
        f"{'':>3} {'Coureur':<28} {'P(victoire)':>12} {'P(podium)':>11} "
        f"{'P(top5)':>9} {'P(top10)':>10} {'P(DNF)':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(results[:max_display], start=1):
        print(
            f"{rank:>3}. {row['name']:<28} {row['p_win']:11.1%} {row['p_podium']:10.1%} "
            f"{row['p_top5']:8.1%} {row['p_top10']:9.1%} {row.get('p_dnf', 0.0):8.1%}"
        )
    print()


def predict_race(races_model, target_distance_km, startlist, race_title, top_n=20, K=2.5, verbose=True, alpha=DEFAULT_ALPHA):
    """Calcule le rating (pondéré par pertinence de distance), estime le bruit
    jour de course et simule la course.

    `alpha` utilise désormais la même constante DEFAULT_ALPHA que le défaut
    de compute_ratings_bayesian (avant : 3.0 codé en dur ici vs 1.0 par
    défaut dans la fonction, sans que ce soit documenté)."""
    races_relevant = relevant_races_for_distance(races_model, target_distance_km)
    if verbose:
        n_runners = len(build_index(races_relevant)[0]) if races_relevant else 0
        print(
            f"Distance cible {target_distance_km} km : {len(races_relevant)}/{len(races_model)} courses "
            f"jugées pertinentes, {n_runners} coureurs distincts\n"
        )

    if not races_model:
        if verbose:
            print(f"⚠️ Aucune course en base : impossible de calculer un classement pour {race_title}.\n")
        return [], [], []

    bayes_cat = compute_ratings_bayesian(races_model, target_distance_km, alpha=alpha, k=K)
    dayof_sigma, global_sigma = compute_dayof_sigma(races_model, bayes_cat, target_distance_km)
    if verbose:
        # Premier classement affiché : mêmes colonnes qu'avant (μ, σ,
        # score conservateur, courses eff., courses), mais trié par μ brut
        # (le rating Bradley-Terry/Plackett-Luce) et non plus par le score
        # conservateur (mu - k*sigma). Le score conservateur mesure la
        # CERTITUDE qu'on a sur le niveau d'un coureur (il pénalise les
        # coureurs peu observés même s'ils affichent un très bon mu) --
        # ce n'est pas ce qu'on veut ici. μ, lui, est directement le
        # rating qui détermine P(i bat j) dans le modèle : il représente
        # la capacité pure à gagner un duel / une course face au plateau,
        # indépendamment de la confiance qu'on a dans cette estimation.
        # Ce classement ne dépend PAS non plus du Monte-Carlo de
        # probabilités de victoire (voir plus bas) : c'est un classement
        # par points totaux, point final.
        bayes_by_points = sorted(bayes_cat, key=lambda row: -row[1])
        print_ranking(
            bayes_by_points,
            f"{race_title} (cible {target_distance_km} km) — classement par points",
        )
        print(f"Sigma jour de course global : {global_sigma:.1f} Elo\n")

    if not startlist:
        if verbose:
            print(f"⚠️ Aucune startlist définie pour {race_title} : impossible de simuler.\n")
        return bayes_cat, [], []

    bayes_startlist, missing = filter_to_startlist(bayes_cat, startlist)

    if missing and verbose:
        print(f"⚠️ Noms de la startlist non trouvés dans le modèle : {missing}\n")

    if not bayes_startlist:
        if verbose:
            print(
                f"⚠️ Aucun coureur de la startlist {race_title} trouvé dans les données : "
                "impossible de simuler.\n"
            )
        return bayes_cat, bayes_startlist, []

    dayof_startlist = {
        row[0]: dayof_sigma.get(row[0], {"sigma_dayof": SIGMA_DAYOF_PRIOR})
        for row in bayes_startlist
    }

    dnf_startlist = compute_dnf_rates(
        races_model,
        [row[0] for row in bayes_startlist],
        target_distance_km,
    )
    dnf_startlist = adjust_dnf_by_dayof_sigma(dnf_startlist, dayof_startlist, global_sigma)

    results = simulate_race_probabilities(
        bayes_startlist,
        dayof_startlist,
        dnf_prob=dnf_startlist,
        n_simulations=50000,
        seed=42,
    )

    if verbose:
        # On affiche TOUS les favoris de la startlist simulée (pas de
        # limite ici : c'est justement une liste réduite et lisible).
        print_topn_probabilities(results, n=len(results), top_display=len(results))

    return bayes_cat, bayes_startlist, results


if __name__ == "__main__":
    # Sous Windows (console cp1252 ou sortie redirigée), print() de 'μ', 'σ'
    # ou '⚠️' lève UnicodeEncodeError : on force l'UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    races_full = RACES
    races_model = truncate_races(races_full, MAX_POSITION)

    print("Que voulez-vous faire ?")
    print("  1. Classement (tous les coureurs sur une même ligne de départ hypothétique)")
    print("  2. Pronos (startlist d'une vraie course, définie dans race_data.py)")
    choice = input("Votre choix [1/2] : ").strip()

    if choice == "1":
        while True:
            raw_n = input("Nombre de coureurs à classer : ").strip()
            try:
                n_runners = int(raw_n)
                if n_runners <= 0:
                    raise ValueError
                break
            except ValueError:
                print("⚠️ Merci d'entrer un nombre entier positif.")

        while True:
            raw_dist = input("Distance cible (km) : ").strip().replace(",", ".")
            try:
                distance_km = float(raw_dist)
                if distance_km <= 0:
                    raise ValueError
                break
            except ValueError:
                print("⚠️ Merci d'entrer un nombre (ex: 176 ou 100).")

        bayes_cat = compute_ratings_bayesian(races_model, distance_km, alpha=DEFAULT_ALPHA)
        bayes_by_points = sorted(bayes_cat, key=lambda row: -row[1])[:n_runners]
        print_ranking(
            bayes_by_points,
            f"Classement par points — cible {distance_km:g} km",
            max_display=n_runners,
        )

    else:
        while True:
            raw_dist = input("Distance de la course (km) : ").strip().replace(",", ".")
            try:
                distance_km = float(raw_dist)
                if distance_km <= 0:
                    raise ValueError
                break
            except ValueError:
                print("⚠️ Merci d'entrer un nombre (ex: 176 ou 100).")

        print(
            "Entrez les coureurs présents au départ, un par ligne "
            "(ligne vide pour terminer) :"
        )
        startlist = []
        while True:
            name = input().strip()
            if not name:
                break
            startlist.append(name)

        if not startlist:
            print("⚠️ Aucun coureur saisi : impossible de faire un pronostic.\n")
        else:
            predict_race(
                races_model,
                target_distance_km=distance_km,
                startlist=startlist,
                race_title=f"Pronos — cible {distance_km:g} km",
                top_n=len(startlist),
            )
