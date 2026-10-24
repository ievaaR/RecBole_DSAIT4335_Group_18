import numpy as np


# Every reranker shares one interface:
#
#     rerank(scores_u, candidates_u, k, lam, **context) -> k item ids, best first
#
# scores_u is one user's row of a score matrix (indexed by item id), candidates_u the item ids it may
# choose from. lam = 0 returns the base ranking of the pool; lam = 1 uses only the beyond-accuracy term.
# Item ids index scores_u, genres and history the same way, so with RecBole internal ids the genre
# matrix needs a row for the padding item 0.


def _relevance(scores_u, candidates, k):
    """Min-max normalised scores of the pool, so lam means the same thing for every model."""
    if k > len(candidates):
        raise ValueError(f"k = {k} is larger than the candidate pool ({len(candidates)} items)")
    s = np.asarray(scores_u, float)[candidates]
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else np.zeros_like(s)


def mmr(scores_u, candidates_u, k, lam=0.5, genres=None):
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998) re-ranking for genre diversity.

    Greedily adds the candidate maximising (1 - lam) * rel(i) + lam * min_{j in S} dist(i, j),
    with dist = 1 - cos(genre_i, genre_j), the same distance ILD uses. The first item is the
    most relevant one, since the diversity term is undefined for an empty list.

    Parameters
    ----------
    scores_u : array
        The base model's scores for one user, indexed by item id.
    candidates_u : array
        Item ids the reranker may choose from.
    k : int
        Length of the returned list.
    lam : float
        Trade-off between relevance (0) and diversity (1).
    genres : array [n_items, n_genres]
        Binary genre matrix, indexed by item id.

    Returns
    -------
    list
        k item ids, best first.
    """
    candidates = np.asarray(candidates_u)
    rel = _relevance(scores_u, candidates, k)
    g = np.asarray(genres, float)[candidates]
    unit = g / np.linalg.norm(g, axis=1, keepdims=True).clip(min=1e-12)
    dist = 1 - unit @ unit.T                        # [n_candidates, n_candidates]

    selected = [int(np.argmax(rel))]
    available = np.ones(len(candidates), dtype=bool)
    available[selected[0]] = False
    min_dist = dist[selected[0]].copy()             # distance of each candidate to the list so far
    while len(selected) < k:
        objective = np.where(available, (1 - lam) * rel + lam * min_dist, -np.inf)
        best = int(np.argmax(objective))            # ties go to the earlier candidate
        selected.append(best)
        available[best] = False
        min_dist = np.minimum(min_dist, dist[best])
    return candidates[selected].tolist()


def steck_2018_re_rank(scores_u, candidates_u, k, lam=0.5, genres=None, history=None, alpha=0.01):
    """
    Calibrated re-ranking (Steck, 2018).

    Greedily adds the candidate maximising (1 - lam) * sum_{i in S} rel(i) - lam * KL(p || q~), where
    p(g|u) is the mean of p(g|i) over the user's training history and q~ = (1 - alpha) q + alpha p,
    with q(g|S) the rank-weighted (1 / log2(rank + 1)) mean of p(g|i) over the list. These are the
    same definitions as metrics.miscalibration, so the reranker optimises the metric that is reported.

    Parameters
    ----------
    scores_u : array
        The base model's scores for one user, indexed by item id.
    candidates_u : array
        Item ids the reranker may choose from.
    k : int
        Length of the returned list.
    lam : float
        Trade-off between relevance (0) and calibration (1).
    genres : array [n_items, n_genres]
        Binary genre matrix, indexed by item id.
    history : array
        Item ids of the user's training interactions.
    alpha : float
        Smoothing of q, so the KL stays finite when a history genre is missing from the list.

    Returns
    -------
    list
        k item ids, best first.
    """
    candidates = np.asarray(candidates_u)
    rel = _relevance(scores_u, candidates, k)
    g = np.asarray(genres, float)
    p_gi = g[candidates] / g[candidates].sum(axis=1, keepdims=True)
    hist = g[np.asarray(history)]
    p = (hist / hist.sum(axis=1, keepdims=True)).mean(axis=0)
    nz = p > 0
    weights = 1.0 / np.log2(np.arange(2, k + 2))

    selected = []
    available = np.ones(len(candidates), dtype=bool)
    q_sum = np.zeros_like(p)                        # rank-weighted genre mass of the list so far
    while len(selected) < k:
        w = weights[len(selected)]
        # q for every candidate appended at the next rank: [n_candidates, n_genres]
        q = (q_sum + w * p_gi) / weights[: len(selected) + 1].sum()
        q_tilde = (1 - alpha) * q + alpha * p
        kl = np.sum(p[nz] * np.log(p[nz] / q_tilde[:, nz]), axis=1)
        # sum of rel over S is the same for every candidate, so only rel(i) changes the argmax
        objective = np.where(available, (1 - lam) * rel - lam * kl, -np.inf)
        best = int(np.argmax(objective))
        selected.append(best)
        available[best] = False
        q_sum += w * p_gi[best]
    return candidates[selected].tolist()


def rerank_all(rerank, scores, users, k, lam, n_candidates=100, histories=None, **context):
    """
    Run one reranker over many users.

    Each user's pool is their top n_candidates items by score (fewer if they have fewer finite
    scores, since seen and padding items are -inf). histories, if given, maps a user id to their
    training items and is passed to the reranker as history=.

    Returns
    -------
    array [len(users), k]
        Reranked item ids, one row per user.
    """
    out = np.empty((len(users), k), dtype=int)
    for row, u in enumerate(users):
        order = np.argsort(-scores[u], kind="stable")[:n_candidates]
        pool = order[np.isfinite(scores[u][order])]
        extra = {} if histories is None else {"history": histories[u]}
        out[row] = rerank(scores[u], pool, k, lam, **context, **extra)
    return out
