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


def xquad(scores_u, candidates_u, k, lam=0.5, short_head=None, history=None, smooth=False):
    """
    xQuAD for popularity bias (Abdollahpouri et al., 2019).

    Greedily adds the candidate maximising
    (1 - lam) * rel(i) + lam * sum_{c in {head, tail}} P(c|u) * 1[i in c] * cov(c, S),
    where cov(c, S) is how far S still is from covering c:
      * binary (the paper's default): prod_{j in S} (1 - 1[j in c]), i.e. 1 until S has an item of c, then 0
      * smooth: 1 - |S ∩ c| / |S|, so the head/tail mix of the list tracks P(c|u)
    P(c|u) is the user's head/tail share of their history (personalised) or 0.5/0.5 if history is None.

    Parameters
    ----------
    scores_u : array
        The base model's scores for one user, indexed by item id.
    candidates_u : array
        Item ids the reranker may choose from.
    k : int
        Length of the returned list.
    lam : float
        Trade-off between relevance (0) and popularity coverage (1).
    short_head : array [n_items] of bool
        True for short-head items (metrics.short_head_mask), indexed by item id.
    history : array or None
        Item ids of the user's training interactions; None gives uniform P(head|u) = P(tail|u) = 0.5.
    smooth : bool
        Use the smooth coverage term instead of the binary one.

    Returns
    -------
    list
        k item ids, best first.
    """
    candidates = np.asarray(candidates_u)
    rel = _relevance(scores_u, candidates, k)
    head = np.asarray(short_head, bool)
    is_head = head[candidates]
    p_head = 0.5 if history is None else head[np.asarray(history)].mean()
    p_cat = np.where(is_head, p_head, 1 - p_head)   # P(c|u) of each candidate's own category

    selected = []
    available = np.ones(len(candidates), dtype=bool)
    n_head = 0                                      # head items in the list so far
    while len(selected) < k:
        n = len(selected)
        n_same = np.where(is_head, n_head, n - n_head)
        if smooth:
            cov = 1 - n_same / n if n else np.ones(len(candidates))
        else:
            cov = (n_same == 0).astype(float)
        objective = np.where(available, (1 - lam) * rel + lam * p_cat * cov, -np.inf)
        best = int(np.argmax(objective))            # ties go to the earlier candidate
        selected.append(best)
        available[best] = False
        n_head += int(is_head[best])
    return candidates[selected].tolist()


def candidate_pool(scores_u, n_candidates=100):
    """A user's top n_candidates item ids by score, best first, dropping -inf (seen and padding) items."""
    order = np.argsort(-scores_u, kind="stable")[:n_candidates]
    return order[np.isfinite(scores_u[order])]


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
        pool = candidate_pool(scores[u], n_candidates)
        extra = {} if histories is None else {"history": histories[u]}
        out[row] = rerank(scores[u], pool, k, lam, **context, **extra)
    return out


def ugf(scores, users, k, eps, in_group_1, n_candidates=100, time_limit=None):
    """
    User-oriented fairness re-ranking (Li et al., 2021), solved as one ILP over all users.

    Unlike the other rerankers this one couples users, so it takes the whole score matrix. With
    binary x_ui over each user's pool and quality Q_u = (1/k) sum_i rel_ui x_ui (rel = min-max
    normalised scores, a label-free proxy for the user's expected accuracy):

        max  sum_{u,i} rel_ui x_ui
        s.t. sum_i x_ui = k                                   for every user
             |mean_{u in G1} Q_u - mean_{u in G2} Q_u| <= eps

    eps plays the role of lam, inverted: a large eps returns the base top-k, smaller eps forces the
    groups' proxy quality closer together. eps = 0 can be infeasible, since the x_ui are integers.

    Parameters
    ----------
    scores : array [n_users, n_items]
        Score matrix, -inf for seen and padding items.
    users : array
        User ids (rows of scores) to rerank.
    k : int
        Length of each list.
    eps : float
        Largest allowed gap in mean proxy quality between the two groups.
    in_group_1 : array [len(users)] of bool
        Group membership, aligned with users (e.g. True for active users).
    n_candidates : int
        Pool size per user; 30 is much faster if 100 is too slow.
    time_limit : float or None
        Solver time limit in seconds.

    Returns
    -------
    array [len(users), k]
        Reranked item ids, one row per user, ordered by base score within each list.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    in_group_1 = np.asarray(in_group_1, bool)
    if in_group_1.all() or not in_group_1.any():
        raise ValueError("both user groups must be non-empty")
    pools = [candidate_pool(scores[u], n_candidates) for u in users]
    rels = [_relevance(scores[u], pool, k) for u, pool in zip(users, pools)]
    sizes = np.array([len(p) for p in pools])
    row = np.repeat(np.arange(len(users)), sizes)  # user row of every variable
    rel = np.concatenate(rels)

    # exactly k items
    pick_k = csr_matrix((np.ones(len(rel)), (row, np.arange(len(rel)))), shape=(len(users), len(rel)))
    # gap = \sum_ui w_u rel_ui x_ui,  w_u = (+/-)1/(k|G1|)
    w = np.where(in_group_1, 1 / (k * in_group_1.sum()), -1 / (k * (~in_group_1).sum()))
    gap = (w[row] * rel)[None, :]

    res = milp(
        c=-rel,
        constraints=[LinearConstraint(pick_k, k, k), LinearConstraint(gap, -eps, eps)],
        integrality=np.ones(len(rel)),
        bounds=Bounds(0, 1),
        options={} if time_limit is None else {"time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"UGF ILP found no solution for eps = {eps}: {res.message}")

    chosen = res.x > 0.5
    out = np.empty((len(users), k), dtype=int)
    start = 0
    for r, pool in enumerate(pools):
        picked = chosen[start : start + len(pool)]
        out[r] = pool[picked]                       # pool in base score order
        start += len(pool)
    return out


def proxy_quality(scores, users, recs, n_candidates=100):
    """Q_u from ugf: the mean min-max normalised score of each user's list, over their top-n pool."""
    out = np.empty(len(users))
    for r, u in enumerate(users):
        pool = candidate_pool(scores[u], n_candidates)
        s = scores[u][pool]
        span = s.max() - s.min()
        out[r] = ((scores[u][recs[r]] - s.min()) / span).mean() if span > 0 else 0.0
    return out
