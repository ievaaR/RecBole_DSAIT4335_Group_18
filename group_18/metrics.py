import numpy as np
import pandas as pd
from scipy.stats import rankdata

K = 10  # main cut-off, same as Task 1
KS = (5, 10, 20)

trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ============================================================================
# Rating-prediction metrics
# ============================================================================

def mae(y_true, y_pred):
    """Mean absolute error over all test ratings."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """Mean squared error over all test ratings."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# ============================================================================
# Top-K metrics (per user)
# ============================================================================

def relevance_vector(ranked, relevant):
    """Binary vector: 1 where ranked[i] is a relevant (held-out) item."""
    relevant = set(relevant)
    return np.fromiter((item in relevant for item in ranked), dtype=float, count=len(ranked))


def _hits(rel, k):
    return rel[:k].sum()


def _precision(rel, n_rel, k):
    return _hits(rel, k) / k


def _recall(rel, n_rel, k):
    return _hits(rel, k) / n_rel


def _hit(rel, n_rel, k):
    return float(_hits(rel, k) > 0)


def _f1(rel, n_rel, k):
    p, r = _precision(rel, n_rel, k), _recall(rel, n_rel, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _fallout(rel, n_rel, k, n_neg):
    # FP / (FP + TN), where FP + TN = number of negative candidates
    if n_neg == 0:
        return np.nan
    fp = min(k, len(rel)) - _hits(rel, k)
    return fp / n_neg


def _dcg(gains, k):
    gains = np.asarray(gains, float)[:k]
    return np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))


def _ndcg(rel, n_rel, k):
    # binary relevance: ideal list has min(K, |G|) relevant items at the top
    idcg = _dcg(np.ones(min(k, n_rel)), k)
    return _dcg(rel, k) / idcg


def _mrr(rel, n_rel, k):
    hit_ranks = np.flatnonzero(rel[:k])
    return 0.0 if len(hit_ranks) == 0 else 1.0 / (hit_ranks[0] + 1)


def _ap(rel, n_rel, k, denom="min"):
    rel_k = rel[:k]
    prec_at_i = np.cumsum(rel_k) / np.arange(1, len(rel_k) + 1)
    total = np.sum(prec_at_i * rel_k)
    if denom == "min":          # min(K, |G|), default, same as RecBole
        d = min(k, n_rel)
    elif denom == "hits":       # number of hits in the top K, as in the slide example
        d = rel_k.sum()
    elif denom == "relevant":   # |G|
        d = n_rel
    else:
        raise ValueError(f"unknown denom {denom!r}")
    return 0.0 if d == 0 else total / d


# ---- public per-user API: ranked list of item ids + collection of relevant ids ----

def precision_at_k(ranked, relevant, k):
    return _precision(relevance_vector(ranked, relevant), len(set(relevant)), k)


def recall_at_k(ranked, relevant, k):
    return _recall(relevance_vector(ranked, relevant), len(set(relevant)), k)


def hit_at_k(ranked, relevant, k):
    return _hit(relevance_vector(ranked, relevant), len(set(relevant)), k)


def f1_at_k(ranked, relevant, k):
    return _f1(relevance_vector(ranked, relevant), len(set(relevant)), k)


def fallout_at_k(ranked, relevant, k):
    """`ranked` must be the FULL ranking of all candidate items, so the number of negatives is known."""
    rel = relevance_vector(ranked, relevant)
    return _fallout(rel, len(set(relevant)), k, n_neg=len(rel) - rel.sum())


def ndcg_at_k(ranked, relevant, k, gains=None):
    """Binary NDCG@K. With `gains={item: grade}` it uses graded relevance 2^grade - 1 instead."""
    if gains is None:
        return _ndcg(relevance_vector(ranked, relevant), len(set(relevant)), k)
    g = np.array([2.0 ** gains.get(item, 0) - 1 for item in ranked])
    ideal = np.sort(2.0 ** np.array(list(gains.values()), float) - 1)[::-1]
    idcg = _dcg(ideal, k)
    return 0.0 if idcg == 0 else _dcg(g, k) / idcg


def mrr_at_k(ranked, relevant, k):
    return _mrr(relevance_vector(ranked, relevant), len(set(relevant)), k)


def ap_at_k(ranked, relevant, k, denom="min"):
    """Average precision@K. denom: 'min' -> min(K,|G|) (default), 'hits' -> hits in top K, 'relevant' -> |G|."""
    return _ap(relevance_vector(ranked, relevant), len(set(relevant)), k, denom)

# ============================================================================
# AUC and ROC
# ============================================================================

def auc_user(pos_scores, neg_scores):
    """P(score_pos > score_neg) + 0.5 * P(tie), over all positive/negative pairs."""
    pos_scores, neg_scores = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(np.concatenate([pos_scores, neg_scores]))  # ascending, ties -> average rank
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def roc_points(rel):
    """(fallout, recall) for K = 0..len(rel), given the relevance vector of a full candidate ranking."""
    rel = np.asarray(rel, float)
    n_pos = rel.sum()
    n_neg = len(rel) - n_pos
    tp = np.concatenate([[0], np.cumsum(rel)])
    fp = np.arange(len(rel) + 1) - tp
    return fp / n_neg, tp / n_pos


def recall_on_grid(fallout, recall, grid):
    """Evaluate a ROC step curve at the fallout values in `grid` (highest recall reached by that fallout)."""
    idx = np.searchsorted(fallout, grid, side="right") - 1
    return np.maximum.accumulate(recall)[idx]

# ============================================================================
# Evaluating a whole model
# ============================================================================

TOPK_METRICS = {
    "precision": _precision,
    "recall": _recall,
    "f1": _f1,
    "hit": _hit,
    "ndcg": _ndcg,
    "mrr": _mrr,
    "map": _ap,
}


def _summarise(rows, ks):
    per_user = pd.DataFrame(rows).set_index("user")
    table = pd.DataFrame(
        {k: {m: per_user[f"{m}@{k}"].mean() for m in list(TOPK_METRICS) + ["fallout"] if f"{m}@{k}" in per_user}
         for k in ks}
    )
    table.columns = [f"@{k}" for k in ks]
    table.index.name = "metric"
    return table, per_user


def evaluate_scores(scores, test_items, ks=KS, roc_grid=1001):
    """Evaluate a score matrix.

    scores     : array [n_users, n_items]; non-candidate items must be -inf.
    test_items : {user_row: iterable of item columns}; users without test items are skipped.
    Returns a dict with:
        'topk'     : DataFrame, metric x K, averaged over users
        'auc'      : mean per-user AUC
        'roc'      : DataFrame(fallout, recall), the user-averaged ROC curve
        'per_user' : DataFrame with every per-user value (for significance tests, breakdowns...)
    """
    scores = np.asarray(scores, dtype=float)
    grid = np.linspace(0, 1, roc_grid)
    rows, roc_sum, n_roc = [], np.zeros_like(grid), 0

    for u, items in test_items.items():
        s = scores[u]
        candidates = np.flatnonzero(np.isfinite(s))
        is_pos = np.zeros(scores.shape[1], dtype=bool)
        is_pos[list(items)] = True
        is_pos &= np.isfinite(s)  # a test item that is also in the history is not a candidate
        n_rel = is_pos.sum()
        if n_rel == 0:
            continue

        order = candidates[np.argsort(-s[candidates], kind="stable")]
        rel = is_pos[order].astype(float)
        n_neg = len(rel) - n_rel

        row = {"user": u, "n_test": n_rel, "n_candidates": len(rel)}
        for k in ks:
            for name, fn in TOPK_METRICS.items():
                row[f"{name}@{k}"] = fn(rel, n_rel, k)
            row[f"fallout@{k}"] = _fallout(rel, n_rel, k, n_neg)
        row["auc"] = auc_user(s[is_pos], s[candidates[~is_pos[candidates]]])
        rows.append(row)

        if n_neg > 0:
            roc_sum += recall_on_grid(*roc_points(rel), grid)
            n_roc += 1

    table, per_user = _summarise(rows, ks)
    return {
        "topk": table,
        "auc": per_user["auc"].mean(),  # NaN users (no negatives) are skipped by mean()
        "roc": pd.DataFrame({"fallout": grid, "recall": roc_sum / max(n_roc, 1)}),
        "per_user": per_user,
    }


def evaluate_topk(recs, test_items, ks=KS):
    """Evaluate ranked top-K lists {user: [item, ...]} (list length >= max(ks))."""
    rows = []
    for u, items in test_items.items():
        relevant = set(items)
        if not relevant:
            continue
        rel = relevance_vector(recs.get(u, []), relevant)
        row = {"user": u, "n_test": len(relevant)}
        for k in ks:
            for name, fn in TOPK_METRICS.items():
                row[f"{name}@{k}"] = fn(rel, len(relevant), k)
        rows.append(row)
    table, per_user = _summarise(rows, ks)
    return {"topk": table, "per_user": per_user}

# ============================================================================
# Beyond-accuracy metrics
# ============================================================================
# Shared conventions: items are integer indices 0..n_items-1 into `genres` / `pop`;
# `recs` is an int array [n_users, K] of top-K lists, best first; `histories` is a list
# (same user order as `recs`) of each user's training items. Per-user metrics return one value
# per row of `recs`; average them over users for the reported number.

def item_popularity(train_users, train_items, n_items):
    """pop(i): number of distinct training users who interacted with item i."""
    pairs = np.unique(np.stack([np.asarray(train_users), np.asarray(train_items)]), axis=1)
    return np.bincount(pairs[1], minlength=n_items)


def _rank_weights(k):
    return 1.0 / np.log2(np.arange(2, k + 2))


def ild(recs, genres):
    """ILD@K per user: mean of 1 - cos(genre_i, genre_j) over all pairs i < j in the list."""
    recs = np.asarray(recs)
    g = np.asarray(genres, float)
    unit = g / np.linalg.norm(g, axis=1, keepdims=True).clip(min=1e-12)
    v = unit[recs]                                  # [n_users, K, G]
    sim = v @ v.transpose(0, 2, 1)                  # [n_users, K, K]
    iu = np.triu_indices(recs.shape[1], k=1)
    return (1 - sim[:, iu[0], iu[1]]).mean(axis=1)


def novelty(recs, pop, n_users):
    """Novelty@K per user: mean of -log2((pop(i) + 1) / (n_users + 1)) over the list."""
    self_info = -np.log2((np.asarray(pop, float) + 1) / (n_users + 1))
    return self_info[np.asarray(recs)].mean(axis=1)


def miscalibration(recs, histories, genres, alpha=0.01):
    """MC_KL@K per user (Steck, 2018): KL(p(g|u) || (1-alpha) q(g|u) + alpha p(g|u)).

    p(g|u): uniform mean of p(g|i) over the training history; q(g|u): mean of p(g|i) over the list
    with rank weights 1/log2(rank + 1). Genres with p(g|u) = 0 contribute 0.
    """
    recs = np.asarray(recs)
    g = np.asarray(genres, float)
    p_gi = g / g.sum(axis=1, keepdims=True)
    w = _rank_weights(recs.shape[1])
    out = np.empty(len(recs))
    for u, (rec, hist) in enumerate(zip(recs, histories)):
        p = p_gi[np.asarray(hist)].mean(axis=0)
        q = w @ p_gi[rec] / w.sum()
        q_tilde = (1 - alpha) * q + alpha * p
        nz = p > 0
        out[u] = np.sum(p[nz] * np.log(p[nz] / q_tilde[nz]))
    return out


def exposure_counts(recs, n_items):
    """How often each catalog item appears across all top-K lists (0 for unrecommended items)."""
    return np.bincount(np.asarray(recs).ravel(), minlength=n_items)


def catalog_coverage(recs, n_items):
    """Coverage@K: |union of all lists| / n_items."""
    return len(np.unique(recs)) / n_items


def gini(x):
    """Gini index of a non-negative vector: 0 = all equal, (n-1)/n = everything on one entry."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum())


def exposure_gini(recs, n_items):
    """Gini@K of the item exposure counts over the whole catalog."""
    return gini(exposure_counts(recs, n_items))


def short_head_mask(pop, frac=0.2):
    """True for the top `frac` of items by popularity (ties broken by item index)."""
    pop = np.asarray(pop)
    mask = np.zeros(len(pop), dtype=bool)
    mask[np.argsort(-pop, kind="stable")[: int(round(frac * len(pop)))]] = True
    return mask


def group_exposure(recs, short_head):
    """Item-side group exposure@K: per-item, rank-discounted exposure of short head vs. long tail.

    Exposure(G) = (1/|G|) * sum_u sum_{i in R_K(u) and G} 1/log2(1 + pos(i)).
    Returns {'short_head', 'long_tail', 'ratio'} with ratio = short_head / long_tail (1 = parity).
    """
    recs = np.asarray(recs)
    short_head = np.asarray(short_head, bool)
    w = np.broadcast_to(_rank_weights(recs.shape[1]), recs.shape)
    in_head = short_head[recs]
    head = w[in_head].sum() / short_head.sum()
    tail = w[~in_head].sum() / (~short_head).sum()
    return {"short_head": head, "long_tail": tail, "ratio": head / tail if tail > 0 else np.inf}


def gru(values, in_group_1):
    """Group recommendation unfairness: |mean F over group 1 - mean F over group 2|."""
    values, in_group_1 = np.asarray(values, float), np.asarray(in_group_1, bool)
    return abs(values[in_group_1].mean() - values[~in_group_1].mean())


HEAD, MID, TAIL = 0, 1, 2


def popularity_bins(pop, head=0.2, tail=0.2):
    """HEAD / MID / TAIL label per item from its share of training interactions.

    Items are sorted by popularity. Head: the most popular items until together they reach `head` of
    all interactions; tail: the least popular items until together they reach `tail`; mid: the rest.
    """
    pop = np.asarray(pop, float)
    order = np.argsort(-pop, kind="stable")
    share = pop[order] / pop.sum()
    from_top = np.cumsum(share) - share             # share of strictly more popular items
    from_bottom = np.cumsum(share[::-1])[::-1] - share
    labels = np.full(len(pop), MID)
    labels[order[from_top < head]] = HEAD
    labels[order[from_bottom < tail]] = TAIL
    return labels


def _bin_shares(items, bins):
    return np.bincount(bins[np.asarray(items)], minlength=3) / len(items)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (0..1)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return np.sum(a[nz] * np.log2(a[nz] / b[nz]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def upd(recs, histories, bins):
    """UPD@K per user: JSD between head/mid/tail shares of the training profile and of the list."""
    bins = np.asarray(bins)
    return np.array([js_divergence(_bin_shares(h, bins), _bin_shares(r, bins))
                     for r, h in zip(np.asarray(recs), histories)])


def arp(recs, pop, n_users):
    """ARP@K per user: mean of pop(i) / n_users over the list."""
    return (np.asarray(pop, float) / n_users)[np.asarray(recs)].mean(axis=1)


def aplt(recs, long_tail):
    """APLT@K per user: fraction of the list that is long-tail."""
    return np.asarray(long_tail, bool)[np.asarray(recs)].mean(axis=1)


def evaluate_beyond_accuracy(recs, histories, genres, pop, n_users, user_groups=None, accuracy=None):
    """All beyond-accuracy metrics for one model.

    recs        : int array [n_eval_users, K] of item indices
    histories   : training items per row of `recs`
    genres      : [n_items, n_genres] binary genre matrix
    pop         : [n_items] distinct training users per item
    n_users     : number of users in training (for novelty / ARP)
    user_groups : {name: bool array per row of recs, True = group 1}, for GRU
    accuracy    : per-user accuracy values per row of recs (e.g. NDCG@K), for GRU
    Returns (summary: pd.Series, per_user: pd.DataFrame).
    """
    recs = np.asarray(recs)
    n_items = len(pop)
    k = recs.shape[1]
    head = short_head_mask(pop)
    per_user = pd.DataFrame({
        f"ild@{k}": ild(recs, genres),
        f"novelty@{k}": novelty(recs, pop, n_users),
        f"mc_kl@{k}": miscalibration(recs, histories, genres),
        f"upd@{k}": upd(recs, histories, popularity_bins(pop)),
        f"arp@{k}": arp(recs, pop, n_users),
        f"aplt@{k}": aplt(recs, ~head),
    })
    summary = per_user.mean()
    summary[f"coverage@{k}"] = catalog_coverage(recs, n_items)
    summary[f"gini@{k}"] = exposure_gini(recs, n_items)
    exp = group_exposure(recs, head)
    summary[f"exposure_head@{k}"] = exp["short_head"]
    summary[f"exposure_tail@{k}"] = exp["long_tail"]
    summary[f"exposure_ratio@{k}"] = exp["ratio"]
    if user_groups is not None and accuracy is not None:
        for name, mask in user_groups.items():
            summary[f"gru_{name}"] = gru(accuracy, mask)
    return summary, per_user
