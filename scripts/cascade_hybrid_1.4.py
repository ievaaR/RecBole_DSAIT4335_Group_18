import torch

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

import math
import numpy as np

EASE_CHECKPOINT = (
    "saved/ml-100k-EASE-Oct-04-2026_00-19-16.pth"
)

SLIM_CHECKPOINT = (
    "saved/ml-100k-SLIMElastic-Oct-04-2026_00-19-36.pth"
)

CANDIDATE_SIZES = [20, 50, 100, 200]
FINAL_K = 10

print("Loading EASE...")

(
    ease_config,
    ease_model,
    ease_dataset,
    ease_train,
    ease_valid,
    ease_test,
) = load_data_and_model(
    model_file=EASE_CHECKPOINT
)

ease_model.eval()


print("Loading SLIMElastic...")

(
    slim_config,
    slim_model,
    slim_dataset,
    slim_train,
    slim_valid,
    slim_test,
) = load_data_and_model(
    model_file=SLIM_CHECKPOINT
)

slim_model.eval()

print("Models loaded.")

print("\nChecking configurations...")

print("EASE seed:", ease_config["seed"])
print("SLIM seed:", slim_config["seed"])

print("EASE eval_args:", ease_config["eval_args"])
print("SLIM eval_args:", slim_config["eval_args"])

assert ease_dataset.user_num == slim_dataset.user_num
assert ease_dataset.item_num == slim_dataset.item_num

print("Users:", ease_dataset.user_num)
print("Items:", ease_dataset.item_num)

uid_field = ease_dataset.uid_field

valid_user_ids = (
    ease_valid.dataset
    .inter_feat[uid_field]
    .numpy()
)

uid_series = sorted(
    set(int(u) for u in valid_user_ids)
)

print(
    f"Number of validation users: {len(uid_series)}"
)

def generate_candidates(
    model,
    data,
    config,
    users,
    candidate_size
):

    scores, items = full_sort_topk(
        users,
        model,
        data,
        k=candidate_size,
        device=config["device"]
    )

    return scores, items

ease_scores, ease_items = generate_candidates(
    ease_model,
    ease_valid,
    ease_config,
    uid_series,
    candidate_size=100
)

print("Candidate tensor shape:")
print(ease_items.shape)

def get_full_scores(
    model,
    data,
    users,
    device
):

    all_scores = []

    for user_id in users:

        interaction = {
            data.dataset.uid_field:
                torch.tensor(
                    [user_id],
                    device=device
                )
        }

        from recbole.data.interaction import Interaction

        interaction = Interaction(interaction)

        scores = model.full_sort_predict(
            interaction
        )

        scores = scores.view(-1)

        all_scores.append(
            scores.detach().cpu()
        )

    return torch.stack(all_scores)

def cascade_rerank(
    candidate_items,
    reranker_scores,
    final_k=10
):

    recommendations = []

    for row in range(
        candidate_items.shape[0]
    ):

        candidates = candidate_items[row]

        candidate_scores = (
            reranker_scores[row][candidates]
        )

        _, order = torch.topk(
            candidate_scores,
            k=final_k
        )

        final_items = candidates[order]

        recommendations.append(final_items)

    return torch.stack(recommendations)

ease_candidate_scores, ease_candidates = (
    generate_candidates(
        ease_model,
        ease_valid,
        ease_config,
        uid_series,
        candidate_size=100
    )
)

slim_full_scores = get_full_scores(
    slim_model,
    slim_valid,
    uid_series,
    slim_config["device"]
)

ease_to_slim = cascade_rerank(
    ease_candidates,
    slim_full_scores,
    final_k=10
)

print(
    "EASE -> SLIM result shape:",
    ease_to_slim.shape
)

def get_ground_truth(data, users):
    """
    Build:
        user_id -> set of relevant item IDs

    from the validation/test interactions.
    """

    dataset = data.dataset

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    user_array = (
        dataset.inter_feat[uid_field]
        .cpu()
        .numpy()
    )

    item_array = (
        dataset.inter_feat[iid_field]
        .cpu()
        .numpy()
    )

    ground_truth = {
        int(user): set()
        for user in users
    }

    for user, item in zip(
        user_array,
        item_array
    ):
        user = int(user)
        item = int(item)

        if user in ground_truth:
            ground_truth[user].add(item)

    return ground_truth

def evaluate_recommendations(
    recommendations,
    users,
    ground_truth,
    k=10
):

    recalls = []
    mrrs = []
    ndcgs = []
    hits = []
    precisions = []

    recommendations = (
        recommendations
        .cpu()
        .numpy()
    )

    for row, user in enumerate(users):

        relevant = ground_truth[user]

        recommended = recommendations[row][:k]

        # --------------------------------
        # Binary relevance vector
        # --------------------------------

        relevance = [
            1 if int(item) in relevant else 0
            for item in recommended
        ]

        number_hits = sum(relevance)

        # --------------------------------
        # Recall@K
        # --------------------------------

        if len(relevant) > 0:
            recall = number_hits / len(relevant)
        else:
            recall = 0.0

        recalls.append(recall)

        # --------------------------------
        # Precision@K
        # --------------------------------

        precision = number_hits / k

        precisions.append(precision)

        # --------------------------------
        # Hit@K
        # --------------------------------

        hit = 1.0 if number_hits > 0 else 0.0

        hits.append(hit)

        # --------------------------------
        # MRR@K
        # --------------------------------

        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(
            relevance,
            start=1
        ):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        mrrs.append(reciprocal_rank)

        # --------------------------------
        # NDCG@K
        # --------------------------------

        dcg = 0.0

        for rank, is_relevant in enumerate(
            relevance,
            start=1
        ):
            if is_relevant:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(
            len(relevant),
            k
        )

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_hits + 1
            )
        )

        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0

        ndcgs.append(ndcg)

    return {
        "recall@10": float(np.mean(recalls)),
        "mrr@10": float(np.mean(mrrs)),
        "ndcg@10": float(np.mean(ndcgs)),
        "hit@10": float(np.mean(hits)),
        "precision@10": float(
            np.mean(precisions)
        ),
    }


valid_ground_truth = get_ground_truth(
    ease_valid,
    uid_series
)

metrics = evaluate_recommendations(
    ease_to_slim,
    uid_series,
    valid_ground_truth,
    k=10
)

print("\n")
print("=" * 60)
print("EASE -> SLIM")
print("Candidate size: 100")
print("=" * 60)

for metric, value in metrics.items():
    print(
        f"{metric}: {value:.4f}"
    )