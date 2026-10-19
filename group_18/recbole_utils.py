"""Glue between RecBole and metrics.py: the Task 1 split, trained models and score matrices.

Score matrices are [n_users, n_items] arrays indexed by RecBole's internal ids. The padding item 0
and each user's history (train for valid_data, train + valid for test_data) are -inf, which is the
candidate convention metrics.evaluate_scores expects.
"""
from pathlib import Path

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import ModelType, get_model, get_trainer, init_seed


def find_project_root(start: Path) -> Path:
    for directory in (start, *start.parents):
        if (directory / "run_recbole.py").exists():
            return directory
    raise FileNotFoundError("Could not find project root (run_recbole.py)")


ROOT = find_project_root(Path(__file__).resolve())
DATASET = "ml-100k"
SEED = 2020
CONFIG_DIR = ROOT / "recbole" / "config"
CHECKPOINT_DIR = ROOT / "saved"


def load_split(model, config_name=None, **overrides):
    """Config + dataset + (train, valid, test) loaders, built exactly like Task 1."""
    config = Config(
        model=model,
        dataset=DATASET,
        config_file_list=[str(CONFIG_DIR / (config_name or model) / f"{DATASET}.yaml")],
        config_dict={"seed": SEED, "reproducibility": True, "use_gpu": False,
                     "data_path": str(ROOT / "dataset"), "checkpoint_dir": str(CHECKPOINT_DIR),
                     **overrides},
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return config, dataset, train_data, valid_data, test_data


def build_model(model, config_name=None, **overrides):
    """Build (and, unless it is fitted at construction, train) a RecBole model on the Task 1 split.

    Traditional models (ItemKNN, EASE, SLIMElastic) compute their weights in __init__.
    Trainable models are fitted with early stopping on validation MRR@10 and the best
    checkpoint is reloaded, as run_recbole does. Returns (model, config, dataset, train, valid, test).
    """
    config, dataset, train_data, valid_data, test_data = load_split(model, config_name, **overrides)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    net = get_model(config["model"])(config, train_data._dataset).to(config["device"])

    if config["MODEL_TYPE"] != ModelType.TRADITIONAL:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, net)
        trainer.fit(train_data, valid_data, saved=True, show_progress=False, verbose=False)
        checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"])
        net.load_state_dict(checkpoint["state_dict"])
        net.load_other_parameter(checkpoint.get("other_parameter"))

    net.eval()
    return net, config, dataset, train_data, valid_data, test_data


@torch.no_grad()
def score_matrix(score_fn, eval_data, n_users, n_items, uid_field="user_id"):
    """Run score_fn over a full-sort dataloader. Returns (scores [n_users, n_items], test_items).

    score_fn(interaction) -> [batch, n_items] tensor: model.full_sort_predict, or any hybrid.
    In these batches positive_u / history_u are row positions in the batch, not user ids.
    """
    scores = np.full((n_users, n_items), -np.inf, dtype=np.float32)
    test_items = {}
    for interaction, history_index, positive_u, positive_i in eval_data:
        batch = score_fn(interaction).view(len(interaction), -1).float().clone()
        batch[:, 0] = -torch.inf
        if history_index is not None:
            batch[history_index[0], history_index[1]] = -torch.inf
        uids = interaction[uid_field].long().cpu().numpy()
        scores[uids] = batch.cpu().numpy()
        for row, item in zip(positive_u.tolist(), positive_i.tolist()):
            test_items.setdefault(int(uids[row]), set()).add(item)
    return scores, test_items


def full_sort_fn(model, dataset):
    """interaction -> [batch, n_items] scores for a RecBole model.

    Models without full_sort_predict (e.g. NeuMF) score every (user, item) pair with predict(),
    the same fallback RecBole's trainer uses.
    """
    items = dataset.get_item_feature()

    def fn(interaction):
        try:
            return model.full_sort_predict(interaction)
        except NotImplementedError:
            pairs = interaction.repeat_interleave(dataset.item_num)
            pairs.update(items.repeat(len(interaction)))
            return model.predict(pairs)

    return fn


def model_scores(model, dataset, eval_data):
    """Score matrix of a RecBole model."""
    model.eval()
    return score_matrix(full_sort_fn(model, dataset), eval_data, dataset.user_num, dataset.item_num, dataset.uid_field)


def recbole_scores(model_file, split="test"):
    """Score matrix + test_items for a saved RecBole checkpoint (internal ids)."""
    from recbole.quick_start import load_data_and_model

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file)
    return model_scores(model, dataset, test_data if split == "test" else valid_data)


def top_k(scores, users, k):
    """Top-k internal item ids per user (rows of `scores`), best first; ties broken by item id."""
    return np.argsort(-scores[users], axis=1, kind="stable")[:, :k]
