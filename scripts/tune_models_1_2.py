import csv
import itertools
import os

from recbole.quick_start import run_recbole


# ============================================================
# SETTINGS
# ============================================================

DATASET = "ml-100k"

# Change this to choose which model to tune.
# Options:
# BPR, FISM, ItemKNN, UserKNN, SLIMElastic, EASE
MODEL_TO_TUNE = "EASE"

OUTPUT_FILE = "tuning_results.csv"


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODELS = {

    "BPR": {
        "recbole_model": "BPR",
        "config_file": "recbole/config/BPR/ml-100k.yaml",
        "grid": {
            "embedding_size": [32, 64, 128],
            "learning_rate": [0.0005, 0.001, 0.005],
        },
    },

    "FISM": {
        "recbole_model": "FISM",
        "config_file": "recbole/config/FISM/ml-100k.yaml",
        "grid": {
            "embedding_size": [32, 64, 128],
            "learning_rate": [0.0005, 0.001, 0.005],
            "reg_weights": [
                [0.001, 0.001],
                [0.01, 0.01],
                [0.1, 0.1],
            ],
        },
    },

    "ItemKNN": {
        "recbole_model": "ItemKNN",
        "config_file": "recbole/config/ItemKNN/ml-100k.yaml",
        "grid": {
            "k": [20, 50, 100, 200],
        },
    },

    "UserKNN": {
        # Important:
        # Your course implementation uses ItemKNN class
        # with knn_method='user' in the UserKNN YAML.
        "recbole_model": "ItemKNN",
        "config_file": "recbole/config/UserKNN/ml-100k.yaml",
        "grid": {
            "k": [20, 50, 100, 200],
        },
    },

    "SLIMElastic": {
        "recbole_model": "SLIMElastic",
        "config_file": "recbole/config/SLIMElastic/ml-100k.yaml",
        "grid": {
            "alpha": [0.01, 0.1, 0.2, 0.5],
            "l1_ratio": [0.01, 0.02, 0.1, 0.5],
        },
    },

    "EASE": {
        "recbole_model": "EASE",
        "config_file": "recbole/config/EASE/ml-100k.yaml",
        "grid": {
            "reg_weight": [50.0, 100.0, 250.0, 500.0, 1000.0],
        },
    },
}


# ============================================================
# HELPER: CREATE PARAMETER COMBINATIONS
# ============================================================

def generate_combinations(grid):

    keys = list(grid.keys())
    values = list(grid.values())

    for combination in itertools.product(*values):

        yield dict(zip(keys, combination))


# ============================================================
# HELPER: SAVE RESULT IMMEDIATELY
# ============================================================

def save_result(row):

    file_exists = os.path.exists(OUTPUT_FILE)

    fieldnames = [
        "model",
        "parameters",
        "valid_recall@10",
        "valid_mrr@10",
        "valid_ndcg@10",
        "valid_hit@10",
        "valid_precision@10",
    ]

    with open(OUTPUT_FILE, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# TUNE MODEL
# ============================================================

def tune_model(model_name):

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    completed = load_completed_experiments(model_name)

    print(f"Already completed: {len(completed)}")

    settings = MODELS[model_name]

    recbole_model = settings["recbole_model"]
    config_file = settings["config_file"]
    grid = settings["grid"]

    combinations = list(generate_combinations(grid))

    print("\n")
    print("=" * 70)
    print(f"TUNING MODEL: {model_name}")
    print(f"Number of experiments: {len(combinations)}")
    print("=" * 70)

    results = []

    for experiment_number, params in enumerate(
        combinations,
        start=1
    ):
        params_string = str(params)

        if params_string in completed:
            print(f"Skipping completed experiment: {params}")
            continue

        print("\n")
        print("-" * 70)
        print(
            f"{model_name} "
            f"Experiment "
            f"{experiment_number}/{len(combinations)}"
        )
        print("-" * 70)

        print("Parameters:")
        print(params)

        try:

            result = run_recbole(
                model=recbole_model,
                dataset=DATASET,
                config_file_list=[config_file],
                config_dict=params,
                saved=False
            )

            valid = result["best_valid_result"]

            row = {
                "model": model_name,
                "parameters": str(params),
                "valid_recall@10": float(
                    valid["recall@10"]
                ),
                "valid_mrr@10": float(
                    valid["mrr@10"]
                ),
                "valid_ndcg@10": float(
                    valid["ndcg@10"]
                ),
                "valid_hit@10": float(
                    valid["hit@10"]
                ),
                "valid_precision@10": float(
                    valid["precision@10"]
                ),
            }

            results.append(row)

            # Save after EVERY run.
            # If the script crashes later,
            # completed experiments remain in CSV.
            save_result(row)

            print("\nVALIDATION RESULT")
            print(
                f"MRR@10: "
                f"{row['valid_mrr@10']:.4f}"
            )

        except Exception as error:

            print("\nERROR")
            print(
                f"Experiment failed: {params}"
            )
            print(error)

            # Continue to next combination
            # instead of killing the entire search.
            continue

    # ========================================================
    # FIND BEST CONFIGURATION
    # ========================================================

    if not results:

        print(
            f"\nNo successful experiments "
            f"for {model_name}."
        )

        return None

    best = max(
        results,
        key=lambda x: x["valid_mrr@10"]
    )

    print("\n")
    print("=" * 70)
    print(f"BEST {model_name} CONFIGURATION")
    print("=" * 70)

    print("Parameters:")
    print(best["parameters"])

    print("\nValidation metrics:")

    print(
        f"MRR@10:       "
        f"{best['valid_mrr@10']:.4f}"
    )

    print(
        f"Recall@10:    "
        f"{best['valid_recall@10']:.4f}"
    )

    print(
        f"NDCG@10:      "
        f"{best['valid_ndcg@10']:.4f}"
    )

    print(
        f"Hit@10:       "
        f"{best['valid_hit@10']:.4f}"
    )

    print(
        f"Precision@10: "
        f"{best['valid_precision@10']:.4f}"
    )

    return best

def load_completed_experiments(model_name):
    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    with open(OUTPUT_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["model"] == model_name:
                completed.add(row["parameters"])

    return completed

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    best = tune_model(MODEL_TO_TUNE)

    print("\n")
    print("=" * 70)
    print("TUNING FINISHED")
    print("=" * 70)

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )