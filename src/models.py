"""Model definitions: which algorithms we try, and how each is built.

Hyperparameter VALUES live in config.CANDIDATE_MODELS; this file only maps
names to code. To try a new algorithm, add one line to MODEL_CLASSES and one
entry to CANDIDATE_MODELS - nothing else needs to change (open/closed principle)."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CANDIDATE_MODELS, RANDOM_SEED

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "decision_tree_deep": DecisionTreeClassifier,
    "decision_tree_pruned": DecisionTreeClassifier,
    # scikit-learn's histogram-based gradient boosting: the same technique as
    # LightGBM/XGBoost, with no extra dependency
    "gradient_boosting": HistGradientBoostingClassifier,
    "knn": KNeighborsClassifier,
    "neural_network": MLPClassifier,
}

# CONCEPT - feature scaling: these models compare feature VALUES directly, so a
# feature with a big range (BMI: 12-98) would drown out one with a small range
# (HighBP: 0-1). k-NN measures distances; logistic regression and neural networks
# learn weights by gradient steps. All three need features on a similar scale.
# Trees split on thresholds ("BMI > 30?"), so scaling changes nothing for them.
NEEDS_SCALING = {"logistic_regression", "knn", "neural_network"}


def build_model(name: str) -> Pipeline:
    """Build an untrained model as a scikit-learn Pipeline.

    CONCEPT - preventing data leakage: the scaler lives INSIDE the pipeline, so
    during cross-validation it is fit on the training folds only. Scaling the
    whole dataset first would leak information from validation rows into training.

    CONCEPT - one artifact: the saved pipeline includes preprocessing, so the API
    applies exactly the same transformations as training (no training/serving skew)."""
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_CLASSES)}")

    model_class = MODEL_CLASSES[name]
    params = dict(CANDIDATE_MODELS[name])
    # Reproducible results. Not every model is random (k-NN isn't), so only set
    # random_state where the model actually has that parameter.
    if "random_state" in model_class().get_params():
        params["random_state"] = RANDOM_SEED

    steps = []
    if name in NEEDS_SCALING:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", model_class(**params)))
    return Pipeline(steps)
