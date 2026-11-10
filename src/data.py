"""Data stage: download, validate, split, and assemble training data.

Every function here is a small, pure step with no knowledge of MLflow or
Prefect, so each one can be unit-tested on its own (see tests/test_data.py)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    MIN_RECENT_HOLDOUT_ROWS,
    NEW_LABELS,
    NEW_LABELS_TRAIN_SHARE,
    RANDOM_SEED,
    RAW_DATA,
    STREAM_DATA,
    STREAM_SHARE,
    TEST_DATA,
    TEST_SHARE,
    TRAIN_DATA,
    TRAIN_SHARE,
)
from src.schema import FEATURE_RANGES, FEATURES, TARGET

UCI_DATASET_ID = 891  # "CDC Diabetes Health Indicators" on the UCI ML Repository


# -----------------------------------------------------------------------------
# Ingestion
# -----------------------------------------------------------------------------
def download_raw_data() -> pd.DataFrame:  # pragma: no cover  (needs the internet)
    """Download the dataset once and cache it in data/raw.csv.

    Caching makes the pipeline idempotent: running it again doesn't
    re-download, and everyone works from the same file."""
    if RAW_DATA.exists():
        return pd.read_csv(RAW_DATA)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ucimlrepo import fetch_ucirepo  # imported here: only needed for downloading

        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    except Exception as exc:
        raise RuntimeError(
            "Automatic download failed. Download "
            "'diabetes_binary_health_indicators_BRFSS2015.csv' from Kaggle "
            "('Diabetes Health Indicators Dataset') and save it as data/raw.csv."
        ) from exc

    df.to_csv(RAW_DATA, index=False)
    return df


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_features(df: pd.DataFrame) -> list[str]:
    """Check the INPUT columns only (no label needed). Returns a list of problems.

    Used on data where the true outcome isn't known, such as the reference
    sample saved with a model (checked by the quality gate).

    Args:
        df: a table that should contain every feature in src/schema.py.

    Returns:
        Plain-language problems found; an empty list means the features are valid."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]  # other checks would crash without these

    errors = []
    null_counts = df[FEATURES].isna().sum()
    for col, count in null_counts[null_counts > 0].items():
        errors.append(f"{col} has {count} missing values")

    for col, (low, high) in FEATURE_RANGES.items():
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range:
            errors.append(f"{col} has {out_of_range} values outside [{low}, {high}]")
    return errors


def validate_data(df: pd.DataFrame, min_rows: int = 100) -> list[str]:
    """Check labeled data (features AND the diabetes label) before training.

    CONCEPT - data validation: bad data produces bad models silently. Checking
    the data contract (columns, missing values, ranges) before training turns
    silent failures into loud, early ones.

    Args:
        df: the table to check, including the TARGET column.
        min_rows: fewer rows than this is reported as a problem (too little
            data to train or evaluate reliably).

    Returns:
        Plain-language problems found; an empty list means the data is valid."""
    if TARGET not in df.columns:
        return [f"Missing columns: ['{TARGET}']"]

    errors = validate_features(df)
    if errors and errors[0].startswith("Missing columns"):
        return errors

    if len(df) < min_rows:
        errors.append(f"Only {len(df)} rows, expected at least {min_rows}")
    if df[TARGET].isna().any():
        errors.append(f"{TARGET} has {int(df[TARGET].isna().sum())} missing values")
    elif not df[TARGET].isin([0, 1]).all():
        errors.append(f"{TARGET} must contain only 0 and 1")
    return errors


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Take a random sample that keeps the diabetes rate the same as in df.

    Used by quick runs (DATA_SAMPLE_ROWS in config). CONCEPT - stratified
    sampling: a plain random sample could, by chance, have more or fewer people
    with diabetes than the full data; stratifying keeps the proportion equal, so
    the small sample behaves like the real thing.

    Args:
        df: the full labeled dataset.
        rows: how many rows to keep. If df has no more rows than this, it's
            returned unchanged.

    Returns:
        The sample. The fixed RANDOM_SEED means the same rows every time.
    """
    if len(df) <= rows:
        return df
    sample, _ = train_test_split(df, train_size=rows, stratify=df[TARGET], random_state=RANDOM_SEED)
    return sample


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / test / stream using the shares in config.

    CONCEPT - stratification: `stratify=` keeps the diabetes rate (~14%) the
    same in every split, so no split is accidentally easier or harder."""
    df = df[FEATURES + [TARGET]]
    train, rest = train_test_split(df, train_size=TRAIN_SHARE, stratify=df[TARGET], random_state=RANDOM_SEED)
    # Of the remaining rows, what fraction is test? e.g. 0.10 / (0.10 + 0.40) = 0.2
    test_fraction_of_rest = TEST_SHARE / (TEST_SHARE + STREAM_SHARE)
    test, stream = train_test_split(
        rest, train_size=test_fraction_of_rest, stratify=rest[TARGET], random_state=RANDOM_SEED
    )
    return train, test, stream


def save_splits(train: pd.DataFrame, test: pd.DataFrame, stream: pd.DataFrame) -> None:
    """Write the three splits to the active profile's data folder (data/ or data/quick/)."""
    TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DATA, index=False)
    test.to_csv(TEST_DATA, index=False)
    stream.to_csv(STREAM_DATA, index=False)


# -----------------------------------------------------------------------------
# Assembling data for a training run
# -----------------------------------------------------------------------------
def load_training_and_evaluation_data(include_new_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (training data, evaluation data, name of the evaluation set).

    CONCEPT - cumulative retraining: training = original data + newly labeled data.
    CONCEPT - evaluate on recent data: new labels are in arrival order, so the
    newest slice is held out for evaluation. It is closest to what the model will
    see next, and it was never used for training (no leakage)."""
    train = pd.read_csv(TRAIN_DATA)

    if include_new_labels and NEW_LABELS.exists():
        new = pd.read_csv(NEW_LABELS)
        cut = int(len(new) * NEW_LABELS_TRAIN_SHARE)
        older, newest = new.iloc[:cut], new.iloc[cut:]
        train = pd.concat([train, older], ignore_index=True)
        if len(newest) >= MIN_RECENT_HOLDOUT_ROWS:
            return train, newest, "recent_labels"

    return train, pd.read_csv(TEST_DATA), "test_set"


if __name__ == "__main__":
    # Quick look at the data: uv run python -m src.data
    data = download_raw_data()
    print(f"Rows: {len(data):,}")
    print(f"Diabetes rate: {data[TARGET].mean():.1%}")
    problems = validate_data(data)
    print("Validation:", "OK" if not problems else problems)
