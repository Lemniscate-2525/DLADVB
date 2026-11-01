
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed"
PROCESSED_DIR = ROOT / "data" / "processed"

TRACE_FILE = RAW_DIR / "Event_traces.csv"
LABEL_FILE = RAW_DIR / "anomaly_label.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    traces = pd.read_csv(TRACE_FILE)
    labels = pd.read_csv(LABEL_FILE)

    dataset = traces.merge(labels, on="BlockId", how="inner")

    if "Label_x" in dataset.columns:
        dataset = dataset.drop(columns=["Label_x"])

    if "Label_y" in dataset.columns:
        dataset = dataset.rename(columns={"Label_y": "Label"})

    dataset["Text"] = (
        dataset["Features"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(",", " ", regex=False)
    )

    dataset["Label"] = dataset["Label"].map({
        "Normal": 0,
        "Anomaly": 1,
    })

    dataset = dataset.dropna(subset=["Label", "BlockId"])
    dataset["Label"] = dataset["Label"].astype(int)

    # Preserve BlockId for group-aware splitting.
    dataset = dataset[["BlockId", "Text", "Label"]]

    # Each BlockId should have one consistent label.
    label_counts = dataset.groupby("BlockId")["Label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("Some BlockIds have conflicting labels.")

    # Check whether each block appears in multiple rows.
    rows_per_block = dataset.groupby("BlockId").size()
    print("Rows per BlockId:")
    print(rows_per_block.describe())

    return dataset


def save_dataset(df):
    # Split by block group, not by individual row.
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, temp_idx = next(
        first_split.split(df, groups=df["BlockId"])
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_idx, test_idx = next(
        second_split.split(temp_df, groups=temp_df["BlockId"])
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Confirm no block appears in multiple splits.
    train_groups = set(train_df["BlockId"])
    val_groups = set(val_df["BlockId"])
    test_groups = set(test_df["BlockId"])

    assert not train_groups & val_groups
    assert not train_groups & test_groups
    assert not val_groups & test_groups

    # Drop BlockId from the model inputs but retain it in the
    # processed dataset so split provenance can be audited.
    df.to_csv(PROCESSED_DIR / "processed_dataset.csv", index=False)

    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_df[["Text", "Label"]].to_csv(
            PROCESSED_DIR / f"{name}.csv",
            index=False,
        )

    print("\nDataset Summary")
    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"\n{name}: {len(split_df)} samples")
        print(split_df["Label"].value_counts().sort_index())
        print("Unique blocks:", split_df["BlockId"].nunique())

    print("\nSplit completed with no BlockId overlap.")


if __name__ == "__main__":
    dataset = load_dataset()
    save_dataset(dataset)
