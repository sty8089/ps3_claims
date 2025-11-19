import pandas as pd


def test_create_sample_column_single_column():
    from ps3.data import create_sample_column

    n = 100
    df = pd.DataFrame({"id": list(range(1, n + 1))})
    out = create_sample_column(df, "id", training_frac=0.8)

    assert "sample" in out.columns
    counts = out["sample"].value_counts()
    # expect roughly 80/20 split; allow a small tolerance
    assert counts.get("train", 0) >= 70
    assert counts.get("test", 0) >= 10


def test_create_sample_column_multiple_columns_grouping():
    from ps3.data import create_sample_column

    # Create duplicated keys across rows and ensure same assignment for duplicates
    rows = []
    for a in ["A", "B"]:
        for b in [1, 2, 3]:
            for dup in range(3):
                rows.append({"col1": a, "col2": b, "val": dup})
    df = pd.DataFrame(rows)
    out = create_sample_column(df, ["col1", "col2"], training_frac=0.5)

    # For each (col1,col2) combination, all rows should have the same sample
    grouped = out.groupby(["col1", "col2"])
    for _, g in grouped:
        assert g["sample"].nunique() == 1


def test_create_sample_column_reproducible():
    from ps3.data import create_sample_column

    df = pd.DataFrame({"id": ["x", "y", "z", "x"]})
    out1 = create_sample_column(df, "id", training_frac=0.6)
    out2 = create_sample_column(df, "id", training_frac=0.6)

    # Assignments should be identical across runs
    assert out1["sample"].tolist() == out2["sample"].tolist()
