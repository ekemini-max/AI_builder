import os
import zipfile
import tempfile
import pandas as pd
from PIL import Image
from modelforge.dataset.validator import validate_vision_dataset, validate_text_dataset
from modelforge.dataset.splitter import stratified_data_split


def test_vision_dataset_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "vision.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for cls in ["cat", "dog"]:
                for i in range(12):
                    img_file = os.path.join(tmpdir, f"{cls}_{i}.png")
                    img = Image.new("RGB", (20, 20), color=(255, 0, 0))
                    img.save(img_file)
                    zf.write(img_file, arcname=f"{cls}/{cls}_{i}.png")

        val_res, extract_dir = validate_vision_dataset(zip_path, min_per_class=10)
        assert val_res.is_valid is True
        assert val_res.sample_count == 24
        assert "cat" in val_res.class_distribution
        assert "dog" in val_res.class_distribution


def test_text_dataset_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "text.csv")
        df = pd.DataFrame({
            "review": [f"Good product sample {i}" for i in range(20)] + [f"Bad item return {i}" for i in range(20)],
            "label": ["positive"] * 20 + ["negative"] * 20
        })
        df.to_csv(csv_path, index=False)

        val_res, clean_df = validate_text_dataset(csv_path, min_per_class=10)
        assert val_res.is_valid is True
        assert val_res.sample_count == 40
        assert val_res.text_column == "review"
        assert val_res.label_column == "label"


def test_stratified_split_integrity_no_leakage():
    items = [f"path/to/item_{i}.png" for i in range(100)]
    labels = ["cat" if i % 2 == 0 else "dog" for i in range(100)]

    tr_i, tr_l, val_i, val_l, te_i, te_l = stratified_data_split(items, labels, test_size=0.2, val_size=0.1)

    assert len(tr_i) + len(val_i) + len(te_i) == 100

    # Assert ZERO leakage between splits
    set_train = set(tr_i)
    set_val = set(val_i)
    set_test = set(te_i)

    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0
    assert len(set_train.intersection(set_val)) == 0
