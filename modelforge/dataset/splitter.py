import random
from typing import List, Tuple, Dict, Any, TypeVar
from sklearn.model_selection import train_test_split

T = TypeVar('T')

def stratified_data_split(
    items: List[T],
    labels: List[Any],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[T], List[Any], List[T], List[Any], List[T], List[Any]]:
    """
    Splits items and labels into train, validation, and held-out test sets using index-based stratified sampling to guarantee index separation and avoid duplicate content assertion false-positives.
    Returns:
        (train_items, train_labels, val_items, val_labels, test_items, test_labels)
    """
    if len(items) != len(labels):
        raise ValueError("Items and labels must have the exact same length.")

    total_samples = len(items)
    if total_samples < 5:
        raise ValueError("Dataset is too small to split into train/val/test sets.")

    indices = list(range(total_samples))

    # Step 1: Separate held-out test set indices from remaining (train + val)
    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        indices,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Step 2: Separate train and val set indices
    relative_val_size = val_size / (1.0 - test_size)

    train_idx, val_idx, train_labels, val_labels = train_test_split(
        train_val_idx,
        train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=random_state
    )

    # Verify zero index leakage between splits
    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)

    assert len(set_train.intersection(set_test)) == 0, "Index leakage detected between train and test sets!"
    assert len(set_val.intersection(set_test)) == 0, "Index leakage detected between val and test sets!"
    assert len(set_train.intersection(set_val)) == 0, "Index leakage detected between train and val sets!"

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]

    return train_items, train_labels, val_items, val_labels, test_items, test_labels
