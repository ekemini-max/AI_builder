import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from modelforge.dataset.splitter import stratified_data_split


def train_text_model(
    texts: List[str],
    class_labels: List[str],
    output_model_path: str,
    output_preprocessor_path: str
) -> Dict[str, Any]:
    """
    Trains a lightweight TF-IDF + LogisticRegression/SGD model on CPU.
    Returns evaluation metrics on the genuine held-out test split.
    """
    unique_classes = sorted(list(set(class_labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[cls] for cls in class_labels]

    # Split dataset into train, val, test splits
    tr_texts, tr_labels, val_texts, val_labels, te_texts, te_labels = stratified_data_split(
        texts, numeric_labels, test_size=0.2, val_size=0.1
    )

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(tr_texts)
    X_test = vectorizer.transform(te_texts)

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, tr_labels)

    # Predict on held-out test set
    y_pred = clf.predict(X_test)
    y_true = te_labels

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(unique_classes))), zero_division=0
    )

    per_class_metrics = {}
    for idx, cls_name in idx_to_class.items():
        per_class_metrics[cls_name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx])
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(unique_classes)))).tolist()

    # Save model and vectorizer artifacts
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)

    joblib.dump(clf, output_model_path)
    joblib.dump({
        "vectorizer": vectorizer,
        "classes": unique_classes,
        "class_to_idx": class_to_idx
    }, output_preprocessor_path)

    return {
        "eval_split_type": "held_out_test",
        "is_trustworthy_held_out": True,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(np.mean(precision)), 4),
        "macro_recall": round(float(np.mean(recall)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_mat,
        "classes": unique_classes,
        "format": "sklearn"
    }
