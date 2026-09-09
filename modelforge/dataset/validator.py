import os
import zipfile
import hashlib
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    is_valid: bool
    status: str  # "passed", "warning", "rejected"
    task_type: str  # "vision" or "text"
    sample_count: int
    class_distribution: Dict[str, int]
    is_balanced: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fingerprint: str = ""
    requires_gpu: bool = False
    text_column: Optional[str] = None
    label_column: Optional[str] = None


def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_vision_dataset(zip_filepath: str, min_per_class: int = 10) -> Tuple[ValidationResult, Optional[str]]:
    """
    Validates a ZIP file containing class folders with images.
    Returns ValidationResult and the path to extracted dataset directory (if valid).
    """
    fingerprint = compute_file_sha256(zip_filepath)
    issues = []
    warnings = []

    if not zipfile.is_zipfile(zip_filepath):
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="vision",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=["File is not a valid ZIP archive."],
            fingerprint=fingerprint
        ), None

    extract_dir = zip_filepath + "_extracted"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="vision",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=[f"Failed to extract ZIP file: {str(e)}"],
            fingerprint=fingerprint
        ), None

    # Scan directory structure for class folders
    # Handled both direct class folders and single root wrapper folder
    subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d)) and not d.startswith("__") and not d.startswith(".")]

    # If single top-level folder wrapping everything
    target_dir = extract_dir
    if len(subdirs) == 1:
        nested_dir = os.path.join(extract_dir, subdirs[0])
        nested_subdirs = [d for d in os.listdir(nested_dir) if os.path.isdir(os.path.join(nested_dir, d)) and not d.startswith("__") and not d.startswith(".")]
        if len(nested_subdirs) >= 2:
            target_dir = nested_dir
            subdirs = nested_subdirs

    if len(subdirs) < 2:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="vision",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=[f"Dataset must contain at least 2 class directories. Found {len(subdirs)} classes."],
            fingerprint=fingerprint
        ), None

    class_distribution: Dict[str, int] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    corrupted_count = 0
    duplicate_hashes = set()
    duplicate_count = 0
    total_samples = 0

    for cls in sorted(subdirs):
        cls_dir = os.path.join(target_dir, cls)
        cls_valid_count = 0

        for root, _, files in os.walk(cls_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    continue

                img_path = os.path.join(root, file)

                # Check readability & corruption
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # Re-open for hash check if needed
                    with open(img_path, "rb") as img_f:
                        img_hash = hashlib.md5(img_f.read()).hexdigest()
                        if img_hash in duplicate_hashes:
                            duplicate_count += 1
                        else:
                            duplicate_hashes.add(img_hash)
                    cls_valid_count += 1
                except Exception:
                    corrupted_count += 1

        class_distribution[cls] = cls_valid_count
        total_samples += cls_valid_count

    if corrupted_count > 0:
        warnings.append(f"Found {corrupted_count} corrupted or unreadable image files which were excluded.")

    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate image files across the dataset.")

    # Check minimum per class
    underpopulated_classes = [c for c, count in class_distribution.items() if count < min_per_class]
    if underpopulated_classes:
        issues.append(f"The following classes have fewer than {min_per_class} valid samples: {', '.join(underpopulated_classes)}. Minimum required per class is {min_per_class}.")

    # Check balance
    counts = list(class_distribution.values())
    max_c = max(counts) if counts else 0
    min_c = min(counts) if counts else 0

    is_balanced = True
    if min_c == 0:
        is_balanced = False
    elif (max_c / min_c) > 3.0:
        is_balanced = False
        warnings.append(f"Dataset is significantly imbalanced (max class has {max_c} samples, min has {min_c}). Model accuracy on rare classes may suffer.")

    # Rule on feasibility of reaching 85% accuracy
    if total_samples < 50:
        warnings.append("Total dataset size is under 50 samples. Achieving high accuracy (>=85%) on held-out test data is unlikely with so few samples.")

    requires_gpu = total_samples > 5000

    is_valid = len(issues) == 0
    status = "rejected" if not is_valid else ("warning" if len(warnings) > 0 else "passed")

    return ValidationResult(
        is_valid=is_valid,
        status=status,
        task_type="vision",
        sample_count=total_samples,
        class_distribution=class_distribution,
        is_balanced=is_balanced,
        issues=issues,
        warnings=warnings,
        fingerprint=fingerprint,
        requires_gpu=requires_gpu
    ), target_dir


def validate_text_dataset(csv_filepath: str, min_per_class: int = 10, text_col: Optional[str] = None, label_col: Optional[str] = None) -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
    """
    Validates a CSV file containing text and label columns.
    Returns ValidationResult and cleaned pandas DataFrame (if valid).
    """
    fingerprint = compute_file_sha256(csv_filepath)
    issues = []
    warnings = []

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="text",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=[f"Failed to parse CSV file: {str(e)}"],
            fingerprint=fingerprint
        ), None

    if df.empty:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="text",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=["CSV file is empty."],
            fingerprint=fingerprint
        ), None

    # Auto-detect text and label columns if not provided
    cols = df.columns.tolist()

    if not text_col:
        text_candidates = ["text", "body", "content", "message", "sentence", "review", "comment"]
        for cand in text_candidates:
            matches = [c for c in cols if cand in c.lower()]
            if matches:
                text_col = matches[0]
                break
        if not text_col:
            # Pick first string-like column
            for c in cols:
                if df[c].dtype == object or isinstance(df[c].iloc[0], str):
                    text_col = c
                    break

    if not label_col:
        label_candidates = ["label", "target", "class", "category", "sentiment", "type"]
        for cand in label_candidates:
            matches = [c for c in cols if cand in c.lower() and c != text_col]
            if matches:
                label_col = matches[0]
                break
        if not label_col:
            # Pick second column or non-text column
            for c in cols:
                if c != text_col:
                    label_col = c
                    break

    if not text_col or not label_col or text_col == label_col:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="text",
            sample_count=len(df),
            class_distribution={},
            is_balanced=False,
            issues=["Could not auto-detect distinct text and label columns in CSV. Please specify column names."],
            fingerprint=fingerprint
        ), None

    # Clean missing / empty values
    initial_count = len(df)
    cleaned_df = df.dropna(subset=[text_col, label_col]).copy()
    cleaned_df[text_col] = cleaned_df[text_col].astype(str).str.strip()
    cleaned_df[label_col] = cleaned_df[label_col].astype(str).str.strip()
    cleaned_df = cleaned_df[cleaned_df[text_col] != ""]
    cleaned_df = cleaned_df[cleaned_df[label_col] != ""]

    dropped_nulls = initial_count - len(cleaned_df)
    if dropped_nulls > 0:
        warnings.append(f"Dropped {dropped_nulls} rows with missing text or label values.")

    # Check duplicate rows
    initial_clean_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=[text_col])
    duplicate_count = initial_clean_count - len(cleaned_df)
    if duplicate_count > 0:
        warnings.append(f"Removed {duplicate_count} duplicate text rows.")

    total_samples = len(cleaned_df)
    if total_samples == 0:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="text",
            sample_count=0,
            class_distribution={},
            is_balanced=False,
            issues=["No valid text samples remain after cleaning missing/empty values."],
            fingerprint=fingerprint
        ), None

    # Class distribution
    class_counts_series = cleaned_df[label_col].value_counts()
    class_distribution = {str(k): int(v) for k, v in class_counts_series.to_dict().items()}

    if len(class_distribution) < 2:
        return ValidationResult(
            is_valid=False,
            status="rejected",
            task_type="text",
            sample_count=total_samples,
            class_distribution=class_distribution,
            is_balanced=False,
            issues=[f"Dataset must contain at least 2 classes. Found {len(class_distribution)} class(es)."],
            fingerprint=fingerprint,
            text_column=text_col,
            label_column=label_col
        ), None

    # Check minimum per class
    underpopulated_classes = [c for c, count in class_distribution.items() if count < min_per_class]
    if underpopulated_classes:
        issues.append(f"The following classes have fewer than {min_per_class} valid samples: {', '.join(underpopulated_classes)}. Minimum required per class is {min_per_class}.")

    counts = list(class_distribution.values())
    max_c = max(counts) if counts else 0
    min_c = min(counts) if counts else 0

    is_balanced = True
    if min_c == 0:
        is_balanced = False
    elif (max_c / min_c) > 3.0:
        is_balanced = False
        warnings.append(f"Dataset is significantly imbalanced (max class has {max_c} samples, min has {min_c}). Model accuracy on rare classes may suffer.")

    if total_samples < 50:
        warnings.append("Total dataset size is under 50 samples. Achieving high accuracy (>=85%) on held-out test data is unlikely with so few samples.")

    requires_gpu = total_samples > 20000

    is_valid = len(issues) == 0
    status = "rejected" if not is_valid else ("warning" if len(warnings) > 0 else "passed")

    return ValidationResult(
        is_valid=is_valid,
        status=status,
        task_type="text",
        sample_count=total_samples,
        class_distribution=class_distribution,
        is_balanced=is_balanced,
        issues=issues,
        warnings=warnings,
        fingerprint=fingerprint,
        requires_gpu=requires_gpu,
        text_column=text_col,
        label_column=label_col
    ), cleaned_df
