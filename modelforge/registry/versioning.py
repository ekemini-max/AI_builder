import os
import shutil
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from modelforge.db.models import Model, ModelVersion, EvaluationMetrics


def register_new_model_version(
    db: Session,
    model_id: str,
    format: str,
    artifact_path: str,
    classes: List[str],
    source: str,  # "upload" or "auto_train"
    metrics_dict: Optional[Dict[str, Any]] = None,
    preprocessor_path: Optional[str] = None,
    set_as_active: bool = True
) -> ModelVersion:
    """
    Registers a new version for an existing model slot, or creates the model slot if it's new.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise ValueError(f"Model with id {model_id} not found.")

    # Calculate next version number
    latest_version = db.query(ModelVersion)\
        .filter(ModelVersion.model_id == model_id)\
        .order_by(ModelVersion.version_number.desc())\
        .first()

    next_version_num = (latest_version.version_number + 1) if latest_version else 1

    model_version = ModelVersion(
        model_id=model_id,
        version_number=next_version_num,
        format=format,
        artifact_path=artifact_path,
        preprocessor_path=preprocessor_path,
        classes=classes,
        source=source,
        status="ready"
    )

    db.add(model_version)
    db.flush()  # Populates model_version.id

    # Attach evaluation metrics if provided
    if metrics_dict:
        eval_metrics = EvaluationMetrics(
            model_version_id=model_version.id,
            eval_split_type=metrics_dict.get("eval_split_type", "held_out_test"),
            is_trustworthy_held_out=metrics_dict.get("is_trustworthy_held_out", True),
            accuracy=metrics_dict.get("accuracy", 0.0),
            macro_precision=metrics_dict.get("macro_precision", 0.0),
            macro_recall=metrics_dict.get("macro_recall", 0.0),
            macro_f1=metrics_dict.get("macro_f1", 0.0),
            per_class_metrics=metrics_dict.get("per_class_metrics", {}),
            confusion_matrix=metrics_dict.get("confusion_matrix", [])
        )
        db.add(eval_metrics)

    if set_as_active:
        model.active_version_id = model_version.id

    db.commit()
    db.refresh(model_version)
    return model_version


def rollback_model_version(db: Session, model_id: str, target_version_number: int) -> ModelVersion:
    """
    Rolls back active version for model_id to a prior version number without deleting history.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise ValueError(f"Model with id {model_id} not found.")

    target_version = db.query(ModelVersion)\
        .filter(ModelVersion.model_id == model_id, ModelVersion.version_number == target_version_number)\
        .first()

    if not target_version:
        raise ValueError(f"Version {target_version_number} not found for model {model_id}.")

    model.active_version_id = target_version.id
    db.commit()
    db.refresh(model)
    return target_version
