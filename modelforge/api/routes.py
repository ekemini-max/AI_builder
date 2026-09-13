import os
import shutil
import uuid
import html
import re
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from modelforge.db.database import get_db
from modelforge.db.models import Model, ModelVersion, DatasetRun, EvaluationMetrics
from modelforge.dataset.validator import validate_vision_dataset, validate_text_dataset
from modelforge.training.text import train_text_model
from modelforge.training.gpu_handoff import generate_gpu_handoff_info
from modelforge.registry.versioning import register_new_model_version, rollback_model_version

router = APIRouter()

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


def sanitize_input_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    clean = re.sub(r'<[^>]*>', '', text)
    clean = html.escape(clean.strip())
    return clean


# --- Request/Response Schemas ---

class CreateModelRequest(BaseModel):
    name: str = Field(..., max_length=100)
    task_type: str  # "vision" or "text"
    description: Optional[str] = Field(None, max_length=1000)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("name", "description", mode="before")
    @classmethod
    def sanitize_strings(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            return sanitize_input_text(v)
        return v


class RollbackRequest(BaseModel):
    version_number: int


class TextPredictRequest(BaseModel):
    text: str


# --- Helper Security Function ---

def verify_api_key(model_id: str, x_api_key: Optional[str], db: Session) -> Model:
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail={"status": "error", "message": f"Model '{model_id}' not found."})

    if x_api_key and x_api_key != model.api_key:
        raise HTTPException(status_code=401, detail={"status": "error", "message": "Invalid API Key provided."})

    return model


# --- Model Management Endpoints ---

@router.post("/models")
def create_model(req: CreateModelRequest, db: Session = Depends(get_db)):
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Model name cannot be empty after removing invalid characters."})

    if req.task_type not in ["vision", "text"]:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Task type must be 'vision' or 'text'."})

    if req.confidence_threshold < 0.0 or req.confidence_threshold > 1.0:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "confidence_threshold must be between 0.0 and 1.0"})

    model = Model(
        name=req.name,
        task_type=req.task_type,
        description=req.description,
        confidence_threshold=req.confidence_threshold
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    return {
        "status": "ok",
        "message": f"Model '{req.name}' created successfully.",
        "model": {
            "id": model.id,
            "name": model.name,
            "task_type": model.task_type,
            "api_key": model.api_key,
            "confidence_threshold": model.confidence_threshold,
            "active_version_id": model.active_version_id
        }
    }


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(Model).all()
    results = []

    for m in models:
        active_version = None
        metrics = None
        if m.active_version_id:
            ver = db.query(ModelVersion).filter(ModelVersion.id == m.active_version_id).first()
            if ver:
                active_version = {
                    "id": ver.id,
                    "version_number": ver.version_number,
                    "format": ver.format,
                    "classes": ver.classes,
                    "source": ver.source
                }
                if ver.evaluation_metrics:
                    eval_m = ver.evaluation_metrics
                    metrics = {
                        "accuracy": eval_m.accuracy,
                        "macro_f1": eval_m.macro_f1,
                        "eval_split_type": eval_m.eval_split_type,
                        "is_trustworthy_held_out": eval_m.is_trustworthy_held_out,
                        "per_class_metrics": eval_m.per_class_metrics
                    }

        results.append({
            "id": m.id,
            "name": m.name,
            "task_type": m.task_type,
            "description": m.description,
            "api_key": m.api_key,
            "confidence_threshold": m.confidence_threshold,
            "active_version": active_version,
            "metrics": metrics,
            "version_count": len(m.versions)
        })

    return {"status": "ok", "models": results}


@router.get("/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Model not found."})

    versions_list = []
    for v in model.versions:
        eval_data = None
        if v.evaluation_metrics:
            em = v.evaluation_metrics
            eval_data = {
                "accuracy": em.accuracy,
                "macro_precision": em.macro_precision,
                "macro_recall": em.macro_recall,
                "macro_f1": em.macro_f1,
                "eval_split_type": em.eval_split_type,
                "is_trustworthy_held_out": em.is_trustworthy_held_out,
                "per_class_metrics": em.per_class_metrics,
                "confusion_matrix": em.confusion_matrix
            }

        versions_list.append({
            "id": v.id,
            "version_number": v.version_number,
            "format": v.format,
            "source": v.source,
            "status": v.status,
            "classes": v.classes,
            "is_active": (v.id == model.active_version_id),
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "evaluation_metrics": eval_data
        })

    return {
        "status": "ok",
        "model": {
            "id": model.id,
            "name": model.name,
            "task_type": model.task_type,
            "description": model.description,
            "api_key": model.api_key,
            "confidence_threshold": model.confidence_threshold,
            "active_version_id": model.active_version_id,
            "versions": versions_list
        }
    }


# --- Model Upload Endpoint ---

@router.post("/models/{model_id}/upload")
async def upload_model_file(
    model_id: str,
    classes: str = Form(...),  # Comma-separated
    file: UploadFile = File(...),
    preprocessor_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    from modelforge.registry.validator import validate_uploaded_model

    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Model not found."})

    class_list = [sanitize_input_text(c) for c in classes.split(",") if c.strip()]
    if len(class_list) < 2:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Must provide at least 2 classes."})

    upload_dir = os.path.join(STORAGE_DIR, model_id, "uploads", str(uuid.uuid4()))
    os.makedirs(upload_dir, exist_ok=True)

    model_filepath = os.path.join(upload_dir, file.filename)
    with open(model_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prep_filepath = None
    if preprocessor_file:
        prep_filepath = os.path.join(upload_dir, preprocessor_file.filename)
        with open(prep_filepath, "wb") as buffer:
            shutil.copyfileobj(preprocessor_file.file, buffer)

    is_valid, detected_format, msg = validate_uploaded_model(model_filepath, model.task_type, class_list, prep_filepath)
    if not is_valid:
        raise HTTPException(status_code=400, detail={"status": "error", "message": msg})

    dummy_metrics = {
        "eval_split_type": "training_sanity",
        "is_trustworthy_held_out": False,
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "per_class_metrics": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in class_list}
    }

    new_ver = register_new_model_version(
        db=db,
        model_id=model.id,
        format=detected_format,
        artifact_path=model_filepath,
        preprocessor_path=prep_filepath,
        classes=class_list,
        source="upload",
        metrics_dict=dummy_metrics
    )

    return {
        "status": "ok",
        "message": "Model uploaded and validated successfully.",
        "version_number": new_ver.version_number,
        "format": new_ver.format
    }


# --- Auto-Train Endpoint ---

@router.post("/models/{model_id}/autotrain")
async def auto_train_model(
    model_id: str,
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Model not found."})

    run_dir = os.path.join(STORAGE_DIR, model_id, "runs", str(uuid.uuid4()))
    os.makedirs(run_dir, exist_ok=True)
    dataset_filepath = os.path.join(run_dir, file.filename)

    with open(dataset_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if model.task_type == "vision":
        from modelforge.training.vision import train_vision_model

        val_res, extract_dir = validate_vision_dataset(dataset_filepath)
        if not val_res.is_valid:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "Dataset validation failed.", "issues": val_res.issues, "warnings": val_res.warnings})

        ds_run = DatasetRun(
            model_id=model.id,
            dataset_name=file.filename,
            task_type="vision",
            file_path=dataset_filepath,
            fingerprint=val_res.fingerprint,
            sample_count=val_res.sample_count,
            class_distribution=val_res.class_distribution,
            is_balanced=val_res.is_balanced,
            validation_status=val_res.status,
            validation_notes="; ".join(val_res.warnings)
        )
        db.add(ds_run)
        db.commit()

        if val_res.requires_gpu:
            gpu_info = generate_gpu_handoff_info("vision", file.filename, val_res.sample_count)
            return {
                "status": "gpu_handoff_required",
                "message": "Dataset exceeds local CPU training limits for high accuracy. External GPU handoff required.",
                "gpu_handoff": gpu_info
            }

        img_paths = []
        labels = []
        for cls_name in sorted(os.listdir(extract_dir)):
            cls_dir = os.path.join(extract_dir, cls_name)
            if os.path.isdir(cls_dir) and not cls_name.startswith("."):
                for root, _, f_list in os.walk(cls_dir):
                    for fname in f_list:
                        if os.path.splitext(fname)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                            img_paths.append(os.path.join(root, fname))
                            labels.append(cls_name)

        output_model_path = os.path.join(run_dir, "model.pt")
        metrics = train_vision_model(img_paths, labels, output_model_path, epochs=3)

        new_ver = register_new_model_version(
            db=db,
            model_id=model.id,
            format="pt",
            artifact_path=output_model_path,
            classes=metrics["classes"],
            source="auto_train",
            metrics_dict=metrics
        )

        return {
            "status": "ok",
            "message": "Model automatically trained and verified on held-out test split.",
            "version_number": new_ver.version_number,
            "metrics": metrics
        }

    elif model.task_type == "text":
        val_res, clean_df = validate_text_dataset(dataset_filepath, text_col=text_column, label_col=label_column)
        if not val_res.is_valid:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "Dataset validation failed.", "issues": val_res.issues, "warnings": val_res.warnings})

        ds_run = DatasetRun(
            model_id=model.id,
            dataset_name=file.filename,
            task_type="text",
            file_path=dataset_filepath,
            fingerprint=val_res.fingerprint,
            sample_count=val_res.sample_count,
            class_distribution=val_res.class_distribution,
            is_balanced=val_res.is_balanced,
            validation_status=val_res.status,
            validation_notes="; ".join(val_res.warnings)
        )
        db.add(ds_run)
        db.commit()

        if val_res.requires_gpu:
            gpu_info = generate_gpu_handoff_info("text", file.filename, val_res.sample_count)
            return {
                "status": "gpu_handoff_required",
                "message": "Dataset exceeds local CPU training limits for high accuracy. External GPU handoff required.",
                "gpu_handoff": gpu_info
            }

        output_model_path = os.path.join(run_dir, "model.joblib")
        output_prep_path = os.path.join(run_dir, "preprocessor.joblib")

        texts = clean_df[val_res.text_column].tolist()
        labels = clean_df[val_res.label_column].tolist()

        metrics = train_text_model(texts, labels, output_model_path, output_prep_path)

        new_ver = register_new_model_version(
            db=db,
            model_id=model.id,
            format="sklearn",
            artifact_path=output_model_path,
            preprocessor_path=output_prep_path,
            classes=metrics["classes"],
            source="auto_train",
            metrics_dict=metrics
        )

        return {
            "status": "ok",
            "message": "Model automatically trained and verified on held-out test split.",
            "version_number": new_ver.version_number,
            "metrics": metrics
        }


# --- Rollback Endpoint ---

@router.post("/models/{model_id}/rollback")
def rollback_version(model_id: str, req: RollbackRequest, db: Session = Depends(get_db)):
    try:
        target_ver = rollback_model_version(db, model_id, req.version_number)
        return {
            "status": "ok",
            "message": f"Successfully rolled back model '{model_id}' to active version {target_ver.version_number}.",
            "active_version_id": target_ver.id,
            "version_number": target_ver.version_number
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)})


# --- Inference Endpoint ---

@router.post("/models/{model_id}/predict")
async def predict_endpoint(
    model_id: str,
    request: Request,
    image_file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    from modelforge.inference.engine import ModelInferenceEngine

    model = verify_api_key(model_id, x_api_key, db)

    if not model.active_version_id:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Model has no active version ready for inference."})

    active_ver = db.query(ModelVersion).filter(ModelVersion.id == model.active_version_id).first()
    if not active_ver:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Active version artifact missing."})

    if model.task_type == "vision":
        if not image_file:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "Vision inference requires an uploaded image file."})

        image_bytes = await image_file.read()
        res = ModelInferenceEngine.predict(
            task_type="vision",
            format=active_ver.format,
            model_path=active_ver.artifact_path,
            preprocessor_path=active_ver.preprocessor_path,
            classes=active_ver.classes,
            input_data=image_bytes,
            confidence_threshold=model.confidence_threshold
        )
        return res

    elif model.task_type == "text":
        body_json = {}
        try:
            body_json = await request.json()
        except Exception:
            pass

        text_input = body_json.get("text")
        if not text_input:
            raise HTTPException(status_code=400, detail={"status": "error", "message": "Text inference requires JSON body with 'text' string."})

        res = ModelInferenceEngine.predict(
            task_type="text",
            format=active_ver.format,
            model_path=active_ver.artifact_path,
            preprocessor_path=active_ver.preprocessor_path,
            classes=active_ver.classes,
            input_data=text_input,
            confidence_threshold=model.confidence_threshold
        )
        return res
