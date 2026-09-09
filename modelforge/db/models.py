import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from modelforge.db.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(String, nullable=False)  # "vision" or "text"
    api_key = Column(String, nullable=False, unique=True, default=generate_uuid)
    active_version_id = Column(String, ForeignKey("model_versions.id", use_alter=True, name="fk_active_version"), nullable=True)
    confidence_threshold = Column(Float, default=0.7)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    versions = relationship("ModelVersion", foreign_keys="ModelVersion.model_id", back_populates="model", cascade="all, delete-orphan")
    dataset_runs = relationship("DatasetRun", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    format = Column(String, nullable=False)  # "onnx", "pt", "h5", "sklearn", "pytorch_weights"
    artifact_path = Column(String, nullable=False)
    preprocessor_path = Column(String, nullable=True)  # TF-IDF, tokenizer, image transforms metadata
    classes = Column(JSON, nullable=False)  # List of class labels e.g. ["cat", "dog"]
    source = Column(String, nullable=False)  # "upload" or "auto_train"
    status = Column(String, nullable=False, default="ready")  # "ready", "failed"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    model = relationship("Model", foreign_keys=[model_id], back_populates="versions")
    evaluation_metrics = relationship("EvaluationMetrics", back_populates="model_version", uselist=False, cascade="all, delete-orphan")


class DatasetRun(Base):
    __tablename__ = "dataset_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, ForeignKey("models.id", ondelete="CASCADE"), nullable=True)
    dataset_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    fingerprint = Column(String, nullable=False)  # SHA-256 hash or summary hash
    sample_count = Column(Integer, nullable=False)
    class_distribution = Column(JSON, nullable=False)  # {"cat": 50, "dog": 50}
    is_balanced = Column(Boolean, nullable=False)
    validation_status = Column(String, nullable=False)  # "passed", "warning", "rejected"
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    model = relationship("Model", back_populates="dataset_runs")


class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_version_id = Column(String, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, unique=True)
    dataset_run_id = Column(String, ForeignKey("dataset_runs.id", ondelete="SET NULL"), nullable=True)

    eval_split_type = Column(String, nullable=False, default="held_out_test")  # "held_out_test", "validation", "training_sanity"
    is_trustworthy_held_out = Column(Boolean, nullable=False, default=True)

    accuracy = Column(Float, nullable=False)
    macro_precision = Column(Float, nullable=False)
    macro_recall = Column(Float, nullable=False)
    macro_f1 = Column(Float, nullable=False)

    # Detailed per-class metrics: {"cat": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 20}, ...}
    per_class_metrics = Column(JSON, nullable=False)
    confusion_matrix = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    model_version = relationship("ModelVersion", back_populates="evaluation_metrics")
