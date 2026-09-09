import os
import tempfile
import pandas as pd
from fastapi.testclient import TestClient
from modelforge.main import app
from modelforge.db.database import init_db, SessionLocal
from modelforge.db.models import Model, ModelVersion
from modelforge.registry.versioning import rollback_model_version

client = TestClient(app)

def test_full_model_and_api_lifecycle():
    init_db()

    # 1. Create Model Slot
    resp = client.post("/api/v1/models", json={
        "name": "E-Commerce Review Classifier",
        "task_type": "text",
        "confidence_threshold": 0.75
    })
    assert resp.status_code == 200
    model_data = resp.json()["model"]
    model_id = model_data["id"]
    api_key = model_data["api_key"]

    # 2. Auto-Train Model v1
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        csv_path = tmp_csv.name
        df = pd.DataFrame({
            "text": [f"I love this item so much {i}" for i in range(25)] + [f"Terrible bad worst broken {i}" for i in range(25)],
            "label": ["positive"] * 25 + ["negative"] * 25
        })
        df.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        resp = client.post(
            f"/api/v1/models/{model_id}/autotrain",
            files={"file": ("dataset_v1.csv", f, "text/csv")}
        )
    assert resp.status_code == 200
    v1_data = resp.json()
    assert v1_data["version_number"] == 1
    assert "metrics" in v1_data
    assert v1_data["metrics"]["eval_split_type"] == "held_out_test"
    assert v1_data["metrics"]["is_trustworthy_held_out"] is True

    # 3. Test Prediction with X-API-Key Header
    # High confidence case
    resp = client.post(
        f"/api/v1/models/{model_id}/predict",
        headers={"X-API-Key": api_key},
        json={"text": "I love this item so much"}
    )
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["status"] == "ok"
    assert pred["is_low_confidence"] is False
    assert pred["predicted_class"] == "positive"
    assert "confidence" in pred
    assert "class_probabilities" in pred

    # Low confidence case trigger
    resp = client.post(
        f"/api/v1/models/{model_id}/predict",
        headers={"X-API-Key": api_key},
        json={"text": "random ambiguous word stream"}
    )
    assert resp.status_code == 200
    pred_low = resp.json()
    assert pred_low["status"] == "ok"
    # Should flag low confidence if below 0.75 threshold
    assert pred_low["confidence_threshold"] == 0.75

    # 4. Retrain model to create Version 2
    with open(csv_path, "rb") as f:
        resp = client.post(
            f"/api/v1/models/{model_id}/autotrain",
            files={"file": ("dataset_v2.csv", f, "text/csv")}
        )
    assert resp.status_code == 200
    v2_data = resp.json()
    assert v2_data["version_number"] == 2

    # 5. Rollback to Version 1
    resp = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={"version_number": 1}
    )
    assert resp.status_code == 200
    assert resp.json()["version_number"] == 1

    # 6. Verify Unauthorized Access without valid API Key
    resp = client.post(
        f"/api/v1/models/{model_id}/predict",
        headers={"X-API-Key": "wrong_key"},
        json={"text": "I love this item"}
    )
    assert resp.status_code == 401
