import os
import tempfile
import pandas as pd
from fastapi.testclient import TestClient
from modelforge.main import app
from modelforge.db.database import init_db

client = TestClient(app)


def test_bug1_stored_xss_sanitization():
    """Bug 1 Regression Test: Model creation with XSS HTML payload is sanitized."""
    init_db()

    xss_payload = "<img src=x onerror=alert(1)>"
    resp = client.post("/api/v1/models", json={
        "name": xss_payload,
        "task_type": "text",
        "description": "<script>alert('xss')</script>",
        "confidence_threshold": 0.6
    })

    assert resp.status_code == 200
    model = resp.json()["model"]

    # Assert raw script and image tags are stripped/escaped
    assert "<img" not in model["name"]
    assert "<script>" not in model["name"]
    assert "onerror=" not in model["name"]


def test_bug2_confidence_threshold_range_validation():
    """Bug 2 Regression Test: confidence_threshold accepts [0.0, 1.0] and rejects out-of-range values with HTTP 400."""
    init_db()

    # Out-of-range high
    resp = client.post("/api/v1/models", json={
        "name": "Bad Threshold High",
        "task_type": "text",
        "confidence_threshold": 5.0
    })
    assert resp.status_code in [400, 422]

    # Out-of-range negative
    resp = client.post("/api/v1/models", json={
        "name": "Bad Threshold Low",
        "task_type": "text",
        "confidence_threshold": -1.0
    })
    assert resp.status_code in [400, 422]

    # Valid thresholds
    for val in [0.0, 0.5, 1.0]:
        resp = client.post("/api/v1/models", json={
            "name": f"Valid Threshold {val}",
            "task_type": "text",
            "confidence_threshold": val
        })
        assert resp.status_code == 200
        assert resp.json()["model"]["confidence_threshold"] == val


def test_bug3_missing_column_autotrain_validation_not_500():
    """Bug 3 Regression Test: Auto-train with missing column returns HTTP 400 validation error, not 500."""
    init_db()

    # Create model
    resp = client.post("/api/v1/models", json={
        "name": "Malformed Dataset Model",
        "task_type": "text",
        "confidence_threshold": 0.7
    })
    assert resp.status_code == 200
    model_id = resp.json()["model"]["id"]

    # Create CSV missing expected label column
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        csv_path = tmp_csv.name
        df = pd.DataFrame({
            "text": [f"Sample text line {i}" for i in range(15)],
            "notlabel": ["cat"] * 15
        })
        df.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        resp = client.post(
            f"/api/v1/models/{model_id}/autotrain",
            files={"file": ("malformed.csv", f, "text/csv")},
            data={"text_column": "text", "label_column": "label"}
        )

    # Must be 400, NOT 500 Internal Server Error
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["status"] == "error"
    assert "not found" in data["detail"]["issues"][0]
    assert "Available columns: text, notlabel" in data["detail"]["issues"][0]
