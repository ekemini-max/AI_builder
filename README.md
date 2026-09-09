# ModelForge - AutoML Training & Model Serving Platform

**ModelForge** is a general-purpose, open-source AutoML platform designed to allow non-technical users to build, evaluate, version, and serve image and text classification models via auto-generated REST APIs.

---

## Key Features & Constraints

### 1. $0 Budget Architecture & CPU vs GPU Handoff
- **$0 Infrastructure Cost:** All dependencies and external handoffs rely exclusively on free-tier services.
- **CPU-Feasible Local Training:**
  - **Vision:** Uses transfer learning on PyTorch MobileNetV2 with automatic data augmentations (flips, rotation, color jitter), frozen backbone feature extraction, and head classification fine-tuning.
  - **Text:** Uses TF-IDF vectorization with n-gram extraction paired with Scikit-learn LogisticRegression / SGD classifiers.
- **Explicit External GPU Handoff:**
  - For datasets exceeding local CPU limits or requiring deep transformer fine-tuning, ModelForge automatically generates downloadable Google Colab / Hugging Face AutoTrain `.ipynb` notebooks. The UI clearly communicates the handoff instead of undertraining or failing silently.

### 2. Honesty Layer & Honest Metrics
- **Strict Held-Out Test Set:** Every dataset is split into stratified train/val/test splits using index-based sampling with zero data leakage.
- **Auditable Metrics:** Reports accuracy, precision, recall, and F1 **per class** alongside macro-averages and confusion matrices.
- **Transparency Flags:** Metrics explicitly distinguish between genuine held-out test evaluations (`is_trustworthy_held_out: true`) versus optimistic training sanity checks.

### 3. Model Upload, Versioning, and Rollback
- **Multi-Format Upload Support:** Validates and auto-detects `.onnx`, `.pt`/`.pth`, `.h5`, and `.joblib` model files, running a sanity inference check prior to acceptance.
- **Version History:** Retraining a model creates a new version slot without overwriting prior versions.
- **Instant Rollback:** Users can switch active model versions with a single click or API call.

### 4. Auto-Generated Inference API & Confidence Behavior
- **REST Endpoints:** Exposes `POST /api/v1/models/{model_id}/predict` returning prediction, confidence score, and class probabilities.
- **API Key Security:** Secured per model slot using `X-API-Key` headers.
- **Low Confidence Flagging:** Predictions below a user-configurable confidence threshold return an `is_low_confidence: true` flag and label suffix `(uncertain)` to enable downstream workflow routing (e.g., n8n).

---

## Directory Structure

```
.
├── modelforge/
│   ├── api/             # FastAPI REST endpoints & route handlers
│   ├── dataset/         # Dataset validation (corruption, balance) & zero-leakage splitter
│   ├── db/              # SQLAlchemy database models & connection setup
│   ├── inference/       # Unified inference engine (PyTorch, ONNX, Scikit-learn)
│   ├── registry/        # Model file validation, storage, and versioning/rollback logic
│   ├── static/          # Web dashboard (HTML, CSS, JS)
│   ├── training/        # Vision (MobileNetV2), text (TF-IDF), and GPU handoff notebook generator
│   └── main.py          # FastAPI application entry point
├── notebooks/           # Generated Google Colab GPU training notebooks
├── tests/               # Automated pytest test suite
└── README.md
```

---

## Getting Started

### Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/modelforge/modelforge.git
   cd modelforge
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core dependencies: `fastapi`, `uvicorn`, `torch`, `torchvision`, `scikit-learn`, `onnxruntime`, `sqlalchemy`, `pillow`, `pandas`, `pydantic`)*

3. **Start the server:**
   ```bash
   uvicorn modelforge.main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the Web Dashboard:**
   Open your browser to `http://localhost:8000/static/index.html`

---

## API Documentation

### Create Model Slot
- **POST** `/api/v1/models`
- **Body:** `{"name": "Sentiment Classifier", "task_type": "text", "confidence_threshold": 0.7}`

### Auto-Train Dataset
- **POST** `/api/v1/models/{model_id}/autotrain`
- **Form Data:** `file` (.zip or .csv), optional `text_column`, `label_column`

### Upload Model File
- **POST** `/api/v1/models/{model_id}/upload`
- **Form Data:** `file` (.onnx, .pt, .h5, .joblib), `classes` ("cat, dog"), optional `preprocessor_file`

### Version Rollback
- **POST** `/api/v1/models/{model_id}/rollback`
- **Body:** `{"version_number": 1}`

### Serve Inference
- **POST** `/api/v1/models/{model_id}/predict`
- **Header:** `X-API-Key: <model_api_key>`
- **Body (Text):** `{"text": "Sample text for prediction"}`
- **Form (Vision):** `image_file` (uploaded image file)

---

## Running Tests

Run the full automated test suite with pytest:

```bash
PYTHONPATH=. pytest tests/
```
