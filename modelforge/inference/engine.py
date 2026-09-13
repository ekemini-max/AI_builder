import os
import io
import joblib
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Optional


def predict_vision_pytorch(
    model_path: str,
    classes: List[str],
    image_bytes: bytes,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    import torch
    from torchvision import models, transforms

    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = len(classes)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = eval_tf(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    predicted_class = classes[top_idx]

    is_low_confidence = confidence < confidence_threshold

    class_probs = {cls_name: round(float(probabilities[i]), 4) for i, cls_name in enumerate(classes)}

    return {
        "status": "ok",
        "predicted_class": predicted_class if not is_low_confidence else f"{predicted_class} (uncertain)",
        "confidence": round(confidence, 4),
        "is_low_confidence": is_low_confidence,
        "confidence_threshold": confidence_threshold,
        "class_probabilities": class_probs
    }


def predict_vision_onnx(
    model_path: str,
    classes: List[str],
    image_bytes: bytes,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    import onnxruntime as ort
    from torchvision import transforms

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_np = eval_tf(image).unsqueeze(0).numpy()

    raw_output = session.run(None, {input_name: tensor_np})[0][0]

    exp_probs = np.exp(raw_output - np.max(raw_output))
    probabilities = exp_probs / exp_probs.sum()

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    predicted_class = classes[top_idx]

    is_low_confidence = confidence < confidence_threshold
    class_probs = {cls_name: round(float(probabilities[i]), 4) for i, cls_name in enumerate(classes)}

    return {
        "status": "ok",
        "predicted_class": predicted_class if not is_low_confidence else f"{predicted_class} (uncertain)",
        "confidence": round(confidence, 4),
        "is_low_confidence": is_low_confidence,
        "confidence_threshold": confidence_threshold,
        "class_probabilities": class_probs
    }


def predict_text_sklearn(
    model_path: str,
    preprocessor_path: str,
    classes: List[str],
    text_input: str,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    clf = joblib.load(model_path)
    prep_data = joblib.load(preprocessor_path)
    vectorizer = prep_data["vectorizer"]

    X = vectorizer.transform([text_input])

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X)[0]
    else:
        decision = clf.decision_function(X)[0]
        if decision.ndim == 0:
            decision = np.array([-decision, decision])
        exp_d = np.exp(decision - np.max(decision))
        probabilities = exp_d / exp_d.sum()

    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    predicted_class = classes[top_idx]

    is_low_confidence = confidence < confidence_threshold
    class_probs = {cls_name: round(float(probabilities[i]), 4) for i, cls_name in enumerate(classes)}

    return {
        "status": "ok",
        "predicted_class": predicted_class if not is_low_confidence else f"{predicted_class} (uncertain)",
        "confidence": round(confidence, 4),
        "is_low_confidence": is_low_confidence,
        "confidence_threshold": confidence_threshold,
        "class_probabilities": class_probs
    }


class ModelInferenceEngine:
    @staticmethod
    def predict(
        task_type: str,
        format: str,
        model_path: str,
        preprocessor_path: Optional[str],
        classes: List[str],
        input_data: Union[bytes, str],
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Unified inference router handling PyTorch, ONNX, and scikit-learn models.
        """
        try:
            if task_type == "vision":
                if not isinstance(input_data, bytes):
                    return {"status": "error", "message": "Vision inference requires image byte data."}

                if format == "pt":
                    return predict_vision_pytorch(model_path, classes, input_data, confidence_threshold)
                elif format == "onnx":
                    return predict_vision_onnx(model_path, classes, input_data, confidence_threshold)
                else:
                    return {"status": "error", "message": f"Unsupported vision model format: {format}"}

            elif task_type == "text":
                if not isinstance(input_data, str):
                    return {"status": "error", "message": "Text inference requires string input."}

                if format in ["sklearn", "joblib"]:
                    if not preprocessor_path or not os.path.exists(preprocessor_path):
                        return {"status": "error", "message": "Text preprocessor artifact missing."}
                    return predict_text_sklearn(model_path, preprocessor_path, classes, input_data, confidence_threshold)
                else:
                    return {"status": "error", "message": f"Unsupported text model format: {format}"}

            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}

        except Exception as e:
            return {"status": "error", "message": f"Inference execution failed: {str(e)}"}
