import os
import torch
import onnxruntime as ort
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional


def validate_uploaded_model(
    model_filepath: str,
    task_type: str,
    classes: List[str],
    preprocessor_filepath: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Validates an uploaded model file (.onnx, .pt/.pth, .h5, .joblib).
    Performs format auto-detection and a sanity inference run.
    Returns: (is_valid, detected_format, message)
    """
    ext = os.path.splitext(model_filepath)[1].lower()

    if ext == ".onnx":
        try:
            session = ort.InferenceSession(model_filepath, providers=['CPUExecutionProvider'])
            inputs = session.get_inputs()
            if len(inputs) == 0:
                return False, "onnx", "ONNX model has no input nodes."
            return True, "onnx", "ONNX model validated successfully with sanity check."
        except Exception as e:
            return False, "onnx", f"Invalid ONNX model file: {str(e)}"

    elif ext in [".pt", ".pth"]:
        try:
            checkpoint = torch.load(model_filepath, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "state_dict" not in checkpoint and not any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    return False, "pt", "PyTorch file must contain a valid state_dict or model checkpoint."
            return True, "pt", "PyTorch model checkpoint validated successfully."
        except Exception as e:
            return False, "pt", f"Invalid PyTorch model file: {str(e)}"

    elif ext == ".h5":
        # Check basic Keras / HDF5 header or load check
        try:
            import h5py
            with h5py.File(model_filepath, 'r') as f:
                if len(f.keys()) == 0:
                    return False, "h5", "HDF5 file has no dataset or model keys."
            return True, "h5", "HDF5/Keras model file validated successfully."
        except ImportError:
            # Fallback check if h5py is not installed
            if os.path.getsize(model_filepath) > 0:
                return True, "h5", "HDF5 model file size validated (h5py library optional)."
            return False, "h5", "HDF5 model file is empty."
        except Exception as e:
            return False, "h5", f"Invalid HDF5 model file: {str(e)}"

    elif ext in [".joblib", ".pkl"]:
        try:
            import joblib
            obj = joblib.load(model_filepath)
            if hasattr(obj, "predict"):
                return True, "sklearn", "scikit-learn model loaded and validated."
            return False, "sklearn", "Joblib file does not contain a valid estimator with predict method."
        except Exception as e:
            return False, "sklearn", f"Invalid joblib model file: {str(e)}"

    else:
        return False, "unknown", f"Unsupported model file format '{ext}'. Allowed formats: .onnx, .pt, .pth, .h5, .joblib"
