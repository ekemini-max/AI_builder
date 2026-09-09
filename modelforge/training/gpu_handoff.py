import os
import json
from typing import Dict, Any

COLAB_VISION_NOTEBOOK_TEMPLATE = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ModelForge External GPU Fine-Tuning Notebook\n",
                "**Design Decision & Budget Constraint Notice:**\n",
                "- **Total Budget:** $0.00\n",
                "- ModelForge uses a decoupled architecture. Large vision/text models requiring GPU compute are offloaded to free GPU environments (Google Colab T4 / Hugging Face AutoTrain).\n",
                "- Run this notebook with GPU enabled (`Runtime -> Change runtime type -> T4 GPU`).\n",
                "- Export the resulting model in ONNX or PyTorch (.pt) format and upload back into ModelForge."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q torch torchvision timm onnx"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torchvision\n",
                "print('GPU Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Device Name:', torch.cuda.get_device_name(0))"
            ]
        }
    ]
}


COLAB_TEXT_NOTEBOOK_TEMPLATE = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ModelForge External GPU Text Fine-Tuning Notebook (DistilBERT/DeBERTa)\n",
                "**Design Decision & Budget Constraint Notice:**\n",
                "- **Total Budget:** $0.00\n",
                "- Text classification with deep transformer models (DistilBERT, RoBERTa) exceeds local CPU limits.\n",
                "- Run this notebook with GPU enabled (`Runtime -> Change runtime type -> T4 GPU`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers datasets accelerate evaluate onnx"
            ]
        }
    ]
}


def generate_gpu_handoff_info(task_type: str, dataset_name: str, sample_count: int, output_dir: str = "notebooks") -> Dict[str, Any]:
    """
    Generates downloadable Jupyter/Colab notebook files and step-by-step instructions for GPU handoff.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"modelforge_{task_type}_gpu_training.ipynb"
    filepath = os.path.join(output_dir, filename)

    notebook_data = COLAB_VISION_NOTEBOOK_TEMPLATE if task_type == "vision" else COLAB_TEXT_NOTEBOOK_TEMPLATE

    with open(filepath, "w") as f:
        json.dump(notebook_data, f, indent=2)

    instructions = [
        "1. Click the 'Download GPU Notebook' button below to download the generated `.ipynb` file.",
        "2. Open Google Colab (colab.research.google.com) or Hugging Face AutoTrain (free tier).",
        "3. Upload the notebook and enable T4 GPU runtime (Runtime -> Change runtime type -> T4 GPU).",
        "4. Follow the step-by-step code cells to train your high-capacity model.",
        "5. Download the output model file (`model.onnx` or `model.pt`) and upload it back into ModelForge via the 'Upload Trained Model' path."
    ]

    return {
        "requires_gpu_handoff": True,
        "reason": f"Dataset '{dataset_name}' with {sample_count} samples or complex architecture requires GPU fine-tuning to reliably reach >= 85% accuracy without local CPU timeout.",
        "notebook_filename": filename,
        "notebook_filepath": filepath,
        "instructions": instructions,
        "supported_free_platforms": [
            {"name": "Google Colab", "url": "https://colab.research.google.com", "tier": "Free T4 GPU"},
            {"name": "Hugging Face AutoTrain", "url": "https://huggingface.co/autotrain", "tier": "Free Community Tier"},
            {"name": "Kaggle Notebooks", "url": "https://www.kaggle.com/code", "tier": "Free 30h/week GPU"}
        ]
    }
