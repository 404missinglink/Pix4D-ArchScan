from huggingface_hub import snapshot_download
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def download_model():
    mistral_models_path = Path.home().joinpath('mistral_models', 'Pixtral')
    mistral_models_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id="mistralai/Pixtral-12B-2409", allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], local_dir=mistral_models_path)

def download_data():
    import dataset_tools as dtools
    dtools.download(dataset='FloodNet 2021: Track 2', dst_dir='~/dataset-ninja/')

download_data()