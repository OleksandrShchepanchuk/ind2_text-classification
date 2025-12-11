import os
import torch
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    TRAIN_PATH = BASE_DIR / "data" / "train.csv"
    TEST_PATH = BASE_DIR / "data" / "test.csv"
    SAVE_DIR = BASE_DIR / "saved_models"
    
    MODEL_NAME = "deepset/gbert-base"
    MAX_LEN = 512
    NUM_CLASSES = 9
    DROPOUT_RATE = 0.1
    
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SEED = 42
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "German_News_Classification_PyTorch"

    os.makedirs(SAVE_DIR, exist_ok=True)