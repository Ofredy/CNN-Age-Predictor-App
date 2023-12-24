# System imports
import os

# Library imports
import torch

# GUI Configs
LOGO_IMG_PATH = os.path.join("misc", "cougar_ai_logo.png")
FACE_DECTECTION_PATH = os.path.join('weights', 'haarcascade_frontalface_default.xml')
ONNX_PATH = os.path.join('weights','age_predictor.onnx')

# CNN CONFIGS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOBILENET_V3_OUT_CHANNELS = 960
MOBILENET_V3_AVG_POOL_OUT_SIZE = 3840
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
LAST_LAYER_TO_FREEZE = 171
AGE_PRED_WEIGHTS_PATH = os.path.join("training_summary")
TRAIN_AGE_PRED_ONNX_PATH = os.path.join("training_summary", "age_predictor.onnx")

# DATASET CONFIGS
BATCH_SIZE = 32
INPUT_CHANNELS = 3
IMG_SIZE = 244
PATH_TO_FOLDER = os.path.join("..", "..", "data", "fairface-img-margin025-trainval")
TRAIN_CSV = os.path.join(PATH_TO_FOLDER, "fairface-label-train.csv")
VAL_CSV = os.path.join(PATH_TO_FOLDER, "fairface-label-val.csv")

# AgePredictor training configs
LOSS_MAE_SUMMARY_PATH = os.path.join("training_summary", "loss_mae_summary.txt")