# System imports
import os

# Library imports
import torch

# GUI Configs
LOGO_IMG_PATH = os.path.join("..", "misc", "cougar_ai_logo.gif")
WINDOW_NAME = "Cougar AI Age Predictor App"
PADDING_IN_BETWEEN_WIDGETS = 10
CAM_WIDTH, CAM_HEIGHT = 640, 640
BUTTON_TEXT = "Make Age Prediction!"
BUTTON_TEXT_SIZE = 15
BUTTON_COLOR = "#C8102E"
BUTTON_WIDTH, BUTTON_HEIGHT = 30, 5

# CNN CONFIGS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MOBILENET_V3_OUT_CHANNELS = 960
MOBILENET_V3_AVG_POOL_OUT_SIZE = 3840
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
LAST_LAYER_TO_FREEZE = 171
AGE_PRED_WEIGHTS_PATH = os.path.join("training_summary")

# DATASET CONFIGS
BATCH_SIZE = 32
IMG_SIZE = 244
PATH_TO_FOLDER = os.path.join("..", "..", "data", "fairface-img-margin025-trainval")
TRAIN_CSV = os.path.join(PATH_TO_FOLDER, "fairface-label-train.csv")
VAL_CSV = os.path.join(PATH_TO_FOLDER, "fairface-label-val.csv")

# AgePredictor training configs
LOSS_MAE_SUMMARY_PATH = os.path.join("training_summary", "loss_mae_summary.txt")