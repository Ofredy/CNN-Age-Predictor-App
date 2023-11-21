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
LEARNING_RATE = 1e-4
MN_V2_OUT_SIZE = 1280