# Library imports
from torch.utils.data import DataLoader
import pandas as pd

# Our imports
from dataset import AgeDataset
from model import AgePredictor
from configs import BATCH_SIZE, TRAIN_CSV, VAL_CSV


train_dataframe = pd.read_csv(TRAIN_CSV)
val_dataframe = pd.read_csv(VAL_CSV)

train = AgeDataset(train_dataframe)
val = AgeDataset(val_dataframe)

train_data_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=train.collate_fn)
val_data_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=val.collate_fn)

age_predictor = AgePredictor()

age_predictor.train_age_predictor(train_data_loader, val_data_loader, continue_training=False)

age_predictor.save_training_results()

age_predictor.save_model_onnx()