# System imports
import time

# Library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2

# Our imports 
from dataset import AgeDataset
from configs import *


class AgePredictor(nn.Module):

    def __init__(self):
        
        super().__init__()

        self._define_age_predictor_model()

        self.age_criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.prediction_dataset = AgeDataset(None)

    def _define_age_predictor_model(self):

        self.mobile_net_v2_age_predictor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze parameters so we don't backprop through them
        for param in self.mobile_net_v2_age_predictor.parameters():
            param.requires_grad = False

        self.mobile_net_v2_age_predictor.classifier = nn.Sequential(
            nn.Linear(self.mobile_net_v2_age_predictor.last_channel, MN_V2_OUT_SIZE//2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(MN_V2_OUT_SIZE//2, MN_V2_OUT_SIZE//4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(MN_V2_OUT_SIZE//4,MN_V2_OUT_SIZE//8),
            nn.ReLU(),
            nn.Linear(MN_V2_OUT_SIZE//8, MN_V2_OUT_SIZE//16),
            nn.ReLU(),
            nn.Linear(MN_V2_OUT_SIZE//16, 1),
            nn.Sigmoid()
        )

        self.mobile_net_v2_age_predictor.to(DEVICE)

    def forward(self, input_image):

        # Inputting the input_image and getting our age prediction
        return self.mobile_net_v2_age_predictor(input_image)

    def _train_batch(self, data):

        self.mobile_net_v2_age_predictor.train()

        img, age = data

        self.optimizer.zero_grad()

        predicted_age = self.mobile_net_v2_age_predictor(img)

        age_loss = self.age_criterion(predicted_age.squeeze(), age)
        age_loss.backward()

        self.optimizer.step()

        return age_loss

    def _val_batch(self, data):
        
        self.mobile_net_v2_age_predictor.eval()

        img, age = data

        with torch.no_grad():
            predicted_age = self.mobile_net_v2_age_predictor(img)

        age_loss = self.age_criterion(predicted_age.squeeze(), age)

        age_mae = torch.abs(age-predicted_age).float().sum()

        return age_loss, age_mae

    def train_age_predictor(self, train_data_loader, val_data_loader):
        
        print("Began training age_predictor, num_epochs: %d" % (NUM_EPOCHS))

        start_time = time.time()

        self.train_losses, self.val_losses = [], []
        self.val_age_maes = []
    
        n_epochs = NUM_EPOCHS

        for epoch in range(n_epochs):

            epoch_train_loss, epoch_val_loss = 0, 0
            val_age_mae, ctr = 0, 0

            for _, data in enumerate(train_data_loader):
                loss = self._train_batch(data)
                epoch_train_loss += loss.item()

            for _, data in enumerate(val_data_loader):
                loss, age_mae = self._val_batch(data)
                epoch_val_loss += loss.item()
                val_age_mae += age_mae
                ctr += len(data[0])

            epoch_train_loss /= len(train_data_loader)
            epoch_val_loss /= len(val_data_loader)
            val_age_mae /= ctr

            self.train_losses.append(epoch_train_loss)
            self.val_losses.append(epoch_val_loss)
            self.val_age_maes.append(val_age_mae)

            time_elasped = time.time() - start_time

            epoch_age_pred_weight_path = os.path.join("training_summary", "{}age_predictor_weights.pt".format(epoch+1))

            torch.save(self.mobile_net_v2_age_predictor.state_dict(), epoch_age_pred_weight_path)

            print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start_time, (n_epochs-epoch)*(time_elasped/(epoch+1))))
            print("train_loss: %.3f, val_loss: %.3f, val_age_mae:%.3f" % (epoch_train_loss, epoch_val_loss, val_age_mae))

    def save_training_results(self):

        with open(LOSS_MAE_SUMMARY_PATH, "w") as output_file:

            for idx in range(len(self.train_losses)):
                output_file.write("%f, %f, %f\n" % (self.train_losses[idx], self.val_losses[idx], self.val_age_maes[idx]))

            output_file.close()

    def load_age_predictor_weights(self):

        self.mobile_net_v2_age_predictor.load_state_dict(torch.load(AGE_PRED_WEIGHTS_PATH))

    def predict_age(self, input_image_path):

        img = cv2.imread(input_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.prediction_dataset.preprocess_image(img).to(DEVICE)
        
        with torch.no_grad():
            self.mobile_net_v2_age_predictor.eval()
            predicted_age = self.mobile_net_v2_age_predictor(img)

        return int(predicted_age * 80)
