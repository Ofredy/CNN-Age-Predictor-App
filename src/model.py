# Library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary

# Our imports 
from configs import MN_V2_OUT_SIZE, DEVICE, LEARNING_RATE


class AgePredictor(nn.Module):

    def __init__(self, train_data_loader, val_data_loader):
        
        super().__init__()

        self._define_age_predictor_model()

        self.age_criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

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

        return age_loss

    def train_age_predictor(self):
        
        train_losses, val_losses = [], []
    
        n_epochs = 10

        for epoch in range(n_epochs):

            epoch_train_loss, epoch_val_loss = 0, 0

            for _, data in enumerate(self.train_data_loader):
                loss = self._train_batch(data)
                epoch_train_loss += loss.item()

            for _, data in enumerate(self.val_data_loader):
                loss = self._val_batch(data)
                epoch_val_loss += loss.item()

        epoch_train_loss /= len(self.train_data_loader)
        epoch_val_loss /= len(self.val_data_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print("Epoch %d, train_loss: %.3f, val_loss: %.3f" % (epoch+1, epoch_train_loss, epoch_val_loss))
