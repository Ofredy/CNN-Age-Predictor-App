# Library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchsummary import summary

# Our imports 
from configs import MN_V2_OUT_SIZE, DEVICE, LEARNING_RATE


class AgePredictor(nn.Module):

    def __init__(self):
        
        super().__init__()

        self._define_age_predictor_model()

        self.age_criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

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

    def train_age_predictor(self):
        
        pass
