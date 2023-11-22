from torchsummary import summary

from model import AgePredictor


age_predictor = AgePredictor()

age_predictor.load_age_predictor_weights()

summary(age_predictor, (3, 244, 244), device='cuda')