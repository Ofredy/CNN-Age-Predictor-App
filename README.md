# CNN-Age-Predictor-App

Welcome to the Cougar AI Spring 2024 Project!

In this project we will build a computer vision application capable of predicting the age of the user through a webcam. We will teach everything needed to understand the application from the ground up. Including convolutional neural networks, transfer learning, resnets, and general user interfaces!

![alt text](blob/app_picture.jpg?raw=true)

The goal of the workshop is to give our members a fundamental understanding of neural networks and computer vision. Allowing them to take on projects of their own!

The project will be taught in weekly workshops, learning the necessary skills piece by piece until the whole application is put together. The detailed outline of the project can be found as workshop/project_outline.pdf in this github repo. The high level project outline of the workshop is as follows:
  - Week 1: Intro to Machine Learning and Single Layer Networks
  - Week 2: Multilayer Networks
  - Week 3 & 4: Convolutional Neural Networks
  - Week 5: Transfer Learning, Resnets, Coding Standards, and GUIs
  - Week 6 & Beyond: Age Predictor App

The project was inspired by the book, Modern Computer Vision with PyTorch: https://www.amazon.com/Modern-Computer-Vision-PyTorch-applications/dp/1839213477. 

The dataset used for Age Predictor is the FairFace Datset. And the labels used to train the network, come from the book, Modern Computer Vision with PyTorch.
  - Fairface Dataset: https://github.com/joojs/fairface
  - Fairface Dataset Train & Val Labels, The labels are stored in a google drive created by the book's author. Refer to the github block on importing the data: https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter05/age_gender_prediction.ipynb

To run the age predictor:
  - Clone or download the repo.
  -  Create an python environment, and install the following libraries: PyTorch, Streamlit, OpenCV 
  -  Activate your python environment and cd into src. Run the command: "streamlit run main.py". The app will open in your default browser.
