# Library imports
import torch
from torchvision import transforms
import cv2
import numpy as np
import streamlit as st
import onnx
import onnxruntime

# Our imports
from age_predictor_cnn.configs import FACE_DECTECTION_PATH, ONNX_PATH, IMG_SIZE


class AgePredictorGUI():

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(FACE_DECTECTION_PATH)
        self.onnx_path = ONNX_PATH
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def _extract_face(self):

        # Read image file buffer with cv2
        bytes_data = self.img_file_buffer.getvalue()
        self.img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert img to gray scale and detect faces
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 4)

    def _preprocess_img(self):

        self.img = cv2.resize(self.img, (IMG_SIZE, IMG_SIZE))
        self.img = torch.tensor(self.img).permute(2, 0, 1)
        self.img = self.normalize(self.img/255)
        self.img = self.img.unsqueeze(0)

        return self.img[None]
    
    def _make_age_prediction(self):

        if len(self.faces) == 0:

            self.age_prediction = None
            return
        
        x, y, w, h = self.faces[0]

        self.img = self.img[ y:y+h, x:x+w ]

        self._preprocess_img()   

        # Loading in the model
        age_predictor_onnx = onnx.load(self.onnx_path)
        onnx.checker.check_model(age_predictor_onnx)

        ort_session = onnxruntime.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(self.img)}
        ort_outs = ort_session.run(None, ort_inputs)

        self.age_prediction = int(80*ort_outs[0][0][0])

    def _display_predicted_age(self):
        
        if self.age_prediction == None:
            st.text("Face not detected")
        else:
            st.text("Age Predicted: %d" % (self.age_prediction))


    def window(self):

        # Webcam display
        self.img_file_buffer = st.camera_input("Take A Picture to Make An Age Prediction")

        if self.img_file_buffer is not None:

            # Extract the face from the image
            self._extract_face()

            # Use extracted face to predict age
            self._make_age_prediction()

            # Display age_prediction
            self._display_predicted_age()
