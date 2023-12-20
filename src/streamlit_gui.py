import cv2
import numpy as np
import streamlit as st


class AgePredictorGUI():

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def _extract_face(self):

        # Read image file buffer with cv2
        bytes_data = self.img_file_buffer.getvalue()
        self.img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert img to gray scale and detect faces
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 4)

    def _make_age_prediction(self):

        x, y, w, h = self.faces[0]

        self.img = self.img[ y:y+h, x:x+w ]
        
        cv2.imwrite("test1.jpg", self.img)

    def window(self):

        # Webcam display
        self.img_file_buffer = st.camera_input("Take A Picture to Make An Age Prediction")

        if self.img_file_buffer is not None:

            # Extract the face from the image
            self._extract_face()

            # Use extracted face to predict age
            self._make_age_prediction()

if __name__ == "__main__":

    age_predictor_gui = AgePredictorGUI()
    age_predictor_gui.window()