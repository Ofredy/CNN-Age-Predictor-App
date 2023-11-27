# System Imports
import tkinter as tk
import tkinter.font

# Library Imports
import cv2
import numpy as np
import PIL
from PIL import Image, ImageTk 

# Our Imports
from age_predictor_cnn.model import AgePredictor
from age_predictor_cnn.configs import *


class AgePredictorGUI:

    def __init__(self):

        # Creating GUI window and initializing the window configs
        self.window = tk.Tk()
        self.window.title(WINDOW_NAME)
        
        # Initializing the make prediction bool
        self.predict_age = False

        # Screenshot capture
        self._webcam_init()
        self._display_webcam()

        # Displaying CougarAI logo once webcam has loaded
        self.window_logo = tk.PhotoImage(file=LOGO_IMG_PATH)
        self.window.iconphoto(False, self.window_logo)

        # Displaying picture taking instructions


        # Displaying button to take picture and execute code
        self.make_age_prediction_button = tk.Button(self.window, text=BUTTON_TEXT, width=BUTTON_WIDTH, 
                                                    height=BUTTON_HEIGHT, 
                                                    font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE),
                                                    bg=BUTTON_COLOR, fg="white",
                                                    command=self._predict_age_selected)
        self.make_age_prediction_button.pack(pady=PADDING_IN_BETWEEN_WIDGETS)

        self.age_predictor = AgePredictor(deployment=True)
        self.ages_predicted = 1

        # Displaying predicted age 

        # Reset button once, age is predicted

        # Commencing the GUI window loop
        self.window.mainloop()

    def _webcam_init(self):
        
        # Initializing web cam and its settings
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # Adding hitting esc as an option to exit the application
        self.window.bind('<Escape>', lambda e: self.window.quit())

        # Adding the cam to the window
        self.cam_label = tk.Label(self.window)
        self.cam_label.pack()

    def _add_boundary_box(self):

        rows, cols = self.captured_img.size

        self.midpoint_x = rows // 2
        self.midpoint_y = cols // 2 

        top_left_corner = (self.midpoint_x-BOUNDARY_BOX_DX_DY, self.midpoint_y-BOUNDARY_BOX_DX_DY)
        bottom_left_corner = (self.midpoint_x-BOUNDARY_BOX_DX_DY, self.midpoint_y+BOUNDARY_BOX_DX_DY)
        top_right_corner = (self.midpoint_x+BOUNDARY_BOX_DX_DY, self.midpoint_y-BOUNDARY_BOX_DX_DY)
        bottom_right_corner = (self.midpoint_x+BOUNDARY_BOX_DX_DY, self.midpoint_y+BOUNDARY_BOX_DX_DY)

        self.captured_img = np.array(self.captured_img)

        self.captured_img = cv2.line(self.captured_img, top_left_corner, top_right_corner, BOUNDARY_BOX_COLOR, BOUNDARY_BOX_THICKNESS)
        self.captured_img = cv2.line(self.captured_img, top_left_corner, bottom_left_corner, BOUNDARY_BOX_COLOR, BOUNDARY_BOX_THICKNESS)
        self.captured_img = cv2.line(self.captured_img, top_right_corner, bottom_right_corner, BOUNDARY_BOX_COLOR, BOUNDARY_BOX_THICKNESS)
        self.captured_img = cv2.line(self.captured_img, bottom_left_corner, bottom_right_corner, BOUNDARY_BOX_COLOR, BOUNDARY_BOX_THICKNESS)

    def _display_webcam(self):

        # Capture the video frame by frame 
        _, frame = self.video_capture.read() 
    
        # Convert image from one color space to other 
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
    
        # Capture the latest frame and transform to image 
        self.captured_img = Image.fromarray(opencv_image) 

        self._add_boundary_box()

        # Convert captured image to photoimage 
        photo_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(self.captured_img)) 
    
        # Displaying photoimage in the label 
        self.cam_label.photo_image = photo_image 
    
        # Configure image in the label 
        self.cam_label.configure(image=photo_image) 
    
        if not self.predict_age:
            # Repeat the same process after every 10 seconds if make prediction button not selected yet
            self.cam_label.after(10, self._display_webcam) 
        else:
            # Convert Imagetk to Image then to cv2 image
            photo_image = ImageTk.getimage(photo_image)
            opencv_img = cv2.cvtColor(np.array(photo_image), cv2.COLOR_RGB2BGR)
            opencv_img = opencv_img[self.midpoint_y-BOUNDARY_BOX_DX_DY:self.midpoint_y+BOUNDARY_BOX_DX_DY,
                                    self.midpoint_x-BOUNDARY_BOX_DX_DY:self.midpoint_x+BOUNDARY_BOX_DX_DY]

            output_img_path = os.path.join("predicted_images", str(self.ages_predicted) + ".jpg")
            cv2.imwrite(output_img_path, opencv_img)

            print(self.age_predictor.predict_age(output_img_path))

            print("THIS LOGIC WAS REACHED")

    def _predict_age_selected(self):
        self.predict_age = True


if __name__ == "__main__":
    
    age_predictor_gui = AgePredictorGUI()
