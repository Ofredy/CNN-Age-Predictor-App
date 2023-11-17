# System Imports
import tkinter as tk
import tkinter.font

# External Imports
import cv2
from PIL import Image, ImageTk 

#Config Imports
from configs import *


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

        # Displaying button to take picture and execute code
        self.make_age_prediction_button = tk.Button(self.window, text=BUTTON_TEXT, width=BUTTON_WIDTH, 
                                                    height=BUTTON_HEIGHT, 
                                                    font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE),
                                                    bg=BUTTON_COLOR, fg="white",
                                                    command=self._predict_age_selected)
        self.make_age_prediction_button.pack(pady=PADDING_IN_BETWEEN_WIDGETS)

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

    def _display_webcam(self):

        # Capture the video frame by frame 
        _, frame = self.video_capture.read() 
    
        # Convert image from one color space to other 
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
    
        # Capture the latest frame and transform to image 
        captured_image = Image.fromarray(opencv_image) 
    
        # Convert captured image to photoimage 
        photo_image = ImageTk.PhotoImage(image=captured_image) 
    
        # Displaying photoimage in the label 
        self.cam_label.photo_image = photo_image 
    
        # Configure image in the label 
        self.cam_label.configure(image=photo_image) 
    
        if not self.predict_age:
            # Repeat the same process after every 10 seconds if make prediction button not selected yet
            self.cam_label.after(10, self._display_webcam) 
        else:
            print("THIS LOGIC WAS REACHED")

    def _predict_age_selected(self):
        self.predict_age = True


if __name__ == "__main__":
    
    age_predictor_gui = AgePredictorGUI()
