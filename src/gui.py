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
        self.instruction_label = tk.Label(self.window,
                                           text=INSTRUCTION_TEXT,
                                           fg=TEXT_COLOR,
                                           font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE))
        self.instruction_label.pack(pady=PADDING_IN_BETWEEN_WIDGETS)

        # Displaying button to take picture and execute code
        self.make_age_prediction_button = tk.Button(self.window, text=BUTTON_TEXT, width=BUTTON_WIDTH, 
                                                    height=BUTTON_HEIGHT, 
                                                    font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE),
                                                    bg=BUTTON_COLOR, fg="white",
                                                    command=self._predict_age_selected)
        self.make_age_prediction_button.pack(pady=PADDING_IN_BETWEEN_WIDGETS)

        # Initializing the AgePredictor network
        self.age_predictor = AgePredictor(deployment=True)
        self.num_ages_predicted = 0
        self.ages_predicted = []

        # Displaying predicted age 
        self.predict_age_label_text = PREDICT_AGE_LABEL_TEXT
        self.predict_age_label = tk.Label(self.window,
                                           text=self.predict_age_label_text,
                                           fg=TEXT_COLOR,
                                           font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE))
        self.predict_age_label.pack(pady=PADDING_IN_BETWEEN_WIDGETS)

        # Adding window termination protocol
        self.window.protocol('WM_DELETE_WINDOW', self._age_predictor_termination)

        # Commencing the GUI window loop
        self.window.mainloop()

    def _webcam_init(self):
        
        # Initializing web cam and its settings
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

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

            self.num_ages_predicted += 1
        
            output_img_path = os.path.join("predicted_images", str(self.num_ages_predicted) + ".jpg")
            cv2.imwrite(output_img_path, opencv_img)

            self.ages_predicted.append(self.age_predictor.predict_age(output_img_path))

            self.predict_age_label_text = self.predict_age_label_text + str(self.ages_predicted[-1]) 
            self.predict_age_label.config(text=self.predict_age_label_text)

            self._display_reset_button()

    def _predict_age_selected(self):

        self.predict_age = True

    def _display_reset_button(self):

        self.reset_button = tk.Button(self.window, text=RESET_BUTTON_TEXT, width=RESET_BUTTON_WIDTH, 
                                                    height=RESET_BUTTON_HEIGHT, 
                                                    font=tkinter.font.Font(family='Helvetica', size=BUTTON_TEXT_SIZE),
                                                    bg=BUTTON_COLOR, fg="white",
                                                    command=self._reset_age_predictor)
        self.reset_button.pack(pady=PADDING_IN_BETWEEN_WIDGETS)    

    def _reset_age_predictor(self):

        # Set predict_age bool to false
        self.predict_age = False

        # Reset predict_age_label_text
        self.predict_age_label_text = PREDICT_AGE_LABEL_TEXT 
        self.predict_age_label.config(text=self.predict_age_label_text)

        # Remove reset button
        self.reset_button.pack_forget()

        # Begin displaying webcam again
        self._display_webcam()

    def _age_predictor_termination(self):

        if self.num_ages_predicted != 0:

            with open(OUTPUT_PREDICTIONS_TXT_PATH, "w") as output_file:

                for idx, age_predicted in enumerate(self.ages_predicted):

                    output_file.write("Image: %d, age_predicted: %d\n" % (idx+1, age_predicted))

                output_file.close()

        self.window.destroy()
